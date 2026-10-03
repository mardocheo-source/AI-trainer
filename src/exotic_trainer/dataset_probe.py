from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import DatasetProbe

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]

TOOL_ALIASES = {
    "exec_command": "bash",
    "run_command": "bash",
    "shell": "bash",
    "terminal": "bash",
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
}

ARGUMENT_ALIASES = {
    "cmd": "command",
    "file_path": "path",
    "filePath": "path",
    "old_string": "oldText",
    "oldString": "oldText",
    "new_string": "newText",
    "newString": "newText",
}


def _canonical_argument_name(tool_name: str, argument_name: str) -> str:
    # LSP's native contract intentionally uses filePath. The old global alias
    # silently rewrote it to path, creating a train/evaluation schema mismatch.
    if tool_name == "lsp" and argument_name in {"file_path", "filePath", "path"}:
        return "filePath"
    return ARGUMENT_ALIASES.get(argument_name, argument_name)


NATIVE_TOOL_BLOCK_PATTERN = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)
NATIVE_CALL_PATTERN = re.compile(r"(?:\[|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _iter_json_samples(path: Path, limit: int = 100) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from (item for item in value[:limit] if isinstance(item, dict))
        elif isinstance(value, dict):
            rows = value.get("data")
            if isinstance(rows, list):
                yield from (item for item in rows[:limit] if isinstance(item, dict))
            else:
                yield value


def _local_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    targets = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for item in targets[:10_000]:
        stat = item.stat()
        digest.update(str(item).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def inspect_dataset(source: str, name: str | None = None) -> DatasetProbe:
    if source.startswith("hf-file:") and "::" in source:
        repository, filename = source.removeprefix("hf-file:").split("::", 1)
        return DatasetProbe(
            name=name or Path(filename).stem,
            source=source,
            source_type="huggingface",
            format=Path(filename).suffix.lower().lstrip(".") or "huggingface-file",
            fingerprint=hashlib.sha256(f"{repository}::{filename}".encode()).hexdigest(),
        )
    candidate = Path(source).expanduser()
    if not candidate.exists():
        normalized = source.removeprefix("hf:")
        if "/" not in normalized:
            raise FileNotFoundError(f"local dataset not found and not a Hugging Face id: {source}")
        return DatasetProbe(
            name=name or normalized.rsplit("/", 1)[-1],
            source=normalized,
            source_type="huggingface",
            format="huggingface",
            fingerprint=hashlib.sha256(normalized.encode()).hexdigest(),
        )

    path = candidate.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    size_bytes = sum(item.stat().st_size for item in files)
    json_files = [item for item in files if item.suffix.lower() in {".json", ".jsonl"}]
    sample_columns: set[str] = set()
    secret_hits = 0
    row_count: int | None = 0 if json_files else None
    for item in json_files[:1000]:
        if item.suffix.lower() == ".jsonl":
            with item.open("rb") as handle:
                count = sum(1 for _ in handle)
            row_count = (row_count or 0) + count
        for row in _iter_json_samples(item):
            sample_columns.update(row.keys())
            serialized = json.dumps(row, ensure_ascii=False)
            secret_hits += sum(bool(pattern.search(serialized)) for pattern in SECRET_PATTERNS)
    format_name = path.suffix.lower().lstrip(".") if path.is_file() else "directory"
    return DatasetProbe(
        name=name or path.stem,
        source=str(path),
        source_type="local",
        format=format_name,
        size_bytes=size_bytes,
        row_count=row_count,
        sample_columns=sorted(sample_columns),
        secret_hits=secret_hits,
        fingerprint=_local_fingerprint(path),
    )


def normalize_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a conversation into prompt/completion examples for every assistant turn."""
    messages = record.get("messages")
    if isinstance(messages, list):
        normalized_messages = []
        output = []
        for message in messages:
            if not isinstance(message, dict) or "role" not in message:
                continue
            content = message.get("content", "") or ""
            if str(message["role"]) == "assistant" and message.get("tool_calls"):
                content = _format_tool_calls(message["tool_calls"], prefix=str(content))
            clean = {"role": str(message["role"]), "content": content}
            if clean["role"] == "assistant" and normalized_messages:
                output.append({"prompt": list(normalized_messages), "completion": [clean]})
            normalized_messages.append(clean)
        return output
    instruction = record.get("instruction") or record.get("question")
    answer = record.get("output") or record.get("answer") or record.get("reference_answer")
    if instruction is not None and answer is not None:
        if record.get("reference_answer") is not None:
            try:
                parsed_answer = ast.literal_eval(str(answer))
            except (SyntaxError, ValueError):
                parsed_answer = None
            if isinstance(parsed_answer, dict) and parsed_answer.get("name"):
                answer = _format_tool_calls(
                    [
                        {
                            "name": parsed_answer["name"],
                            "arguments": parsed_answer.get("arguments", {}),
                        }
                    ]
                )
        return [
            {
                "prompt": [{"role": "user", "content": str(instruction)}],
                "completion": [{"role": "assistant", "content": str(answer)}],
            }
        ]
    if "prompt" in record and "completion" in record:
        prompt = record["prompt"]
        completion = record["completion"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if isinstance(completion, str):
            completion = [{"role": "assistant", "content": completion}]
        return [{"prompt": prompt, "completion": completion}]
    return []


def _format_tool_calls(tool_calls: Any, prefix: str = "") -> str:
    if not isinstance(tool_calls, list):
        return prefix
    rendered = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function", call)
        if not isinstance(function, dict) or not function.get("name"):
            continue
        name = TOOL_ALIASES.get(str(function["name"]), str(function["name"]))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        arguments = {
            _canonical_argument_name(name, str(key)): value for key, value in arguments.items()
        }
        params = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        rendered.append(f"{name}({params})")
    if not rendered:
        return prefix
    native = f"<|tool_call_start|>[{', '.join(rendered)}]<|tool_call_end|>"
    return f"{prefix}{native}"


def tool_names_in_example(example: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for field in ("prompt", "completion"):
        messages = example.get(field, [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if isinstance(message, dict):
                content = str(message.get("content", ""))
                for block in NATIVE_TOOL_BLOCK_PATTERN.findall(content):
                    names.update(NATIVE_CALL_PATTERN.findall(block))
    return names
