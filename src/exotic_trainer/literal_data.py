from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_probe import inspect_dataset, normalize_record
from .paths import project_root
from .preflight import example_hash
from .registry import Registry
from .sealed import list_sealed_suites, register_sealed_suite
from .tool_schema import KNOWN_TOOLS, STANDARD_TOOLS, schema_conditioned_system_prompt

TOOLS = tuple(sorted(STANDARD_TOOLS))
TRAIN_ROWS = 5_200
EVAL_ROWS = 520
TRAIN_PATH = project_root() / "downloads" / "generated" / "literal-lock-train-v2-5200.jsonl"
DEV_PATH = project_root() / "examples" / "dev-literal-lock-v2-520.jsonl"
SEALED_PATH = project_root() / "examples" / "sealed-literal-lock-v5-520.jsonl"

POLICY = (
    "You are a precise coding assistant with declared tools. Call exactly one tool only "
    "when the user authorizes a concrete external action and supplies all required values. "
    "Preserve literal argument values exactly. Answer directly otherwise."
)


def _assistant_tool(name: str, arguments: dict[str, Any], case_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": f"call_{case_id}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def _arguments(tool: str, split: str, index: int) -> dict[str, Any]:
    tag = f"{split}-{index:04d}"
    root = f"workspaces/{split}/batch-{index % 41:02d}/{tag}"
    marker = f"LITERAL_{split.upper()}_{index:04d}"
    if tool == "read":
        return {"path": f"{root}/manifest.toml", "offset": index % 31, "limit": 17 + index % 83}
    if tool == "write":
        return {"path": f"{root}/state.ini", "content": f"case={tag}\nmarker={marker}\n"}
    if tool == "edit":
        return {
            "path": f"{root}/settings.ini",
            "oldText": f"state=pending-{tag}",
            "newText": f"state=verified-{tag}",
            "replaceAll": False,
        }
    if tool == "bash":
        return {
            "command": f"python -m pytest qa/{split}/test_{index:04d}.py -k case_{index} -q",
            "description": f"Verify literal case {tag}",
            "timeout": 40 + index % 161,
        }
    if tool == "apply_patch":
        return {"patchText": (
            "*** Begin Patch\n"
            f"*** Update File: {root}/feature.py\n"
            "@@\n"
            f'-FLAG = "pending-{tag}"\n'
            f'+FLAG = "verified-{tag}"\n'
            "*** End Patch\n"
        )}
    if tool == "glob":
        return {"pattern": f"**/{tag}_*.py", "path": root}
    if tool == "grep":
        return {"pattern": marker, "path": root, "include": "*.py"}
    if tool == "lsp":
        operation = ("documentSymbol", "goToDefinition", "workspaceSymbol")[index % 3]
        if operation == "workspaceSymbol":
            return {"operation": operation, "symbol": f"literal_{split}_{index:04d}"}
        result: dict[str, Any] = {"operation": operation, "filePath": f"{root}/router.py"}
        if operation == "goToDefinition":
            result.update(line=2 + index % 97, character=1 + index % 23)
        return result
    if tool == "skill":
        return {"name": f"literal-audit-{split}-{index:04d}"}
    if tool == "task":
        return {
            "description": f"Audit literal case {tag}",
            "prompt": f"Inspect {root} for marker {marker}; report only and do not edit files.",
            "subagent_type": "general",
        }
    if tool == "todowrite":
        return {"todos": [
            {"content": f"inspect {tag}", "status": "in_progress"},
            {"content": f"verify {marker}", "status": "pending"},
            {"content": f"record {tag}", "status": "pending"},
        ]}
    if tool == "webfetch":
        return {
            "url": f"https://{split}.example/literal/{tag}",
            "prompt": f"Extract the compatibility note {marker}.",
        }
    if tool == "websearch":
        return {"query": f"{tag} structured tool contract revision {2030 + index % 7}"}
    raise KeyError(tool)


def _tool_prompt(tool: str, arguments: dict[str, Any], split: str, index: int) -> str:
    block = json.dumps(arguments, ensure_ascii=False, indent=2)
    wrappers = (
        "Perform one authorized action now.",
        "Use the declared menu for this concrete operation.",
        "Do not explain; carry out exactly the specified operation.",
        "All required values are supplied below. Execute once.",
        "For this disposable workspace, perform the following action.",
    )
    return (
        f"{wrappers[(index + len(tool)) % len(wrappers)]}\n"
        f"Tool to use: `{tool}`\n"
        "Copy this complete argument object exactly; do not infer or paraphrase fields:\n"
        f"```json\n{block}\n```\n"
        f"Request id: `{split}-request-{index:04d}`."
    )


def _direct_prompt(split: str, index: int) -> tuple[str, str, str]:
    tag = f"{split}-direct-{index:04d}"
    path = f"workspaces/{split}/{tag}/router.py"
    command = f"python -m pytest qa/{split}/test_{index:04d}.py -q"
    kind = index % 5
    if kind == 0:
        return (
            f"Define tool abstention for glossary id `{tag}`. Answer directly; no action.",
            f"Tool abstention means answering directly when no external action is required ({tag}).",
            "clean-concept",
        )
    if kind == 1:
        return (
            f"Do not read `{path}`. Explain in one sentence what a read tool normally does.",
            f"A read tool retrieves file content without intentionally modifying it ({tag}).",
            "negated-action",
        )
    if kind == 2:
        return (
            f"The text `{command}` is an example command; quote its purpose without running it.",
            f"It is an example of invoking a focused test file without executing it here ({tag}).",
            "quoted-operation",
        )
    if kind == 3:
        return (
            f"Hypothetically, which tool category would inspect `{path}`? Do not call it.",
            f"A read-only file inspection tool would be the relevant category ({tag}).",
            "hypothetical-tool",
        )
    return (
        (
            f"Improve `{path}` but no change, file content, or acceptance criterion is supplied. "
            "State what is missing without using a tool."
        ),
        f"The requested change and acceptance criterion are missing, so no safe action is defined ({tag}).",
        "underspecified-action",
    )


def _row(
    split: str,
    case_id: str,
    user: str,
    assistant: dict[str, Any],
    tool: str | None,
    stratum: str,
) -> dict[str, Any]:
    metadata = {
        "case_id": case_id,
        "split": split,
        "stratum": stratum,
        "decision": "tool" if tool else "direct",
        "tool": tool,
        "schema_version": 2,
        "all_expected_values_visible": True,
    }
    key = "training_metadata" if split == "train" else "evaluation_metadata"
    return {
        key: metadata,
        "messages": [
            {"role": "system", "content": schema_conditioned_system_prompt(POLICY, KNOWN_TOOLS)},
            {"role": "user", "content": user},
            assistant,
        ],
    }


def build_rows(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "dev", "sealed"}:
        raise ValueError(f"unsupported LiteralLock split: {split}")
    per_tool = 200 if split == "train" else 20
    direct_count = 2_600 if split == "train" else 260
    rows: list[dict[str, Any]] = []
    split_offset = {"train": 0, "dev": 10_000, "sealed": 20_000}[split]
    for tool_index, tool in enumerate(TOOLS):
        for local_index in range(per_tool):
            index = split_offset + tool_index * per_tool + local_index
            arguments = _arguments(tool, split, index)
            case_id = f"{split}-literal-{tool}-{local_index:04d}"
            rows.append(_row(
                split,
                case_id,
                _tool_prompt(tool, arguments, split, index),
                _assistant_tool(tool, arguments, case_id),
                tool,
                f"tool-{tool}",
            ))
    for local_index in range(direct_count):
        index = split_offset + local_index
        prompt, answer, stratum = _direct_prompt(split, index)
        rows.append(_row(
            split,
            f"{split}-direct-{local_index:04d}",
            prompt,
            {"role": "assistant", "content": answer},
            None,
            stratum,
        ))
    random.Random({"train": 2026081201, "dev": 2026081202, "sealed": 2026081203}[split]).shuffle(rows)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    if not path.exists() or path.read_text(encoding="utf-8") != payload:
        path.write_text(payload, encoding="utf-8")


def _signatures(rows: list[dict[str, Any]]) -> set[str]:
    return {
        example_hash(sample)
        for row in rows
        for sample in normalize_record(row)
    }


def audit_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    signatures = {name: _signatures(rows) for name, rows in splits.items()}
    problems: list[str] = []
    for name, rows in splits.items():
        expected = TRAIN_ROWS if name == "train" else EVAL_ROWS
        metadata_key = "training_metadata" if name == "train" else "evaluation_metadata"
        decisions = Counter(row[metadata_key]["decision"] for row in rows)
        tools = Counter(
            row[metadata_key]["tool"]
            for row in rows
            if row[metadata_key]["tool"] is not None
        )
        per_tool = 200 if name == "train" else 20
        schema_conditioned_rows = sum(
            "List of tools:" in str(row["messages"][0].get("content") or "")
            for row in rows
        )
        grounded_tool_rows = 0
        for row in rows:
            metadata = row[metadata_key]
            if metadata["decision"] != "tool":
                continue
            call = row["messages"][-1]["tool_calls"][0]["function"]
            arguments = json.loads(call["arguments"])
            visible_block = json.dumps(arguments, ensure_ascii=False, indent=2)
            grounded_tool_rows += int(visible_block in row["messages"][1]["content"])
        reports[name] = {
            "rows": len(rows),
            "unique_normalized_samples": len(signatures[name]),
            "decisions": dict(decisions),
            "tool_distribution": dict(sorted(tools.items())),
            "schema_conditioned_rows": schema_conditioned_rows,
            "grounded_tool_rows": grounded_tool_rows,
        }
        if len(rows) != expected or len(signatures[name]) != expected:
            problems.append(f"{name}: expected {expected} unique rows")
        if set(tools) != set(TOOLS) or set(tools.values()) != {per_tool}:
            problems.append(f"{name}: tool distribution is not uniform across 13 tools")
        if decisions["tool"] != expected // 2 or decisions["direct"] != expected // 2:
            problems.append(f"{name}: tool/direct split is not 50/50")
        if schema_conditioned_rows != expected:
            problems.append(f"{name}: not every prompt contains the locked tool menu")
        if grounded_tool_rows != expected // 2:
            problems.append(f"{name}: not every tool target is shown literally in its prompt")
    for left, right in (("train", "dev"), ("train", "sealed"), ("dev", "sealed")):
        overlap = len(signatures[left] & signatures[right])
        reports[f"{left}_{right}_overlap"] = overlap
        if overlap:
            problems.append(f"{left}/{right}: {overlap} normalized samples overlap")
    return {
        "status": "blocked" if problems else "ready",
        "schema_version": 2,
        "all_expected_values_visible": True,
        "reports": reports,
        "problems": problems,
        "fingerprint": hashlib.sha256(
            json.dumps(reports, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def ensure_literal_lock_data() -> dict[str, Any]:
    splits = {name: build_rows(name) for name in ("train", "dev", "sealed")}
    audit = audit_splits(splits)
    if audit["status"] != "ready":
        raise ValueError(f"LiteralLock dataset audit failed: {audit['problems']}")
    paths = {"train": TRAIN_PATH, "dev": DEV_PATH, "sealed": SEALED_PATH}
    for name, path in paths.items():
        _write_jsonl(path, splits[name])
    registry = Registry()
    names = {
        "train": "LiteralLock train v2 — 5200 balanced — 13 tools",
        "dev": "DEV LiteralLock v2 — 520 balanced — 13 tools",
    }
    dataset_ids = {
        name: registry.upsert_dataset(inspect_dataset(str(paths[name]), display_name))
        for name, display_name in names.items()
    }
    resolved_sealed = str(SEALED_PATH.resolve())
    suite = next(
        (item for item in list_sealed_suites() if str(Path(item["source"]).resolve()) == resolved_sealed),
        None,
    )
    if suite is None:
        suite = register_sealed_suite(
            resolved_sealed,
            "FINAL LiteralLock v5 — 520 balanced — 13 tools",
            max_rows=EVAL_ROWS,
        )
    return {
        **audit,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "dataset_ids": dataset_ids,
        "suite_id": suite["id"],
        "suite_name": suite["name"],
    }
