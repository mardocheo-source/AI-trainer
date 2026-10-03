from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_probe import inspect_dataset, normalize_record
from .paths import project_root
from .registry import Registry
from .tool_schema import KNOWN_TOOLS, STANDARD_TOOLS, schema_conditioned_system_prompt

ROUTING_SUPPLEMENT_ROWS = 5_200
ROUTING_SUPPLEMENT_NAME = "Routing supplement v1 — 5200 balanced — 13 tools"
ROUTING_SUPPLEMENT_PATH = (
    project_root() / "downloads" / "generated" / "routing-supplement-v1-5200.jsonl"
)

TOOLS = tuple(sorted(STANDARD_TOOLS))
DIRECT_STRATA = (
    "clean-concept",
    "negated-action",
    "quoted-operation",
    "hypothetical-tool",
    "underspecified-action",
)

NEUTRAL_POLICY = "You are a precise coding assistant. Follow the user's request accurately."
ROUTING_POLICY = (
    "You are a coding assistant with declared tools. Call a tool only for an authorized "
    "external action with sufficient arguments. Answer directly for conceptual, quoted, "
    "hypothetical, negated, or underspecified requests. Never invent missing arguments."
)

TOOL_REQUEST_STYLES = (
    "Perform this fully specified action now: {request}",
    "Use the most appropriate declared tool to {request}",
    "For the disposable checkout, {request}",
    "Complete this bounded operation and then stop: {request}",
    "The maintainer authorizes you to {request}",
    "Act on the repository rather than explaining how to {request}",
    "Carry out exactly one operation: {request}",
    "For this diagnostic run, {request}",
    "All required details are present; {request}",
    "Proceed now and {request}",
    "The next concrete coding-agent step is to {request}",
    "Use a declared capability to {request}",
    "Execute only the following request: {request}",
    "In the current project workspace, {request}",
    "Take the requested external action: {request}",
    "Resolve this operational request by using one tool: {request}",
)


def _assistant_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_train_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _tool_request(name: str, index: int) -> tuple[str, dict[str, Any]]:
    code = f"traincase-{index:04d}"
    bucket = f"batch-{index % 37:02d}"
    root = f"workspaces/training/{bucket}/{code}"
    token = f"ROUTE_TRAIN_{index:04d}"
    if name == "read":
        arguments: dict[str, Any] = {"path": f"{root}/manifest-{index:04d}.toml"}
        if index % 3:
            arguments.update({"offset": index % 41, "limit": 12 + index % 73})
        return f"read `{arguments['path']}` without changing it.", arguments
    if name == "write":
        content = f"case={code}\nstatus=screen-{index:04d}\n"
        return (
            f"write `{root}/screen-{index:04d}.ini` with the supplied two lines and final newline.",
            {"path": f"{root}/screen-{index:04d}.ini", "content": content},
        )
    if name == "edit":
        return (
            (
                f"in `{root}/settings.ini`, replace `phase=pending-{index:04d}` with "
                f"`phase=verified-{index:04d}` exactly once."
            ),
            {
                "path": f"{root}/settings.ini",
                "oldText": f"phase=pending-{index:04d}",
                "newText": f"phase=verified-{index:04d}",
                "replaceAll": False,
            },
        )
    if name == "bash":
        command = f"python -m pytest qa/generated/test_route_{index:04d}.py -k case_{index} -q"
        return (
            f"run exactly `{command}`.",
            {
                "command": command,
                "description": f"Check generated route case {index:04d}",
                "timeout": 45 + index % 151,
            },
        )
    if name == "apply_patch":
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {root}/feature.py\n"
            "@@\n"
            f'-{token} = "pending-{index:04d}"\n'
            f'+{token} = "verified-{index:04d}"\n'
            "*** End Patch\n"
        )
        return f"apply the supplied change to `{root}/feature.py`.", {"patchText": patch}
    if name == "glob":
        arguments = {"pattern": f"**/route_{index:04d}_*.py", "path": root}
        return (
            f"find paths below `{root}` matching `**/route_{index:04d}_*.py`.",
            arguments,
        )
    if name == "grep":
        arguments = {"pattern": token, "path": root, "include": "*.py"}
        return f"search Python files below `{root}` for `{token}`.", arguments
    if name == "lsp":
        operation = ("documentSymbol", "goToDefinition", "workspaceSymbol")[index % 3]
        if operation == "workspaceSymbol":
            arguments = {"operation": operation, "symbol": f"route_train_{index:04d}"}
            return f"locate workspace symbol `route_train_{index:04d}`.", arguments
        arguments = {"operation": operation, "filePath": f"{root}/router.py"}
        if operation == "goToDefinition":
            arguments.update({"line": 3 + index % 97, "character": 1 + index % 19})
        location = (
            f" at line {arguments['line']}, character {arguments['character']}"
            if operation == "goToDefinition"
            else ""
        )
        return f"request `{operation}` for `{root}/router.py`{location}.", arguments
    if name == "skill":
        skill_name = f"training-route-audit-{index:04d}"
        return f"load specialist skill `{skill_name}`.", {"name": skill_name}
    if name == "task":
        arguments = {
            "description": f"Inspect training case {index:04d}",
            "prompt": (
                f"Inspect {root}, identify the routing defect tagged {token}, and report it "
                "without modifying files."
            ),
            "subagent_type": "general",
        }
        return f"delegate a read-only audit of `{root}`.", arguments
    if name == "todowrite":
        todos = [
            {"content": f"inspect training case {index:04d}", "status": "in_progress"},
            {"content": f"verify marker {token}", "status": "pending"},
            {"content": f"record result {index:04d}", "status": "pending"},
        ]
        return f"replace the task list with the three supplied steps for `{code}`.", {
            "todos": todos
        }
    if name == "webfetch":
        url = f"https://training.example/route-cases/{bucket}/{code}"
        return (
            f"fetch `{url}` and extract compatibility note `{token}`.",
            {"url": url, "prompt": f"Extract compatibility note {token}."},
        )
    if name == "websearch":
        query = f"{code} structured tool routing contract revision {2028 + index % 7}"
        return f"search the web for `{query}`.", {"query": query}
    raise KeyError(name)


def _direct_case(stratum: str, index: int) -> tuple[str, str]:
    code = f"train-direct-{index:04d}"
    path = f"sandbox/training/{code}/router.py"
    command = f"python -m pytest qa/generated/test_direct_{index:04d}.py -q"
    if stratum == "clean-concept":
        concepts = (
            ("tool routing", "chooses between a direct answer and an external operation"),
            ("call serialization", "encodes a tool name and arguments in a structured form"),
            ("argument validation", "checks names, types, and required values before execution"),
            ("tool abstention", "avoids an external action when a direct answer is sufficient"),
            ("bounded delegation", "assigns a narrow objective with an explicit stopping point"),
            ("exact replacement", "changes only text that matches a supplied source fragment"),
            ("schema conditioning", "shows a model the callable contracts available at inference"),
            ("read-only inspection", "observes project state without intentionally modifying it"),
        )
        concept, definition = concepts[index % len(concepts)]
        return (
            f"For glossary entry `{code}`, define `{concept}` without using a tool.",
            f"For `{code}`, {concept} {definition}.",
        )
    if stratum == "negated-action":
        return (
            f"Do not run `{command}`. Explain what the `-q` flag would change for case `{code}`.",
            f"For `{code}`, `-q` would reduce the amount of test-runner output.",
        )
    if stratum == "quoted-operation":
        return (
            f"Paraphrase the quoted instruction “read `{path}`”; do not access the file.",
            f"The quotation asks a future operator to inspect `{path}` without changing it.",
        )
    if stratum == "hypothetical-tool":
        return (
            (
                f"Hypothetically, what would searching `{path}` for `CASE_{index:04d}` "
                "return? Do not perform a search."
            ),
            f"It would return locations and context containing `CASE_{index:04d}` if any existed.",
        )
    if stratum == "underspecified-action":
        return (
            f"Please fix the `{code}` project configuration.",
            f"Which file, setting, current value, and desired value should I change for `{code}`?",
        )
    raise KeyError(stratum)


def _conditioned_system(condition: str) -> str:
    policy = ROUTING_POLICY if condition == "explicit-routing-policy" else NEUTRAL_POLICY
    return schema_conditioned_system_prompt(policy, KNOWN_TOOLS)


def _row(
    *,
    case_id: str,
    condition: str,
    stratum: str,
    user: str,
    assistant: dict[str, Any],
    decision: str,
    tool: str | None,
) -> dict[str, Any]:
    return {
        "training_metadata": {
            "case_id": case_id,
            "condition": condition,
            "stratum": stratum,
            "decision": decision,
            "tool": tool,
            "schema_version": 1,
        },
        "messages": [
            {"role": "system", "content": _conditioned_system(condition)},
            {"role": "user", "content": user},
            assistant,
        ],
    }


def build_routing_supplement_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool_index, tool in enumerate(TOOLS):
        for index in range(200):
            request, arguments = _tool_request(tool, index)
            condition = "explicit-routing-policy" if (index + tool_index) % 2 else "neutral"
            rows.append(
                _row(
                    case_id=f"train-tool-{tool}-{index:04d}",
                    condition=condition,
                    stratum=f"tool-{tool}",
                    user=TOOL_REQUEST_STYLES[index % len(TOOL_REQUEST_STYLES)].format(
                        request=request
                    ),
                    assistant=_assistant_tool(tool, arguments),
                    decision="tool",
                    tool=tool,
                )
            )
    for stratum_index, stratum in enumerate(DIRECT_STRATA):
        for index in range(520):
            prompt, answer = _direct_case(stratum, index)
            condition = "explicit-routing-policy" if (index + stratum_index) % 2 else "neutral"
            rows.append(
                _row(
                    case_id=f"train-direct-{stratum}-{index:04d}",
                    condition=condition,
                    stratum=stratum,
                    user=prompt,
                    assistant={"role": "assistant", "content": answer},
                    decision="direct",
                    tool=None,
                )
            )
    random.Random(20260812).shuffle(rows)
    return rows


def _normalized_signatures(rows: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for row in rows:
        for sample in normalize_record(row):
            canonical = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            signatures.add(hashlib.sha256(canonical.encode()).hexdigest())
    return signatures


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def audit_routing_supplement(
    rows: list[dict[str, Any]], dev_source: str | Path | None = None
) -> dict[str, Any]:
    decisions = Counter(row["training_metadata"]["decision"] for row in rows)
    conditions = Counter(row["training_metadata"]["condition"] for row in rows)
    strata = Counter(row["training_metadata"]["stratum"] for row in rows)
    tools = Counter(
        row["training_metadata"]["tool"]
        for row in rows
        if row["training_metadata"]["tool"] is not None
    )
    case_ids = {row["training_metadata"]["case_id"] for row in rows}
    normalized = _normalized_signatures(rows)
    system_conditioned = sum(
        bool(row.get("messages"))
        and str(row["messages"][0].get("role")) == "system"
        and "List of tools: [" in str(row["messages"][0].get("content"))
        for row in rows
    )
    dev_rows = _read_jsonl(Path(dev_source).expanduser().resolve()) if dev_source else []
    dev_signatures = _normalized_signatures(dev_rows)
    dev_case_ids = {
        str((row.get("evaluation_metadata") or {}).get("pair_id")) for row in dev_rows
    }
    report = {
        "status": "ready",
        "rows": len(rows),
        "normalized_samples": len(normalized),
        "decisions": dict(decisions),
        "conditions": dict(conditions),
        "tool_distribution": dict(sorted(tools.items())),
        "direct_strata": {
            name: count for name, count in sorted(strata.items()) if name in DIRECT_STRATA
        },
        "system_and_menu_conditioned": system_conditioned,
        "unique_case_ids": len(case_ids),
        "dev_rows_checked": len(dev_rows),
        "dev_normalized_overlap": len(normalized & dev_signatures),
        "dev_case_id_overlap": len(case_ids & dev_case_ids),
    }
    problems = []
    if len(rows) != ROUTING_SUPPLEMENT_ROWS or len(normalized) != ROUTING_SUPPLEMENT_ROWS:
        problems.append("expected 5200 unique normalized rows")
    if decisions != {"tool": 2_600, "direct": 2_600}:
        problems.append("decision split is not exactly 2600 tool / 2600 direct")
    if set(tools) != set(TOOLS) or set(tools.values()) != {200}:
        problems.append("tool distribution is not exactly 200 examples for each of 13 tools")
    if any(strata.get(name) != 520 for name in DIRECT_STRATA):
        problems.append("direct strata are not exactly 520 examples each")
    if system_conditioned != ROUTING_SUPPLEMENT_ROWS:
        problems.append("not every row contains the full schema-conditioned tool menu")
    if report["dev_normalized_overlap"] or report["dev_case_id_overlap"]:
        problems.append("training supplement overlaps the selected DEV set")
    if problems:
        report.update(status="blocked", problems=problems)
    return report


def ensure_routing_supplement(dev_source: str | Path | None = None) -> dict[str, Any]:
    """Build, audit and register the deterministic routing supplement."""
    rows = build_routing_supplement_rows()
    report = audit_routing_supplement(rows, dev_source=dev_source)
    if report["status"] != "ready":
        raise ValueError(f"routing supplement audit failed: {report.get('problems')}")
    ROUTING_SUPPLEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    if not ROUTING_SUPPLEMENT_PATH.exists() or ROUTING_SUPPLEMENT_PATH.read_text(
        encoding="utf-8"
    ) != payload:
        ROUTING_SUPPLEMENT_PATH.write_text(payload, encoding="utf-8")
    probe = inspect_dataset(str(ROUTING_SUPPLEMENT_PATH), name=ROUTING_SUPPLEMENT_NAME)
    dataset_id = Registry().upsert_dataset(probe)
    report.update(
        dataset_id=dataset_id,
        dataset_name=ROUTING_SUPPLEMENT_NAME,
        source=str(ROUTING_SUPPLEMENT_PATH),
        fingerprint=probe.fingerprint,
        size_bytes=probe.size_bytes,
    )
    return report
