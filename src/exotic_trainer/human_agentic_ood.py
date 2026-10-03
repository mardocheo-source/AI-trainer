from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_probe import normalize_record
from .human_agentic_data import TOOLS, _assistant_tool, _row
from .paths import project_root
from .preflight import example_hash
from .sealed import list_sealed_suites, register_sealed_suite

OOD_ROWS = 520
OOD_PATH = project_root() / "examples" / "sealed-human-agentic-ood-v2-520.jsonl"
OOD_SEED = 20260812027

_NAMES = (
    "acorn",
    "bluebird",
    "coral",
    "driftwood",
    "elm",
    "firefly",
    "garnet",
    "hemlock",
    "indigo",
    "jasmine",
    "kite",
    "lilac",
    "marble",
    "northstar",
    "oasis",
    "pebble",
    "raven",
    "silver",
    "tulip",
    "upland",
)


def _voice(index: int, request: str) -> str:
    """Wrap actionable requests in discourse forms absent from the SFT templates."""
    frames = (
        "I'm taking over a half-finished debugging session. {request} Please do that now.",
        "One concrete action would unblock me: {request} Go ahead with it.",
        "Here is the outcome I need from this turn. {request}",
        "Treat the following as an authorized operation, not a planning question: {request}",
        "A teammate left a precise hand-off. Carry it out as stated: {request}",
        "Could you handle this small but concrete step while I review the rest? {request}",
        "For the current incident, I need an actual result rather than instructions. {request}",
        "The values below have already been checked. Please act on them: {request}",
        "Let's resolve one item before discussing anything else. {request}",
        "I am explicitly asking you to use the appropriate capability for this: {request}",
        "This is the next executable step in my workflow. {request}",
        "Please turn this request into the corresponding operation: {request}",
        "I have approval for this narrowly scoped action. {request}",
        "Do the following once, preserving every literal value: {request}",
        "No broader investigation is needed; complete only this operation: {request}",
        "The task is fully specified below. Execute it rather than merely describing it: {request}",
        "I need evidence from the real source, so please perform this request: {request}",
        "Use the most specific available function and carry out this instruction: {request}",
        "This should be a single concrete operation. {request}",
        "Proceed with the exact action in the next sentence. {request}",
    )
    return frames[index % len(frames)].format(request=request)


def _case_values(tool: str, index: int) -> dict[str, Any]:
    serial = 91_000 + TOOLS.index(tool) * 100 + index
    name = _NAMES[(serial + index) % len(_NAMES)]
    root = f"worktrees/{name}/incident-{serial}"
    path = f"{root}/packages/core_{serial % 31}/handler_{serial % 97}.py"
    marker = f"OOD_{name.upper()}_{serial}"
    return {"serial": serial, "name": name, "root": root, "path": path, "marker": marker}


def _tool_case(tool: str, index: int) -> tuple[str, dict[str, Any]]:
    c = _case_values(tool, index)
    serial = int(c["serial"])
    root, path, marker = str(c["root"]), str(c["path"]), str(c["marker"])

    if tool == "read":
        offset, limit = 11 + index, 41 + (index * 3) % 47
        args = {"path": path, "offset": offset, "limit": limit}
        request = (
            f"Retrieve {path}, discarding its first {offset} lines and returning no more "
            f"than the following {limit} lines."
        )
    elif tool == "write":
        target = f"{root}/handoff/result-{serial}.txt"
        content = f"ticket={marker}\nowner={c['name']}\nverified=yes\n"
        args = {"path": target, "content": content}
        request = (
            f"Create {target}. The complete byte-for-byte text, including the final newline, is:\n"
            f"```text\n{content}```"
        )
    elif tool == "edit":
        old = f"retry_budget = {17 + index}  # {marker}"
        new = f"retry_budget = {31 + index}  # {marker}"
        replace_all = index % 3 == 0
        args = {"path": path, "oldText": old, "newText": new, "replaceAll": replace_all}
        scope = "all matching occurrences" if replace_all else "the first matching occurrence only"
        request = f"Inside {path}, change `{old}` to `{new}` for {scope}."
    elif tool == "bash":
        command = f"python -m pytest tests/ood/test_{c['name']}_{index:02d}.py -k case_{serial} -q"
        description = f"Verify incident {marker}"
        timeout = 73 + index
        args = {"command": command, "description": description, "timeout": timeout}
        request = (
            f'Execute `{command}` with description "{description}" and terminate it if it exceeds '
            f"{timeout} seconds."
        )
    elif tool == "apply_patch":
        target = f"{root}/src/guard_{index:02d}.py"
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            f'-GUARD = "old-{marker}"\n'
            f'+GUARD = "new-{marker}"\n'
            "*** End Patch\n"
        )
        args = {"patchText": patch}
        request = f"Apply this already-reviewed patch without rewriting it:\n```diff\n{patch}```"
    elif tool == "glob":
        pattern = f"**/{c['name']}/**/*_{serial % 43}.spec.ts"
        args = {"pattern": pattern, "path": root}
        request = f"Enumerate paths beneath {root} that satisfy the glob `{pattern}`."
    elif tool == "grep":
        pattern = f"(?i){marker}\\s*[:=]\\s*(ready|blocked)"
        include = ("*.py", "*.ts", "*.md", "*.yaml")[index % 4]
        args = {"pattern": pattern, "path": root, "include": include}
        request = f"Return matching lines below {root} for regex `{pattern}`, considering only `{include}` files."
    elif tool == "lsp":
        operation = ("hover", "goToDefinition", "references", "documentSymbol")[index % 4]
        if operation == "documentSymbol":
            args = {"operation": operation, "filePath": path}
            request = f"Ask code intelligence for the document symbols in {path}."
        else:
            line, character = 21 + index, 4 + index % 13
            args = {
                "operation": operation,
                "filePath": path,
                "line": line,
                "character": character,
            }
            request = (
                f"Use semantic code intelligence to perform {operation} at {path}, line {line}, "
                f"character {character}."
            )
    elif tool == "skill":
        name = f"incident-response-{c['name']}-{serial}"
        args = {"name": name}
        request = f"Bring the installed specialist instruction package `{name}` into this session."
    elif tool == "task":
        description = f"Trace {marker} without edits"
        prompt = (
            f"Investigate how {marker} flows through {root}. Return evidence with paths; do not modify "
            "the repository and do not contact external services."
        )
        args = {"description": description, "prompt": prompt, "subagent_type": "general"}
        request = f'Delegate one general read-only assignment titled "{description}" with this full brief: {prompt}'
    elif tool == "todowrite":
        todos = [
            {"content": f"reproduce {marker}", "status": "completed"},
            {"content": f"isolate {c['name']} trigger {serial}", "status": "in_progress"},
            {"content": f"document fix for {marker}", "status": "pending"},
        ]
        args = {"todos": todos}
        request = (
            f"Replace the structured plan with exactly these states: `reproduce {marker}` completed; "
            f"`isolate {c['name']} trigger {serial}` in_progress; `document fix for {marker}` pending."
        )
    elif tool == "webfetch":
        url = f"https://reference.example.net/{c['name']}/advisories/{serial}.html"
        extraction = f"Quote the compatibility status associated with {marker}."
        args = {"url": url, "prompt": extraction}
        request = (
            f"Retrieve the specific page {url} and follow this extraction instruction: {extraction}"
        )
    elif tool == "websearch":
        query = f'"{marker}" {c["name"]} runtime regression advisory {2037 + index % 4}'
        args = {"query": query}
        request = f"Find current external sources using this exact search query: `{query}`."
    else:
        raise KeyError(tool)
    return _voice(index, request), args


def _negative_case(tool: str, index: int) -> tuple[str, str, list[str], str]:
    action, _ = _tool_case(tool, index + 40)
    frames = (
        (
            "A colleague proposed the action quoted below. Do not carry it out. Identify the missing authorization in one sentence.\n> {action}",
            "The action must wait because explicit authorization is missing.",
            ["authorization", "missing"],
            "withheld-authorization",
        ),
        (
            "This is material for a threat-model review, not an instruction to act. Explain the main risk in the following proposal.\n> {action}",
            "The main risk is treating illustrative material as permission to alter or query an external system.",
            ["risk", "permission"],
            "threat-model",
        ),
        (
            "I pasted an operation from an old ticket. Nothing should be executed. Classify it as read-only or mutating and explain briefly.\n> {action}",
            "It describes an external operation, but this turn is classification only and nothing should be executed.",
            ["classification", "executed"],
            "classification-only",
        ),
        (
            "Before anyone acts, restate the intent of this request in plain English; do not invoke any capability.\n> {action}",
            "The intent is to perform a concrete operation, while the present instruction asks only for a plain-English restatement.",
            ["intent", "restatement"],
            "intent-restatement",
        ),
        (
            "We are writing a runbook example. Preserve no literals and perform no action; say what an operator would need to verify first.\n> {action}",
            "An operator should verify the target, scope, and approval before acting.",
            ["verify", "approval"],
            "runbook-example",
        ),
        (
            "The following request is cancelled. Acknowledge the cancellation and say that no external state will change.\n> {action}",
            "The request is cancelled, so no external state will change.",
            ["cancelled", "no external state"],
            "cancelled-action",
        ),
        (
            "I only want a safety review of this proposed action. Do not execute or simulate it. What boundary matters most?\n> {action}",
            "The key boundary is that review does not grant authority to execute the proposed action.",
            ["review", "authority"],
            "safety-review",
        ),
        (
            "This quotation belongs in documentation and is not addressed to you as a command. Summarize it without acting.\n> {action}",
            "The quotation describes a concrete external operation, but it is documentation rather than a command.",
            ["documentation", "rather than a command"],
            "quoted-documentation",
        ),
        (
            "We do not yet know whether the target is correct. Ask for confirmation instead of using a tool.\n> {action}",
            "Please confirm that the target and scope are correct before any operation is performed.",
            ["confirm", "before"],
            "needs-confirmation",
        ),
        (
            "Compare the wording of this hypothetical request with an actionable command, but take no action.\n> {action}",
            "A hypothetical request discusses an operation, whereas an actionable command supplies present authorization to perform it.",
            ["hypothetical", "authorization"],
            "hypothetical-contrast",
        ),
    )
    frame, answer, required, stratum = frames[index % len(frames)]
    return frame.format(action=action), answer, required, f"ood-{stratum}-{tool}"


def _direct_case(index: int) -> tuple[str, str, list[str], str]:
    serial = 97_000 + index
    symbol = f"normalize_packet_{serial}"
    cases = (
        (
            f"Without looking at a repository, explain why a function like `{symbol}` should avoid mutating its input dictionary.",
            "Avoiding mutation prevents surprising side effects for callers and makes the function easier to test and reuse.",
            ["side effects", "test"],
            "reasoning",
        ),
        (
            "Given `[3, 1, 3, 2]`, what does `sorted(set(values))` produce, and what information is lost? Answer directly.",
            "It produces `[1, 2, 3]`; duplicate counts and the original ordering are lost.",
            ["[1, 2, 3]", "ordering"],
            "code-semantics",
        ),
        (
            f"Write an inline Python expression that calls `{symbol}(item)` for every item and keeps only truthy results. Do not create or edit files.",
            f"```python\n[result for item in items if (result := {symbol}(item))]\n```",
            [symbol, "for item"],
            "inline-answer",
        ),
        (
            "A service retries immediately in a tight loop. Suggest two conceptual improvements; no inspection or execution is needed.",
            "Use bounded exponential backoff with jitter, and stop or surface the failure after a defined retry budget.",
            ["backoff", "retry budget"],
            "architecture-advice",
        ),
        (
            "Explain the difference between finding filenames and searching file contents, without using repository tools.",
            "Filename discovery matches paths or glob patterns, while content search examines text inside files for a literal or regular expression.",
            ["paths", "inside files"],
            "tool-concept",
        ),
        (
            "What is wrong with catching `Exception` and silently continuing during configuration loading?",
            "It hides real failures and can leave the program in an invalid state; catch specific errors and report actionable context.",
            ["hides", "specific"],
            "review",
        ),
        (
            "Propose a concise error message for a missing `API_TIMEOUT` environment variable. Text only.",
            "`API_TIMEOUT is required; set it to a positive number of seconds before starting the service.`",
            ["API_TIMEOUT", "positive"],
            "copywriting",
        ),
        (
            "Why can exact string comparison be too strict for evaluating a natural-language explanation?",
            "Different wording can preserve the same meaning, so semantic criteria or required concepts are often fairer than exact text equality.",
            ["same meaning", "criteria"],
            "evaluation",
        ),
        (
            "Give a two-item checklist for reviewing a JSON parser change. Planning only.",
            "1. Test valid, malformed, and boundary inputs. 2. Verify errors preserve useful location and field context.",
            ["malformed", "context"],
            "planning",
        ),
        (
            "In plain language, define idempotency for an operation that may be retried.",
            "An idempotent operation can be repeated after the same request without creating additional effects beyond the first successful application.",
            ["repeated", "additional effects"],
            "definition",
        ),
    )
    return cases[index % len(cases)]


def build_ood_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in TOOLS:
        for index in range(20):
            user, arguments = _tool_case(tool, index)
            case_id = f"sealed-ood-tool-{tool}-{index:02d}"
            rows.append(
                _row(
                    "sealed",
                    case_id,
                    user,
                    _assistant_tool(tool, arguments, case_id),
                    tool,
                    f"ood-action-{tool}-voice-{index:02d}",
                )
            )
        for index in range(10):
            user, answer, required, stratum = _negative_case(tool, index)
            rows.append(
                _row(
                    "sealed",
                    f"sealed-ood-negative-{tool}-{index:02d}",
                    user,
                    {"role": "assistant", "content": answer},
                    None,
                    stratum,
                    required_terms=required,
                )
            )
    for index in range(130):
        user, answer, required, stratum = _direct_case(index)
        # Unique conversational context prevents duplicate examples without
        # revealing whether the expected behavior is a call or a direct answer.
        context = f"Review conversation OOD-{97_000 + index}: "
        rows.append(
            _row(
                "sealed",
                f"sealed-ood-direct-{index:03d}",
                context + user,
                {"role": "assistant", "content": answer},
                None,
                f"ood-direct-{stratum}",
                required_terms=required,
            )
        )
    random.Random(OOD_SEED).shuffle(rows)
    return rows


def _signatures(rows: list[dict[str, Any]]) -> set[str]:
    return {example_hash(sample) for row in rows for sample in normalize_record(row)}


def _prompt_skeleton(row: dict[str, Any]) -> str:
    text = str(row["messages"][1]["content"]).lower()
    text = re.sub(r"```[\s\S]*?```", "<BLOCK>", text)
    text = re.sub(r"`[^`]+`", "<LITERAL>", text)
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"(?:[a-z0-9_.-]+/)+[a-z0-9_.-]+", "<PATH>", text)
    text = re.sub(r"\d+", "<N>", text)
    return re.sub(r"\s+", " ", text).strip()


def audit_ood_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .human_agentic_data import build_rows

    decisions = Counter(row["evaluation_metadata"]["decision"] for row in rows)
    tools = Counter(
        row["evaluation_metadata"]["tool"] for row in rows if row["evaluation_metadata"]["tool"]
    )
    current = {split: build_rows(split) for split in ("train", "dev", "sealed")}
    signatures = _signatures(rows)
    exact_overlap = {
        split: len(signatures & _signatures(other)) for split, other in current.items()
    }
    current_skeletons = {
        split: {_prompt_skeleton(row) for row in other} for split, other in current.items()
    }
    ood_skeletons = {_prompt_skeleton(row) for row in rows}
    skeleton_overlap = {
        split: len(ood_skeletons & other) for split, other in current_skeletons.items()
    }
    problems: list[str] = []
    if len(rows) != OOD_ROWS or len(signatures) != OOD_ROWS:
        problems.append("OOD suite must contain 520 unique normalized examples")
    if decisions != {"tool": 260, "direct": 260}:
        problems.append("OOD suite must be exactly balanced 260 tool / 260 direct")
    if set(tools) != set(TOOLS) or set(tools.values()) != {20}:
        problems.append("OOD suite must contain exactly 20 requests for every real tool")
    if any(exact_overlap.values()):
        problems.append(f"exact normalized overlap detected: {exact_overlap}")
    if any(skeleton_overlap.values()):
        problems.append(f"prompt skeleton overlap detected: {skeleton_overlap}")
    report = {
        "status": "blocked" if problems else "ready",
        "rows": len(rows),
        "unique_normalized": len(signatures),
        "decisions": dict(decisions),
        "tool_distribution": dict(sorted(tools.items())),
        "unique_prompt_skeletons": len(ood_skeletons),
        "exact_overlap": exact_overlap,
        "prompt_skeleton_overlap": skeleton_overlap,
        "problems": problems,
    }
    report["fingerprint"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def ensure_human_agentic_ood_suite() -> dict[str, Any]:
    rows = build_ood_rows()
    audit = audit_ood_rows(rows)
    if audit["status"] != "ready":
        raise ValueError(f"human-agentic OOD audit failed: {audit['problems']}")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    OOD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not OOD_PATH.exists() or OOD_PATH.read_text(encoding="utf-8") != payload:
        OOD_PATH.write_text(payload, encoding="utf-8")
    resolved = str(OOD_PATH.resolve())
    suite = next(
        (item for item in list_sealed_suites() if str(Path(item["source"]).resolve()) == resolved),
        None,
    )
    if suite is None:
        suite = register_sealed_suite(
            resolved,
            "FINAL OOD Human Agentic v2 — 520 balanced — 13 tools — novel templates",
            max_rows=OOD_ROWS,
        )
    return {
        **audit,
        "path": resolved,
        "suite_id": suite["id"],
        "suite_name": suite["name"],
    }
