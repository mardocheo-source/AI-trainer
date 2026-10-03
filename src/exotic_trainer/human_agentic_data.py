from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .dataset_probe import inspect_dataset, normalize_record
from .paths import project_root
from .preflight import example_hash
from .registry import Registry
from .sealed import list_sealed_suites, register_sealed_suite
from .tool_schema import KNOWN_TOOLS, REQUIRED_ARGUMENTS, STANDARD_TOOLS

TOOLS = tuple(sorted(STANDARD_TOOLS))
TRAIN_ROWS = 5_200
EVAL_ROWS = 520
TRAIN_PATH = project_root() / "downloads" / "generated" / "human-agentic-train-v1-5200.jsonl"
DEV_PATH = project_root() / "examples" / "dev-human-agentic-v1-520.jsonl"
SEALED_PATH = project_root() / "examples" / "sealed-human-agentic-v1-520.jsonl"

POLICY = (
    "You are a careful coding assistant. Infer the user's intent from ordinary language. "
    "Use one declared tool when a concrete external action is requested and sufficiently "
    "specified; otherwise answer naturally without a tool. Never invent missing literal values."
)

_SPLIT_SEEDS = {"train": 20260812011, "dev": 20260812012, "sealed": 20260812013}
_SPLIT_OFFSET = {"train": 0, "dev": 30_000, "sealed": 60_000}

_PROJECTS = (
    "harbor",
    "lantern",
    "juniper",
    "atlas",
    "mosaic",
    "kestrel",
    "willow",
    "quartz",
    "ember",
    "cobalt",
    "orchid",
    "meridian",
    "pioneer",
    "delta",
    "cirrus",
    "saffron",
    "vector",
    "birch",
    "aurora",
    "canyon",
)
_SYMBOLS = (
    "resolve_route",
    "parse_manifest",
    "build_index",
    "render_report",
    "load_profile",
    "validate_token",
    "dispatch_job",
    "merge_options",
    "format_result",
    "open_session",
)
_TOPICS = (
    "dependency injection",
    "idempotency",
    "structured logging",
    "retry budgets",
    "type narrowing",
    "cache invalidation",
    "path traversal",
    "streaming parsers",
    "property tests",
    "graceful shutdown",
)


def _assistant_tool(name: str, arguments: dict[str, Any], case_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{case_id}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _scalars(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from _scalars(item)
    elif isinstance(value, bool):
        yield "true" if value else "false"
    elif value is not None:
        yield str(value)


def _context(split: str, index: int) -> dict[str, Any]:
    serial = _SPLIT_OFFSET[split] + index
    project = _PROJECTS[serial % len(_PROJECTS)]
    branch = ("feature", "maintenance", "release", "experiment")[serial % 4]
    root = f"projects/{project}/{branch}-{serial:05d}"
    return {
        "serial": serial,
        "project": project,
        "root": root,
        "path": f"{root}/{('src', 'lib', 'tests', 'config')[serial % 4]}/"
        f"{('router.py', 'settings.toml', 'worker.ts', 'manifest.yaml')[serial % 4]}",
        "symbol": f"{_SYMBOLS[serial % len(_SYMBOLS)]}_{serial % 97}",
        "marker": f"{project.upper()}_{serial:05d}",
        "topic": _TOPICS[serial % len(_TOPICS)],
    }


def _tool_case(tool: str, split: str, local_index: int) -> tuple[str, dict[str, Any], str]:
    c = _context(split, TOOLS.index(tool) * 1_000 + local_index)
    serial = int(c["serial"])
    path = str(c["path"])
    root = str(c["root"])
    marker = str(c["marker"])
    family = local_index % 5

    if tool == "read":
        offset, limit = 2 + serial % 37, 20 + serial % 71
        args = {"path": path, "offset": offset, "limit": limit}
        prompts = (
            f"Could you open {path} for me, starting at line {offset}, and show the next {limit} lines?",
            f"I need to inspect {limit} lines of {path}; skip the first {offset} lines before showing it.",
            f"Please look inside {path}. Begin at offset {offset} and limit the output to {limit} lines.",
            f"Before I change anything, retrieve lines {offset} onward from {path}, at most {limit} lines.",
            f"Can you check the portion of {path} after line {offset}? Stop after {limit} lines.",
        )
    elif tool == "write":
        target = f"{root}/notes/{c['project']}-{serial % 101}.ini"
        content = f"project={c['project']}\nmarker={marker}\nmode=review\n"
        args = {"path": target, "content": content}
        prompts = (
            f"Create {target} with exactly this content:\n```ini\n{content}```",
            f"Please save the following small INI file as {target}; keep the trailing newline.\n```ini\n{content}```",
            f"Add a new file at {target}. Its entire contents should be:\n```ini\n{content}```",
            f"Write this configuration verbatim to {target}:\n```ini\n{content}```",
            f"I want {target} created now, containing only these three lines:\n```ini\n{content}```",
        )
    elif tool == "edit":
        old = f'RELEASE_STATE = "candidate-{serial}"'
        new = f'RELEASE_STATE = "approved-{serial}"'
        replace_all = bool(serial % 2)
        args = {"path": path, "oldText": old, "newText": new, "replaceAll": replace_all}
        extent = "every occurrence" if replace_all else "only the first occurrence"
        prompts = (
            f"In {path}, replace `{old}` with `{new}`; change {extent}.",
            f"Please update {path}: the exact text `{old}` should become `{new}`. Apply it to {extent}.",
            f"Make a precise text replacement in {path}, from `{old}` to `{new}`, affecting {extent}.",
            f"Edit {path} now. Find the literal `{old}` and substitute `{new}` for {extent}.",
            f"The state in {path} is stale. Replace `{old}` by `{new}` for {extent}.",
        )
    elif tool == "bash":
        command = (
            f"python -m pytest tests/{c['project']}/test_{serial % 997:03d}.py -k {c['symbol']} -q"
        )
        timeout = 45 + serial % 136
        description = f"Run the focused {c['project']} regression"
        args = {"command": command, "description": description, "timeout": timeout}
        prompts = (
            f"Run `{command}` in the repository. Give it a timeout of {timeout} seconds; this is the focused {c['project']} regression.",
            f"Please execute the focused test command `{command}` and stop it after {timeout} seconds if it hangs.",
            f'From the project terminal, launch `{command}` with a {timeout}-second timeout. Describe it as "{description}".',
            f'I authorize running `{command}` now. Use {timeout} seconds as the limit and label it "{description}".',
            f'Can you run this check for me: `{command}`? Cap it at {timeout} seconds; description: "{description}".',
        )
    elif tool == "apply_patch":
        target = f"{root}/src/feature_{serial % 83}.py"
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            f'-STATUS = "pending-{serial}"\n'
            f'+STATUS = "ready-{serial}"\n'
            "*** End Patch\n"
        )
        args = {"patchText": patch}
        prompts = (
            f"Apply this patch to the repository:\n```diff\n{patch}```",
            f"Please make the following repository patch exactly as written:\n```diff\n{patch}```",
            f"Here is the reviewed change. Apply it now without altering the diff:\n```diff\n{patch}```",
            f"Patch the working tree with this approved change:\n```diff\n{patch}```",
            f"Use the patch below to update the file:\n```diff\n{patch}```",
        )
    elif tool == "glob":
        pattern = f"**/{c['project']}_*_{serial % 29}.py"
        args = {"pattern": pattern, "path": root}
        prompts = (
            f"Find every path under {root} matching `{pattern}`.",
            f"Which files below {root} match the glob `{pattern}`? Please search the tree.",
            f"Search {root} for filenames described by `{pattern}` and show the matches.",
            f"I need a path listing from {root} using the pattern `{pattern}`.",
            f"Locate matching files in {root}; the glob to use is `{pattern}`.",
        )
    elif tool == "grep":
        pattern = f"{marker}.*{c['symbol']}"
        include = ("*.py", "*.ts", "*.toml", "*.yaml")[serial % 4]
        args = {"pattern": pattern, "path": root, "include": include}
        prompts = (
            f"Search under {root} for the regex `{pattern}`, but only in `{include}` files.",
            f"Please look through `{include}` files in {root} and find occurrences matching `{pattern}`.",
            f"Run a repository text search rooted at {root}: pattern `{pattern}`, include filter `{include}`.",
            f"Find where `{pattern}` appears below {root}; restrict the search to `{include}`.",
            f"I need matching lines, not just filenames: search {root} for `{pattern}` in `{include}` files.",
        )
    elif tool == "lsp":
        operation = ("documentSymbol", "goToDefinition", "references", "workspaceSymbol")[
            serial % 4
        ]
        if operation == "workspaceSymbol":
            args = {"operation": operation, "symbol": str(c["symbol"])}
            detail = f"the workspace symbol `{c['symbol']}`"
        elif operation == "documentSymbol":
            args = {"operation": operation, "filePath": path}
            detail = f"all document symbols in {path}"
        else:
            line, character = 3 + serial % 89, 1 + serial % 27
            args = {"operation": operation, "filePath": path, "line": line, "character": character}
            detail = f"{operation} at line {line}, character {character} of {path}"
        prompts = (
            f"Ask the language server for {detail}.",
            f"Use code intelligence to retrieve {detail}.",
            f"Please query the editor's language service for {detail}.",
            f"I need a semantic code lookup: get {detail}.",
            f"Without a text search, request {detail} from the language server.",
        )
    elif tool == "skill":
        name = f"{c['project']}-migration-{serial % 113}"
        args = {"name": name}
        prompts = (
            f"Load the specialist workflow named `{name}` before we continue.",
            f"Please activate the `{name}` specialist instructions for this task.",
            f"Bring the named workflow `{name}` into context now.",
            f"I want to use the installed specialist guide called `{name}`.",
            f"Open the reusable instruction package named `{name}` for me.",
        )
    elif tool == "task":
        description = f"Audit {c['project']} routing"
        brief = (
            f"Inspect {root} for routing risks involving {c['symbol']}. "
            "Report findings with file references and do not modify files."
        )
        args = {"description": description, "prompt": brief, "subagent_type": "general"}
        prompts = (
            f'Delegate a read-only investigation titled "{description}". Brief: {brief}',
            f'Please hand this bounded audit to a general subagent. Title it "{description}" and tell it: {brief}',
            f'Start a delegated research task with the title "{description}". The full assignment is: {brief}',
            f'Have a general-purpose subagent perform this audit, named "{description}": {brief}',
            f'Spin up one read-only delegated task. Description: "{description}". Instructions: {brief}',
        )
    elif tool == "todowrite":
        todos = [
            {"content": f"inspect {c['project']} request {serial}", "status": "in_progress"},
            {"content": f"verify {c['symbol']}", "status": "pending"},
            {"content": f"record decision {marker}", "status": "pending"},
        ]
        args = {"todos": todos}
        prompts = (
            f'Update the task list to these three items: 1) "{todos[0]["content"]}" is in progress; 2) "{todos[1]["content"]}" is pending; 3) "{todos[2]["content"]}" is pending.',
            f'Create a structured todo list: mark "{todos[0]["content"]}" in progress, then add pending items "{todos[1]["content"]}" and "{todos[2]["content"]}".',
            f"Please track this work: in progress — {todos[0]['content']}; pending — {todos[1]['content']}; pending — {todos[2]['content']}.",
            f'Replace the current plan with three tasks. The active task is "{todos[0]["content"]}"; the next two pending tasks are "{todos[1]["content"]}" and "{todos[2]["content"]}".',
            f"Write the following statuses into the structured task list: `{todos[0]['content']}` = in_progress, `{todos[1]['content']}` = pending, `{todos[2]['content']}` = pending.",
        )
    elif tool == "webfetch":
        url = f"https://docs.example.org/{c['project']}/releases/{serial}"
        extraction = f"Extract the compatibility note for marker {marker}."
        args = {"url": url, "prompt": extraction}
        prompts = (
            f"Fetch {url} and {extraction}",
            f"Open the page at {url}; from it, {extraction.lower()}",
            f"Retrieve this specific documentation page: {url}. {extraction}",
            f"Please visit {url} and return only what this instruction asks: {extraction}",
            f"Read the online document {url}. {extraction}",
        )
    elif tool == "websearch":
        query = f"{c['project']} {c['topic']} compatibility {2031 + serial % 7} reference {serial}"
        args = {"query": query}
        prompts = (
            f"Search the web for `{query}` and show me current sources.",
            f"Look online for this exact research query: `{query}`.",
            f"Please find up-to-date web results about `{query}`.",
            f"Run an internet search using the phrase `{query}`.",
            f"I need current external information; search for `{query}`.",
        )
    else:
        raise KeyError(tool)
    return prompts[family], args, f"human-tool-{tool}-f{family}"


def _tool_result(tool: str, arguments: dict[str, Any], case_id: str) -> str:
    if tool in {"read", "grep", "glob", "lsp", "webfetch", "websearch"}:
        return f"Result for {case_id}: one verified match was returned."
    if tool in {"write", "edit", "apply_patch"}:
        return f"Result for {case_id}: the requested repository change was applied successfully."
    if tool == "bash":
        return f"Result for {case_id}: command completed with exit code 0; 1 test passed."
    return f"Result for {case_id}: request accepted successfully."


def _matched_direct(tool: str, split: str, local_index: int) -> tuple[str, str, list[str], str]:
    action, _arguments, _ = _tool_case(tool, split, local_index + 500)
    c = _context(split, 20_000 + TOOLS.index(tool) * 1_000 + local_index)
    family = local_index % 5
    wrappers = (
        f"Do not perform the action. In one sentence, explain which information should be confirmed first. Context only: {action}",
        f"This is a hypothetical example, not authorization to use a tool. Briefly describe what the requested operation would do: {action}",
        f"I am drafting documentation. Do not execute anything; summarize the safety consideration in this example: {action}",
        f"No repository or network action is allowed. Tell me whether this request is read-only or mutating and why: {action}",
        f"Pause before acting. Explain the intent of this request in plain language, without calling a tool: {action}",
    )
    answers = (
        f"The exact scope and authorization should be confirmed before any external action on {c['project']}.",
        f"Hypothetically, it would carry out a concrete operation involving {c['project']}, but nothing should be executed now.",
        f"The safety consideration is to preserve the supplied literals and avoid acting without explicit authorization for {c['project']}.",
        f"This example describes an external operation involving {c['project']}; the instruction explicitly forbids performing it.",
        f"Its intent is to prepare a concrete operation for {c['project']}, while the current request asks only for an explanation.",
    )
    required = {
        0: ["confirm", "before"],
        1: ["hypothetically", "nothing"],
        2: ["safety", "authorization"],
        3: ["forbids", "operation"],
        4: ["intent", "explanation"],
    }[family]
    # The near-negative preserves the actionable request and its concrete values,
    # then changes only the human intent (hypothetical/explain/do-not-act).
    return wrappers[family], answers[family], required, f"near-negative-{tool}-f{family}"


def _natural_direct(split: str, index: int) -> tuple[str, str, list[str], str]:
    c = _context(split, 50_000 + index)
    family = index % 10
    topic, symbol = str(c["topic"]), str(c["symbol"])
    note = f"{str(c['project']).title()} review {int(c['serial']):05d}"
    cases = (
        (
            f"In two short sentences, explain {topic} to a Python developer. No project inspection is needed.",
            f"{topic.title()} keeps a system predictable by making its behavior explicit. In Python, apply it at clear boundaries and test failure cases.",
            [topic.split()[0], "Python"],
            "concept",
        ),
        (
            "Why is `items = [[]] * 3` risky when the inner lists are later mutated? Answer directly.",
            "All three entries reference the same inner list, so mutating one changes every entry; use a comprehension to create independent lists.",
            ["same", "comprehension"],
            "debug-snippet",
        ),
        (
            f"Write a small pure Python function named `{symbol}` that returns whether an integer is even. Do not create a file.",
            f"```python\ndef {symbol}(value: int) -> bool:\n    return value % 2 == 0\n```",
            [symbol, "% 2"],
            "inline-code",
        ),
        (
            "Review this expression for readability: `ready = True if count > 0 else False`. Suggest a direct replacement.",
            "Use `ready = count > 0`; the comparison already produces a Boolean value.",
            ["count > 0", "Boolean"],
            "code-review",
        ),
        (
            f"Give me a three-step plan for adding tests around `{symbol}`. Planning only; do not inspect files.",
            f"1. Define the observable contract for `{symbol}`. 2. Cover normal and boundary inputs. 3. Add a regression case for the original failure.",
            ["contract", "boundary", "regression"],
            "plan-only",
        ),
        (
            "What does `python -m pytest -q` normally do? Explain it, but do not run it.",
            "It invokes pytest as a Python module and runs discovered tests with quieter output; this explanation does not execute the command.",
            ["pytest", "quieter"],
            "command-explanation",
        ),
        (
            "Name two defenses against path traversal in a file-serving endpoint. No external lookup is needed.",
            "Resolve and constrain paths to an allowed root, and reject untrusted absolute paths or parent-directory traversal components.",
            ["allowed root", "parent"],
            "security",
        ),
        (
            "Rewrite `The service failed because config was bad` in a more actionable style.",
            "The service could not start because the configuration was invalid; validate the configuration fields and report the failing key.",
            ["configuration", "failing key"],
            "rewrite",
        ),
        (
            f"Suggest two unit-test cases for a parser called `{symbol}` that accepts comma-separated names.",
            f"Test `{symbol}` with multiple valid names including whitespace, and test malformed input such as an empty element between commas.",
            ["whitespace", "empty"],
            "test-design",
        ),
        (
            "When would a queue be preferable to calling a slow background job synchronously? Answer conceptually.",
            "A queue is preferable when the caller should not wait, work needs retries or rate control, or producers and workers must scale independently.",
            ["retries", "scale"],
            "architecture",
        ),
    )
    prompt, answer, required, stratum = cases[family]
    # A harmless, human-style review-note reference makes every task an
    # independent example without revealing its label or answer.
    return (
        f"Context: {note}.\n{prompt}",
        f"{answer}\nReference: {note}.",
        required,
        stratum,
    )


def _row(
    split: str,
    case_id: str,
    user: str,
    assistant: dict[str, Any],
    tool: str | None,
    stratum: str,
    required_terms: list[str] | None = None,
    tool_result: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "case_id": case_id,
        "split": split,
        "condition": "initial-human-request",
        "stratum": stratum,
        "decision": "tool" if tool else "direct",
        "tool": tool,
        "required_terms": required_terms or [],
        "schema_version": 1,
    }
    key = "training_metadata" if split == "train" else "evaluation_metadata"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": user},
        assistant,
    ]
    if split == "train" and tool is not None:
        messages.extend(
            [
                {"role": "tool", "content": tool_result or "The operation completed."},
                {
                    "role": "assistant",
                    "content": "Done. I completed the requested action and checked its result.",
                },
            ]
        )
    return {key: metadata, "messages": messages}


def build_rows(split: str) -> list[dict[str, Any]]:
    if split not in _SPLIT_SEEDS:
        raise ValueError(f"unsupported human-agentic split: {split}")
    per_tool = 200 if split == "train" else 20
    near_negative_per_tool = 100 if split == "train" else 10
    natural_direct_count = 1_300 if split == "train" else 130
    rows: list[dict[str, Any]] = []
    for tool in TOOLS:
        for local_index in range(per_tool):
            request, arguments, family = _tool_case(tool, split, local_index)
            case_id = f"{split}-human-{tool}-{local_index:04d}"
            rows.append(
                _row(
                    split,
                    case_id,
                    request,
                    _assistant_tool(tool, arguments, case_id),
                    tool,
                    family,
                    tool_result=_tool_result(tool, arguments, case_id),
                )
            )
        for local_index in range(near_negative_per_tool):
            request, answer, required, family = _matched_direct(tool, split, local_index)
            rows.append(
                _row(
                    split,
                    f"{split}-near-negative-{tool}-{local_index:04d}",
                    request,
                    {"role": "assistant", "content": answer},
                    None,
                    family,
                    required_terms=required,
                )
            )
    for index in range(natural_direct_count):
        request, answer, required, family = _natural_direct(split, index)
        rows.append(
            _row(
                split,
                f"{split}-natural-direct-{index:04d}",
                request,
                {"role": "assistant", "content": answer},
                None,
                f"natural-{family}",
                required_terms=required,
            )
        )
    random.Random(_SPLIT_SEEDS[split]).shuffle(rows)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    if not path.exists() or path.read_text(encoding="utf-8") != payload:
        path.write_text(payload, encoding="utf-8")


def _signatures(rows: list[dict[str, Any]]) -> set[str]:
    return {example_hash(sample) for row in rows for sample in normalize_record(row)}


def _initial_target(row: dict[str, Any]) -> dict[str, Any]:
    return row["messages"][2]


def _grounded_required_arguments(tool: str, arguments: dict[str, Any], prompt: str) -> bool:
    def human_normalize(value: str) -> str:
        value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
        return re.sub(r"[_\s]+", " ", value.lower()).strip()

    normalized_prompt = human_normalize(prompt)

    def visible(value: Any) -> bool:
        if isinstance(value, dict):
            return all(visible(item) for item in value.values())
        if isinstance(value, list):
            return all(visible(item) for item in value)
        if isinstance(value, bool):
            return True  # expressed naturally as first/every occurrence
        normalized_value = human_normalize(str(value))
        return normalized_value in normalized_prompt

    if tool == "todowrite":
        # Natural prompts express statuses as "active", "in progress" and
        # "next"; the literal task content itself must still be fully grounded.
        return all(
            visible(item.get("content", ""))
            for item in arguments.get("todos", [])
            if isinstance(item, dict)
        )

    return all(
        key in arguments and visible(arguments[key]) for key in REQUIRED_ARGUMENTS.get(tool, set())
    )


def audit_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    signatures = {name: _signatures(rows) for name, rows in splits.items()}
    reports: dict[str, Any] = {}
    problems: list[str] = []
    for name, rows in splits.items():
        expected_rows = TRAIN_ROWS if name == "train" else EVAL_ROWS
        expected_normalized = 7_800 if name == "train" else EVAL_ROWS
        metadata_key = "training_metadata" if name == "train" else "evaluation_metadata"
        decisions = Counter(row[metadata_key]["decision"] for row in rows)
        tools = Counter(row[metadata_key]["tool"] for row in rows if row[metadata_key]["tool"])
        explicit_answers = sum(
            "tool to use" in row["messages"][1]["content"].lower()
            or "complete argument object" in row["messages"][1]["content"].lower()
            for row in rows
        )
        menu_in_raw_prompt = sum(
            "List of tools: [" in row["messages"][0]["content"] for row in rows
        )
        grounded = 0
        for row in rows:
            metadata = row[metadata_key]
            if metadata["decision"] != "tool":
                continue
            call = _initial_target(row)["tool_calls"][0]["function"]
            args = json.loads(call["arguments"])
            prompt = row["messages"][1]["content"]
            grounded += int(_grounded_required_arguments(str(metadata["tool"]), args, prompt))
        reports[name] = {
            "raw_rows": len(rows),
            "normalized_samples": sum(len(normalize_record(row)) for row in rows),
            "unique_normalized_samples": len(signatures[name]),
            "initial_decisions": dict(decisions),
            "tool_distribution": dict(sorted(tools.items())),
            "explicit_solution_leaks": explicit_answers,
            "raw_prompts_with_embedded_menu": menu_in_raw_prompt,
            "grounded_tool_requests": grounded,
            "template_families": len({row[metadata_key]["stratum"] for row in rows}),
        }
        if len(rows) != expected_rows:
            problems.append(f"{name}: expected {expected_rows} raw rows")
        if sum(len(normalize_record(row)) for row in rows) != expected_normalized:
            problems.append(f"{name}: expected {expected_normalized} normalized samples")
        if len(signatures[name]) != expected_normalized:
            problems.append(f"{name}: normalized samples are not unique")
        if decisions != {"tool": expected_rows // 2, "direct": expected_rows // 2}:
            problems.append(f"{name}: initial tool/direct decisions are not exactly balanced")
        expected_per_tool = 200 if name == "train" else 20
        if set(tools) != set(TOOLS) or set(tools.values()) != {expected_per_tool}:
            problems.append(f"{name}: 13-tool distribution is not uniform")
        if explicit_answers:
            problems.append(f"{name}: {explicit_answers} prompts reveal the solution label")
        if menu_in_raw_prompt:
            problems.append(f"{name}: raw prompts embed a menu that inference would duplicate")
        if grounded != expected_rows // 2:
            problems.append(f"{name}: not every target argument is grounded in human prose")
    for left, right in (("train", "dev"), ("train", "sealed"), ("dev", "sealed")):
        overlap = len(signatures[left] & signatures[right])
        reports[f"{left}_{right}_exact_overlap"] = overlap
        if overlap:
            problems.append(f"{left}/{right}: {overlap} normalized samples overlap")
    canonical = json.dumps(reports, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "status": "blocked" if problems else "ready",
        "schema_version": 1,
        "human_like": True,
        "tool_menu_contract": list(TOOLS),
        "reports": reports,
        "problems": problems,
        "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def ensure_human_agentic_data() -> dict[str, Any]:
    splits = {name: build_rows(name) for name in ("train", "dev", "sealed")}
    audit = audit_splits(splits)
    if audit["status"] != "ready":
        raise ValueError(f"human-agentic data audit failed: {audit['problems']}")
    paths = {"train": TRAIN_PATH, "dev": DEV_PATH, "sealed": SEALED_PATH}
    for name, path in paths.items():
        _write_jsonl(path, splits[name])
    registry = Registry()
    dataset_ids = {
        "train": registry.upsert_dataset(
            inspect_dataset(str(TRAIN_PATH), "Human Agentic train v1 — 5200 — 13 tools")
        ),
        "dev": registry.upsert_dataset(
            inspect_dataset(str(DEV_PATH), "DEV Human Agentic v1 — 520 — 13 tools")
        ),
    }
    resolved_sealed = str(SEALED_PATH.resolve())
    suite = next(
        (
            item
            for item in list_sealed_suites()
            if str(Path(item["source"]).resolve()) == resolved_sealed
        ),
        None,
    )
    if suite is None:
        suite = register_sealed_suite(
            resolved_sealed,
            "FINAL Human Agentic v1 — 520 balanced — 13 tools",
            max_rows=EVAL_ROWS,
        )
    return {
        **audit,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "dataset_ids": dataset_ids,
        "suite_id": suite["id"],
        "suite_name": suite["name"],
    }
