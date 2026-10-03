from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .human_agentic_data import TOOLS, _assistant_tool, _row, build_rows
from .human_agentic_ood import (
    _direct_case as _ood_direct_case,
)
from .human_agentic_ood import (
    _negative_case as _ood_negative_case,
)
from .human_agentic_ood import (
    _signatures,
    build_ood_rows,
)
from .human_agentic_ood import (
    _tool_case as _ood_tool_case,
)
from .paths import project_root
from .sealed import list_sealed_suites, register_sealed_suite

FOCUSED_ROWS = 520
FOCUSED_PATH = project_root() / "examples" / "sealed-human-agentic-focused-v3-520.jsonl"
FOCUSED_SEED = 20260813031

_HANDOFFS = (
    "I'm picking up a real coding session after a short break. {request}",
    "Please handle one precise item from my current workbench: {request}",
    "A human teammate sent this request in ordinary language. Respond to its actual intent: {request}",
    "For this turn, decide carefully whether action is warranted, then satisfy the request: {request}",
    "This came from an active development conversation, not a benchmark template: {request}",
    "Keep the response natural while respecting any exact literals in the request: {request}",
    "I need a useful result from this concrete engineering exchange: {request}",
    "Read the whole instruction before deciding whether to call a capability: {request}",
    "Treat this as a normal collaborator request with the usual safety boundaries: {request}",
    "Resolve this single issue without inventing any unspecified value: {request}",
)


def _human_handoff(index: int, request: str) -> str:
    return _HANDOFFS[index % len(_HANDOFFS)].format(request=request)


def build_focused_rows() -> list[dict[str, Any]]:
    """Create a fresh human-like final suite without changing target semantics.

    It uses new literal instances and an unseen discourse layer.  Every tool has
    20 positive cases; 130 matched near-negatives plus 130 ordinary direct
    answers keep routing exactly balanced.
    """

    rows: list[dict[str, Any]] = []
    for tool in TOOLS:
        for index in range(20):
            request, arguments = _ood_tool_case(tool, index + 20)
            case_id = f"sealed-focus-tool-{tool}-{index:02d}"
            rows.append(
                _row(
                    "sealed",
                    case_id,
                    _human_handoff(index, request),
                    _assistant_tool(tool, arguments, case_id),
                    tool,
                    f"focus-action-{tool}-voice-{index:02d}",
                )
            )
        for index in range(10):
            request, answer, required, stratum = _ood_negative_case(tool, index + 10)
            rows.append(
                _row(
                    "sealed",
                    f"sealed-focus-negative-{tool}-{index:02d}",
                    _human_handoff(index + 20, request),
                    {"role": "assistant", "content": answer},
                    None,
                    f"focus-{stratum}",
                    required_terms=required,
                )
            )
    for index in range(130):
        request, answer, required, stratum = _ood_direct_case(index + 130)
        reference = f"Focus review {120_000 + index}"
        rows.append(
            _row(
                "sealed",
                f"sealed-focus-direct-{index:03d}",
                _human_handoff(index + 40, f"Context label: {reference}. {request}"),
                {"role": "assistant", "content": f"{answer}\nReference: {reference}."},
                None,
                f"focus-direct-{stratum}",
                required_terms=required,
            )
        )
    random.Random(FOCUSED_SEED).shuffle(rows)
    return rows


def _prompt_skeleton(row: dict[str, Any]) -> str:
    text = str(row["messages"][1]["content"]).lower()
    text = re.sub(r"```[\s\S]*?```", "<BLOCK>", text)
    text = re.sub(r"`[^`]+`", "<LITERAL>", text)
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"(?:[a-z0-9_.-]+/)+[a-z0-9_.-]+", "<PATH>", text)
    text = re.sub(r"\d+", "<N>", text)
    return re.sub(r"\s+", " ", text).strip()


def audit_focused_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(row["evaluation_metadata"]["decision"] for row in rows)
    tools = Counter(
        row["evaluation_metadata"]["tool"] for row in rows if row["evaluation_metadata"]["tool"]
    )
    current = {split: build_rows(split) for split in ("train", "dev", "sealed")}
    current["ood-v2"] = build_ood_rows()
    signatures = _signatures(rows)
    exact_overlap = {
        split: len(signatures & _signatures(other)) for split, other in current.items()
    }
    skeletons = {_prompt_skeleton(row) for row in rows}
    skeleton_overlap = {
        split: len(skeletons & {_prompt_skeleton(row) for row in other})
        for split, other in current.items()
    }
    problems: list[str] = []
    if len(rows) != FOCUSED_ROWS or len(signatures) != FOCUSED_ROWS:
        problems.append("focused suite must contain 520 unique normalized examples")
    if decisions != {"tool": 260, "direct": 260}:
        problems.append("focused suite must be exactly balanced 260 tool / 260 direct")
    if set(tools) != set(TOOLS) or set(tools.values()) != {20}:
        problems.append("focused suite must contain exactly 20 cases for each of 13 tools")
    if any(exact_overlap.values()):
        problems.append(f"exact normalized overlap detected: {exact_overlap}")
    if any(skeleton_overlap.values()):
        problems.append(f"prompt skeleton overlap detected: {skeleton_overlap}")
    report: dict[str, Any] = {
        "status": "blocked" if problems else "ready",
        "schema_version": 1,
        "human_like": True,
        "strict_and_elastic_agentic": True,
        "rows": len(rows),
        "unique_normalized": len(signatures),
        "decisions": dict(decisions),
        "tool_distribution": dict(sorted(tools.items())),
        "unique_prompt_skeletons": len(skeletons),
        "exact_overlap": exact_overlap,
        "prompt_skeleton_overlap": skeleton_overlap,
        "problems": problems,
    }
    report["fingerprint"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def ensure_focused_human_agentic_suite() -> dict[str, Any]:
    rows = build_focused_rows()
    audit = audit_focused_rows(rows)
    if audit["status"] != "ready":
        raise ValueError(f"focused human-agentic audit failed: {audit['problems']}")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    FOCUSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FOCUSED_PATH.exists() or FOCUSED_PATH.read_text(encoding="utf-8") != payload:
        FOCUSED_PATH.write_text(payload, encoding="utf-8")
    resolved = str(FOCUSED_PATH.resolve())
    suite = next(
        (item for item in list_sealed_suites() if str(Path(item["source"]).resolve()) == resolved),
        None,
    )
    if suite is None:
        suite = register_sealed_suite(
            resolved,
            "FINAL Focused Human Agentic v3 — strict + elastic — 520 — 13 tools",
            max_rows=FOCUSED_ROWS,
        )
    return {
        **audit,
        "path": resolved,
        "suite_id": suite["id"],
        "suite_name": suite["name"],
    }
