from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .human_agentic_data import TOOLS, _assistant_tool, _row, build_rows
from .human_agentic_focus_data import _prompt_skeleton, build_focused_rows
from .human_agentic_ood import (
    _direct_case,
    _negative_case,
    _signatures,
    _tool_case,
    build_ood_rows,
)
from .paths import project_root
from .sealed import list_sealed_suites, register_sealed_suite

CONTROL_ROWS = 520
CONTROL_PATH = project_root() / "examples" / "sealed-decimal-control-v4-520.jsonl"
CONTROL_SEED = 20260813047

_FRAMES = (
    "Consider this the next turn in an ordinary pair-programming session. {request}",
    "Please respond to the concrete intent of this teammate message: {request}",
    "This request came from a live engineering hand-off. Read it fully before acting: {request}",
    "Handle this as a normal collaborator would, preserving every supplied literal: {request}",
    "Decide first whether an operation is actually requested, then answer appropriately: {request}",
    "Here is one self-contained item from today's work. {request}",
    "Treat the wording below as authoritative; do not infer a different task: {request}",
    "I need a useful, human-readable outcome for this specific request: {request}",
    "Resolve only the requested scope and keep exact values unchanged: {request}",
    "This is not a template exercise; it is a realistic developer request. {request}",
    "Please distinguish discussion from authorization while handling this message: {request}",
    "Continue the workbench conversation with the most fitting response: {request}",
    "A teammate expects one precise outcome from this turn. {request}",
)


def _frame(index: int, request: str) -> str:
    return _FRAMES[index % len(_FRAMES)].format(request=request)


def build_decimal_control_rows() -> list[dict[str, Any]]:
    """Fresh 13-tool, 50/50 final suite for decimal-schedule controls."""

    rows: list[dict[str, Any]] = []
    for tool in TOOLS:
        for index in range(20):
            request, arguments = _tool_case(tool, index + 80)
            case_id = f"sealed-decimal-tool-{tool}-{index:02d}"
            rows.append(
                _row(
                    "sealed",
                    case_id,
                    _frame(index, request),
                    _assistant_tool(tool, arguments, case_id),
                    tool,
                    f"decimal-action-{tool}-{index:02d}",
                )
            )
        for index in range(10):
            request, answer, required, stratum = _negative_case(tool, index + 70)
            rows.append(
                _row(
                    "sealed",
                    f"sealed-decimal-negative-{tool}-{index:02d}",
                    _frame(index + 30, request),
                    {"role": "assistant", "content": answer},
                    None,
                    f"decimal-negative-{stratum}",
                    required_terms=required,
                )
            )
    for index in range(130):
        request, answer, required, stratum = _direct_case(index + 500)
        reference = f"Decimal control conversation {310_000 + index}"
        rows.append(
            _row(
                "sealed",
                f"sealed-decimal-direct-{index:03d}",
                _frame(index + 60, f"{reference}. {request}"),
                {"role": "assistant", "content": f"{answer}\nReference: {reference}."},
                None,
                f"decimal-direct-{stratum}",
                required_terms=required,
            )
        )
    random.Random(CONTROL_SEED).shuffle(rows)
    return rows


def audit_decimal_control_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(row["evaluation_metadata"]["decision"] for row in rows)
    tools = Counter(
        row["evaluation_metadata"]["tool"]
        for row in rows
        if row["evaluation_metadata"]["tool"]
    )
    comparison = {split: build_rows(split) for split in ("train", "dev", "sealed")}
    comparison["ood-v2"] = build_ood_rows()
    comparison["focused-v3"] = build_focused_rows()
    signatures = _signatures(rows)
    skeletons = {_prompt_skeleton(row) for row in rows}
    exact_overlap = {
        split: len(signatures & _signatures(other)) for split, other in comparison.items()
    }
    skeleton_overlap = {
        split: len(skeletons & {_prompt_skeleton(row) for row in other})
        for split, other in comparison.items()
    }
    problems: list[str] = []
    if len(rows) != CONTROL_ROWS or len(signatures) != CONTROL_ROWS:
        problems.append("control suite must contain 520 unique normalized examples")
    if decisions != {"tool": 260, "direct": 260}:
        problems.append("control suite must be exactly balanced 260 tool / 260 direct")
    if set(tools) != set(TOOLS) or set(tools.values()) != {20}:
        problems.append("control suite must contain 20 cases for each of 13 tools")
    if any(exact_overlap.values()):
        problems.append(f"exact normalized overlap detected: {exact_overlap}")
    if any(skeleton_overlap.values()):
        problems.append(f"prompt skeleton overlap detected: {skeleton_overlap}")
    report: dict[str, Any] = {
        "status": "blocked" if problems else "ready",
        "schema_version": 1,
        "purpose": "fresh final comparison of decimal-pair and constant noise schedules",
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


def ensure_decimal_control_suite() -> dict[str, Any]:
    rows = build_decimal_control_rows()
    audit = audit_decimal_control_rows(rows)
    if audit["status"] != "ready":
        raise ValueError(f"decimal-control suite audit failed: {audit['problems']}")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONTROL_PATH.exists() or CONTROL_PATH.read_text(encoding="utf-8") != payload:
        CONTROL_PATH.write_text(payload, encoding="utf-8")
    resolved = str(CONTROL_PATH.resolve())
    suite = next(
        (item for item in list_sealed_suites() if str(Path(item["source"]).resolve()) == resolved),
        None,
    )
    if suite is None:
        suite = register_sealed_suite(
            resolved,
            "FINAL Decimal Noise Controls v4 — untouched — 520 — 13 tools",
            max_rows=CONTROL_ROWS,
        )
    return {
        **audit,
        "path": resolved,
        "suite_id": suite["id"],
        "suite_name": suite["name"],
    }
