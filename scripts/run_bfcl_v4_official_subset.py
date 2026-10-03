#!/usr/bin/env python3
"""Run a selected BFCL v4 JSONL subset with the pinned official evaluator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exotic_trainer.bfcl import (
    BFCL_GENERATION_TIMEOUT_SECONDS,
    BFCL_LONG_PROMPT_CACHE_THRESHOLD,
    BFCL_REQUEST_TIMEOUT_SECONDS,
    BFCL_SPECS,
    run_bfcl,
)

DEFAULT_MODEL = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
DEFAULT_DATA = REPO_ROOT / "downloads/bfcl-v4-25pct"
OFFICIAL_SOURCE = REPO_ROOT / ".cache/bfcl/source-v4/berkeley-function-call-leaderboard"


def _read_ids(path: Path) -> list[str]:
    identifiers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict) and "id" in item:
            identifiers.append(str(item["id"]))
    return identifiers


def _dependency_closure(entries: list[dict[str, Any]], wanted: set[str]) -> list[str]:
    by_id = {str(entry["id"]): entry for entry in entries}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"Selected IDs are absent from official BFCL data: {missing[:5]}")
    closure = set(wanted)
    pending = list(wanted)
    while pending:
        current = pending.pop()
        for dependency in by_id[current].get("depends_on") or []:
            dependency = str(dependency)
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return [str(entry["id"]) for entry in entries if str(entry["id"]) in closure]


def build_official_id_map(
    data_dir: Path, scope: str, limit_per_category: int | None = None
) -> dict[str, list[str]]:
    sys.path.insert(0, str(OFFICIAL_SOURCE))
    from bfcl_eval.utils import load_dataset_entry

    selected: dict[str, list[str]] = {}
    for path in sorted(data_dir.glob("BFCL_v4_*.json")):
        category = path.stem.removeprefix("BFCL_v4_")
        if category == "format_sensitivity":
            continue
        identifiers = _read_ids(path)
        if limit_per_category is not None:
            identifiers = identifiers[:limit_per_category]
        if not identifiers:
            continue
        if category.startswith("multi_turn") and scope not in {
            "multi-turn",
            "single-and-multi",
            "all",
        }:
            continue
        if category in {"memory", "web_search"} and scope not in {"agentic", "all"}:
            continue
        if (
            not category.startswith("multi_turn")
            and category not in {"memory", "web_search"}
            and scope in {"multi-turn", "agentic"}
        ):
            continue

        if category == "memory":
            for backend in ("kv", "vector", "rec_sum"):
                official_category = f"memory_{backend}"
                wanted = {
                    identifier.replace("memory", official_category, 1) for identifier in identifiers
                }
                entries = load_dataset_entry(official_category)
                selected[official_category] = _dependency_closure(entries, wanted)
        elif category == "web_search":
            for variant in ("base", "no_snippet"):
                official_category = f"web_search_{variant}"
                wanted = [
                    identifier.replace("web_search", official_category, 1)
                    for identifier in identifiers
                ]
                available = {str(entry["id"]) for entry in load_dataset_entry(official_category)}
                missing = sorted(set(wanted) - available)
                if missing:
                    raise ValueError(
                        f"Selected IDs are absent from {official_category}: {missing[:5]}"
                    )
                selected[official_category] = wanted
        else:
            available = {str(entry["id"]) for entry in load_dataset_entry(category)}
            missing = sorted(set(identifiers) - available)
            if missing:
                raise ValueError(f"Selected IDs are absent from {category}: {missing[:5]}")
            selected[category] = identifiers
    if not selected:
        raise ValueError(f"No BFCL cases found in {data_dir}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Resume an existing official subset output directory",
    )
    parser.add_argument(
        "--scope",
        choices=("single-turn", "multi-turn", "agentic", "single-and-multi", "all"),
        default="all",
        help="Official sections to include from the selected subset",
    )
    parser.add_argument("--device", default="xpu")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Local generation cap; 256 is sufficient for BFCL function-call syntax",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        help="Diagnostic cap applied before official agentic category expansion",
    )
    parser.add_argument(
        "--xpu-memory-fraction",
        type=float,
        default=0.70,
        help="Hard per-process XPU allocator cap (audited maximum: 0.70).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_dir is not None:
        output = args.resume_dir.resolve()
        config_path = output / "config.json"
        state_path = output / "state.json"
        if not config_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"Incomplete BFCL resume directory: {output}")
        print(f"Resuming official BFCL v4 subset: {output}", flush=True)
        state = run_bfcl(config_path)
        print(json.dumps(state.get("score_summary", {}), indent=2), flush=True)
        return

    if args.adapter is None or args.output_dir is None:
        raise ValueError("--adapter and --output-dir are required for a new run")
    adapter = args.adapter.resolve()
    data_dir = args.data_dir.resolve()
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Invalid adapter directory: {adapter}")
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    if args.limit_per_category is not None and args.limit_per_category < 1:
        raise ValueError("--limit-per-category must be positive")
    if args.max_new_tokens < 32:
        raise ValueError("--max-new-tokens must be at least 32")
    if not 0.50 <= args.xpu_memory_fraction <= 0.70:
        raise ValueError("--xpu-memory-fraction must stay in the audited 0.50-0.70 range")
    id_map = build_official_id_map(data_dir, args.scope, limit_per_category=args.limit_per_category)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = (args.output_dir / f"{stamp}_{adapter.parent.name}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected_count = sum(len(ids) for ids in id_map.values())
    run_id = f"bfcl-v4-official-subset-{stamp}"
    config = {
        "schema_version": 1,
        "version": "v4",
        "bfcl_commit": BFCL_SPECS["v4"]["commit"],
        "profile": f"official-subset-{args.scope}",
        "categories": sorted(id_map),
        "partial": True,
        "test_case_ids": id_map,
        "handler": "liquid-lfm2",
        "handler_provenance": "pinned official BFCL v4 evaluator with local Liquid handler",
        "transport": "managed-local-xpu",
        "target": str(adapter),
        "target_name": adapter.parent.name,
        "model_path": str(Path(args.model).resolve()),
        "adapter_path": str(adapter),
        "endpoint": None,
        "api_model": "liquid-lfm2",
        "smoke_samples": selected_count,
        "device": args.device,
        "runs_dir": str(REPO_ROOT / "runs"),
        "bfcl_run_id": run_id,
        "output_dir": str(output),
        "runtime_limits": {
            "request_timeout_seconds": BFCL_REQUEST_TIMEOUT_SECONDS,
            "generation_timeout_seconds": BFCL_GENERATION_TIMEOUT_SECONDS,
            "openai_max_retries": 0,
            "long_prompt_cache_threshold": BFCL_LONG_PROMPT_CACHE_THRESHOLD,
            "xpu_memory_fraction": args.xpu_memory_fraction,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "state.json").write_text(
        json.dumps(
            {
                "bfcl_run_id": run_id,
                "status": "queued",
                "stage": "queued",
                "created_at": time.time(),
                "config": config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "selection_manifest.json").write_text(
        json.dumps(
            {
                "source": str(data_dir),
                "scope": args.scope,
                "limit_per_category": args.limit_per_category,
                "selected_entries_including_prerequisites": selected_count,
                "categories": {key: len(value) for key, value in id_map.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Official BFCL v4 subset: {selected_count} entries across "
        f"{len(id_map)} categories; output={output}",
        flush=True,
    )
    state = run_bfcl(config_path)
    print(json.dumps(state.get("score_summary", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
