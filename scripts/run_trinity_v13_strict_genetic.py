#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exotic_trainer.strict_genetic_pipeline import run_tournament
from exotic_trainer.v13_campaign import default_campaign_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the contamination-free, dual-seed Trinity v13 strict genetic tournament."
    )
    parser.add_argument("--population", type=int, default=5)
    parser.add_argument(
        "--round-steps",
        default="12,24,40",
        help="Comma-separated SFT optimizer steps for successive genetic rounds.",
    )
    parser.add_argument("--micro-rows", type=int, default=420)
    parser.add_argument("--tournament-seed", type=int, default=20260823)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_campaign_root(),
        help="Self-contained V13 run root (docs, tournament, final training and benchmarks).",
    )
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--final-steps", type=int, default=240)
    parser.add_argument("--final-preference-steps", type=int, default=48)
    parser.add_argument(
        "--execute-xpu",
        action="store_true",
        help="Required acknowledgement after reviewing the XPU memory safety report.",
    )
    args = parser.parse_args()
    if not args.execute_xpu:
        parser.error(
            "XPU execution is disabled by default after a Level Zero device loss; "
            "inspect the safety report and pass --execute-xpu explicitly."
        )
    steps = tuple(int(value) for value in args.round_steps.split(",") if value.strip())
    result = run_tournament(
        population=args.population,
        round_steps=steps,
        micro_rows=args.micro_rows,
        tournament_seed=args.tournament_seed,
        output_dir=args.output_dir,
        run_final=not args.no_final,
        final_steps=args.final_steps,
        final_preference_steps=args.final_preference_steps,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "output_dir": result["output_dir"],
                "champion": result.get("champion"),
                "final": result.get("final"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
