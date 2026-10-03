#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exotic_trainer.strict_genetic_data import ensure_strict_genetic_corpus
from exotic_trainer.v13_campaign import (
    create_campaign_layout,
    default_campaign_root,
    materialize_selected_adapter,
    validate_campaign_layout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate the self-contained V13 run.")
    parser.add_argument("--run-root", type=Path, default=default_campaign_root())
    parser.add_argument(
        "--materialize-adapter",
        action="store_true",
        help="Copy the already-selected final adapter into RUN_ROOT/adapter-final.",
    )
    args = parser.parse_args()

    campaign = create_campaign_layout(args.run_root)
    corpus = ensure_strict_genetic_corpus()
    run_root = args.run_root.resolve()
    (run_root / "data" / "strict_genetic_corpus_manifest.json").write_text(
        json.dumps(corpus, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "campaign": campaign,
        "layout": validate_campaign_layout(run_root),
        "corpus_status": corpus.get("status"),
        "benchmark_sources_used": corpus.get("benchmark_sources_used"),
        "sealed_sources_used": corpus.get("sealed_sources_used"),
    }
    if args.materialize_adapter:
        result["adapter"] = materialize_selected_adapter(run_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
