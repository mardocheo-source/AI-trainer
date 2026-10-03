#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exotic_trainer.v13_campaign import default_campaign_root, generate_campaign_reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate V13 benchmark CSV, Markdown reports and category PNGs from raw files."
    )
    parser.add_argument("--run-root", type=Path, default=default_campaign_root())
    args = parser.parse_args()
    print(json.dumps(generate_campaign_reports(args.run_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
