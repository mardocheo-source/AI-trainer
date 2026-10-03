#!/usr/bin/env python3
"""Create a deterministic stratified subset of the pinned official BFCL v3 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = (
    REPO_ROOT
    / ".cache"
    / "bfcl"
    / "source-v3"
    / "berkeley-function-call-leaderboard"
    / "data"
)
DEFAULT_OUTPUT = REPO_ROOT / "downloads" / "bfcl-v3-25pct"
BFCL_V3_COMMIT = "70b6a4a2144597b1f99d1f4d3185d35d7ee532a4"


def _read_jsonl(path: Path, *, require_id: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict) or (require_id and "id" not in item):
                raise ValueError(f"Invalid BFCL row at {path}:{line_number}")
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _category_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _manifest_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def prepare_subset(source: Path, output: Path, *, ratio: float, seed: int) -> dict[str, Any]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be strictly between 0 and 1")
    source = source.resolve()
    output = output.resolve()
    files = sorted(source.glob("BFCL_v3_*.json"))
    if not files:
        raise FileNotFoundError(f"Pinned BFCL v3 data not found: {source}")

    manifest_core: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "bfcl-v3-official-stratified-subset",
        "bfcl_commit": BFCL_V3_COMMIT,
        "ratio": ratio,
        "seed": seed,
        "selection": "per-category deterministic random sample without replacement",
        "partial": True,
        "publishable": False,
        "categories": {},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bfcl-v3-subset-", dir=output.parent) as temp_name:
        temporary = Path(temp_name)
        answer_source = source / "possible_answer"
        for data_path in files:
            rows = _read_jsonl(data_path)
            sample_count = max(1, min(len(rows), math.ceil(len(rows) * ratio)))
            rng = random.Random(_category_seed(seed, data_path.stem))
            indices = sorted(rng.sample(range(len(rows)), sample_count))
            selected = [rows[index] for index in indices]
            selected_ids = [
                str(row.get("id") or f"{data_path.stem}:source-index-{index}")
                for index, row in zip(indices, selected, strict=True)
            ]
            _write_jsonl(temporary / data_path.name, selected)

            answer_path = answer_source / data_path.name
            answer_count = 0
            if answer_path.is_file():
                answers = _read_jsonl(answer_path, require_id=True)
                by_id = {str(row["id"]): row for row in answers}
                missing = [identifier for identifier in selected_ids if identifier not in by_id]
                if missing:
                    raise ValueError(
                        f"Missing possible answers for {data_path.name}: {missing[:5]}"
                    )
                selected_answers = [by_id[identifier] for identifier in selected_ids]
                _write_jsonl(temporary / "possible_answer" / data_path.name, selected_answers)
                answer_count = len(selected_answers)

            manifest_core["categories"][data_path.stem] = {
                "source_cases": len(rows),
                "selected_cases": len(selected),
                "possible_answers": answer_count,
                "selected_ids": selected_ids,
            }

        manifest_core["source_cases"] = sum(
            item["source_cases"] for item in manifest_core["categories"].values()
        )
        manifest_core["selected_cases"] = sum(
            item["selected_cases"] for item in manifest_core["categories"].values()
        )
        manifest = {
            **manifest_core,
            "fingerprint": _manifest_fingerprint(manifest_core),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        existing_manifest = output / "manifest.json"
        if existing_manifest.is_file():
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != manifest["fingerprint"]:
                raise RuntimeError(
                    f"Refusing to overwrite a different frozen BFCL v3 subset: {output}"
                )
            return existing
        if output.exists():
            raise RuntimeError(f"Output exists without a matching frozen manifest: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temporary, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    manifest = prepare_subset(
        args.source_dir,
        args.output_dir,
        ratio=args.ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "output_dir": str(args.output_dir.resolve()),
                "source_cases": manifest["source_cases"],
                "selected_cases": manifest["selected_cases"],
                "ratio": manifest["ratio"],
                "fingerprint": manifest["fingerprint"],
                "partial": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
