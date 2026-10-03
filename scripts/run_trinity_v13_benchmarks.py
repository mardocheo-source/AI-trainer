#!/usr/bin/env python3
"""Run V13 final benchmarks sequentially after the adapter has been frozen."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exotic_trainer.bfcl import (
    BFCL_GENERATION_TIMEOUT_SECONDS,
    BFCL_LONG_PROMPT_CACHE_THRESHOLD,
    BFCL_REQUEST_TIMEOUT_SECONDS,
    BFCL_SPECS,
    bfcl_environment_status,
    run_bfcl,
    xpu_vram_status,
)
from exotic_trainer.v13_campaign import default_campaign_root

DEFAULT_MODEL = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
GIB = 1024**3


def _assert_resource_headroom(run_root: Path) -> None:
    free_disk = shutil.disk_usage(run_root).free / GIB
    if free_disk < 8.0:
        raise RuntimeError(f"Benchmark preflight refused: only {free_disk:.2f} GiB disk is free")
    status = xpu_vram_status()
    if status is None:
        raise RuntimeError("Benchmark preflight refused: /run/xe-gpu-vram is unavailable")
    free_vram = float(status["total_gib"]) - float(status["used_gib"])
    if free_vram < 7.0:
        raise RuntimeError(
            f"Benchmark preflight refused: only {free_vram:.2f} GiB XPU VRAM is free"
        )


def _run_sealed(args: argparse.Namespace, adapter: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_trinity_v13_sealed_eval.py"),
        "--run-root",
        str(args.run_root),
        "--adapter",
        str(adapter),
        "--model",
        str(args.model),
        "--xpu-memory-fraction",
        str(args.xpu_memory_fraction),
        "--execute-xpu",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_bfcl_v3(args: argparse.Namespace, adapter: Path) -> None:
    environment = bfcl_environment_status("v3")
    if environment["status"] != "ready":
        raise RuntimeError("Pinned BFCL v3 is not ready; run the existing bfcl-setup v3 first")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.run_root / "benchmarks" / "bfcl_v3" / "raw" / stamp
    output.mkdir(parents=True, exist_ok=False)
    subset_dir = REPO_ROOT / "downloads" / "bfcl-v3-25pct"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_bfcl_v3_subset.py"),
            "--output-dir",
            str(subset_dir),
            "--ratio",
            "0.25",
            "--seed",
            "20260823",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subset_manifest = json.loads((subset_dir / "manifest.json").read_text(encoding="utf-8"))
    config = {
        "schema_version": 1,
        "version": "v3",
        "bfcl_commit": BFCL_SPECS["v3"]["commit"],
        "profile": "official-stratified-25pct",
        "categories": ["all"],
        "partial": True,
        "data_dir": str(subset_dir),
        "subset_manifest": {
            "ratio": subset_manifest["ratio"],
            "seed": subset_manifest["seed"],
            "source_cases": subset_manifest["source_cases"],
            "selected_cases": subset_manifest["selected_cases"],
            "fingerprint": subset_manifest["fingerprint"],
        },
        "handler": "liquid-lfm2",
        "handler_provenance": "pinned official BFCL v3 evaluator with local Liquid handler",
        "transport": "managed-local-xpu",
        "target": str(adapter),
        "target_name": "trinity-v13-strict-genetic",
        "model_path": str(Path(args.model).resolve()),
        "adapter_path": str(adapter),
        "endpoint": None,
        "api_model": "liquid-lfm2",
        "smoke_samples": 20,
        "device": "xpu",
        "runs_dir": str(args.run_root),
        "bfcl_run_id": f"trinity-v13-bfcl-v3-{stamp}",
        "output_dir": str(output),
        "runtime_limits": {
            "request_timeout_seconds": BFCL_REQUEST_TIMEOUT_SECONDS,
            "generation_timeout_seconds": BFCL_GENERATION_TIMEOUT_SECONDS,
            "openai_max_retries": 0,
            "long_prompt_cache_threshold": BFCL_LONG_PROMPT_CACHE_THRESHOLD,
            "xpu_memory_fraction": args.xpu_memory_fraction,
            "max_new_tokens": 1024,
        },
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "state.json").write_text(
        json.dumps(
            {
                "bfcl_run_id": config["bfcl_run_id"],
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
    run_bfcl(config_path)


def _run_bfcl_v4(args: argparse.Namespace, adapter: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_bfcl_v4_official_subset.py"),
        "--adapter",
        str(adapter),
        "--model",
        str(args.model),
        "--data-dir",
        str(REPO_ROOT / "downloads" / "bfcl-v4-25pct"),
        "--output-dir",
        str(args.run_root / "benchmarks" / "bfcl_v4" / "raw"),
        "--scope",
        "all",
        "--xpu-memory-fraction",
        str(args.xpu_memory_fraction),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=default_campaign_root())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", choices=("sealed", "bfcl_v3", "bfcl_v4", "all"), default="all")
    parser.add_argument("--xpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--execute-xpu", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if not args.execute_xpu:
        parser.error("benchmark execution is disabled unless --execute-xpu is explicit")
    if not 0.50 <= args.xpu_memory_fraction <= 0.70:
        parser.error("--xpu-memory-fraction must stay in the audited 0.50-0.70 range")
    adapter = args.run_root / "adapter-final"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Frozen V13 adapter is missing: {adapter}")
    selected = ("sealed", "bfcl_v3", "bfcl_v4") if args.suite == "all" else (args.suite,)
    for suite in selected:
        _assert_resource_headroom(args.run_root)
        if suite == "sealed":
            _run_sealed(args, adapter)
        elif suite == "bfcl_v3":
            _run_bfcl_v3(args, adapter)
        else:
            _run_bfcl_v4(args, adapter)
    print(
        json.dumps(
            {"status": "complete", "suites": selected, "run_root": str(args.run_root)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
