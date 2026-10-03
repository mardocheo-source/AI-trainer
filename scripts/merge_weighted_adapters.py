#!/usr/bin/env python3
"""Merge two compatible LoRA adapters without lossy SVD compression.

For adapters trained from the same base model, each update is
``delta_i = scale_i * B_i @ A_i``. Concatenating the factors represents
``weight1 * delta_1 + weight2 * delta_2`` exactly at rank ``rank_1 + rank_2``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact weighted LoRA adapter merge")
    parser.add_argument(
        "--adapter1",
        default=str(
            REPO_ROOT
            / "runs/trinity_3d_v3_5_state_binding_H01_s720_prod/adapter-final"
        ),
        help="Anchor adapter directory",
    )
    parser.add_argument(
        "--adapter2",
        default=str(REPO_ROOT / "runs/trinity_4d_golden_r24_s750_prod/adapter-final"),
        help="Specialist adapter directory",
    )
    parser.add_argument(
        "--weight1", type=float, default=1.0, help="Anchor coefficient (default: 1.0)"
    )
    parser.add_argument(
        "--weight2", type=float, default=0.05, help="Specialist coefficient (default: 0.05)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs/trinity_exact_anchor_4d_golden_l005/adapter-final"),
        help="New merged adapter directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing merge artifacts in an existing output directory",
    )
    return parser.parse_args()


def _read_config(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def adapter_scaling(config: dict[str, Any]) -> float:
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank


def _module_pairs(tensors: dict[str, torch.Tensor]) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    for key in tensors:
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in tensors:
            raise ValueError(f"Missing paired LoRA B tensor for {key}")
        pairs[key.split(".lora_A.", 1)[0]] = (key, b_key)
    return pairs


def _validate_compatibility(
    config1: dict[str, Any],
    config2: dict[str, Any],
    tensors1: dict[str, torch.Tensor],
    tensors2: dict[str, torch.Tensor],
) -> dict[str, tuple[tuple[str, str], tuple[str, str]]]:
    if config1.get("use_dora") or config2.get("use_dora"):
        raise ValueError("Exact concatenation currently supports LoRA/rsLoRA, not DoRA")
    base1 = config1.get("base_model_name_or_path")
    base2 = config2.get("base_model_name_or_path")
    if base1 and base2 and Path(base1).name != Path(base2).name:
        raise ValueError(f"Adapters use different base models: {base1!r} != {base2!r}")

    pairs1 = _module_pairs(tensors1)
    pairs2 = _module_pairs(tensors2)
    if set(pairs1) != set(pairs2):
        missing1 = sorted(set(pairs2) - set(pairs1))
        missing2 = sorted(set(pairs1) - set(pairs2))
        raise ValueError(
            f"LoRA module sets differ; missing from adapter1={missing1}, "
            f"missing from adapter2={missing2}"
        )
    unsupported1 = sorted(
        key for key in tensors1 if ".lora_A." not in key and ".lora_B." not in key
    )
    unsupported2 = sorted(
        key for key in tensors2 if ".lora_A." not in key and ".lora_B." not in key
    )
    if unsupported1 or unsupported2:
        raise ValueError(
            "Adapters contain unsupported non-LoRA tensors; refusing a partial merge: "
            f"adapter1={unsupported1}, adapter2={unsupported2}"
        )

    result: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    for module in sorted(pairs1):
        a1_key, b1_key = pairs1[module]
        a2_key, b2_key = pairs2[module]
        a1, b1 = tensors1[a1_key], tensors1[b1_key]
        a2, b2 = tensors2[a2_key], tensors2[b2_key]
        if a1.ndim != 2 or b1.ndim != 2 or a2.ndim != 2 or b2.ndim != 2:
            raise ValueError(f"Expected 2-D LoRA tensors for module {module}")
        if a1.shape[1] != a2.shape[1] or b1.shape[0] != b2.shape[0]:
            raise ValueError(
                f"Incompatible projection shapes for {module}: "
                f"A={tuple(a1.shape)}/{tuple(a2.shape)}, "
                f"B={tuple(b1.shape)}/{tuple(b2.shape)}"
            )
        result[module] = (pairs1[module], pairs2[module])
    return result


def merge_exact_adapters(
    adapter1: Path,
    adapter2: Path,
    output_dir: Path,
    *,
    weight1: float,
    weight2: float,
    force: bool = False,
) -> dict[str, Any]:
    """Create an exact concatenated-factor merge and return its manifest."""
    if not math.isfinite(weight1) or not math.isfinite(weight2):
        raise ValueError("Merge weights must be finite")
    adapter1 = adapter1.resolve()
    adapter2 = adapter2.resolve()
    output_dir = output_dir.resolve()
    weights1_path = adapter1 / "adapter_model.safetensors"
    weights2_path = adapter2 / "adapter_model.safetensors"
    if not weights1_path.is_file() or not weights2_path.is_file():
        raise FileNotFoundError("adapter_model.safetensors is missing from an input adapter")

    managed_names = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "merge_manifest.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    existing = [output_dir / name for name in managed_names if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"Output already contains merge artifacts: {output_dir}; pass --force to replace them"
        )

    config1 = _read_config(adapter1)
    config2 = _read_config(adapter2)
    tensors1 = load_file(str(weights1_path), device="cpu")
    tensors2 = load_file(str(weights2_path), device="cpu")
    modules = _validate_compatibility(config1, config2, tensors1, tensors2)

    rank1 = int(config1["r"])
    rank2 = int(config2["r"])
    target_rank = rank1 + rank2
    scale1 = adapter_scaling(config1)
    scale2 = adapter_scaling(config2)

    # Keep the anchor scaling unchanged even though the concatenated rank is larger.
    output_uses_rslora = bool(config1.get("use_rslora"))
    target_alpha = (
        scale1 * math.sqrt(target_rank) if output_uses_rslora else scale1 * target_rank
    )
    target_scale = (
        target_alpha / math.sqrt(target_rank)
        if output_uses_rslora
        else target_alpha / target_rank
    )
    coefficient1 = weight1 * scale1 / target_scale
    coefficient2 = weight2 * scale2 / target_scale

    merged_tensors: dict[str, torch.Tensor] = {}
    for (a1_key, b1_key), (a2_key, b2_key) in modules.values():
        a1, b1 = tensors1[a1_key], tensors1[b1_key]
        a2, b2 = tensors2[a2_key], tensors2[b2_key]
        merged_tensors[a1_key] = torch.cat((a1, a2), dim=0).contiguous()
        merged_tensors[b1_key] = torch.cat(
            (
                (b1.float() * coefficient1).to(dtype=b1.dtype),
                (b2.float() * coefficient2).to(dtype=b1.dtype),
            ),
            dim=1,
        ).contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged_tensors, str(output_dir / "adapter_model.safetensors"))
    output_config = dict(config1)
    output_config["r"] = target_rank
    output_config["lora_alpha"] = target_alpha
    output_config["use_rslora"] = output_uses_rslora
    (output_dir / "adapter_config.json").write_text(
        json.dumps(output_config, indent=2) + "\n", encoding="utf-8"
    )
    for meta_name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        source = adapter1 / meta_name
        if source.is_file():
            shutil.copy2(source, output_dir / meta_name)

    manifest = {
        "fusion_type": "exact_lora_factor_concatenation",
        "formula": "weight1 * delta_adapter1 + weight2 * delta_adapter2",
        "weight_adapter1": weight1,
        "weight_adapter2": weight2,
        "adapter1_source": str(adapter1),
        "adapter2_source": str(adapter2),
        "rank_adapter1": rank1,
        "rank_adapter2": rank2,
        "rank_target": target_rank,
        "scaling_adapter1": scale1,
        "scaling_adapter2": scale2,
        "scaling_target": target_scale,
        "svd_compression": False,
        "anchor_exact_when_weight1_is_one": weight1 == 1.0 and coefficient1 == 1.0,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("EXACT LORA ADAPTER MERGE (NO SVD COMPRESSION)")
    print(f"Anchor:     {args.adapter1} (weight={args.weight1})")
    print(f"Specialist: {args.adapter2} (weight={args.weight2})")
    print(f"Output:     {args.output_dir}")
    print("=" * 80)
    try:
        manifest = merge_exact_adapters(
            Path(args.adapter1),
            Path(args.adapter2),
            Path(args.output_dir),
            weight1=args.weight1,
            weight2=args.weight2,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(
        f"Created exact rank-{manifest['rank_target']} adapter; "
        f"anchor_exact={manifest['anchor_exact_when_weight1_is_one']}"
    )


if __name__ == "__main__":
    main()
