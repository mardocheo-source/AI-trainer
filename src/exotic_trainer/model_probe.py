from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import ModelProbe

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        if len(header_length_raw) != 8:
            raise ValueError(f"invalid safetensors file: {path}")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        if header_length > 512 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header length: {header_length}")
        return json.loads(handle.read(header_length))


def _fingerprint(directory: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        stat = path.stat()
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        if path.suffix in {".json", ".jinja"} and stat.st_size < 10_000_000:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def inspect_model(path: str | Path) -> ModelProbe:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {directory}")
    config_path = directory / "config.json"
    if not config_path.exists():
        raise ValueError(f"missing config.json in {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_files = sorted(directory.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"no .safetensors weights found in {directory}")

    tensor_names: list[str] = []
    parameter_count = 0
    dtype_counts: Counter[str] = Counter()
    for weight_file in weight_files:
        for name, metadata in _safetensors_header(weight_file).items():
            if name == "__metadata__":
                continue
            tensor_names.append(name)
            shape = metadata.get("shape", [])
            count = 1
            for dimension in shape:
                count *= int(dimension)
            parameter_count += count
            dtype_counts[str(metadata.get("dtype", "unknown"))] += count

    module_names = sorted({name.rsplit(".weight", 1)[0] for name in tensor_names if name.endswith(".weight")})
    linear_suffixes = sorted(
        {
            module.rsplit(".", 1)[-1]
            for module in module_names
            if any(
                marker in module.rsplit(".", 1)[-1]
                for marker in ("proj", "w1", "w2", "w3", "linear", "dense")
            )
        }
    )
    all_files = [config_path, *weight_files]
    all_files.extend(path for path in directory.glob("tokenizer*"))
    disk_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    architecture = (config.get("architectures") or ["unknown"])[0]
    dominant_dtype = dtype_counts.most_common(1)[0][0] if dtype_counts else str(config.get("dtype"))

    return ModelProbe(
        name=directory.name,
        path=str(directory),
        architecture=architecture,
        model_type=str(config.get("model_type", "unknown")),
        dtype=dominant_dtype.lower(),
        parameter_count=parameter_count,
        disk_bytes=disk_bytes,
        hidden_size=config.get("hidden_size"),
        layer_count=config.get("num_hidden_layers"),
        context_length=config.get("max_position_embeddings"),
        vocab_size=config.get("vocab_size"),
        linear_suffixes=linear_suffixes,
        tensor_modules=module_names,
        fingerprint=_fingerprint(directory, all_files),
    )


def choose_lora_targets(probe: ModelProbe, profile: str = "auto") -> list[str]:
    modules = probe.tensor_modules
    attention = [
        name
        for name in modules
        if ".self_attn." in name
        and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "out_proj"}
    ]
    mlp = [
        name
        for name in modules
        if ".feed_forward." in name and name.rsplit(".", 1)[-1] in {"w1", "w2", "w3"}
    ]
    other_linear = [
        name
        for name in modules
        if name.rsplit(".", 1)[-1] in set(probe.linear_suffixes)
        and "embed_tokens" not in name
        and "lm_head" not in name
    ]
    if profile == "attention":
        selected = attention
    elif profile == "attention-mlp":
        selected = attention + mlp
    elif profile == "all-linear":
        selected = other_linear
    elif profile == "auto":
        selected = attention + mlp if probe.parameter_count < 3_000_000_000 else attention
    else:
        raise ValueError(f"unknown target profile: {profile}")
    if not selected:
        raise ValueError(f"no LoRA-compatible tensor targets found for profile {profile}")
    return sorted(set(selected))

