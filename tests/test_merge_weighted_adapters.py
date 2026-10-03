from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


def _load_merge_module():
    script = Path(__file__).resolve().parents[1] / "scripts/merge_weighted_adapters.py"
    spec = importlib.util.spec_from_file_location("merge_weighted_adapters", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_adapter(path: Path, a: torch.Tensor, b: torch.Tensor, *, alpha: int) -> None:
    path.mkdir(parents=True)
    config = {
        "r": a.shape[0],
        "lora_alpha": alpha,
        "use_rslora": True,
        "use_dora": False,
        "base_model_name_or_path": "/models/test-base",
        "target_modules": ["q_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": a,
            "base_model.model.q_proj.lora_B.weight": b,
        },
        str(path / "adapter_model.safetensors"),
    )


def test_exact_concat_reconstructs_weighted_delta(tmp_path: Path) -> None:
    merge = _load_merge_module()
    adapter1, adapter2, output = tmp_path / "a1", tmp_path / "a2", tmp_path / "out"
    a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b1 = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    a2 = torch.tensor([[2.0, -1.0]])
    b2 = torch.tensor([[3.0], [-2.0]])
    _write_adapter(adapter1, a1, b1, alpha=4)
    _write_adapter(adapter2, a2, b2, alpha=3)

    manifest = merge.merge_exact_adapters(
        adapter1, adapter2, output, weight1=1.0, weight2=0.25
    )
    tensors = load_file(str(output / "adapter_model.safetensors"))
    config = json.loads((output / "adapter_config.json").read_text())
    a_out = tensors["base_model.model.q_proj.lora_A.weight"]
    b_out = tensors["base_model.model.q_proj.lora_B.weight"]
    scale_out = config["lora_alpha"] / math.sqrt(config["r"])
    expected = (4 / math.sqrt(2)) * (b1 @ a1) + 0.25 * 3 * (b2 @ a2)

    assert config["r"] == 3
    assert manifest["svd_compression"] is False
    assert manifest["anchor_exact_when_weight1_is_one"] is True
    assert torch.equal(a_out[:2], a1)
    assert torch.allclose(scale_out * (b_out @ a_out), expected, atol=1e-6, rtol=1e-6)


def test_existing_output_requires_force(tmp_path: Path) -> None:
    merge = _load_merge_module()
    adapter1, adapter2, output = tmp_path / "a1", tmp_path / "a2", tmp_path / "out"
    a = torch.ones((1, 2))
    b = torch.ones((2, 1))
    _write_adapter(adapter1, a, b, alpha=2)
    _write_adapter(adapter2, a, b, alpha=2)
    merge.merge_exact_adapters(adapter1, adapter2, output, weight1=1.0, weight2=0.1)

    with pytest.raises(FileExistsError):
        merge.merge_exact_adapters(adapter1, adapter2, output, weight1=1.0, weight2=0.1)
