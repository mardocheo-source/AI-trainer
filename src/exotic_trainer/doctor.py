from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys
from typing import Any


def system_report() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "gradio", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "venv": sys.prefix != sys.base_prefix,
        "xpu_smi": shutil.which("xpu-smi"),
        "packages": packages,
        "torch_xpu_available": False,
        "xpu_devices": [],
        "errors": [],
    }
    try:
        import torch

        available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        report["torch_xpu_available"] = available
        if available:
            for index in range(torch.xpu.device_count()):
                properties = torch.xpu.get_device_properties(index)
                report["xpu_devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory": getattr(properties, "total_memory", None),
                    }
                )
    except Exception as error:  # noqa: BLE001 - diagnostics must report arbitrary driver failures
        report["errors"].append(f"torch/xpu: {type(error).__name__}: {error}")
    return report
