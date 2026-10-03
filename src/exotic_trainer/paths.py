from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    configured = os.environ.get("EXOTIC_TRAINER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").exists():
        return package_root
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").exists():
        return current
    return current


def registry_path() -> Path:
    configured = os.environ.get("EXOTIC_TRAINER_REGISTRY")
    if configured:
        return Path(configured).expanduser().resolve()
    return project_root() / "data" / "registry.sqlite3"


def runs_dir() -> Path:
    configured = os.environ.get("EXOTIC_TRAINER_RUNS_DIR")
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else project_root() / "runs"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_huggingface_cache() -> Path:
    root = project_root() / ".cache" / "huggingface"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    return root
