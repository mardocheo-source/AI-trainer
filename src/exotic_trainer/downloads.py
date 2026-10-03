from __future__ import annotations

import shutil
from pathlib import Path

from .paths import configure_huggingface_cache, project_root


def split_hf_file_source(source: str) -> tuple[str, str] | None:
    if not source.startswith("hf-file:") or "::" not in source:
        return None
    repository, filename = source.removeprefix("hf-file:").split("::", 1)
    if not repository or not filename:
        raise ValueError("HF file source must be hf-file:owner/dataset::path/to/file.jsonl")
    return repository, filename


def fetch_huggingface_file(source: str, destination: str | Path | None = None) -> Path:
    configure_huggingface_cache()
    from huggingface_hub import hf_hub_download

    parsed = split_hf_file_source(source)
    if parsed is None:
        raise ValueError("not a Hugging Face file source")
    repository, filename = parsed
    cached = Path(hf_hub_download(repository, filename=filename, repo_type="dataset"))
    target = (
        Path(destination).expanduser().resolve()
        if destination
        else project_root() / "downloads" / repository.replace("/", "--") / filename
    )
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
        target = target / Path(filename).name
    shutil.copy2(cached, target)
    return target


def fetch_huggingface_dataset(source: str, destination: str | Path | None = None) -> Path:
    if split_hf_file_source(source):
        return fetch_huggingface_file(source, destination)
    configure_huggingface_cache()
    from datasets import Dataset, DatasetDict, load_dataset

    source = source.removeprefix("hf:")
    target = (
        Path(destination).expanduser().resolve()
        if destination
        else project_root() / "downloads" / source.replace("/", "--")
    )
    target.mkdir(parents=True, exist_ok=True)
    loaded = load_dataset(source)
    if isinstance(loaded, Dataset):
        loaded.to_parquet(target / "train.parquet")
    elif isinstance(loaded, DatasetDict):
        for split, dataset in loaded.items():
            dataset.to_parquet(target / f"{split}.parquet")
    else:
        raise TypeError(f"unsupported downloaded dataset object: {type(loaded).__name__}")
    return target
