from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .dataset_probe import normalize_record, tool_names_in_example
from .downloads import fetch_huggingface_file, split_hf_file_source
from .paths import configure_huggingface_cache
from .tool_schema import schema_conditioned_system_prompt
from .token_roles import TokenRole, role_spans


CAPABILITY_COLUMNS = (
    "cap_is_tool",
    "cap_is_parallel",
    "cap_is_multiturn",
    "cap_is_typed",
    "turn_index",
    "turn_count",
    "trajectory_weight",
    "cap_is_docstring_derived",
    "cap_expected_call_count",
)


def _completion_text(example: dict[str, Any]) -> str:
    completion = example.get("completion")
    if not isinstance(completion, list):
        return str(completion or "")
    return "\n".join(
        str(message.get("content") or "")
        for message in completion
        if isinstance(message, dict)
    )


def _capability_targets(
    record: dict[str, Any],
    example: dict[str, Any],
    *,
    turn_index: int,
    turn_count: int,
) -> dict[str, int | float]:
    metadata = record.get("training_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    per_turn = metadata.get("assistant_turn_capabilities")
    turn_metadata: dict[str, Any] = {}
    if isinstance(per_turn, list) and turn_index < len(per_turn):
        candidate = per_turn[turn_index]
        if isinstance(candidate, dict):
            turn_metadata = candidate
    category = str(record.get("category") or metadata.get("category") or "canonical").lower()
    text = _completion_text(example)
    spans = role_spans(text)
    inferred_tool = any(role == TokenRole.TOOL_NAME for _, _, role in spans)
    tool = bool(
        turn_metadata.get(
            "is_tool",
            metadata.get("is_tool", record.get("expected_tool") is not None or inferred_tool),
        )
    )
    parallel = bool(
        turn_metadata.get(
            "is_parallel",
            metadata.get(
                "is_parallel",
                "parallel" in category,
            ),
        )
    )
    multiturn = bool(
        turn_metadata.get(
            "is_multiturn",
            metadata.get(
                "is_multiturn",
                turn_count > 1 or "multi_turn" in category or "multiturn" in category,
            ),
        )
    )
    typed = bool(
        turn_metadata.get(
            "is_typed",
            metadata.get(
                "is_typed",
                tool or any(name in category for name in ("java", "javascript", "sql", "typed")),
            ),
        )
    )
    structure = metadata.get("structure")
    if not isinstance(structure, dict):
        structure = {}
    docstring_derived = bool(
        turn_metadata.get(
            "is_docstring_derived",
            metadata.get(
                "is_docstring_derived",
                structure.get("is_docstring_derived", False),
            ),
        )
    )
    expected_call_count = int(
        turn_metadata.get(
            "expected_call_count",
            metadata.get(
                "expected_call_count",
                structure.get("expected_call_count", 1),
            ),
        )
        or 1
    )
    return {
        "cap_is_tool": int(tool),
        "cap_is_parallel": int(parallel),
        "cap_is_multiturn": int(multiturn),
        "cap_is_typed": int(typed),
        "turn_index": int(turn_index),
        "turn_count": int(turn_count),
        "trajectory_weight": 1.0 / max(1, int(turn_count)),
        "cap_is_docstring_derived": int(docstring_derived),
        "cap_expected_call_count": max(1, expected_call_count),
    }


def _load_one(source: str, max_rows: int, seed: int) -> Any:
    cache_dir = configure_huggingface_cache() / "datasets"
    from datasets import Dataset, load_dataset

    path = Path(source).expanduser()
    if split_hf_file_source(source):
        path = fetch_huggingface_file(source)
    if not path.exists():
        stream = load_dataset(source, split="train", streaming=True, cache_dir=str(cache_dir))
        stream = stream.shuffle(seed=seed, buffer_size=min(10_000, max_rows * 2))
        return Dataset.from_list(list(stream.take(max_rows)))
    path = path.resolve()
    if path.is_dir():
        parquet = sorted(str(item) for item in path.rglob("*.parquet"))
        json_files = sorted(
            str(item) for item in path.rglob("*") if item.suffix.lower() in {".json", ".jsonl"}
        )
        if parquet:
            loaded = load_dataset(
                "parquet", data_files=parquet, split="train", cache_dir=str(cache_dir)
            )
            return _limit_local(loaded, max_rows, seed)
        if json_files:
            loaded = load_dataset(
                "json", data_files=json_files, split="train", cache_dir=str(cache_dir)
            )
            return _limit_local(loaded, max_rows, seed)
        raise ValueError(f"no JSON, JSONL or Parquet files found in {path}")
    if path.suffix.lower() in {".json", ".jsonl"}:
        loaded = load_dataset("json", data_files=str(path), split="train", cache_dir=str(cache_dir))
        return _limit_local(loaded, max_rows, seed)
    if path.suffix.lower() == ".parquet":
        loaded = load_dataset(
            "parquet", data_files=str(path), split="train", cache_dir=str(cache_dir)
        )
        return _limit_local(loaded, max_rows, seed)
    raise ValueError(f"unsupported local dataset format: {path.suffix}")


def _limit_local(dataset: Any, max_rows: int, seed: int) -> Any:
    if len(dataset) <= max_rows:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_rows))


def _normalize_dataset(dataset: Any, allowed_tools: set[str] | None = None) -> Any:
    columns = list(dataset.column_names)

    def expand(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = {
            "prompt": [],
            "completion": [],
            "group_id": [],
            "evaluation_metadata_json": [],
            "sampling_stratum": [],
            **{name: [] for name in CAPABILITY_COLUMNS},
        }
        count = len(next(iter(batch.values()))) if batch else 0
        for index in range(count):
            record = {column: batch[column][index] for column in columns}
            canonical = json.dumps(
                record, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
            )
            group_id = hashlib.sha256(canonical.encode()).hexdigest()
            evaluation_metadata = json.dumps(
                record.get("evaluation_metadata") or {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            training_metadata = record.get("training_metadata")
            if not isinstance(training_metadata, dict):
                training_metadata = {}
            sampling_stratum = str(
                record.get("category") or training_metadata.get("category") or "canonical"
            )
            normalized_items = normalize_record(record)
            turn_count = len(normalized_items)
            for turn_index, item in enumerate(normalized_items):
                output["prompt"].append(item["prompt"])
                output["completion"].append(item["completion"])
                output["group_id"].append(group_id)
                output["evaluation_metadata_json"].append(evaluation_metadata)
                output["sampling_stratum"].append(sampling_stratum)
                targets = _capability_targets(
                    record,
                    item,
                    turn_index=turn_index,
                    turn_count=turn_count,
                )
                for name in CAPABILITY_COLUMNS:
                    output[name].append(targets[name])
        return output

    normalized = dataset.map(
        expand,
        batched=True,
        batch_size=250,
        remove_columns=columns,
        desc="Normalizing agent trajectories",
    )
    if len(normalized) == 0:
        raise ValueError(
            "dataset produced no training samples; expected messages, instruction/output, "
            "question/answer or prompt/completion columns"
        )
    if allowed_tools is not None:
        normalized = normalized.filter(
            lambda example: tool_names_in_example(example).issubset(allowed_tools),
            desc="Filtering incompatible tool schemas",
        )
        if len(normalized) == 0:
            raise ValueError("all normalized rows were rejected by the allowed-tools filter")
    return normalized


def _condition_dataset_on_tool_menu(dataset: Any, tool_names: list[str]) -> Any:
    """Inject the LFM-compatible tool menu into normalized SFT prompts once."""
    if not tool_names:
        return dataset

    def condition(prompt: Any) -> dict[str, Any]:
        messages = [dict(message) for message in prompt] if isinstance(prompt, list) else []
        if messages and messages[0].get("role") == "system":
            content = str(messages[0].get("content") or "")
            if "List of tools: [" not in content:
                messages[0]["content"] = schema_conditioned_system_prompt(content, tool_names)
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": schema_conditioned_system_prompt(
                        "You are a precise coding assistant.", tool_names
                    ),
                },
            )
        return {"prompt": messages}

    return dataset.map(
        condition,
        input_columns=["prompt"],
        desc="Conditioning prompts on the declared tool menu",
    )


def _example_content_hash(example: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"prompt": example.get("prompt"), "completion": example.get("completion")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _group_aware_split(dataset: Any, eval_ratio: float, seed: int) -> tuple[Any, Any]:
    groups = sorted(set(dataset["group_id"]))
    if len(groups) < 2:
        raise ValueError("group-aware validation requires at least two independent records")
    random.Random(seed).shuffle(groups)
    eval_group_count = max(1, min(len(groups) - 1, round(len(groups) * eval_ratio)))
    eval_groups = set(groups[:eval_group_count])
    evaluation = dataset.filter(
        lambda group_id: group_id in eval_groups,
        input_columns=["group_id"],
        desc="Selecting whole validation trajectories",
    )
    training = dataset.filter(
        lambda group_id: group_id not in eval_groups,
        input_columns=["group_id"],
        desc="Selecting whole training trajectories",
    )
    return training, evaluation


def _drop_group_id(dataset: Any) -> Any:
    removable = [
        name
        for name in ("group_id", "evaluation_metadata_json", "sampling_stratum")
        if name in dataset.column_names
    ]
    return dataset.remove_columns(removable) if removable else dataset


def _deterministic_draw(
    pool: list[int], count: int, rng: random.Random
) -> tuple[list[int], bool]:
    """Draw exactly count indices, using replacement only after exhausting the pool."""
    if count <= len(pool):
        return rng.sample(pool, count), False
    drawn = list(pool)
    rng.shuffle(drawn)
    drawn.extend(rng.choice(pool) for _ in range(count - len(pool)))
    return drawn, True


def _compose_replay_stream(
    dataset: Any,
    *,
    replay_ratio: float,
    replay_stratum: str,
    seed: int,
    total_draws: int,
) -> tuple[Any, dict[str, Any]]:
    """Build a fixed-length deterministic canonical/replay training stream."""
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    replay_pool = [
        index
        for index, stratum in enumerate(dataset["sampling_stratum"])
        if str(stratum) == replay_stratum
    ]
    canonical_pool = [
        index
        for index, stratum in enumerate(dataset["sampling_stratum"])
        if str(stratum) != replay_stratum
    ]
    if replay_ratio > 0.0 and not replay_pool:
        raise ValueError(f"replay stratum {replay_stratum!r} is empty")
    if not canonical_pool:
        raise ValueError("canonical sampling pool is empty")

    replay_draws = round(total_draws * replay_ratio)
    canonical_draws = total_draws - replay_draws
    rng = random.Random(seed ^ 0x5EED5EED)
    replay_indices, replay_replacement = _deterministic_draw(
        replay_pool, replay_draws, rng
    ) if replay_draws else ([], False)
    canonical_indices, canonical_replacement = _deterministic_draw(
        canonical_pool, canonical_draws, rng
    )
    selected_indices = canonical_indices + replay_indices
    rng.shuffle(selected_indices)
    composed = dataset.select(selected_indices)

    source_ids = [str(group_id) for group_id in composed["group_id"]]
    content_hashes = [_example_content_hash(example) for example in composed]
    manifest = {
        "semantics": "fixed_length_training_stream_quota",
        "replay_stratum": replay_stratum,
        "seed": seed,
        "input_rows": len(dataset),
        "total_draws": total_draws,
        "canonical_pool_rows": len(canonical_pool),
        "replay_pool_rows": len(replay_pool),
        "n_canonical_draws": canonical_draws,
        "n_replay_draws": replay_draws,
        "requested_ratio": replay_ratio,
        "effective_ratio": replay_draws / max(1, total_draws),
        "canonical_with_replacement": canonical_replacement,
        "replay_with_replacement": replay_replacement,
        "source_ids_sha256": hashlib.sha256(
            "\n".join(source_ids).encode("utf-8")
        ).hexdigest(),
        "composition_sha256": hashlib.sha256(
            "\n".join(content_hashes).encode("utf-8")
        ).hexdigest(),
    }
    return composed, manifest


def load_training_data(
    sources: list[str],
    eval_ratio: float,
    seed: int,
    max_source_rows: int,
    max_training_samples: int,
    allowed_tools: list[str] | None = None,
    validation_sources: list[str] | None = None,
    max_validation_samples: int = 512,
    tool_menu_conditioning: bool = False,
    replay_ratio: float | None = None,
    replay_stratum: str = "rehearsal",
    return_composition_manifest: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, dict[str, Any] | None]:
    from datasets import concatenate_datasets

    datasets = [
        _normalize_dataset(
            _load_one(source, max_source_rows, seed + index),
            allowed_tools=set(allowed_tools) if allowed_tools is not None else None,
        )
        for index, source in enumerate(sources)
    ]
    combined = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if tool_menu_conditioning:
        combined = _condition_dataset_on_tool_menu(combined, list(allowed_tools or []))
    if validation_sources:
        validation_parts = [
            _normalize_dataset(
                _load_one(source, max_validation_samples, seed + 10_000 + index),
                allowed_tools=set(allowed_tools) if allowed_tools is not None else None,
            )
            for index, source in enumerate(validation_sources)
        ]
        evaluation = (
            validation_parts[0]
            if len(validation_parts) == 1
            else concatenate_datasets(validation_parts)
        )
        if tool_menu_conditioning:
            evaluation = _condition_dataset_on_tool_menu(
                evaluation, list(allowed_tools or [])
            )
        training_groups = set(combined["group_id"])
        evaluation_groups = set(evaluation["group_id"])
        if training_groups & evaluation_groups:
            raise ValueError("training and external validation contain identical source records")
        training_hashes = {_example_content_hash(example) for example in combined}
        content_overlap = sum(
            _example_content_hash(example) in training_hashes for example in evaluation
        )
        if content_overlap:
            raise ValueError(
                f"training and external validation overlap by {content_overlap} normalized samples"
            )
    else:
        combined, evaluation = _group_aware_split(combined, eval_ratio=eval_ratio, seed=seed)
    replay_manifest: dict[str, Any] | None = None
    if replay_ratio is not None:
        combined, replay_manifest = _compose_replay_stream(
            combined,
            replay_ratio=replay_ratio,
            replay_stratum=replay_stratum,
            seed=seed,
            total_draws=min(len(combined), max_training_samples),
        )
    elif len(combined) > max_training_samples:
        combined = combined.shuffle(seed=seed).select(range(max_training_samples))
    if len(evaluation) > max_validation_samples:
        evaluation = evaluation.shuffle(seed=seed + 1).select(range(max_validation_samples))
    result = (_drop_group_id(combined), _drop_group_id(evaluation))
    if return_composition_manifest:
        return (*result, replay_manifest)
    return result


def validate_training_source(
    source: str, rows: int = 5, allowed_tools: list[str] | None = None
) -> dict[str, Any]:
    raw = _load_one(source, max_rows=rows, seed=42)
    normalized = _normalize_dataset(raw, allowed_tools=set(allowed_tools or []))
    sample = normalized[0]
    return {
        "source": source,
        "raw_rows": len(raw),
        "raw_columns": list(raw.column_names),
        "normalized_rows": len(normalized),
        "sample_prompt_roles": [item.get("role") for item in sample["prompt"]],
        "sample_completion": str(sample["completion"])[:500],
    }
