from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .data_loading import _load_one, _normalize_dataset
from .dataset_probe import tool_names_in_example
from .paths import configure_huggingface_cache
from .schema import TrainingRecipe


def example_hash(example: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"prompt": example.get("prompt"), "completion": example.get("completion")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    selected = sorted(values)
    index = min(len(selected) - 1, round((len(selected) - 1) * fraction))
    return int(selected[index])


def _token_lengths(tokenizer: Any, example: dict[str, Any]) -> tuple[int, int, int]:
    prompt = example.get("prompt") or []
    completion = example.get("completion") or []
    try:
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [*prompt, *completion],
            add_generation_prompt=False,
            tokenize=True,
        )
        prompt_length = _encoded_length(prompt_ids)
        total_length = _encoded_length(full_ids)
        return prompt_length, max(0, total_length - prompt_length), total_length
    except (AttributeError, TypeError, ValueError):
        prompt_ids = tokenizer(json.dumps(prompt, ensure_ascii=False))["input_ids"]
        completion_ids = tokenizer(json.dumps(completion, ensure_ascii=False))["input_ids"]
        return len(prompt_ids), len(completion_ids), len(prompt_ids) + len(completion_ids)


def _encoded_length(value: Any) -> int:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "shape"):
        return int(value.shape[-1])
    if isinstance(value, list) and value and isinstance(value[0], list):
        return len(value[0])
    return len(value)


def preflight_training_sources(
    model_path: str,
    sources: list[str],
    sequence_length: int,
    allowed_tools: list[str],
    max_rows_per_source: int = 500,
    sealed_hashes: set[str] | None = None,
    sealed_sources: set[str] | None = None,
    role: str = "training",
    include_hashes: bool = False,
    filter_unknown_tools: bool = True,
    require_tool_coverage: bool = True,
) -> dict[str, Any]:
    configure_huggingface_cache()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    allowed = set(allowed_tools)
    known_sealed_hashes = sealed_hashes or set()
    known_sealed_sources = sealed_sources or set()
    reports: list[dict[str, Any]] = []
    aggregate_tools: Counter[str] = Counter()
    warnings: list[dict[str, str]] = []
    total_compatible = 0
    total_normalized = 0
    total_truncated = 0
    total_prompt_overflow = 0
    total_overlap = 0
    total_initial_tool_calls = 0
    total_initial_direct_answers = 0
    total_post_tool_direct_answers = 0
    total_system_conditioned_samples = 0
    compatible_hashes: set[str] = set()

    for source_index, source in enumerate(sources):
        raw = _load_one(source, max_rows=max_rows_per_source, seed=42 + source_index)
        normalized = _normalize_dataset(raw, allowed_tools=None)
        prompt_lengths: list[int] = []
        completion_lengths: list[int] = []
        total_lengths: list[int] = []
        source_tools: Counter[str] = Counter()
        unknown_tools: Counter[str] = Counter()
        compatible = 0
        truncated = 0
        prompt_overflow = 0
        overlap = 0
        initial_tool_calls = 0
        initial_direct_answers = 0
        post_tool_direct_answers = 0
        system_conditioned_samples = 0
        source_groups: set[str] = set()
        for example in normalized:
            if example.get("group_id"):
                source_groups.add(str(example["group_id"]))
            tools = tool_names_in_example(example)
            completion_tools = tool_names_in_example(
                {"prompt": [], "completion": example.get("completion", [])}
            )
            prompt_messages = example.get("prompt") or []
            prompt_roles = [
                str(message.get("role")) for message in prompt_messages if isinstance(message, dict)
            ]
            has_prior_tool = "tool" in prompt_roles
            is_initial_request = not any(role in {"assistant", "tool"} for role in prompt_roles)
            system_conditioned_samples += int("system" in prompt_roles)
            if is_initial_request and completion_tools:
                initial_tool_calls += 1
            elif is_initial_request and not completion_tools:
                initial_direct_answers += 1
            elif has_prior_tool and not completion_tools:
                post_tool_direct_answers += 1
            source_tools.update(completion_tools)
            aggregate_tools.update(completion_tools)
            unknown_tools.update(tools - allowed)
            is_compatible = not filter_unknown_tools or tools.issubset(allowed)
            if is_compatible:
                compatible += 1
                compatible_hashes.add(example_hash(example))
            prompt_tokens, completion_tokens, total_tokens = _token_lengths(tokenizer, example)
            prompt_lengths.append(prompt_tokens)
            completion_lengths.append(completion_tokens)
            total_lengths.append(total_tokens)
            truncated += int(total_tokens > sequence_length)
            prompt_overflow += int(prompt_tokens >= sequence_length - 8)
            overlap += int(is_compatible and example_hash(example) in known_sealed_hashes)
        source_report = {
            "source": source,
            "raw_rows_sampled": len(raw),
            "normalized_samples": len(normalized),
            "trajectory_groups": len(source_groups),
            "compatible_samples": compatible,
            "rejected_unknown_tool_samples": len(normalized) - compatible,
            "tool_distribution": dict(source_tools.most_common()),
            "unknown_tool_distribution": dict(unknown_tools.most_common()),
            "tokens": {
                "prompt_p50": _percentile(prompt_lengths, 0.5),
                "prompt_p95": _percentile(prompt_lengths, 0.95),
                "completion_p50": _percentile(completion_lengths, 0.5),
                "completion_p95": _percentile(completion_lengths, 0.95),
                "total_p50": _percentile(total_lengths, 0.5),
                "total_p95": _percentile(total_lengths, 0.95),
                "total_max": max(total_lengths, default=0),
                "sequence_limit": sequence_length,
                "truncated_samples": truncated,
                "prompt_overflow_samples": prompt_overflow,
            },
            "sealed_overlap_samples": overlap,
            "routing_examples": {
                "initial_tool_calls": initial_tool_calls,
                "initial_direct_answers": initial_direct_answers,
                "post_tool_direct_answers": post_tool_direct_answers,
                "system_conditioned_samples": system_conditioned_samples,
            },
        }
        reports.append(source_report)
        total_compatible += compatible
        total_normalized += len(normalized)
        total_truncated += truncated
        total_prompt_overflow += prompt_overflow
        total_overlap += overlap
        total_initial_tool_calls += initial_tool_calls
        total_initial_direct_answers += initial_direct_answers
        total_post_tool_direct_answers += post_tool_direct_answers
        total_system_conditioned_samples += system_conditioned_samples
        if source in known_sealed_sources:
            warnings.append(
                {
                    "severity": "error",
                    "code": "SEALED_SOURCE_REUSE",
                    "message": (
                        f"{source} is registered as a sealed test and cannot be used for {role}."
                    ),
                }
            )
        if unknown_tools and filter_unknown_tools:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "UNKNOWN_TOOLS",
                    "message": f"{source} contains unsupported tools: {dict(unknown_tools)}.",
                }
            )
        if initial_tool_calls and initial_direct_answers == 0:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "NO_INITIAL_ABSTENTION_EXAMPLES",
                    "message": (
                        f"{source} teaches {initial_tool_calls} initial tool calls but zero "
                        "initial user-to-direct-answer examples. Post-tool answers do not "
                        "teach tool abstention and can create an always-call bias."
                    ),
                }
            )
        if initial_tool_calls and system_conditioned_samples == 0:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "NO_SCHEMA_CONDITIONED_EXAMPLES",
                    "message": (
                        f"{source} contains tool-call targets without any system-conditioned "
                        "examples. Evaluation with an explicit tool menu is a protocol shift."
                    ),
                }
            )
        rejected_ratio = (len(normalized) - compatible) / max(1, len(normalized))
        if filter_unknown_tools and rejected_ratio >= 0.5:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "HIGH_TOOL_FILTER_REJECTION",
                    "message": (
                        f"{rejected_ratio:.1%} of normalized samples from {source} would be "
                        "discarded because their tool vocabulary is incompatible with the "
                        f"configured tools {sorted(allowed)}."
                    ),
                }
            )
        if compatible < 100:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "LOW_COMPATIBLE_SAMPLE_COUNT",
                    "message": f"Only {compatible} compatible normalized samples were found in {source}.",
                }
            )

    if total_overlap:
        warnings.append(
            {
                "severity": "error",
                "code": "SEALED_CONTENT_OVERLAP",
                "message": (
                    f"{total_overlap} sampled {role} examples overlap sealed test content."
                ),
            }
        )
    truncation_ratio = total_truncated / max(1, total_normalized)
    if total_prompt_overflow:
        warnings.append(
            {
                "severity": "error",
                "code": "PROMPT_OVERFLOW",
                "message": (
                    f"{total_prompt_overflow} prompts consume the sequence window before the "
                    "answer, so required completion tokens would be lost."
                ),
            }
        )
    elif truncation_ratio > 0.25:
        warnings.append(
            {
                "severity": "error",
                "code": "EXCESSIVE_TRUNCATION",
                "message": f"{truncation_ratio:.1%} of sampled examples exceed the sequence limit.",
            }
        )
    elif total_truncated:
        warnings.append(
            {
                "severity": "warning",
                "code": "SOME_TRUNCATION",
                "message": f"{truncation_ratio:.1%} of sampled examples exceed the sequence limit.",
            }
        )
    expected_tools = sorted(allowed)
    missing_tools = [tool for tool in expected_tools if aggregate_tools[tool] == 0]
    tool_total = sum(aggregate_tools.values())
    dominant_share = max(aggregate_tools.values(), default=0) / max(1, tool_total)
    if require_tool_coverage and missing_tools:
        warnings.append(
            {
                "severity": "warning",
                "code": "MISSING_TOOL_COVERAGE",
                "message": f"No completion examples were found for tools: {missing_tools}.",
            }
        )
    if tool_total and dominant_share > 0.75:
        warnings.append(
            {
                "severity": "warning",
                "code": "TOOL_IMBALANCE",
                "message": f"One tool represents {dominant_share:.1%} of sampled tool calls.",
            }
        )
    if total_compatible == 0:
        warnings.append(
            {
                "severity": "error",
                "code": "NO_COMPATIBLE_SAMPLES",
                "message": "No compatible training samples remain after tool filtering.",
            }
        )
    result = {
        "status": "blocked"
        if any(item["severity"] == "error" for item in warnings)
        else "warning"
        if warnings
        else "ready",
        "model": model_path,
        "sequence_length": sequence_length,
        "sources": reports,
        "summary": {
            "normalized_samples": total_normalized,
            "compatible_samples": total_compatible,
            "compatible_retention_rate": total_compatible / max(1, total_normalized),
            "truncated_samples": total_truncated,
            "prompt_overflow_samples": total_prompt_overflow,
            "sealed_overlap_samples": total_overlap,
            "tool_distribution": dict(aggregate_tools.most_common()),
            "routing_examples": {
                "initial_tool_calls": total_initial_tool_calls,
                "initial_direct_answers": total_initial_direct_answers,
                "post_tool_direct_answers": total_post_tool_direct_answers,
                "system_conditioned_samples": total_system_conditioned_samples,
            },
        },
        "warnings": warnings,
    }
    if include_hashes:
        result["_compatible_hashes"] = sorted(compatible_hashes)
    return result


def preflight_recipe(
    recipe: TrainingRecipe,
    sealed_hashes: set[str] | None = None,
    sealed_sources: set[str] | None = None,
) -> dict[str, Any]:
    training = preflight_training_sources(
        model_path=recipe.model,
        sources=recipe.datasets,
        sequence_length=recipe.sequence_length,
        allowed_tools=recipe.allowed_tools,
        max_rows_per_source=min(500, recipe.max_source_rows),
        sealed_hashes=sealed_hashes,
        sealed_sources=sealed_sources,
        role="training",
        include_hashes=True,
        filter_unknown_tools=recipe.filter_unknown_tools,
        require_tool_coverage=recipe.benchmark_profile in {"pi-agent", "mixed"},
    )
    training_hashes = set(training.pop("_compatible_hashes", []))
    warnings = list(training["warnings"])
    if recipe.validation_mode == "external":
        validation = preflight_training_sources(
            model_path=recipe.model,
            sources=recipe.validation_datasets,
            sequence_length=recipe.sequence_length,
            allowed_tools=recipe.allowed_tools,
            max_rows_per_source=min(500, recipe.max_validation_samples),
            sealed_hashes=sealed_hashes,
            sealed_sources=sealed_sources,
            role="validation",
            include_hashes=True,
            filter_unknown_tools=recipe.filter_unknown_tools,
            require_tool_coverage=recipe.benchmark_profile in {"pi-agent", "mixed"},
        )
        validation_hashes = set(validation.pop("_compatible_hashes", []))
        overlap = training_hashes & validation_hashes
        if overlap:
            warnings.append(
                {
                    "severity": "error",
                    "code": "TRAIN_VALIDATION_OVERLAP",
                    "message": (
                        f"{len(overlap)} sampled compatible examples occur in both training "
                        "and external validation."
                    ),
                }
            )
        warnings.extend(validation["warnings"])
    else:
        validation = {
            "status": "group_holdout",
            "policy": "whole normalized trajectories are assigned to only one split",
            "eval_ratio": recipe.eval_ratio,
            "source": "training datasets",
        }
    status = (
        "blocked"
        if any(item["severity"] == "error" for item in warnings)
        else "warning"
        if warnings
        else "ready"
    )
    return {
        "status": status,
        "benchmark_profile": recipe.benchmark_profile,
        "comparison_group": recipe.comparison_group,
        "experiment_variant": recipe.experiment_variant,
        "split_policy": recipe.validation_mode,
        "training": training,
        "validation": validation,
        "warnings": warnings,
    }
