from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .agent_eval import evaluate_agent_behavior
from .bfcl import xpu_vram_status
from .data_loading import _load_one, _normalize_dataset
from .paths import project_root
from .schema import (
    Boolean3DConfig,
    GeometryConfig,
    NoiseConfig,
    RoleLossConfig,
    TrainingRecipe,
)
from .strict_genetic_data import (
    CATEGORIES,
    DEV_FOLDS,
    PREFERENCE_PATH,
    TRAIN_PATH,
    dev_path,
    ensure_strict_genetic_corpus,
)
from .tool_schema import KNOWN_TOOLS
from .trainer import run_training
from .v13_campaign import create_campaign_layout, default_campaign_root

MODEL_PATH = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
REHEARSAL_PATH = (
    project_root() / "downloads" / "generated" / "human-agentic-pure-dedup-v9-5500.jsonl"
)
DUAL_SEEDS = (6492, 8778)
GIB = 1024**3

SEARCH_SPACE = {
    "origin": "v13-unified-exotic; all exotic optimizations co-active with continuous tuning",
    "lora_rank": [16],
    "lora_alpha": [32],
    "lora_dropout": [0.03, 0.08],
    "learning_rate_log": [7e-5, 2.2e-4],
    "weight_decay": [0.005, 0.05],
    "warmup_ratio": [0.03, 0.08],
    "sequence_length": [1408],
    "target_profile": ["attention"],
    "noise_source": ["pi"],
    "noise_amplitude_mode": ["digit_pairs", "diff", "single_digit"],
    "noise_alpha": [1.8, 2.6],
    "noise_modulation": [0.10, 0.20],
    "noise_envelope": ["cosine"],
    "noise_clean_tail": [0.25, 0.40],
    "geometry_mode": ["relational"],
    "geometry_weight": [0.007, 0.015],
    "geometry_margin": [0.32, 0.45],
    "boolean_weight": [0.006, 0.018],
    "boolean_margin": [0.28, 0.45],
    "boolean_dim_action": [1.0, 1.6],
    "boolean_dim_arity": [1.0, 1.6],
    "boolean_dim_syntax": [1.0, 1.6],
    "role_ordinary_weight": [1.8, 2.6],
    "role_delimiter_weight": [1.8, 2.3],
    "role_tool_name_weight": [3.0, 3.8],
    "role_argument_key_weight": [2.2, 3.0],
    "role_argument_value_weight": [2.4, 3.2],
    "role_turn_horizon_beta": [0.40, 0.70],
    "state_binding_weight": [0.002, 0.020],
    "order_strategy": ["random", "interleaved", "hard-first", "rehearsal-spaced"],
    "repeat_focus": ["strict-routing", "agentic", "all-weak"],
    "category_weights": list(CATEGORIES),
}


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _normalized_weights(rng: random.Random) -> dict[str, float]:
    priors = {
        "strict_ast": 1.60,
        "routing_negative": 1.30,
        "web_multihop": 1.60,
        "memory": 1.60,
        "multi_step": 1.60,
        "polyglot_java": 1.60,
        "polyglot_javascript": 1.60,
        "parallel_multiple": 1.50,
        "rehearsal": 1.20,
    }
    raw = {
        category: priors[category] * math.exp(rng.uniform(-0.6, 0.6)) for category in CATEGORIES
    }
    raw["rehearsal"] = max(raw["rehearsal"], 0.90)
    total = sum(raw.values())
    return {category: raw[category] / total for category in CATEGORIES}


def sample_gene(rng: random.Random, index: int) -> dict[str, Any]:
    amplitude_mode = rng.choice(SEARCH_SPACE["noise_amplitude_mode"])
    dim_w = (
        round(rng.uniform(*SEARCH_SPACE["boolean_dim_action"]), 2),
        round(rng.uniform(*SEARCH_SPACE["boolean_dim_arity"]), 2),
        round(rng.uniform(*SEARCH_SPACE["boolean_dim_syntax"]), 2),
    )
    gene = {
        "id": f"g{index:02d}",
        "regularizer_family": "unified_exotic",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": round(rng.uniform(*SEARCH_SPACE["lora_dropout"]), 4),
        "use_rslora": True,
        "target_profile": "attention",
        "sequence_length": 1408,
        "learning_rate": _log_uniform(rng, *SEARCH_SPACE["learning_rate_log"]),
        "weight_decay": round(rng.uniform(*SEARCH_SPACE["weight_decay"]), 4),
        "warmup_ratio": round(rng.uniform(*SEARCH_SPACE["warmup_ratio"]), 4),
        "noise": {
            "enabled": True,
            "source": "pi",
            "amplitude_mode": amplitude_mode,
            "digit_order": "natural",
            "alpha": round(rng.uniform(*SEARCH_SPACE["noise_alpha"]), 3),
            "modulation": round(rng.uniform(*SEARCH_SPACE["noise_modulation"]), 3),
            "scope": "prompt",
            "envelope": "cosine",
            "clean_tail_fraction": round(rng.uniform(*SEARCH_SPACE["noise_clean_tail"]), 3),
            "seed_offset": 10007,
            "protect_prompt_literals": False,
            "value_consistency_weight": 0.0,
            "intent_consistency_weight": 0.0,
        },
        "geometry": {
            "enabled": True,
            "mode": "relational",
            "scope": "structured",
            "weight": round(rng.uniform(*SEARCH_SPACE["geometry_weight"]), 5),
            "layer": -1,
            "margin": round(rng.uniform(*SEARCH_SPACE["geometry_margin"]), 3),
            "sample_tokens": 32,
        },
        "boolean_3d": {
            "enabled": True,
            "weight": round(rng.uniform(*SEARCH_SPACE["boolean_weight"]), 5),
            "polytope_margin": round(rng.uniform(*SEARCH_SPACE["boolean_margin"]), 3),
            "polytope_weight": 1.0,
            "refusal_weight": 1.0,
            "refusal_action_threshold": 0.15,
            "dimension_weights": dim_w,
            "sample_tokens": 48,
            "layer": -1,
        },
        "role_loss": {
            "enabled": True,
            "ordinary_weight": round(rng.uniform(*SEARCH_SPACE["role_ordinary_weight"]), 3),
            "delimiter_weight": round(rng.uniform(*SEARCH_SPACE["role_delimiter_weight"]), 3),
            "tool_name_weight": round(rng.uniform(*SEARCH_SPACE["role_tool_name_weight"]), 3),
            "argument_key_weight": round(rng.uniform(*SEARCH_SPACE["role_argument_key_weight"]), 3),
            "argument_value_weight": round(rng.uniform(*SEARCH_SPACE["role_argument_value_weight"]), 3),
            "turn_horizon_beta": round(rng.uniform(*SEARCH_SPACE["role_turn_horizon_beta"]), 3),
        },
        "state_binding_weight": round(rng.uniform(*SEARCH_SPACE["state_binding_weight"]), 5),
        "category_weights": _normalized_weights(rng),
        "order_strategy": rng.choice(SEARCH_SPACE["order_strategy"]),
        "repeat_focus": rng.choice(SEARCH_SPACE["repeat_focus"]),
        "hard_repeat": rng.choice([1, 2]),
    }
    return _enforce_safe_gene(gene)


def _gene_signature(gene: dict[str, Any]) -> str:
    payload = {key: value for key, value in gene.items() if key != "id"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def _enforce_safe_gene(gene: dict[str, Any]) -> dict[str, Any]:
    gene["regularizer_family"] = "unified_exotic"
    gene["noise"]["enabled"] = True
    gene["noise"]["source"] = "pi"
    gene["noise"]["value_consistency_weight"] = 0.0
    gene["noise"]["intent_consistency_weight"] = 0.0
    gene["geometry"]["enabled"] = True
    gene["boolean_3d"]["enabled"] = True
    gene["role_loss"]["enabled"] = True
    gene["state_binding_weight"] = max(0.0001, float(gene.get("state_binding_weight", 0.005)))
    gene["sequence_length"] = 1408
    gene["target_profile"] = "attention"
    gene["lora_rank"] = min(16, int(gene.get("lora_rank", 16)))
    gene["lora_alpha"] = max(gene["lora_rank"], int(gene.get("lora_alpha", 32)))
    gene["use_rslora"] = True
    return gene


def audit_gene_safety(gene: dict[str, Any]) -> dict[str, Any]:
    problems = []
    if int(gene.get("sequence_length", 1408)) > 1408:
        problems.append("sequence length exceeds the audited 1408-token cap")
    if int(gene.get("lora_rank", 16)) > 16:
        problems.append("LoRA rank exceeds the audited cap of 16")
    if gene.get("target_profile") != "attention":
        problems.append("only the attention target profile is allowed")
    if (
        float(gene.get("noise", {}).get("value_consistency_weight", 0.0)) > 0.0
        or float(gene.get("noise", {}).get("intent_consistency_weight", 0.0)) > 0.0
    ):
        problems.append("noise consistency would add an unapproved second clean forward")
    return {
        "status": "blocked" if problems else "safe",
        "enabled_memory_family": "unified_exotic",
        "xpu_memory_fraction": 0.70,
        "problems": problems,
    }


def mutate_gene(parent: dict[str, Any], rng: random.Random, identifier: str) -> dict[str, Any]:
    fresh = sample_gene(rng, 0)
    child = json.loads(json.dumps(parent))
    child["id"] = identifier
    top_level = [
        "lora_dropout",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "state_binding_weight",
        "order_strategy",
        "repeat_focus",
        "hard_repeat",
    ]
    for key in rng.sample(top_level, k=rng.randint(2, 4)):
        child[key] = fresh[key]
    for block in ("noise", "geometry", "boolean_3d", "role_loss"):
        for key in rng.sample(list(child[block]), k=max(1, len(child[block]) // 2)):
            child[block][key] = fresh[block][key]
    if rng.random() < 0.6:
        child["category_weights"] = fresh["category_weights"]
    return _enforce_safe_gene(child)


def crossover_gene(
    left: dict[str, Any], right: dict[str, Any], rng: random.Random, identifier: str
) -> dict[str, Any]:
    child = json.loads(json.dumps(left))
    child["id"] = identifier
    for key in (
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "lora_dropout",
        "state_binding_weight",
        "order_strategy",
        "repeat_focus",
        "hard_repeat",
    ):
        if rng.random() < 0.5:
            child[key] = right[key]
    for block in ("noise", "geometry", "boolean_3d", "role_loss", "category_weights"):
        for key in child[block]:
            if rng.random() < 0.5:
                child[block][key] = right[block][key]
    if rng.random() < 0.5:
        child = mutate_gene(child, rng, identifier)
    return _enforce_safe_gene(child)


def initial_population(rng: random.Random, population: int) -> list[dict[str, Any]]:
    genes = []
    for index in range(population):
        gene = sample_gene(rng, index + 1)
        genes.append(_enforce_safe_gene(gene))
    return genes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case_id(row: dict[str, Any]) -> str:
    metadata = row.get("training_metadata") or row.get("evaluation_metadata") or {}
    return str(metadata.get("case_id") or "")


def _category(row: dict[str, Any]) -> str:
    metadata = row.get("training_metadata") or row.get("evaluation_metadata") or {}
    return str(metadata.get("category") or metadata.get("stratum") or "rehearsal")


def _weighted_quotas(weights: dict[str, float], total_rows: int) -> dict[str, int]:
    minimum = min(12, max(1, total_rows // (len(CATEGORIES) * 4)))
    quotas = {category: minimum for category in CATEGORIES}
    remaining = total_rows - sum(quotas.values())
    if remaining < 0:
        raise ValueError("micro corpus is too small for category coverage")
    fractions = {category: remaining * weights[category] for category in CATEGORIES}
    for category in CATEGORIES:
        quotas[category] += int(fractions[category])
    while sum(quotas.values()) < total_rows:
        category = max(CATEGORIES, key=lambda item: fractions[item] - int(fractions[item]))
        quotas[category] += 1
        fractions[category] = int(fractions[category])
    return quotas


def _order_rows(
    rows: list[dict[str, Any]], strategy: str, rng: random.Random
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_category(row)].append(row)
    for values in buckets.values():
        rng.shuffle(values)
    if strategy == "random":
        rng.shuffle(rows)
        return rows
    if strategy == "interleaved":
        ordered = []
        while any(buckets.values()):
            for category in CATEGORIES:
                if buckets[category]:
                    ordered.append(buckets[category].pop())
        return ordered
    if strategy == "hard-first":
        order = (
            "routing_negative",
            "strict_ast",
            "polyglot_java",
            "polyglot_javascript",
            "parallel_multiple",
            "memory",
            "web_multihop",
            "multi_step",
            "rehearsal",
        )
        return [row for category in order for row in buckets[category]]
    if strategy == "rehearsal-spaced":
        rehearsal = buckets.pop("rehearsal")
        hard = [
            row for category in CATEGORIES if category != "rehearsal" for row in buckets[category]
        ]
        ordered = []
        for index, row in enumerate(hard):
            ordered.append(row)
            if rehearsal and index % 3 == 2:
                ordered.append(rehearsal.pop())
        return ordered + rehearsal
    raise ValueError(f"unknown order strategy: {strategy}")


def build_candidate_mix(
    gene: dict[str, Any], seed: int, total_rows: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed ^ int(_gene_signature(gene), 16))
    master = _read_jsonl(TRAIN_PATH)
    preferences = {
        pair["preference_metadata"]["case_id"]: pair for pair in _read_jsonl(PREFERENCE_PATH)
    }
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in master:
        buckets[_category(row)].append(row)
    quotas = _weighted_quotas(gene["category_weights"], total_rows)
    selected: list[dict[str, Any]] = []
    repeat_categories = {
        "strict-routing": {
            "strict_ast",
            "routing_negative",
            "polyglot_java",
            "polyglot_javascript",
            "parallel_multiple",
        },
        "agentic": {
            "web_multihop",
            "memory",
            "multi_step",
            "polyglot_java",
            "polyglot_javascript",
        },
        "all-weak": {
            "strict_ast",
            "routing_negative",
            "web_multihop",
            "memory",
            "multi_step",
            "polyglot_java",
            "polyglot_javascript",
            "parallel_multiple",
        },
    }[str(gene["repeat_focus"])]
    repeated_rows = 0
    for category in CATEGORIES:
        pool = list(buckets[category])
        rng.shuffle(pool)
        repeats = int(gene["hard_repeat"]) if category in repeat_categories else 1
        unique_needed = min(len(pool), math.ceil(quotas[category] / repeats))
        chosen = pool[:unique_needed]
        expanded = [row for row in chosen for _ in range(repeats)]
        while len(expanded) < quotas[category]:
            expanded.extend(chosen)
        selected.extend(expanded[: quotas[category]])
        repeated_rows += quotas[category] - len(
            {_case_id(row) for row in expanded[: quotas[category]]}
        )
    selected = _order_rows(selected, gene["order_strategy"], rng)
    selected_pairs = [preferences[_case_id(row)] for row in selected if _case_id(row) in preferences]
    manifest = {
        "rows": len(selected),
        "unique_cases": len({_case_id(row) for row in selected}),
        "quotas": quotas,
        "actual_categories": dict(sorted(Counter(_category(row) for row in selected).items())),
        "order_strategy": gene["order_strategy"],
        "repeat_focus": gene["repeat_focus"],
        "hard_repeat": gene["hard_repeat"],
        "repeated_rows": repeated_rows,
    }
    return selected, selected_pairs, manifest


def build_round_dev(round_index: int) -> tuple[list[dict[str, Any]], str]:
    folds = {fold: _read_jsonl(dev_path(fold)) for fold in DEV_FOLDS}
    if round_index == 0:
        return folds["a"], "dev-a"
    left = DEV_FOLDS[min(round_index - 1, len(DEV_FOLDS) - 2)]
    right = DEV_FOLDS[min(round_index, len(DEV_FOLDS) - 1)]
    rows = []
    for category in CATEGORIES:
        old = [row for row in folds[left] if _category(row) == category]
        new = [row for row in folds[right] if _category(row) == category]
        rows.extend(old[: len(old) // 2])
        rows.extend(new[len(new) // 2 :])
    random.Random(2026082390 + round_index).shuffle(rows)
    return rows, f"half-{left}-half-{right}"


def fitness_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    predictions = list(metrics.get("agent_predictions") or [])
    tool_rows = [item for item in predictions if item.get("expected_tool") is not None]

    def success(categories: set[str]) -> float:
        selected = [
            item
            for item in predictions
            if str((item.get("evaluation_metadata") or {}).get("category")) in categories
        ]
        return sum(bool(item.get("strict_human_task_success")) for item in selected) / max(
            1, len(selected)
        )

    strict = sum(bool(item.get("argument_values_exact")) for item in tool_rows) / max(
        1, len(tool_rows)
    )
    elastic = sum(bool(item.get("required_argument_values_elastic")) for item in tool_rows) / max(
        1, len(tool_rows)
    )
    multi = success({"web_multihop", "memory", "multi_step"})
    rehearsal = success({"rehearsal"})
    category_strict = {category: success({category}) for category in CATEGORIES}
    routing = float(metrics.get("agent_balanced_tool_decision_accuracy") or 0.0)
    mcc = (float(metrics.get("agent_tool_decision_mcc") or 0.0) + 1.0) / 2.0
    format_score = float(metrics.get("agent_expected_format_accuracy") or 0.0)
    gap = max(0.0, elastic - strict)
    regression_guard = min(routing, format_score, rehearsal)

    score = (
        0.35 * strict
        + 0.15 * routing
        + 0.10 * mcc
        + 0.20 * multi
        + 0.10 * format_score
        + 0.10 * rehearsal
        + 0.05 * regression_guard
        - 0.10 * gap
    )
    return {
        "fitness": score,
        "strict_exact": strict,
        "elastic_required": elastic,
        "elastic_strict_gap": gap,
        "balanced_routing": routing,
        "routing_mcc_scaled": mcc,
        "multi_category_strict": multi,
        "expected_format": format_score,
        "rehearsal_strict": rehearsal,
        "regression_guard": regression_guard,
        "weakest_category_strict": min(category_strict.values()),
        "category_strict": category_strict,
    }


def aggregate_dual_seed(seed_scores: list[float]) -> dict[str, float]:
    if len(seed_scores) != len(DUAL_SEEDS):
        raise ValueError(f"exactly {len(DUAL_SEEDS)} seed scores are required")
    mean = fmean(seed_scores)
    worst = min(seed_scores)
    deviation = pstdev(seed_scores)
    robust = 0.7 * mean + 0.3 * worst - 0.15 * deviation
    return {"mean": mean, "worst": worst, "std": deviation, "robust_fitness": robust}


def _recipe(
    gene: dict[str, Any],
    *,
    seed: int,
    steps: int,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    final_training: bool = False,
) -> TrainingRecipe:
    return TrainingRecipe(
        name=f"trinity-v13-{gene['id']}-seed-{seed}",
        model=MODEL_PATH,
        datasets=[str(train_path.resolve())],
        validation_mode="external",
        validation_datasets=[str(validation_path.resolve())],
        benchmark_profile="mixed",
        comparison_group="trinity-v13-strict-genetic",
        experiment_variant="custom",
        output_dir=str(output_dir.resolve()),
        seed=seed,
        dtype="bf16",
        device="xpu",
        xpu_memory_fraction=0.70,
        sequence_length=int(gene["sequence_length"]),
        micro_batch_size=1,
        gradient_accumulation=8 if final_training else 4,
        learning_rate=float(gene["learning_rate"]),
        warmup_ratio=float(gene["warmup_ratio"]),
        weight_decay=float(gene["weight_decay"]),
        max_steps=steps,
        budget_mode="steps",
        time_limit_minutes=240 if final_training else 60,
        reserve_minutes=20 if final_training else 5,
        save_every_minutes=30,
        logging_steps=max(1, min(10, steps // 4)),
        max_source_rows=20_000 if final_training else 2_000,
        max_training_samples=20_000 if final_training else 2_000,
        max_validation_samples=200,
        agent_eval_samples=0,
        agent_eval_max_new_tokens=128,
        allowed_tools=sorted(KNOWN_TOOLS),
        filter_unknown_tools=True,
        tool_menu_conditioning=True,
        packing=False,
        train_sampling_strategy=("random" if gene["order_strategy"] == "random" else "sequential"),
        gradient_checkpointing=True,
        lora_rank=int(gene["lora_rank"]),
        lora_alpha=int(gene["lora_alpha"]),
        lora_dropout=float(gene["lora_dropout"]),
        use_rslora=bool(gene["use_rslora"]),
        target_profile=str(gene["target_profile"]),
        noise=NoiseConfig.model_validate(gene["noise"]),
        geometry=GeometryConfig.model_validate(gene["geometry"]),
        role_loss=RoleLossConfig.model_validate(gene["role_loss"]),
        boolean_3d=Boolean3DConfig.model_validate(gene["boolean_3d"]),
        state_binding_weight=float(gene["state_binding_weight"]),
    )


def _evaluate_adapter(
    *,
    gene: dict[str, Any],
    seed: int,
    adapter_path: Path,
    dev_source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)
    torch.xpu.empty_cache()
    torch.xpu.set_per_process_memory_fraction(0.70, device=0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path)).to("xpu")
    model.eval()

    raw = _load_one(str(dev_source), max_rows=500, seed=seed + 50_000)
    evaluation = _normalize_dataset(raw, allowed_tools=set(KNOWN_TOOLS))
    metrics = evaluate_agent_behavior(
        model=model,
        tokenizer=tokenizer,
        dataset=evaluation,
        max_samples=len(evaluation),
        max_new_tokens=128,
        deadline=float("inf"),
        tool_names=sorted(KNOWN_TOOLS),
        include_predictions=True,
    )
    fitness = fitness_from_metrics(metrics)
    result_payload = {
        "adapter_path": str(adapter_path.resolve()),
        "metrics": metrics,
        "fitness": fitness,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result_payload, indent=2, default=str), encoding="utf-8"
    )
    del model, base
    gc.collect()
    torch.xpu.empty_cache()
    return result_payload


def run_candidate(
    *,
    root: Path,
    gene: dict[str, Any],
    round_index: int,
    seed: int,
    steps: int,
    micro_rows: int,
    dev_rows: list[dict[str, Any]],
    dev_label: str,
    force: bool = True,
) -> dict[str, Any]:
    candidate_dir = root / f"round-{round_index}" / gene["id"] / f"seed-{seed}"
    result_path = candidate_dir / "result.json"
    if result_path.exists() and not force:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("gene_signature") == _gene_signature(gene):
            return cached
    safety = audit_gene_safety(gene)
    if safety["status"] != "safe":
        raise RuntimeError(f"genotype blocked by XPU safety audit: {safety['problems']}")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    train_rows, preference_rows, mix_manifest = build_candidate_mix(gene, seed, micro_rows)
    train_source = candidate_dir / "train.jsonl"
    preference_source = candidate_dir / "preferences.jsonl"
    dev_source = candidate_dir / f"{dev_label}.jsonl"
    _write_jsonl(train_source, train_rows)
    _write_jsonl(preference_source, preference_rows)
    _write_jsonl(dev_source, dev_rows)
    (candidate_dir / "gene.json").write_text(json.dumps(gene, indent=2), encoding="utf-8")
    (candidate_dir / "safety.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")
    (candidate_dir / "mix-manifest.json").write_text(
        json.dumps(mix_manifest, indent=2), encoding="utf-8"
    )
    sft_dir = candidate_dir / "sft"
    recipe = _recipe(
        gene,
        seed=seed,
        steps=steps,
        train_path=train_source,
        validation_path=dev_source,
        output_dir=sft_dir,
    )
    cached_sft_metrics = sft_dir / "metrics.json"
    cached_sft_adapter = sft_dir / "adapter-final"
    if cached_sft_metrics.exists() and (cached_sft_adapter / "adapter_model.safetensors").exists():
        sft_metrics = json.loads(cached_sft_metrics.read_text(encoding="utf-8"))
        sft_metrics["output_dir"] = str(sft_dir.resolve())
    else:
        sft_metrics = run_training(recipe)

    gc.collect()
    try:
        import torch

        torch.xpu.empty_cache()
    except (ImportError, RuntimeError):
        pass

    result = _evaluate_adapter(
        gene=gene,
        seed=seed,
        adapter_path=Path(str(sft_metrics["output_dir"])) / "adapter-final",
        dev_source=dev_source,
        output_dir=candidate_dir,
    )
    result["sft"] = {
        key: sft_metrics.get(key)
        for key in ("run_id", "global_step", "train_loss", "eval_loss", "elapsed_total_seconds")
    }
    result["gene_id"] = gene["id"]
    result["gene_signature"] = _gene_signature(gene)
    result["seed"] = seed
    result["round"] = round_index
    result["dev_label"] = dev_label
    result["mix"] = mix_manifest
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def resource_preflight(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    vram = xpu_vram_status()
    report = {
        "status": "ready",
        "disk_free_gib": round(disk.free / GIB, 3),
        "disk_minimum_gib": 10.0,
        "xpu": vram,
        "xpu_minimum_free_gib": 7.0,
        "xpu_memory_fraction": 0.70,
        "automatic_retry_after_device_loss": False,
    }
    problems = []
    if disk.free < 10 * GIB:
        problems.append("less than 10 GiB free on the run filesystem")
    if vram is None:
        problems.append("the external /run/xe-gpu-vram monitor is unavailable")
    else:
        free_gib = float(vram["total_gib"]) - float(vram["used_gib"])
        report["xpu_free_gib"] = round(free_gib, 3)
        if free_gib < 7.0:
            problems.append(f"only {free_gib:.2f} GiB XPU VRAM is free")
    if problems:
        report.update(status="blocked", problems=problems)
    return report


@contextmanager
def _pipeline_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".pipeline.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _spawn_next_generation(
    ranked: list[dict[str, Any]], rng: random.Random, round_index: int
) -> list[dict[str, Any]]:
    keep = max(2, math.ceil(len(ranked) / 2)) if len(ranked) > 2 else len(ranked)
    survivors = [item["gene"] for item in ranked[:keep]]
    if len(survivors) < 2:
        return survivors
    child_id = f"r{round_index + 1}-child"
    child = crossover_gene(survivors[0], survivors[1], rng, child_id)
    if _gene_signature(child) in {_gene_signature(item) for item in survivors}:
        child = mutate_gene(survivors[0], rng, child_id)
    return survivors + [child]


def _combine_final_training_data(final_root: Path, gene: dict[str, Any], seed: int) -> Path:
    strict_rows, _pairs, _manifest = build_candidate_mix(gene, seed, total_rows=3_120)
    rehearsal = _read_jsonl(REHEARSAL_PATH)
    combined = rehearsal + strict_rows
    random.Random(seed).shuffle(combined)
    output = final_root / f"seed-{seed}" / "full-training.jsonl"
    _write_jsonl(output, combined)
    return output


def run_final_dual_seed(
    *,
    final_root: Path,
    gene: dict[str, Any],
    steps: int,
    preference_steps: int = 0,
    force: bool = True,
) -> dict[str, Any]:
    all_dev = [row for fold in DEV_FOLDS for row in _read_jsonl(dev_path(fold))]
    final_results: dict[str, Any] = {}
    for seed in DUAL_SEEDS:
        final_dir = final_root / f"seed-{seed}"
        result_path = final_dir / "result.json"
        if result_path.exists() and not force:
            final_results[str(seed)] = json.loads(result_path.read_text(encoding="utf-8"))
            continue
        final_dir.mkdir(parents=True, exist_ok=True)
        train_source = _combine_final_training_data(final_root, gene, seed)
        dev_source = final_dir / "dev-all.jsonl"
        _write_jsonl(dev_source, all_dev)
        recipe = _recipe(
            gene,
            seed=seed,
            steps=steps,
            train_path=train_source,
            validation_path=dev_source,
            output_dir=final_dir / "sft",
            final_training=True,
        )
        sft_metrics = run_training(recipe)
        result = _evaluate_adapter(
            gene=gene,
            seed=seed,
            adapter_path=Path(str(sft_metrics["output_dir"])) / "adapter-final",
            dev_source=dev_source,
            output_dir=final_dir,
        )
        result.update(
            seed=seed,
            sft={
                key: sft_metrics.get(key)
                for key in (
                    "run_id",
                    "global_step",
                    "train_loss",
                    "eval_loss",
                    "elapsed_total_seconds",
                )
            },
        )
        result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        final_results[str(seed)] = result
    selected_seed = max(
        DUAL_SEEDS,
        key=lambda seed: float(final_results[str(seed)]["fitness"]["fitness"]),
    )
    selected = {
        "selected_seed": selected_seed,
        "selected_adapter": final_results[str(selected_seed)]["adapter_path"],
        "results": final_results,
        "selection_policy": (
            "highest strict composite on the preregistered local DEV A/B/C folds; "
            "SEALED and BFCL remain untouched"
        ),
        "benchmark_evaluations_run": [],
    }
    (final_root / "selection.json").write_text(
        json.dumps(selected, indent=2, default=str), encoding="utf-8"
    )
    return selected


def run_tournament(
    *,
    population: int = 5,
    round_steps: tuple[int, ...] = (20, 35),
    micro_rows: int = 420,
    tournament_seed: int = 20260823,
    output_dir: Path | None = None,
    run_final: bool = True,
    final_steps: int = 240,
    final_preference_steps: int = 0,
    force: bool = True,
) -> dict[str, Any]:
    if population < 2:
        raise ValueError("population must contain at least two genes")
    if not round_steps or any(step < 1 for step in round_steps):
        raise ValueError("round_steps must contain positive integers")
    if len(round_steps) > len(DEV_FOLDS):
        raise ValueError("at most three rounds are allowed by the preregistered A/B/C DEV rotation")
    corpus = ensure_strict_genetic_corpus()
    if corpus["benchmark_sources_used"] or corpus["sealed_sources_used"]:
        raise RuntimeError("benchmark or sealed content cannot enter genetic selection")
    if not REHEARSAL_PATH.exists():
        raise FileNotFoundError(f"rehearsal corpus not found: {REHEARSAL_PATH}")
    root = output_dir or default_campaign_root()
    root = root.resolve()
    create_campaign_layout(root)
    tournament_root = root / "genetic_tournament"
    final_root = root / "final_training"
    state_path = tournament_root / "state.json"
    with _pipeline_lock(root):
        resources = resource_preflight(root)
        (tournament_root / "resource_preflight.json").write_text(
            json.dumps(resources, indent=2) + "\n", encoding="utf-8"
        )
        if resources["status"] != "ready":
            raise RuntimeError(f"V13 resource preflight blocked: {resources['problems']}")
        rng = random.Random(tournament_seed)
        genes = initial_population(rng, population)
        state: dict[str, Any] = {
            "status": "running",
            "stage": "initialized",
            "created_at": datetime.now(UTC).isoformat(),
            "output_dir": str(root),
            "protocol": {
                "schema": "trinity-strict-genetic-v13-unified",
                "dual_seeds": list(DUAL_SEEDS),
                "round_steps": list(round_steps),
                "micro_rows": micro_rows,
                "fold_rotation": ["dev-a", "half-dev-a-half-dev-b", "half-dev-b-half-dev-c"],
                "fitness": (
                    "0.35 strict + 0.15 balanced routing + 0.10 scaled MCC + 0.20 multi-category "
                    "+ 0.10 format + 0.10 rehearsal + 0.05 min(routing,format,rehearsal) "
                    "- 0.10 elastic/strict gap"
                ),
                "dual_seed_aggregate": "0.70 mean + 0.30 worst - 0.15 population std",
                "benchmark_policy": "no SEALED/BFCL during selection; official evaluation only after final selection",
                "hardware_safety": {
                    "xpu_memory_fraction": 0.70,
                    "sequence_length_cap": 1408,
                    "lora_rank_cap": 16,
                    "target_profile": "attention",
                    "all_exotic_regularizers_coactive": True,
                    "automatic_retry_after_device_loss": False,
                },
            },
            "search_space": SEARCH_SPACE,
            "resource_preflight": resources,
            "corpus": corpus,
            "rounds": [],
        }
        _write_state(state_path, state)
        try:
            for round_index, steps in enumerate(round_steps):
                dev_rows, dev_label = build_round_dev(round_index)
                round_state: dict[str, Any] = {
                    "round": round_index,
                    "steps": steps,
                    "dev_label": dev_label,
                    "genes": [],
                }
                state.update(
                    stage=f"round-{round_index}", current_genes=[gene["id"] for gene in genes]
                )
                _write_state(state_path, state)
                for gene in genes:
                    seed_results = []
                    for seed in DUAL_SEEDS:
                        result = run_candidate(
                            root=tournament_root,
                            gene=gene,
                            round_index=round_index,
                            seed=seed,
                            steps=steps,
                            micro_rows=micro_rows,
                            dev_rows=dev_rows,
                            dev_label=dev_label,
                            force=force,
                        )
                        seed_results.append(result)
                        state["active_result"] = {
                            "round": round_index,
                            "gene": gene["id"],
                            "seed": seed,
                            "fitness": result["fitness"],
                        }
                        _write_state(state_path, state)
                    aggregate = aggregate_dual_seed(
                        [float(item["fitness"]["fitness"]) for item in seed_results]
                    )
                    round_state["genes"].append(
                        {
                            "gene": gene,
                            "signature": _gene_signature(gene),
                            "seeds": seed_results,
                            **aggregate,
                        }
                    )
                round_state["genes"].sort(key=lambda item: item["robust_fitness"], reverse=True)
                round_state["ranking"] = [item["gene"]["id"] for item in round_state["genes"]]
                state["rounds"].append(round_state)
                _write_state(state_path, state)
                if round_index < len(round_steps) - 1:
                    genes = _spawn_next_generation(round_state["genes"], rng, round_index)
            champion = state["rounds"][-1]["genes"][0]
            state["champion"] = {
                "gene": champion["gene"],
                "signature": champion["signature"],
                "robust_fitness": champion["robust_fitness"],
            }
            if run_final:
                state.update(stage="final-dual-seed-training")
                _write_state(state_path, state)
                state["final"] = run_final_dual_seed(
                    final_root=final_root,
                    gene=champion["gene"],
                    steps=final_steps,
                    preference_steps=final_preference_steps,
                    force=force,
                )
            state.update(
                status="complete",
                stage="benchmark-ready" if run_final else "tournament-complete",
                finished_at=datetime.now(UTC).isoformat(),
            )
            _write_state(state_path, state)
            return state
        except Exception as error:
            state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
            _write_state(state_path, state)
            raise


def remove_intermediate_checkpoints(root: Path) -> dict[str, Any]:
    targets = [path for path in root.rglob("checkpoint-*") if path.is_dir()]
    bytes_before = sum(
        file.stat().st_size for target in targets for file in target.rglob("*") if file.is_file()
    )
    for target in targets:
        shutil.rmtree(target)
    return {"removed_checkpoints": len(targets), "reclaimed_bytes": bytes_before}
