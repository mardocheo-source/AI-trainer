from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ModelProbe(BaseModel):
    name: str
    path: str
    architecture: str
    model_type: str
    dtype: str
    parameter_count: int
    disk_bytes: int
    hidden_size: int | None = None
    layer_count: int | None = None
    context_length: int | None = None
    vocab_size: int | None = None
    linear_suffixes: list[str] = Field(default_factory=list)
    tensor_modules: list[str] = Field(default_factory=list)
    fingerprint: str
    created_at: str = Field(default_factory=utc_now)


class DatasetProbe(BaseModel):
    name: str
    source: str
    source_type: Literal["local", "huggingface"]
    format: str
    size_bytes: int | None = None
    row_count: int | None = None
    sample_columns: list[str] = Field(default_factory=list)
    secret_hits: int = 0
    fingerprint: str
    created_at: str = Field(default_factory=utc_now)


class NoiseConfig(BaseModel):
    enabled: bool = False
    amplitude_mode: Literal[
        "single_digit", "digit_pairs", "triplets", "diff", "constant"
    ] = "digit_pairs"
    source: Literal["pi", "e", "sqrt2", "phi"] = "pi"
    digit_order: Literal["natural", "shuffled"] = "natural"
    alpha: float = Field(default=5.0, ge=0.0, le=50.0)
    modulation: float = Field(default=0.35, ge=0.0, le=1.0)
    scope: Literal["all", "prompt"] = "all"
    envelope: Literal["constant", "linear", "cosine"] = "constant"
    clean_tail_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    seed_offset: int = Field(default=10_007, ge=0)
    protect_prompt_literals: bool = False
    value_consistency_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    intent_consistency_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class GeometryConfig(BaseModel):
    enabled: bool = False
    weight: float = Field(default=0.02, ge=0.0, le=1.0)
    mode: Literal["orthogonal", "margin", "relational"] = "orthogonal"
    scope: Literal["all_completion", "structured", "anchors"] = "all_completion"
    layer: int = -1
    margin: float = Field(default=0.2, gt=0.0)
    sample_tokens: int = Field(default=64, ge=4, le=512)


class Boolean3DConfig(BaseModel):
    enabled: bool = False
    weight: float = Field(default=0.01, ge=0.0, le=1.0)
    polytope_margin: float = Field(default=0.35, ge=0.0, le=2.0)
    polytope_weight: float = Field(default=0.5, ge=0.0, le=2.0)
    refusal_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    refusal_action_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    # Optional asymmetric row-level barrier. Neutral zero margins preserve all
    # historical recipes; V9.4 enables both explicitly.
    call_margin: float = Field(default=0.0, ge=0.0, le=2.0)
    refuse_margin: float = Field(default=0.0, ge=0.0, le=2.0)
    asymmetric_refusal_multiplier: float = Field(default=1.0, ge=1.0, le=8.0)
    dimension_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_tokens: int = Field(default=48, ge=4, le=512)
    layer: int = -1


class RoleLossConfig(BaseModel):
    enabled: bool = False
    ordinary_weight: float = Field(default=1.0, gt=0.0, le=20.0)
    delimiter_weight: float = Field(default=0.5, gt=0.0, le=20.0)
    tool_name_weight: float = Field(default=1.5, gt=0.0, le=20.0)
    argument_key_weight: float = Field(default=2.0, gt=0.0, le=20.0)
    argument_value_weight: float = Field(default=2.5, gt=0.0, le=20.0)
    turn_horizon_beta: float = Field(default=0.0, ge=0.0, le=2.0)
    # Contextual multiplier applied only to TOOL_NAME tokens on examples whose
    # normalized capability metadata marks them as parallel.  The neutral
    # default preserves every pre-V9.3 recipe bit-for-bit.
    parallel_tool_name_multiplier: float = Field(default=1.0, gt=0.0, le=4.0)
    # V9.8 targeted grounding weights.  Neutral defaults preserve all sealed
    # historical contracts.  The masks are derived from supervised token-role
    # spans: exact prompt literals, parallel argument values, Boolean values,
    # assignment targets, and explicitly mentioned optional keys.
    literal_copy_span_multiplier: float = Field(default=1.0, gt=0.0, le=12.0)
    # V12.0 dual-mask supervision.  The neutral defaults keep every historical
    # recipe unchanged.  Dataset rows explicitly marked as docstring-derived
    # receive a distinct value-token multiplier, while the cardinality term is
    # an EOS unlikelihood penalty at non-final parallel call boundaries.
    docstring_derived_value_multiplier: float = Field(
        default=1.0, gt=0.0, le=12.0
    )
    cardinality_completion_penalty: float = Field(default=0.0, ge=0.0, le=4.0)
    parallel_argument_value_multiplier: float = Field(default=1.0, gt=0.0, le=8.0)
    # Parallel-envelope span anchoring.  Zero is neutral for all contracts
    # predating V11.0.  Positive values reinforce the call-boundary/tool-name
    # skeleton and exact prompt literals inside parallel completions without
    # globally increasing ordinary argument-token gradients.
    pe_span_anchor_weight: float = Field(default=0.0, ge=0.0, le=2.0)
    boolean_argument_multiplier: float = Field(default=1.0, gt=0.0, le=12.0)
    assignment_target_multiplier: float = Field(default=1.0, gt=0.0, le=8.0)
    explicit_optional_key_multiplier: float = Field(default=1.0, gt=0.0, le=8.0)
    # Closing tokens can be reinforced independently from opening/punctuation
    # delimiters. Refusal opening penalty is an unlikelihood term and is zero
    # by default so older contracts are unchanged.
    closing_delimiter_multiplier: float = Field(default=1.0, gt=0.0, le=8.0)
    refusal_opening_penalty: float = Field(default=0.0, ge=0.0, le=20.0)


class TrainingRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "agentic-rslora"
    model: str
    datasets: list[str]
    validation_mode: Literal["group_holdout", "external"] = "group_holdout"
    validation_datasets: list[str] = Field(default_factory=list)
    benchmark_profile: Literal["general", "pi-agent", "mixed"] = "general"
    comparison_group: str = "default"
    experiment_variant: Literal["baseline", "exotic", "custom"] = "custom"
    output_dir: str | None = None
    seed: int = 42
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    device: Literal["auto", "xpu", "cpu"] = "auto"
    xpu_memory_fraction: float = Field(default=0.70, ge=0.5, le=0.90)
    sequence_length: int = Field(default=2048, ge=128, le=32768)
    micro_batch_size: int = Field(default=1, ge=1, le=32)
    gradient_accumulation: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=0.01)
    lr_scheduler_type: Literal["cosine", "cosine_with_min_lr"] = "cosine"
    cosine_min_learning_rate: float = Field(default=0.0, ge=0.0, le=0.01)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_steps: int = Field(default=100_000, ge=1)
    budget_mode: Literal["time", "steps"] = "time"
    time_limit_minutes: int = Field(default=180, ge=5, le=1440)
    reserve_minutes: int = Field(default=20, ge=1, le=240)
    save_every_minutes: int = Field(default=15, ge=1, le=120)
    logging_steps: int = Field(default=5, ge=1)
    eval_ratio: float = Field(default=0.05, gt=0.0, lt=0.5)
    max_source_rows: int = Field(default=25_000, ge=100, le=2_000_000)
    max_training_samples: int = Field(default=50_000, ge=100, le=5_000_000)
    max_validation_samples: int = Field(default=512, ge=8, le=100_000)
    agent_eval_samples: int = Field(default=16, ge=0, le=256)
    agent_eval_max_new_tokens: int = Field(default=128, ge=8, le=2048)
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["read", "write", "edit", "bash"]
    )
    filter_unknown_tools: bool = True
    tool_menu_conditioning: bool = False
    packing: bool = False
    train_sampling_strategy: Literal["random", "sequential"] = "random"
    replay_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)
    replay_stratum: str = "rehearsal"
    # The dataset may already be a sealed, quota-controlled optimizer stream.
    # In that case replay_ratio remains recorded as a contract parameter, while
    # the loader must not reshuffle/recompose the precomputed order.
    precomposed_training_stream: bool = False
    gradient_checkpointing: bool = True
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    use_rslora: bool = True
    target_profile: Literal["attention", "attention-mlp", "all-linear", "auto"] = "auto"
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    role_loss: RoleLossConfig = Field(default_factory=RoleLossConfig)
    boolean_3d: Boolean3DConfig = Field(default_factory=Boolean3DConfig)
    state_binding_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    auxiliary_loss_policy: Literal["strict", "degrade"] = "strict"
    save_steps: int | None = None
    init_adapter_path: str | None = None
    # Optional, explicit continuation point for curricula that replace the
    # training stream between sealed sentinel boundaries.  The checkpoint
    # carries model, optimizer, scheduler and RNG state; ``ignore_data_skip``
    # is required when the replacement stream intentionally starts with a new
    # replay prefix.
    resume_from_checkpoint: str | None = None
    ignore_data_skip: bool = False
    # Stop at an optimizer step while keeping ``max_steps`` equal to the final
    # schedule horizon.  This preserves the original cosine scheduler when an
    # adaptive controller evaluates and replaces replay data between segments.
    stop_after_steps: int | None = Field(default=None, ge=1)
    workbook_sheet_id: str | None = None
    exploration_trial: int | None = None
    exploration_signature: str | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> TrainingRecipe:
        if self.reserve_minutes >= self.time_limit_minutes:
            raise ValueError("reserve_minutes must be smaller than time_limit_minutes")
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        if self.cosine_min_learning_rate >= self.learning_rate:
            raise ValueError(
                "cosine_min_learning_rate must be smaller than learning_rate"
            )
        if self.lr_scheduler_type == "cosine" and self.cosine_min_learning_rate != 0.0:
            raise ValueError(
                "cosine_min_learning_rate requires lr_scheduler_type='cosine_with_min_lr'"
            )
        if self.validation_mode == "external" and not self.validation_datasets:
            raise ValueError(
                "external validation mode requires at least one validation dataset"
            )
        if self.validation_mode == "group_holdout" and self.validation_datasets:
            raise ValueError(
                "group_holdout validation must not define external datasets"
            )
        if self.experiment_variant == "baseline" and (
            self.noise.enabled
            or self.geometry.enabled
            or self.role_loss.enabled
            or self.boolean_3d.enabled
        ):
            raise ValueError(
                "baseline variants must keep noise, geometry, role loss and boolean 3d disabled"
            )
        return self

    def resolved_output(self, root: Path, timestamp: str) -> Path:
        if self.output_dir:
            return Path(self.output_dir).expanduser().resolve()
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.name)
        return root / f"{timestamp}_{safe_name}"


class ExplorationConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["random", "grid"] = "random"
    trial_count: int = Field(default=3, ge=1, le=32)
    total_time_limit_minutes: int = Field(default=180, ge=15, le=1440)
    seed: int = 42
    adapter_methods: list[Literal["lora", "rslora"]] = Field(
        default_factory=lambda: ["rslora"]
    )
    lora_ranks: list[int] = Field(default_factory=lambda: [8, 16])
    target_profiles: list[
        Literal["attention", "attention-mlp", "all-linear", "auto"]
    ] = Field(default_factory=lambda: ["attention", "auto"])
    sequence_lengths: list[int] = Field(default_factory=lambda: [1024, 2048])
    learning_rate_min: float = Field(default=5e-5, gt=0.0, le=0.01)
    learning_rate_max: float = Field(default=2e-4, gt=0.0, le=0.01)
    noise_modes: list[Literal["off", "pi", "e", "sqrt2", "phi"]] = Field(
        default_factory=lambda: ["off", "pi"]
    )
    noise_alpha_min: float = Field(default=1.0, ge=0.0, le=50.0)
    noise_alpha_max: float = Field(default=8.0, ge=0.0, le=50.0)
    noise_scopes: list[Literal["all", "prompt"]] = Field(
        default_factory=lambda: ["all", "prompt"]
    )
    noise_digit_orders: list[Literal["natural", "shuffled"]] = Field(
        default_factory=lambda: ["natural"]
    )
    noise_envelopes: list[Literal["constant", "linear", "cosine"]] = Field(
        default_factory=lambda: ["constant"]
    )
    noise_modulation_min: float = Field(default=0.1, ge=0.0, le=1.0)
    noise_modulation_max: float = Field(default=0.35, ge=0.0, le=1.0)
    noise_clean_tail_min: float = Field(default=0.0, ge=0.0, lt=1.0)
    noise_clean_tail_max: float = Field(default=0.3, ge=0.0, lt=1.0)
    geometry_modes: list[Literal["off", "on", "orthogonal", "margin", "relational"]] = (
        Field(default_factory=lambda: ["off", "relational"])
    )
    geometry_scopes: list[Literal["all_completion", "structured"]] = Field(
        default_factory=lambda: ["structured"]
    )
    geometry_layers: list[int] = Field(default_factory=lambda: [-1, -4])
    geometry_weight_min: float = Field(default=0.005, ge=0.0, le=1.0)
    geometry_weight_max: float = Field(default=0.05, ge=0.0, le=1.0)
    geometry_margin_min: float = Field(default=0.1, gt=0.0, le=1.0)
    geometry_margin_max: float = Field(default=0.3, gt=0.0, le=1.0)
    role_loss_modes: list[Literal["off", "on"]] = Field(
        default_factory=lambda: ["off", "on"]
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> ExplorationConfig:
        # Workbook migration: older sheets used the binary value "on" for
        # the original Gram-to-identity geometry.
        self.geometry_modes = [
            "orthogonal" if mode == "on" else mode for mode in self.geometry_modes
        ]
        if self.learning_rate_min > self.learning_rate_max:
            raise ValueError("learning_rate_min must not exceed learning_rate_max")
        if self.noise_alpha_min > self.noise_alpha_max:
            raise ValueError("noise_alpha_min must not exceed noise_alpha_max")
        if self.noise_modulation_min > self.noise_modulation_max:
            raise ValueError(
                "noise_modulation_min must not exceed noise_modulation_max"
            )
        if self.noise_clean_tail_min > self.noise_clean_tail_max:
            raise ValueError(
                "noise_clean_tail_min must not exceed noise_clean_tail_max"
            )
        if self.geometry_weight_min > self.geometry_weight_max:
            raise ValueError("geometry_weight_min must not exceed geometry_weight_max")
        if self.geometry_margin_min > self.geometry_margin_max:
            raise ValueError("geometry_margin_min must not exceed geometry_margin_max")
        if not all(
            (
                self.adapter_methods,
                self.lora_ranks,
                self.target_profiles,
                self.sequence_lengths,
                self.noise_modes,
                self.noise_scopes,
                self.noise_digit_orders,
                self.noise_envelopes,
                self.geometry_modes,
                self.geometry_scopes,
                self.geometry_layers,
                self.role_loss_modes,
            )
        ):
            raise ValueError(
                "all exploration choice lists must contain at least one value"
            )
        return self


class ExperimentSheet(BaseModel):
    id: str
    name: str
    recipe: TrainingRecipe
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
