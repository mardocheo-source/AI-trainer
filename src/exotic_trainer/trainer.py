from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_eval import evaluate_agent_behavior
from .boolean_3d import Boolean3DLoss, Boolean3DProjector
from .data_loading import load_training_data
from .model_probe import choose_lora_targets, inspect_model
from .noise import EmbeddingNoiseController
from .paths import runs_dir
from .preflight import example_hash
from .progress import EventWriter
from .recipe import save_recipe
from .registry import Registry
from .schema import Boolean3DConfig, GeometryConfig, RoleLossConfig, TrainingRecipe
from .sealed import sealed_guard
from .token_roles import TokenRole, batch_token_roles

_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _turn_credit_scale(
    turn_index: Any, turn_count: Any, beta: float, torch: Any
) -> Any:
    """Return per-row credit based on the actual assistant-turn depth."""
    indices = turn_index.float()
    counts = turn_count.float()
    depth = torch.where(
        counts.gt(1),
        indices / (counts - 1).clamp_min(1),
        torch.zeros_like(indices),
    ).clamp(0.0, 1.0)
    return 1.0 + float(beta) * depth.pow(1.5)


def _decoded_token_pieces(
    tokenizer: Any, token_ids: list[int]
) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pieces.append(piece)
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return "".join(pieces), offsets


def _contiguous_role_spans(
    labels: list[int], roles: list[int], role: TokenRole
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, (label, observed_role) in enumerate(zip(labels, roles, strict=True)):
        selected = label != -100 and observed_role == int(role)
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def _token_positions_for_chars(
    offsets: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if left < end and right > start
    ]
    if not positions:
        return None
    return positions[0], positions[-1] + 1


def _state_binding_token_pairs(
    tokenizer: Any,
    input_ids: Any,
    labels: Any,
    roles: Any,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Find local and prompt-to-completion Python variable bindings.

    Each returned tuple is ``producer_start, producer_end, consumer_start,
    consumer_end`` in token coordinates. Prompt producers must be assignment
    targets, which avoids treating repeated schema words as state variables.
    """
    ids = [int(value) for value in input_ids.detach().cpu().tolist()]
    label_values = [int(value) for value in labels.detach().cpu().tolist()]
    role_values = [int(value) for value in roles.detach().cpu().tolist()]
    supervised = [index for index, value in enumerate(label_values) if value != -100]
    if not supervised:
        return [], []
    completion_start = supervised[0]
    text, offsets = _decoded_token_pieces(tokenizer, ids)
    prompt_end = (
        offsets[completion_start][0] if completion_start < len(offsets) else len(text)
    )
    prompt_text = text[:prompt_end]
    local_producers: dict[str, list[tuple[int, int]]] = {}
    local_pairs: list[tuple[int, int, int, int]] = []
    cross_pairs: list[tuple[int, int, int, int]] = []

    for start, end in _contiguous_role_spans(
        label_values, role_values, TokenRole.ARGUMENT_VALUE
    ):
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        name = text[char_start:char_end].strip()
        if not _PYTHON_IDENTIFIER.fullmatch(name):
            continue
        is_assignment = re.match(r"\s*=", text[char_end:]) is not None
        if is_assignment:
            local_producers.setdefault(name, []).append((start, end))
            continue

        producers = local_producers.get(name) or []
        if producers:
            producer_start, producer_end = producers[-1]
            local_pairs.append((producer_start, producer_end, start, end))

        assignment = re.compile(
            rf"(?<![A-Za-z0-9_])({re.escape(name)})(?![A-Za-z0-9_])\s*="
        )
        matches = list(assignment.finditer(prompt_text))
        if matches:
            match = matches[-1]
            producer = _token_positions_for_chars(offsets, match.start(1), match.end(1))
            if producer is not None:
                cross_pairs.append((*producer, start, end))

    return local_pairs, cross_pairs


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "cpu":
        return "cpu"
    available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    if requested == "xpu" and not available:
        raise RuntimeError("XPU requested but torch.xpu.is_available() is false")
    return "xpu" if available else "cpu"


def _build_callback(
    trainer_callback: type,
    writer: EventWriter,
    deadline: float,
    save_interval_seconds: int,
    noise: EmbeddingNoiseController | None,
    torch: Any,
    stop_file: Path | None,
    budget_mode: str = "time",
    stop_after_steps: int | None = None,
) -> Any:
    class BudgetCallback(trainer_callback):
        def __init__(self) -> None:
            self.started = time.monotonic()
            self.last_save = self.started
            self.stop_requested = False
            self.time_budget_reached = False
            self.sentinel_reached = False

        def _apply_external_stop(self, state: Any, control: Any) -> Any:
            if stop_file is not None and stop_file.exists() and not self.stop_requested:
                self.stop_requested = True
                writer.write("stop_requested", step=state.global_step)
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            writer.write("train_begin", max_steps=state.max_steps, deadline=deadline)
            return self._apply_external_stop(state, control)

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            now = time.monotonic()
            if noise:
                if budget_mode == "steps":
                    progress = state.global_step / max(1, state.max_steps)
                else:
                    progress = (now - self.started) / max(1.0, deadline - self.started)
                noise.advance(progress)
            if now - self.last_save >= save_interval_seconds:
                control.should_save = True
                self.last_save = now
            if now >= deadline:
                self.time_budget_reached = True
                writer.write("time_budget_reached", step=state.global_step)
                control.should_save = True
                control.should_training_stop = True
            if (
                stop_after_steps is not None
                and state.global_step >= stop_after_steps
                and state.global_step < state.max_steps
            ):
                self.sentinel_reached = True
                writer.write(
                    "adaptive_sentinel_reached",
                    step=state.global_step,
                    max_steps=state.max_steps,
                )
                control.should_save = True
                control.should_training_stop = True
            return self._apply_external_stop(state, control)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            payload = dict(logs or {})
            payload["step"] = state.global_step
            payload["elapsed_seconds"] = time.monotonic() - self.started
            if noise:
                payload["noise"] = noise.state()
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    payload["xpu_memory_bytes"] = int(torch.xpu.memory_allocated())
                    payload["xpu_peak_memory_bytes"] = int(
                        torch.xpu.max_memory_allocated()
                    )
            except RuntimeError as error:
                payload["xpu_memory_error"] = str(error)
            writer.write("log", **payload)
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            writer.write("checkpoint", step=state.global_step)
            return control

    return BudgetCallback()


def _time_aware_trainer_class(
    base: type,
    torch: Any,
    schedule_start: float,
    deadline: float,
    warmup_ratio: float,
) -> type:
    class TimeAwareSFTTrainer(base):
        """Cosine schedule whose progress follows wall time instead of guessed step count."""

        def create_scheduler(
            self, num_training_steps: int, optimizer: Any | None = None
        ) -> Any:
            if self.lr_scheduler is not None:
                return self.lr_scheduler
            selected_optimizer = self.optimizer if optimizer is None else optimizer
            total_seconds = max(1.0, deadline - schedule_start)
            warmup_seconds = total_seconds * warmup_ratio

            def multiplier(_step: int) -> float:
                elapsed = max(0.0, time.monotonic() - schedule_start)
                if warmup_seconds > 0 and elapsed < warmup_seconds:
                    return max(1e-3, elapsed / warmup_seconds)
                decay_seconds = max(1.0, total_seconds - warmup_seconds)
                progress = min(
                    1.0, max(0.0, (elapsed - warmup_seconds) / decay_seconds)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                selected_optimizer, lr_lambda=multiplier
            )
            return self.lr_scheduler

    return TimeAwareSFTTrainer


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            state["xpu"] = torch.xpu.get_rng_state()
    except (AttributeError, RuntimeError):
        pass
    return state


def _restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if "xpu" in state:
        try:
            torch.xpu.set_rng_state(state["xpu"])
        except (AttributeError, RuntimeError):
            pass


class _CapabilityMetadataCollator:
    """Preserve compact numeric v7 targets alongside TRL's token batch."""

    columns = (
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

    def __init__(self, base: Any, torch: Any) -> None:
        self.base = base
        self.torch = torch

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        clean = [
            {key: value for key, value in example.items() if key not in self.columns}
            for example in examples
        ]
        batch = self.base(clean)
        for name in self.columns:
            values = [
                example.get(name, 1.0 if name == "trajectory_weight" else 0)
                for example in examples
            ]
            dtype = (
                self.torch.float32 if name == "trajectory_weight" else self.torch.long
            )
            batch[name] = self.torch.tensor(values, dtype=dtype)
        return batch


def _regularized_trainer_class(
    base: type,
    torch: Any,
    tokenizer: Any | None,
    geometry: GeometryConfig,
    role_loss: RoleLossConfig,
    boolean_3d: Boolean3DConfig | None = None,
    noise: EmbeddingNoiseController | None = None,
    writer: EventWriter | None = None,
    logging_steps: int = 5,
    state_binding_weight: float = 0.0,
    auxiliary_loss_policy: str = "strict",
) -> type:
    b3d_config = boolean_3d or Boolean3DConfig()

    class StructuredSFTTrainer(base):
        """Combine role CE, noise, geometry and the 3D Boolean latent space."""

        _last_geometry_step = -1

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.auxiliary_failures: dict[str, str] = {}
            self.auxiliary_activity = {
                "state_binding_active_pairs": 0,
                "state_binding_active_batches": 0,
                "state_binding_local_active_pairs": 0,
                "state_binding_cross_turn_active_pairs": 0,
                "multiturn_active_batches": 0,
                "boolean_3d_active_batches": 0,
            }
            self.boolean_3d_projector = None
            self.boolean_3d_loss_fn = None
            if (
                b3d_config.enabled
                and hasattr(self.model, "config")
                and hasattr(self.model.config, "hidden_size")
            ):
                hidden_dim = self.model.config.hidden_size
                self.boolean_3d_projector = Boolean3DProjector(hidden_dim).to(
                    device=self.model.device, dtype=self.model.dtype
                )
                # This is an explicit fixed coordinate probe. Training it would
                # let the small projector absorb the objective instead of
                # shaping the model hidden states, and it is not an inference
                # artifact saved by PEFT.
                self.boolean_3d_projector.requires_grad_(False)
                self.boolean_3d_loss_fn = Boolean3DLoss(b3d_config)

        def _roles(self, inputs: dict[str, Any]) -> Any | None:
            labels = inputs.get("labels")
            input_ids = inputs.get("input_ids")
            if tokenizer is None or labels is None or input_ids is None:
                return None
            return batch_token_roles(tokenizer, input_ids, labels, torch)

        def _weighted_ce(
            self,
            outputs: Any,
            input_ids: Any,
            labels: Any,
            roles: Any,
            is_parallel: Any | None = None,
            is_docstring_derived: Any | None = None,
        ) -> Any:
            # Do not make a second contiguous copy of the full [tokens, vocab]
            # tensor.  On a 12 GiB XPU that copy plus the cross-entropy workspace
            # can make Level Zero reset the device for long seed-dependent batches.
            logits = outputs.logits[..., :-1, :]
            targets = labels[..., 1:].contiguous()
            selected_roles = roles[..., 1:].contiguous()
            weights = torch.full(
                targets.shape,
                float(role_loss.ordinary_weight),
                device=targets.device,
                dtype=logits.dtype,
            )
            mapping = {
                TokenRole.DELIMITER: role_loss.delimiter_weight,
                TokenRole.TOOL_NAME: role_loss.tool_name_weight,
                TokenRole.ARGUMENT_KEY: role_loss.argument_key_weight,
                TokenRole.ARGUMENT_VALUE: role_loss.argument_value_weight,
                TokenRole.CLOSING_DELIMITER: (
                    role_loss.delimiter_weight * role_loss.closing_delimiter_multiplier
                ),
            }
            for token_role, value in mapping.items():
                weights.masked_fill_(selected_roles.eq(int(token_role)), float(value))
            parallel_multiplier = float(
                getattr(role_loss, "parallel_tool_name_multiplier", 1.0)
            )
            parallel_rows = None
            if is_parallel is not None:
                parallel_rows = is_parallel.to(targets.device).bool()
                parallel_rows = parallel_rows.reshape(
                    parallel_rows.shape[0], *([1] * (targets.ndim - 1))
                )
            if parallel_rows is not None and parallel_multiplier != 1.0:
                parallel_tool_names = (
                    selected_roles.eq(int(TokenRole.TOOL_NAME)) & parallel_rows
                )
                weights = torch.where(
                    parallel_tool_names,
                    weights * parallel_multiplier,
                    weights,
                )

            targeted = {
                "literal": float(
                    getattr(role_loss, "literal_copy_span_multiplier", 1.0)
                ),
                "derived": float(
                    getattr(role_loss, "docstring_derived_value_multiplier", 1.0)
                ),
                "pe_literal": 1.0
                + float(getattr(role_loss, "pe_span_anchor_weight", 0.0)),
                "boolean": float(
                    getattr(role_loss, "boolean_argument_multiplier", 1.0)
                ),
                "assignment": float(
                    getattr(role_loss, "assignment_target_multiplier", 1.0)
                ),
                "optional_key": float(
                    getattr(role_loss, "explicit_optional_key_multiplier", 1.0)
                ),
            }
            parallel_value_multiplier = float(
                getattr(role_loss, "parallel_argument_value_multiplier", 1.0)
            )
            if parallel_rows is not None and parallel_value_multiplier != 1.0:
                weights = torch.where(
                    selected_roles.eq(int(TokenRole.ARGUMENT_VALUE)) & parallel_rows,
                    weights * parallel_value_multiplier,
                    weights,
                )

            # PE-Span is intentionally local: it strengthens only the repeated
            # call-envelope skeleton on parallel rows.  Exact cross-call IDs are
            # reinforced below through ``pe_literal`` using prompt membership,
            # so arbitrary semantic values do not receive a broad extra weight.
            pe_span_weight = float(getattr(role_loss, "pe_span_anchor_weight", 0.0))
            if parallel_rows is not None and pe_span_weight > 0.0:
                pe_roles = (
                    selected_roles.eq(int(TokenRole.DELIMITER))
                    | selected_roles.eq(int(TokenRole.CLOSING_DELIMITER))
                    | selected_roles.eq(int(TokenRole.TOOL_NAME))
                )
                weights = torch.where(
                    pe_roles & parallel_rows,
                    weights * (1.0 + pe_span_weight),
                    weights,
                )

            if tokenizer is not None and any(
                value != 1.0 for value in targeted.values()
            ):
                masks = {
                    name: torch.zeros_like(labels, dtype=torch.bool, device="cpu")
                    for name in targeted
                }
                cpu_ids = input_ids.detach().cpu()
                cpu_labels = labels.detach().cpu()
                cpu_roles = roles.detach().cpu()
                derived_rows = None
                if is_docstring_derived is not None:
                    derived_rows = is_docstring_derived.detach().cpu().bool()
                for row in range(cpu_ids.shape[0]):
                    ids = [int(value) for value in cpu_ids[row].tolist()]
                    label_values = [int(value) for value in cpu_labels[row].tolist()]
                    role_values = [int(value) for value in cpu_roles[row].tolist()]
                    supervised = [
                        index
                        for index, value in enumerate(label_values)
                        if value != -100
                    ]
                    if not supervised:
                        continue
                    text, offsets = _decoded_token_pieces(tokenizer, ids)
                    completion_start = supervised[0]
                    prompt_end = offsets[completion_start][0]
                    prompt_text = text[:prompt_end]
                    for start, end in _contiguous_role_spans(
                        label_values, role_values, TokenRole.ARGUMENT_VALUE
                    ):
                        char_start = offsets[start][0]
                        char_end = offsets[end - 1][1]
                        value = text[char_start:char_end].strip().strip("\"'")
                        suffix = text[char_end : char_end + 8]
                        is_assignment = re.match(r"^\s*=", suffix) is not None
                        is_prompt_literal = bool(value) and value in prompt_text
                        if is_prompt_literal and not is_assignment:
                            masks["literal"][row, start:end] = True
                            if parallel_rows is not None and bool(
                                parallel_rows[row].reshape(-1)[0].item()
                            ):
                                masks["pe_literal"][row, start:end] = True
                        if (
                            derived_rows is not None
                            and bool(derived_rows[row].item())
                            and not is_prompt_literal
                            and not is_assignment
                        ):
                            masks["derived"][row, start:end] = True
                        if value.lower() in {"true", "false"}:
                            masks["boolean"][row, start:end] = True
                        if re.match(r"^\s*=", suffix) and re.fullmatch(
                            r"(?:state|parallel_result)_\d+", value
                        ):
                            masks["assignment"][row, start:end] = True
                    for start, end in _contiguous_role_spans(
                        label_values, role_values, TokenRole.ARGUMENT_KEY
                    ):
                        char_start = offsets[start][0]
                        char_end = offsets[end - 1][1]
                        key = text[char_start:char_end].strip().strip("\"'")
                        if key in {"include_archived", "include_history"}:
                            masks["optional_key"][row, start:end] = True
                for name, multiplier in targeted.items():
                    if multiplier == 1.0:
                        continue
                    shifted_mask = masks[name][..., 1:].to(targets.device)
                    weights = torch.where(shifted_mask, weights * multiplier, weights)
            weights = weights * targets.ne(-100).to(weights.dtype)

            # Chunking bounds the temporary CE workspace independently of sequence
            # length while preserving exactly the same token-weighted objective.
            weighted_sum = logits.new_zeros(())
            chunk_tokens = 128
            vocabulary = logits.shape[-1]
            for start in range(0, logits.shape[-2], chunk_tokens):
                end = min(start + chunk_tokens, logits.shape[-2])
                chunk_loss = torch.nn.functional.cross_entropy(
                    logits[..., start:end, :].reshape(-1, vocabulary),
                    targets[..., start:end].reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape_as(targets[..., start:end])
                weighted_sum = (
                    weighted_sum + (chunk_loss * weights[..., start:end]).sum()
                )
            return weighted_sum / weights.sum().clamp_min(1.0)

        def _cardinality_completion_loss(
            self,
            outputs: Any,
            labels: Any,
            roles: Any,
            is_parallel: Any | None,
            expected_call_count: Any | None,
        ) -> Any:
            """Discourage EOS after a non-final call in supervised multi-call rows."""
            penalty = float(
                getattr(role_loss, "cardinality_completion_penalty", 0.0)
            )
            if (
                penalty <= 0.0
                or tokenizer is None
                or is_parallel is None
                or expected_call_count is None
            ):
                return outputs.logits.new_zeros(())
            eos_id = tokenizer.eos_token_id
            if not isinstance(eos_id, int) or eos_id < 0:
                raise ValueError("tokenizer has no scalar eos_token_id")

            parallel = is_parallel.to(labels.device).bool().reshape(-1)
            expected = expected_call_count.to(labels.device).long().reshape(-1)
            terms: list[Any] = []
            for row in range(labels.shape[0]):
                if not bool(parallel[row].item()) or int(expected[row].item()) <= 1:
                    continue
                valid_close = labels[row].ne(-100) & roles[row].eq(
                    int(TokenRole.CLOSING_DELIMITER)
                )
                boundary = valid_close.clone()
                boundary[:-1] &= ~valid_close[1:]
                positions = boundary.nonzero(as_tuple=False).flatten()
                if positions.numel() <= 1:
                    continue
                for position in positions[:-1].tolist():
                    if position >= outputs.logits.shape[1] - 1:
                        continue
                    next_logits = outputs.logits[row, position].float()
                    log_probability = next_logits[eos_id] - torch.logsumexp(
                        next_logits, dim=-1
                    )
                    probability = torch.exp(log_probability).clamp(
                        max=1.0 - 1e-6
                    )
                    terms.append(-torch.log1p(-probability))
            if not terms:
                return outputs.logits.new_zeros(())
            return penalty * torch.stack(terms).mean().to(outputs.logits.dtype)

        def _refusal_opening_loss(
            self,
            outputs: Any,
            labels: Any,
            is_tool_query: Any | None,
        ) -> Any:
            """Penalize tool-block opening probability on explicit no-call rows."""
            penalty = float(role_loss.refusal_opening_penalty)
            if penalty <= 0 or is_tool_query is None or tokenizer is None:
                return outputs.logits.new_zeros(())
            token_id = tokenizer.convert_tokens_to_ids("<|tool_call_start|>")
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError("tokenizer has no <|tool_call_start|> token")
            targets = labels[..., 1:]
            valid = targets.ne(-100)
            first_completion = valid & valid.long().cumsum(dim=-1).eq(1)
            no_call_rows = ~is_tool_query.to(targets.device).bool().reshape(-1, 1)
            positions = first_completion & no_call_rows
            if not positions.any():
                return outputs.logits.new_zeros(())
            logits = outputs.logits[..., :-1, :][positions].float()
            probability = torch.softmax(logits, dim=-1)[:, token_id]
            unlikelihood = -torch.log1p(-probability.clamp(max=1.0 - 1e-6))
            return penalty * unlikelihood.mean().to(outputs.logits.dtype)

        def _disable_or_raise(self, name: str, error: Exception) -> None:
            message = f"{type(error).__name__}: {error}"
            if auxiliary_loss_policy == "strict":
                raise RuntimeError(
                    f"auxiliary loss {name} failed: {message}"
                ) from error
            if name not in self.auxiliary_failures:
                self.auxiliary_failures[name] = message
                if writer is not None:
                    writer.write("auxiliary_loss_disabled", name=name, error=message)

        def _state_binding_loss(
            self, hidden: Any, input_ids: Any, labels: Any, roles: Any
        ) -> tuple[Any, int, int, int]:
            """Align Python assignment producers with their later consumers."""
            if roles is None or input_ids is None or labels is None or hidden is None:
                return hidden.new_zeros(()), 0, 0, 0

            norm_h = torch.nn.functional.normalize(hidden.float(), dim=-1)
            batch_loss = hidden.new_zeros(())
            pairs_count = 0
            local_count = 0
            cross_count = 0

            for row in range(input_ids.shape[0]):
                local_pairs, cross_pairs = _state_binding_token_pairs(
                    tokenizer, input_ids[row], labels[row], roles[row]
                )
                for producer_start, producer_end, consumer_start, consumer_end in (
                    local_pairs + cross_pairs
                ):
                    producer = norm_h[row, producer_start:producer_end].mean(dim=0)
                    consumer = norm_h[row, consumer_start:consumer_end].mean(dim=0)
                    batch_loss = batch_loss + (1.0 - (producer * consumer).sum())
                    pairs_count += 1
                local_count += len(local_pairs)
                cross_count += len(cross_pairs)

            if pairs_count > 0:
                return batch_loss / pairs_count, pairs_count, local_count, cross_count
            return hidden.new_zeros(()), 0, 0, 0

        def _select_geometry(
            self,
            hidden: Any,
            labels: Any,
            roles: Any | None,
        ) -> tuple[Any, Any]:
            mask = labels.ne(-100)
            if geometry.scope == "structured" and roles is not None:
                mask = mask & roles.ge(int(TokenRole.TOOL_NAME))
            elif geometry.scope == "anchors" and roles is not None:
                mask = mask & (
                    roles.eq(int(TokenRole.TOOL_NAME))
                    | roles.eq(int(TokenRole.ARGUMENT_KEY))
                )
            selected = hidden[mask]
            positions = mask.nonzero(as_tuple=False)
            if selected.shape[0] > geometry.sample_tokens:
                indices = torch.linspace(
                    0,
                    selected.shape[0] - 1,
                    steps=geometry.sample_tokens,
                    device=selected.device,
                ).long()
                selected = selected.index_select(0, indices)
                positions = positions.index_select(0, indices)
            return selected, positions

        def _geometry_loss(self, selected: Any, reference: Any | None) -> Any:
            selected = torch.nn.functional.normalize(selected.float(), dim=-1)
            gram = selected @ selected.transpose(0, 1)
            if geometry.mode == "relational" and reference is not None:
                reference = torch.nn.functional.normalize(reference.float(), dim=-1)
                reference_gram = reference @ reference.transpose(0, 1)
                return ((gram - reference_gram.detach()) ** 2).mean()
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            if geometry.mode == "margin":
                off_diagonal = ~torch.eye(
                    gram.shape[0], device=gram.device, dtype=torch.bool
                )
                excess = torch.nn.functional.relu(gram.abs() - geometry.margin)
                return (excess[off_diagonal] ** 2).mean()
            return ((gram - eye) ** 2).mean()

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            regularized_inputs = dict(inputs)
            capability_targets = {
                name: regularized_inputs.pop(name, None)
                for name in _CapabilityMetadataCollator.columns
            }
            turn_count = capability_targets.get("turn_count")
            if (
                model.training
                and turn_count is not None
                and bool(turn_count.gt(1).any().item())
            ):
                self.auxiliary_activity["multiturn_active_batches"] += 1
            if geometry.enabled or b3d_config.enabled or state_binding_weight > 0.0:
                regularized_inputs["output_hidden_states"] = True
            labels = regularized_inputs.get("labels")
            input_ids = regularized_inputs.get("input_ids")
            attention_mask = regularized_inputs.get("attention_mask")
            roles = self._roles(regularized_inputs) if model.training else None
            if noise:
                noise.begin_batch(labels, attention_mask, input_ids=input_ids)
            clean_value_logits = None
            clean_intent_logits = None
            clean_geometry_reference = None
            try:
                needs_clean_reference = bool(
                    model.training
                    and noise
                    and (
                        (geometry.enabled and geometry.mode == "relational")
                        or noise.config.value_consistency_weight > 0
                        or noise.config.intent_consistency_weight > 0
                    )
                )
                if needs_clean_reference:
                    rng_state = _capture_rng_state(torch)
                    with noise.suspended(), torch.no_grad():
                        clean_outputs = model(**regularized_inputs)
                    _restore_rng_state(torch, rng_state)
                    if (
                        noise.config.value_consistency_weight > 0
                        and labels is not None
                        and roles is not None
                    ):
                        clean_value_mask = labels[..., 1:].ne(-100) & roles[..., 1:].eq(
                            int(TokenRole.ARGUMENT_VALUE)
                        )
                        if clean_value_mask.any():
                            clean_value_logits = clean_outputs.logits[..., :-1, :][
                                clean_value_mask
                            ].detach()
                    if (
                        noise.config.intent_consistency_weight > 0
                        and labels is not None
                        and roles is not None
                    ):
                        shifted_valid = labels[..., 1:].ne(-100)
                        first_target = shifted_valid & shifted_valid.long().cumsum(
                            dim=-1
                        ).eq(1)
                        intent_mask = first_target | (
                            shifted_valid & roles[..., 1:].eq(int(TokenRole.TOOL_NAME))
                        )
                        if intent_mask.any():
                            clean_intent_logits = clean_outputs.logits[..., :-1, :][
                                intent_mask
                            ].detach()
                    if geometry.enabled and geometry.mode == "relational":
                        try:
                            clean_hidden = clean_outputs.hidden_states[geometry.layer]
                        except IndexError as error:
                            raise ValueError(
                                f"geometry layer {geometry.layer} is outside the model "
                                "hidden-state range"
                            ) from error
                        clean_geometry_reference, _clean_positions = (
                            self._select_geometry(clean_hidden, labels, roles)
                        )
                        clean_geometry_reference = clean_geometry_reference.detach()
                    # Retaining every clean vocabulary logit and hidden layer across the
                    # noisy forward wastes several GiB at sequence length 2048.  LiteralLock
                    # only needs the selected value logits and selected geometry anchors.
                    del clean_outputs
                loss, outputs = super().compute_loss(
                    model,
                    regularized_inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                if noise:
                    noise.end_batch()

            # The parent path is always called first so TRL continues logging
            # mean_token_accuracy, entropy and token counts.
            if (
                model.training
                and role_loss.enabled
                and labels is not None
                and roles is not None
            ):
                try:
                    weighted = self._weighted_ce(
                        outputs,
                        input_ids,
                        labels,
                        roles,
                        capability_targets.get("cap_is_parallel"),
                        capability_targets.get("cap_is_docstring_derived"),
                    ).to(loss.dtype)
                    row_weight = capability_targets.get("trajectory_weight")
                    turn_index = capability_targets.get("turn_index")
                    turn_count = capability_targets.get("turn_count")
                    if row_weight is not None:
                        scale = row_weight.to(weighted.device).float()
                        if turn_index is not None and turn_count is not None:
                            scale = scale * _turn_credit_scale(
                                turn_index.to(weighted.device),
                                turn_count.to(weighted.device),
                                float(getattr(role_loss, "turn_horizon_beta", 0.0)),
                                torch,
                            )
                        weighted = weighted * scale.mean().to(weighted.dtype)
                    loss = weighted + self._refusal_opening_loss(
                        outputs,
                        labels,
                        capability_targets.get("cap_is_tool"),
                    ).to(weighted.dtype)
                    loss = loss + self._cardinality_completion_loss(
                        outputs,
                        labels,
                        roles,
                        capability_targets.get("cap_is_parallel"),
                        capability_targets.get("cap_expected_call_count"),
                    ).to(weighted.dtype)
                except Exception as error:  # noqa: BLE001 - v7 can explicitly degrade
                    self._disable_or_raise("role_loss", error)

            # LiteralLock keeps the noisy view close to the clean teacher only
            # on target argument-value tokens. The π digit pairs still govern
            # prompt noise; exact literals themselves are protected separately.
            if (
                model.training
                and noise
                and noise.config.value_consistency_weight > 0
                and clean_value_logits is not None
                and labels is not None
                and roles is not None
            ):
                shifted_roles = roles[..., 1:]
                shifted_labels = labels[..., 1:]
                value_mask = shifted_labels.ne(-100) & shifted_roles.eq(
                    int(TokenRole.ARGUMENT_VALUE)
                )
                if value_mask.any():
                    noisy_logits = outputs.logits[..., :-1, :][value_mask].float()
                    consistency = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(noisy_logits, dim=-1),
                        torch.nn.functional.softmax(clean_value_logits.float(), dim=-1),
                        reduction="batchmean",
                    )
                    loss = (
                        loss
                        + noise.config.value_consistency_weight
                        * consistency.to(loss.dtype)
                    )

            # Human-intent consistency protects the autoregressive decision
            # boundary (act vs answer) and the selected tool name.  It keeps
            # these small, semantically critical regions close between the
            # clean and natural-π-noisy views without constraining every token.
            if (
                model.training
                and noise
                and noise.config.intent_consistency_weight > 0
                and clean_intent_logits is not None
                and labels is not None
                and roles is not None
            ):
                shifted_valid = labels[..., 1:].ne(-100)
                first_target = shifted_valid & shifted_valid.long().cumsum(dim=-1).eq(1)
                intent_mask = first_target | (
                    shifted_valid & roles[..., 1:].eq(int(TokenRole.TOOL_NAME))
                )
                if intent_mask.any():
                    noisy_intent_logits = outputs.logits[..., :-1, :][
                        intent_mask
                    ].float()
                    intent_consistency = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(noisy_intent_logits, dim=-1),
                        torch.nn.functional.softmax(
                            clean_intent_logits.float(), dim=-1
                        ),
                        reduction="batchmean",
                    )
                    loss = loss + noise.config.intent_consistency_weight * (
                        intent_consistency.to(loss.dtype)
                    )

            # Geometry is training-only so validation loss remains directly
            # comparable with a clean baseline.
            if model.training and geometry.enabled and labels is not None:
                try:
                    hidden = outputs.hidden_states[geometry.layer]
                except IndexError as error:
                    raise ValueError(
                        f"geometry layer {geometry.layer} is outside the model hidden-state range"
                    ) from error
                selected, positions = self._select_geometry(hidden, labels, roles)
                minimum_geometry_tokens = 2 if geometry.scope == "anchors" else 4
                if selected.shape[0] >= minimum_geometry_tokens:
                    reference = None
                    if clean_geometry_reference is not None:
                        reference = clean_geometry_reference
                    elif geometry.mode == "relational":
                        reference_layer = geometry.layer - 1
                        try:
                            prior_hidden = outputs.hidden_states[reference_layer]
                        except IndexError as error:
                            raise ValueError(
                                "relational geometry requires a valid preceding hidden layer"
                            ) from error
                        reference = prior_hidden[positions[:, 0], positions[:, 1]]
                    geometry_loss = self._geometry_loss(selected, reference)
                    loss = loss + geometry.weight * geometry_loss.to(loss.dtype)
                    step = int(self.state.global_step)
                    if step % logging_steps == 0 and step != self._last_geometry_step:
                        sample = torch.nn.functional.normalize(
                            selected[:32].detach().float(), dim=-1
                        ).cpu()
                        sample = sample - sample.mean(dim=0, keepdim=True)
                        try:
                            _u, _s, vectors = torch.pca_lowrank(
                                sample, q=2, center=False
                            )
                            coordinates = sample @ vectors[:, :2]
                            writer.write(
                                "geometry",
                                step=step,
                                loss=float(geometry_loss.detach().cpu()),
                                mode=geometry.mode,
                                scope=geometry.scope,
                                layer=geometry.layer,
                                selected_tokens=int(selected.shape[0]),
                                points=coordinates.tolist(),
                            )
                            self._last_geometry_step = step
                        except RuntimeError as error:
                            writer.write("geometry_error", step=step, error=str(error))

            # 3D Geometric Boolean Latent Space regularizer
            if (
                model.training
                and b3d_config.enabled
                and labels is not None
                and getattr(self, "boolean_3d_projector", None) is not None
                and getattr(self, "boolean_3d_loss_fn", None) is not None
            ):
                try:
                    b3d_hidden = outputs.hidden_states[b3d_config.layer]
                except (IndexError, AttributeError):
                    b3d_hidden = (
                        outputs.hidden_states[-1]
                        if hasattr(outputs, "hidden_states") and outputs.hidden_states
                        else None
                    )
                if b3d_hidden is not None:
                    if "boolean_3d" not in self.auxiliary_failures:
                        try:
                            b3d_loss = self.boolean_3d_loss_fn(
                                self.boolean_3d_projector,
                                b3d_hidden,
                                labels,
                                roles,
                                is_tool_query=capability_targets.get("cap_is_tool"),
                                is_parallel=capability_targets.get("cap_is_parallel"),
                                is_multiturn=capability_targets.get("cap_is_multiturn"),
                                is_typed=capability_targets.get("cap_is_typed"),
                            )
                            loss = loss + b3d_loss.to(loss.dtype)
                            self.auxiliary_activity["boolean_3d_active_batches"] += 1
                        except (
                            Exception
                        ) as error:  # noqa: BLE001 - v7 can explicitly degrade
                            self._disable_or_raise("boolean_3d", error)

            # State-Binding Anchor regularizer (Long-Context Variable Consistency)
            if (
                model.training
                and labels is not None
                and roles is not None
                and input_ids is not None
                and state_binding_weight > 0.0
                and hasattr(outputs, "hidden_states")
                and outputs.hidden_states
            ):
                if "state_binding" not in self.auxiliary_failures:
                    try:
                        last_hidden = outputs.hidden_states[-1]
                        state_loss, active_pairs, local_pairs, cross_pairs = (
                            self._state_binding_loss(
                                last_hidden, input_ids, labels, roles
                            )
                        )
                        if active_pairs:
                            loss = loss + state_binding_weight * state_loss.to(
                                loss.dtype
                            )
                            self.auxiliary_activity[
                                "state_binding_active_pairs"
                            ] += active_pairs
                            self.auxiliary_activity["state_binding_active_batches"] += 1
                            self.auxiliary_activity[
                                "state_binding_local_active_pairs"
                            ] += local_pairs
                            self.auxiliary_activity[
                                "state_binding_cross_turn_active_pairs"
                            ] += cross_pairs
                    except (
                        Exception
                    ) as error:  # noqa: BLE001 - v7 can explicitly degrade
                        self._disable_or_raise("state_binding", error)

            return (loss, outputs) if return_outputs else loss

    return StructuredSFTTrainer


def _geometric_trainer_class(
    base: type,
    torch: Any,
    weight: float,
    writer: EventWriter,
    logging_steps: int,
) -> type:
    """Compatibility wrapper for the original geometry trainer tests/API."""

    return _regularized_trainer_class(
        base,
        torch,
        tokenizer=None,
        geometry=GeometryConfig(enabled=True, weight=weight),
        role_loss=RoleLossConfig(),
        noise=None,
        writer=writer,
        logging_steps=logging_steps,
    )


@contextmanager
def _exclusive_training_lock():
    """Allow only one model-training worker to allocate the accelerator."""
    lock_path = runs_dir() / ".accelerator-training.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "another training worker"
            raise RuntimeError(
                f"accelerator training already running ({owner}); wait for it or stop it first"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} started={datetime.now(UTC).isoformat()}")
        lock_file.flush()
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def run_training(recipe: TrainingRecipe, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return _run_training(recipe, dry_run=True)
    with _exclusive_training_lock():
        return _run_training(recipe, dry_run=False)


def _run_training(recipe: TrainingRecipe, dry_run: bool = False) -> dict[str, Any]:
    registry = Registry()
    model_path = registry.resolve_model(recipe.model)
    dataset_sources = [registry.resolve_dataset(item) for item in recipe.datasets]
    validation_sources = [
        registry.resolve_dataset(item) for item in recipe.validation_datasets
    ]
    sealed_hashes, sealed_sources = sealed_guard()
    reused_sources = sorted(set(dataset_sources + validation_sources) & sealed_sources)
    if reused_sources:
        raise ValueError(
            f"sealed test sources cannot be used for training: {reused_sources}"
        )
    probe = inspect_model(model_path)
    targets = choose_lora_targets(probe, recipe.target_profile)
    plan = {
        "model": model_path,
        "datasets": dataset_sources,
        "validation_mode": recipe.validation_mode,
        "validation_datasets": validation_sources,
        "benchmark_profile": recipe.benchmark_profile,
        "comparison_group": recipe.comparison_group,
        "experiment_variant": recipe.experiment_variant,
        "parameter_count": probe.parameter_count,
        "target_modules": targets,
        "trainable_rank": recipe.lora_rank,
        "adapter_method": "rslora" if recipe.use_rslora else "lora",
        "xpu_memory_fraction": recipe.xpu_memory_fraction,
        "time_limit_minutes": recipe.time_limit_minutes,
        "training_budget_minutes": recipe.time_limit_minutes - recipe.reserve_minutes,
        "budget_mode": recipe.budget_mode,
        "max_steps": recipe.max_steps,
        "packing": recipe.packing,
        "train_sampling_strategy": recipe.train_sampling_strategy,
        "replay_ratio": recipe.replay_ratio,
        "replay_stratum": recipe.replay_stratum,
        "precomposed_training_stream": recipe.precomposed_training_stream,
        "noise": recipe.noise.model_dump(mode="json"),
        "geometry": recipe.geometry.model_dump(mode="json"),
        "role_loss": recipe.role_loss.model_dump(mode="json"),
        "boolean_3d": recipe.boolean_3d.model_dump(mode="json"),
        "state_binding_weight": recipe.state_binding_weight,
        "auxiliary_loss_policy": recipe.auxiliary_loss_policy,
        "tool_menu_conditioning": recipe.tool_menu_conditioning,
        "resume_from_checkpoint": recipe.resume_from_checkpoint,
        "ignore_data_skip": recipe.ignore_data_skip,
        "stop_after_steps": recipe.stop_after_steps,
    }
    if dry_run:
        return {"status": "dry-run", **plan}

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainerCallback,
            set_seed,
        )
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    except ImportError as error:
        raise RuntimeError(
            "training dependencies are missing; install a hardware-compatible PyTorch build and pip install -e '.[train]'"
        ) from error

    device = _resolve_device(recipe.device, torch)
    if device == "xpu" and hasattr(torch.xpu, "set_per_process_memory_fraction"):
        torch.xpu.set_per_process_memory_fraction(recipe.xpu_memory_fraction, device=0)
    set_seed(recipe.seed)
    random.seed(recipe.seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{random.randrange(0, 65536):04x}"
    output_dir = recipe.resolved_output(runs_dir(), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(output_dir / "events.jsonl")
    save_recipe(recipe, output_dir / "recipe.yaml")
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    registry.create_run(run_id, recipe.model_dump(mode="json"), output_dir)
    overall_start = time.monotonic()
    deadline = overall_start + (recipe.time_limit_minutes - recipe.reserve_minutes) * 60
    writer.write("run_created", run_id=run_id, device=device, **plan)

    noise_controller: EmbeddingNoiseController | None = None
    try:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype_map[recipe.dtype],
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        if recipe.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        lora = LoraConfig(
            r=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            use_rslora=recipe.use_rslora,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        if recipe.init_adapter_path:
            init_p = Path(recipe.init_adapter_path).resolve()
            weights_p = init_p / "adapter_model.safetensors"
            if not weights_p.is_file():
                raise FileNotFoundError(f"init adapter weights not found: {weights_p}")
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            init_state = load_file(str(weights_p), device="cpu")
            set_peft_model_state_dict(model, init_state)
            init_sha256 = hashlib.sha256(weights_p.read_bytes()).hexdigest()
            writer.write(
                "init_adapter_loaded",
                source=str(weights_p),
                sha256=init_sha256,
            )
            print(f"✓ Pesi iniziali caricati con successo da: {weights_p}")

        trainable, total = model.get_nb_trainable_parameters()
        writer.write(
            "model_loaded",
            trainable_parameters=trainable,
            total_parameters=total,
            trainable_percent=100 * trainable / total,
        )
        if recipe.noise.enabled:
            noise_controller = EmbeddingNoiseController(
                recipe.noise,
                seed=recipe.seed,
                tokenizer=tokenizer,
            )
            noise_controller.attach(model)

        train_data, eval_data, replay_manifest = load_training_data(
            dataset_sources,
            eval_ratio=recipe.eval_ratio,
            seed=recipe.seed,
            max_source_rows=recipe.max_source_rows,
            max_training_samples=recipe.max_training_samples,
            allowed_tools=recipe.allowed_tools if recipe.filter_unknown_tools else None,
            validation_sources=(
                validation_sources if recipe.validation_mode == "external" else None
            ),
            max_validation_samples=recipe.max_validation_samples,
            tool_menu_conditioning=recipe.tool_menu_conditioning,
            replay_ratio=(
                None if recipe.precomposed_training_stream else recipe.replay_ratio
            ),
            replay_stratum=recipe.replay_stratum,
            return_composition_manifest=True,
        )
        if sealed_hashes:
            overlap = sum(
                example_hash(example) in sealed_hashes
                for dataset in (train_data, eval_data)
                for example in dataset
            )
            if overlap:
                raise ValueError(
                    f"training blocked: {overlap} normalized samples overlap sealed test content"
                )
        writer.write(
            "dataset_ready",
            train_rows=len(train_data),
            eval_rows=len(eval_data),
            replay_manifest=replay_manifest,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 60:
            raise TimeoutError(
                "model and dataset preparation exhausted the training time budget"
            )

        training_args = SFTConfig(
            output_dir=str(output_dir),
            max_steps=recipe.max_steps,
            max_length=recipe.sequence_length,
            per_device_train_batch_size=recipe.micro_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=recipe.gradient_accumulation,
            learning_rate=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
            lr_scheduler_type=recipe.lr_scheduler_type,
            lr_scheduler_kwargs=(
                {"min_lr": recipe.cosine_min_learning_rate}
                if recipe.lr_scheduler_type == "cosine_with_min_lr"
                else {}
            ),
            warmup_ratio=recipe.warmup_ratio,
            logging_steps=recipe.logging_steps,
            eval_strategy="steps",
            eval_steps=max(25, recipe.logging_steps * 10),
            save_strategy="steps",
            save_steps=recipe.save_steps or recipe.max_steps,
            save_total_limit=5,
            packing=recipe.packing,
            completion_only_loss=True,
            gradient_checkpointing=recipe.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=recipe.dtype == "bf16" and device == "xpu",
            fp16=recipe.dtype == "fp16" and device == "xpu",
            use_cpu=device == "cpu",
            dataloader_pin_memory=False,
            report_to=[],
            train_sampling_strategy=recipe.train_sampling_strategy,
            seed=recipe.seed,
            data_seed=recipe.seed,
            remove_unused_columns=False,
            ignore_data_skip=recipe.ignore_data_skip,
        )
        callback = _build_callback(
            TrainerCallback,
            writer,
            deadline,
            recipe.save_every_minutes * 60,
            noise_controller,
            torch,
            (
                Path(os.environ["EXOTIC_TRAINER_STOP_FILE"])
                if os.environ.get("EXOTIC_TRAINER_STOP_FILE")
                else None
            ),
            budget_mode=recipe.budget_mode,
            stop_after_steps=recipe.stop_after_steps,
        )
        schedule_start = time.monotonic()
        trainer_type = SFTTrainer
        if recipe.budget_mode == "time":
            trainer_type = _time_aware_trainer_class(
                trainer_type,
                torch,
                schedule_start=schedule_start,
                deadline=deadline,
                warmup_ratio=recipe.warmup_ratio,
            )
        if (
            recipe.noise.enabled
            or recipe.geometry.enabled
            or recipe.role_loss.enabled
            or recipe.boolean_3d.enabled
        ):
            trainer_type = _regularized_trainer_class(
                trainer_type,
                torch,
                tokenizer=tokenizer,
                geometry=recipe.geometry,
                role_loss=recipe.role_loss,
                boolean_3d=recipe.boolean_3d,
                noise=noise_controller,
                writer=writer,
                logging_steps=recipe.logging_steps,
                state_binding_weight=recipe.state_binding_weight,
                auxiliary_loss_policy=recipe.auxiliary_loss_policy,
            )
        pad_token = tokenizer.pad_token or tokenizer.eos_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        if pad_token_id is None:
            raise ValueError(
                f"tokenizer pad token is not in the vocabulary: {pad_token!r}"
            )
        data_collator = _CapabilityMetadataCollator(
            DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=True,
            ),
            torch,
        )
        trainer = trainer_type(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            processing_class=tokenizer,
            callbacks=[callback],
            data_collator=data_collator,
        )
        registry.update_run(run_id, "training")
        resume_checkpoint = recipe.resume_from_checkpoint
        if resume_checkpoint:
            resume_path = Path(resume_checkpoint).expanduser().resolve()
            if not resume_path.is_dir():
                raise FileNotFoundError(
                    f"resume checkpoint directory not found: {resume_path}"
                )
            writer.write(
                "training_resume_requested",
                checkpoint=str(resume_path),
                ignore_data_skip=recipe.ignore_data_skip,
            )
            result = trainer.train(resume_from_checkpoint=str(resume_path))
        else:
            result = trainer.train()
        auxiliary_activity = dict(getattr(trainer, "auxiliary_activity", {}))
        auxiliary_failures = dict(getattr(trainer, "auxiliary_failures", {}))
        if (
            recipe.state_binding_weight > 0.0
            and int(auxiliary_activity.get("state_binding_active_pairs", 0)) == 0
            and "state_binding" not in auxiliary_failures
        ):
            inactive = "configured but no producer-consumer token pair became active"
            if recipe.auxiliary_loss_policy == "strict":
                raise RuntimeError(f"auxiliary loss state_binding failed: {inactive}")
            auxiliary_failures["state_binding"] = inactive
            writer.write(
                "auxiliary_loss_disabled", name="state_binding", error=inactive
            )
        if (
            recipe.state_binding_weight > 0.0
            and int(auxiliary_activity.get("multiturn_active_batches", 0)) > 0
            and int(auxiliary_activity.get("state_binding_cross_turn_active_pairs", 0))
            == 0
            and "state_binding" not in auxiliary_failures
        ):
            inactive = "multiturn rows were seen but no prompt-to-completion binding became active"
            if recipe.auxiliary_loss_policy == "strict":
                raise RuntimeError(f"auxiliary loss state_binding failed: {inactive}")
            auxiliary_failures["state_binding"] = inactive
            writer.write(
                "auxiliary_loss_disabled", name="state_binding", error=inactive
            )
        trainer.save_model(output_dir / "adapter-final")
        tokenizer.save_pretrained(output_dir / "adapter-final")
        writer.write("adapter_saved", path=str(output_dir / "adapter-final"))
        metrics = dict(result.metrics)
        final_deadline = overall_start + recipe.time_limit_minutes * 60
        stopped = bool(callback.stop_requested)
        adaptive_sentinel_reached = bool(callback.sentinel_reached)
        validation_metrics: dict[str, Any] = {}
        if (
            not stopped
            and not adaptive_sentinel_reached
            and time.monotonic() < final_deadline - 60
        ):
            try:
                writer.write(
                    "validation_eval_begin",
                    samples=len(eval_data),
                    policy=recipe.validation_mode,
                )
                # SFTTrainer preprocesses and stores its evaluation dataset at
                # construction time. Passing the original prompt/completion
                # dataset here bypasses that preparation and breaks the final
                # summary evaluation, even though periodic evaluation works.
                validation_metrics = dict(trainer.evaluate())
                if validation_metrics.get("eval_loss") is not None:
                    validation_metrics["eval_perplexity"] = math.exp(
                        min(20.0, float(validation_metrics["eval_loss"]))
                    )
                writer.write("validation_eval_complete", **validation_metrics)
            except Exception as error:  # noqa: BLE001 - adapter is already safely saved
                validation_metrics = {
                    "validation_eval_error": f"{type(error).__name__}: {error}"
                }
                writer.write("validation_eval_failed", **validation_metrics)
        metrics.update(validation_metrics)
        if stopped or adaptive_sentinel_reached:
            agent_metrics = {"agent_eval_samples": 0, "stopped_by_user": True}
            if adaptive_sentinel_reached and not stopped:
                agent_metrics = {
                    "agent_eval_samples": 0,
                    "adaptive_sentinel_reached": True,
                }
        else:
            try:
                writer.write("agent_eval_begin", max_samples=recipe.agent_eval_samples)
                agent_metrics = evaluate_agent_behavior(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=eval_data,
                    max_samples=recipe.agent_eval_samples,
                    max_new_tokens=recipe.agent_eval_max_new_tokens,
                    deadline=final_deadline,
                    tool_names=recipe.allowed_tools,
                )
            except (
                Exception
            ) as error:  # noqa: BLE001 - optional eval must not lose an adapter
                agent_metrics = {
                    "agent_eval_samples": 0,
                    "agent_eval_error": f"{type(error).__name__}: {error}",
                }
                writer.write(
                    "agent_eval_failed", error=agent_metrics["agent_eval_error"]
                )
            else:
                writer.write("agent_eval_complete", **agent_metrics)
        metrics.update(agent_metrics)
        metrics.update(
            {
                "run_id": run_id,
                "device": device,
                "xpu_memory_fraction": (
                    recipe.xpu_memory_fraction if device == "xpu" else None
                ),
                "elapsed_total_seconds": time.monotonic() - overall_start,
                "global_step": trainer.state.global_step,
                "time_budget_reached": bool(callback.time_budget_reached),
                "adaptive_sentinel_reached": adaptive_sentinel_reached,
                "fixed_step_target_reached": (
                    recipe.budget_mode != "steps"
                    or int(trainer.state.global_step) >= int(recipe.max_steps)
                ),
                "steps_shortfall": (
                    max(0, int(recipe.max_steps) - int(trainer.state.global_step))
                    if recipe.budget_mode == "steps"
                    else 0
                ),
                "noise": noise_controller.state() if noise_controller else None,
                "auxiliary_activity": auxiliary_activity,
                "auxiliary_failures": auxiliary_failures,
                "technique_status": "DEGRADED" if auxiliary_failures else "FULL",
            }
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
        final_status = (
            "stopped"
            if stopped
            else "checkpointed" if adaptive_sentinel_reached else "complete"
        )
        writer.write(
            (
                "run_stopped"
                if stopped
                else (
                    "adaptive_checkpoint_complete"
                    if adaptive_sentinel_reached
                    else "run_complete"
                )
            ),
            **metrics,
        )
        registry.update_run(run_id, final_status, metrics)
        return {"status": final_status, "output_dir": str(output_dir), **metrics}
    except Exception as error:
        writer.write("run_failed", error_type=type(error).__name__, error=str(error))
        registry.update_run(run_id, "failed", {"error": str(error)})
        raise
    finally:
        if noise_controller:
            noise_controller.detach()
