from __future__ import annotations

import math
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .schema import NoiseConfig

# A finite, reviewed window is intentionally cycled. The pointer is included in run metrics.
DIGITS = {
    "pi": "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679",
    "e": "7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274",
    "sqrt2": "4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727",
    "phi": "6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374",
}


@dataclass
class DigitSchedule:
    source: str
    base_alpha: float
    modulation: float
    mode: str = "digit_pairs"
    order: str = "natural"
    seed: int = 42
    index: int = 0
    last_pair: int | None = None

    def _units(self) -> list[tuple[int, float]]:
        digits = DIGITS[self.source]
        if self.mode == "single_digit":
            units = [(int(d), (int(d) - 4.5) / 4.5) for d in digits]
        elif self.mode == "triplets":
            triplets = [int(digits[i : i + 3]) for i in range(0, len(digits) - 2, 3)]
            units = [(t, (t - 499.5) / 499.5) for t in triplets]
        elif self.mode == "diff":
            diffs = [abs(int(digits[i]) - int(digits[i - 1])) for i in range(1, len(digits))]
            units = [(df, (df - 4.5) / 4.5) for df in diffs]
        else:  # default "digit_pairs"
            pairs = [int(digits[i : i + 2]) for i in range(0, len(digits), 2)]
            units = [(p, (p - 49.5) / 49.5) for p in pairs]

        if self.order == "shuffled":
            random.Random(self.seed).shuffle(units)
        return units

    def next_alpha(self) -> float:
        units = self._units()
        val, centered = units[self.index % len(units)]
        self.last_pair = val
        self.index = (self.index + 1) % len(units)
        return max(0.0, self.base_alpha * (1.0 + self.modulation * centered))


class EmbeddingNoiseController:
    """π/e/√2/φ digit-modulated embedding noise with isolated RNG state.

    The decimal granularity is the primary amplitude modulation. An optional
    envelope controls how that digit signal is phased out near the end of
    training.
    """

    def __init__(self, config: NoiseConfig, seed: int = 42, tokenizer: Any | None = None) -> None:
        self.config = config
        self.schedule = (
            DigitSchedule(
                config.source,
                config.alpha,
                config.modulation,
                mode=config.amplitude_mode,
                order=config.digit_order,
                seed=int(seed) + int(config.seed_offset),
            )
            if config.amplitude_mode in {"single_digit", "digit_pairs", "triplets", "diff"}
            else None
        )
        self.alpha = 0.0
        self.digit_alpha = config.alpha
        self.progress = 0.0
        self.handle: Any | None = None
        self._labels: Any | None = None
        self._attention_mask: Any | None = None
        self._literal_mask: Any | None = None
        self._tokenizer = tokenizer
        self._suspend_depth = 0
        self._seed = int(seed) + int(config.seed_offset)
        self._generators: dict[str, Any] = {}
        self.advance(0.0)

    def _envelope(self, progress: float) -> float:
        active_fraction = 1.0 - self.config.clean_tail_fraction
        if progress >= active_fraction:
            return 0.0
        active_progress = min(1.0, max(0.0, progress / max(active_fraction, 1e-9)))
        if self.config.envelope == "linear":
            return 1.0 - active_progress
        if self.config.envelope == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * active_progress))
        return 1.0

    def advance(self, progress: float | None = None) -> None:
        if progress is not None:
            self.progress = min(1.0, max(0.0, float(progress)))
        self.digit_alpha = (
            self.schedule.next_alpha() if self.schedule is not None else self.config.alpha
        )
        self.alpha = self.digit_alpha * self._envelope(self.progress)

    def begin_batch(
        self,
        labels: Any | None,
        attention_mask: Any | None = None,
        input_ids: Any | None = None,
    ) -> None:
        self._labels = labels
        self._attention_mask = attention_mask
        self._literal_mask = None
        if self.config.protect_prompt_literals and self._tokenizer is not None and input_ids is not None:
            import torch

            from .token_roles import batch_prompt_literal_mask

            self._literal_mask = batch_prompt_literal_mask(self._tokenizer, input_ids, torch)

    def end_batch(self) -> None:
        self._labels = None
        self._attention_mask = None
        self._literal_mask = None

    @contextmanager
    def suspended(self) -> Iterator[None]:
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1

    def _generator(self, torch: Any, device: Any) -> Any:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._seed)
            self._generators[key] = generator
        return generator

    def attach(self, model: Any) -> None:
        embedding = model.get_input_embeddings()
        if embedding is None:
            raise ValueError("model does not expose input embeddings")

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if (
                not getattr(_module, "training", False)
                or self.alpha <= 0
                or self._suspend_depth
            ):
                return output
            import torch

            if not isinstance(output, torch.Tensor) or output.ndim < 2:
                return output
            sequence = output.shape[-2]
            width = output.shape[-1]
            magnitude = self.alpha / math.sqrt(max(1, sequence * width))
            noise = torch.empty_like(output).uniform_(
                -1.0,
                1.0,
                generator=self._generator(torch, output.device),
            )
            if self.config.scope == "prompt":
                labels = self._labels
                if labels is None or output.ndim != 3 or labels.shape != output.shape[:2]:
                    return output
                mask = labels.eq(-100)
                if self._attention_mask is not None and self._attention_mask.shape == labels.shape:
                    mask = mask & self._attention_mask.bool()
                if self._literal_mask is not None and self._literal_mask.shape == labels.shape:
                    mask = mask & ~self._literal_mask.bool()
                noise = noise * mask.unsqueeze(-1).to(noise.dtype)
            return output + noise * magnitude

        self.handle = embedding.register_forward_hook(hook)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def state(self) -> dict[str, Any]:
        return {
            "amplitude_mode": self.config.amplitude_mode,
            "source": self.config.source,
            "digit_order": self.config.digit_order,
            "alpha": self.alpha,
            "digit_alpha": self.digit_alpha,
            "digit_pair": self.schedule.last_pair if self.schedule is not None else None,
            "digit_index": self.schedule.index if self.schedule is not None else None,
            "scope": self.config.scope,
            "envelope": self.config.envelope,
            "clean_tail_fraction": self.config.clean_tail_fraction,
            "protect_prompt_literals": self.config.protect_prompt_literals,
            "value_consistency_weight": self.config.value_consistency_weight,
            "progress": self.progress,
            "rng_seed": self._seed,
        }
