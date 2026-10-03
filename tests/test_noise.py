from exotic_trainer.noise import DigitSchedule
from exotic_trainer.noise import EmbeddingNoiseController
from exotic_trainer.schema import NoiseConfig
from exotic_trainer.token_roles import literal_mask_for_token_ids

import torch


def test_digit_schedule_cycles_and_is_deterministic() -> None:
    first = DigitSchedule("pi", 5.0, 0.35)
    second = DigitSchedule("pi", 5.0, 0.35)
    assert [first.next_alpha() for _ in range(60)] == [second.next_alpha() for _ in range(60)]
    # The reviewed source window contains 100 decimal digits = 50 pairs.
    # Sixty optimizer advances therefore wrap once and stop at pair index 10.
    assert first.index == 10


def test_pi_pair_remains_primary_modulation_and_clean_tail_turns_it_off() -> None:
    controller = EmbeddingNoiseController(
        NoiseConfig(
            enabled=True,
            source="pi",
            alpha=2.0,
            modulation=0.1,
            clean_tail_fraction=0.3,
        )
    )
    expected = 2.0 * (1.0 + 0.1 * ((14 - 49.5) / 49.5))
    assert controller.state()["digit_pair"] == 14
    assert controller.alpha == expected

    controller.advance(0.7)

    assert controller.alpha == 0.0


def test_shuffled_pi_uses_same_decimal_pairs_in_a_different_order() -> None:
    natural = DigitSchedule("pi", 2.0, 0.1, order="natural", seed=9)
    shuffled = DigitSchedule("pi", 2.0, 0.1, order="shuffled", seed=9)
    natural_values = [natural.next_alpha() for _ in range(50)]
    shuffled_values = [shuffled.next_alpha() for _ in range(50)]

    assert sorted(natural_values) == sorted(shuffled_values)
    assert natural_values != shuffled_values


def test_prompt_only_noise_uses_isolated_rng_and_preserves_completion_embeddings() -> None:
    embedding = torch.nn.Embedding(8, 4)

    class Model:
        def get_input_embeddings(self):
            return embedding

    controller = EmbeddingNoiseController(
        NoiseConfig(enabled=True, alpha=2.0, scope="prompt"), seed=123
    )
    controller.attach(Model())
    embedding.train()
    ids = torch.tensor([[1, 2]])
    clean = embedding.weight.detach().index_select(0, ids.flatten()).reshape(1, 2, 4)
    labels = torch.tensor([[-100, 2]])
    controller.begin_batch(labels, torch.ones_like(labels))

    torch.manual_seed(77)
    expected_global = torch.rand(3)
    torch.manual_seed(77)
    noised = embedding(ids)
    actual_global = torch.rand(3)
    controller.end_batch()
    controller.detach()

    assert not torch.equal(noised[:, 0], clean[:, 0])
    assert torch.equal(noised[:, 1], clean[:, 1])
    assert torch.equal(actual_global, expected_global)


class CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(value) for value in token_ids)


def test_literal_lock_marks_inline_and_fenced_prompt_values() -> None:
    text = "Use `read` with:\n```json\n{\"path\": \"x.py\"}\n``` now"
    mask = literal_mask_for_token_ids(CharacterTokenizer(), [ord(char) for char in text])

    assert mask[text.index("read")] is True
    assert mask[text.index("path")] is True
    assert mask[text.index("Use")] is False
