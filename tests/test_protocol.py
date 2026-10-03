from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from exotic_trainer.agent_eval import _elastic_values_equal
from exotic_trainer.decimal_control_data import (
    audit_decimal_control_rows,
    build_decimal_control_rows,
)
from exotic_trainer.gui import _protocol_signature
from exotic_trainer.human_agentic_focus_data import audit_focused_rows, build_focused_rows
from exotic_trainer.noise import EmbeddingNoiseController
from exotic_trainer.schema import GeometryConfig, NoiseConfig, RoleLossConfig, TrainingRecipe
from exotic_trainer.trainer import _regularized_trainer_class


def test_baseline_variant_rejects_exotic_controls() -> None:
    with pytest.raises(ValidationError, match="baseline variants"):
        TrainingRecipe(
            model="model",
            datasets=["dataset"],
            experiment_variant="baseline",
            noise=NoiseConfig(enabled=True),
        )


def test_protocol_id_changes_only_when_a_control_changes() -> None:
    baseline = TrainingRecipe(
        model="model",
        datasets=["train"],
        validation_mode="external",
        validation_datasets=["validation"],
        experiment_variant="baseline",
    )
    exotic = baseline.model_copy(
        update={
            "experiment_variant": "exotic",
            "noise": NoiseConfig(enabled=True),
        }
    )
    changed_adapter = exotic.model_copy(update={"lora_rank": 32})

    assert _protocol_signature(baseline.model_dump()) == _protocol_signature(exotic.model_dump())
    assert _protocol_signature(exotic.model_dump()) != _protocol_signature(
        changed_adapter.model_dump()
    )


def test_relational_geometry_uses_previous_layer_as_reference() -> None:
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    class Model:
        training = True

        def __call__(self, **_inputs):
            prior = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
            current = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]])
            return SimpleNamespace(loss=torch.tensor(1.0), hidden_states=[prior, current])

    class Writer:
        def write(self, *_args, **_kwargs):
            pass

    trainer_type = _regularized_trainer_class(
        BaseTrainer,
        torch,
        tokenizer=None,
        geometry=GeometryConfig(enabled=True, mode="relational", weight=0.2),
        role_loss=RoleLossConfig(),
        noise=None,
        writer=Writer(),
        logging_steps=5,
    )
    trainer = trainer_type()
    trainer.state = SimpleNamespace(global_step=1)

    loss = trainer.compute_loss(Model(), {"labels": torch.tensor([[0, 1, 2, 3]])})

    assert loss.item() > 1.0


def test_elastic_arguments_accept_only_semantic_representation_changes() -> None:
    assert _elastic_values_equal("timeout", "30", 30)
    assert _elastic_values_equal("path", "src/./pkg/../main.py", "src/main.py")
    assert _elastic_values_equal("content", "hello\r\n", "hello")
    assert _elastic_values_equal(
        "command",
        "python -m tool 'two words'",
        'python -m tool "two words"',
    )
    assert not _elastic_values_equal("query", "near match", "exact match")
    assert not _elastic_values_equal("path", "src/other.py", "src/main.py")
    assert not _elastic_values_equal("command", "pytest tests/a.py", "pytest tests/b.py")


def test_focused_final_suite_is_balanced_uniform_and_unseen() -> None:
    report = audit_focused_rows(build_focused_rows())

    assert report["status"] == "ready"
    assert report["rows"] == 520
    assert report["decisions"] == {"tool": 260, "direct": 260}
    assert set(report["tool_distribution"].values()) == {20}
    assert all(value == 0 for value in report["exact_overlap"].values())
    assert all(value == 0 for value in report["prompt_skeleton_overlap"].values())


def test_constant_embedding_noise_does_not_consume_decimal_pairs() -> None:
    controller = EmbeddingNoiseController(
        NoiseConfig(
            enabled=True,
            amplitude_mode="constant",
            alpha=2.0,
            modulation=0.9,
            clean_tail_fraction=0.3,
        )
    )

    controller.advance(0.25)
    first = controller.state()
    controller.advance(0.50)
    second = controller.state()

    assert first["alpha"] == 2.0
    assert second["alpha"] == 2.0
    assert first["digit_pair"] is None
    assert second["digit_index"] is None


def test_decimal_control_final_is_balanced_uniform_and_fresh() -> None:
    report = audit_decimal_control_rows(build_decimal_control_rows())

    assert report["status"] == "ready"
    assert report["rows"] == 520
    assert report["decisions"] == {"tool": 260, "direct": 260}
    assert set(report["tool_distribution"].values()) == {20}
    assert all(value == 0 for value in report["exact_overlap"].values())
    assert all(value == 0 for value in report["prompt_skeleton_overlap"].values())
