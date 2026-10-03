from __future__ import annotations

import torch
from torch import nn

from exotic_trainer.agent_eval import _prompt_has_embedded_tool_menu
from exotic_trainer.boolean_3d import Boolean3DLoss
from exotic_trainer.schema import Boolean3DConfig, TrainingRecipe
from exotic_trainer.token_roles import TokenRole, role_spans
from exotic_trainer.trainer import _state_binding_token_pairs, _turn_credit_scale


class _ConstantProjector(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_full((hidden.shape[0], 3), 0.9, dtype=torch.float32)


class _CharacterTokenizer:
    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _mixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.zeros((2, 5, 4), dtype=torch.float32)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    roles = torch.full((2, 5), int(TokenRole.ORDINARY), dtype=torch.long)
    labels[:, 1:] = 1
    roles[1, 2] = int(TokenRole.TOOL_NAME)
    roles[1, 3] = int(TokenRole.ARGUMENT_KEY)
    roles[1, 4] = int(TokenRole.ARGUMENT_VALUE)
    return hidden, labels, roles


def test_boolean_3d_infers_no_tool_rows_and_applies_refusal_anchor() -> None:
    config = Boolean3DConfig(
        enabled=True,
        weight=1.0,
        polytope_weight=0.0,
        refusal_weight=1.0,
        refusal_action_threshold=0.1,
        sample_tokens=32,
    )
    loss_fn = Boolean3DLoss(config)
    hidden, labels, roles = _mixed_batch()

    inferred = loss_fn(_ConstantProjector(), hidden, labels, roles, is_tool_query=None)
    forced_tool = loss_fn(_ConstantProjector(), hidden, labels, roles, is_tool_query=True)

    assert inferred > forced_tool


def test_boolean_3d_refusal_weight_is_not_an_arity_dimension_alias() -> None:
    hidden, labels, roles = _mixed_batch()
    projector = _ConstantProjector()
    no_refusal = Boolean3DLoss(
        Boolean3DConfig(
            enabled=True,
            weight=1.0,
            polytope_weight=0.0,
            refusal_weight=0.0,
            dimension_weights=(1.0, 1.0, 1.0),
            sample_tokens=32,
        )
    )(projector, hidden, labels, roles, is_tool_query=None)
    with_refusal = Boolean3DLoss(
        Boolean3DConfig(
            enabled=True,
            weight=1.0,
            polytope_weight=0.0,
            refusal_weight=0.9,
            dimension_weights=(1.0, 1.0, 1.0),
            sample_tokens=32,
        )
    )(projector, hidden, labels, roles, is_tool_query=None)

    assert with_refusal > no_refusal


def test_boolean_3d_accepts_explicit_per_row_capability_targets() -> None:
    config = Boolean3DConfig(enabled=True, weight=1.0, sample_tokens=32)
    loss_fn = Boolean3DLoss(config)
    hidden, labels, roles = _mixed_batch()

    value = loss_fn(
        _ConstantProjector(),
        hidden,
        labels,
        roles,
        is_tool_query=torch.tensor([0, 1]),
        is_parallel=torch.tensor([0, 1]),
        is_multiturn=torch.tensor([0, 0]),
        is_typed=torch.tensor([0, 1]),
    )

    assert torch.isfinite(value)


def test_state_binding_is_opt_in_for_new_recipes() -> None:
    recipe = TrainingRecipe(model="model", datasets=["dataset"])
    assert recipe.state_binding_weight == 0.0


def test_turn_credit_uses_actual_turn_depth() -> None:
    scale = _turn_credit_scale(
        torch.tensor([0, 1, 0]),
        torch.tensor([2, 2, 1]),
        0.7,
        torch,
    )

    assert torch.allclose(scale, torch.tensor([1.0, 1.7, 1.0]))


def test_state_binding_finds_prompt_producer_and_completion_consumer() -> None:
    prompt = "observed_state = inspect_meter()\n"
    completion = "plan = build_plan(state=observed_state)\nsave(plan=plan)"
    text = prompt + completion
    input_ids = torch.tensor([ord(char) for char in text], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[len(prompt) :] = input_ids[len(prompt) :]
    roles = torch.full_like(input_ids, int(TokenRole.ORDINARY))
    assignment_start = text.index("plan =", len(prompt))
    observed_start = text.index("observed_state", len(prompt))
    consumer_start = text.rindex("plan")
    roles[assignment_start : assignment_start + len("plan")] = int(TokenRole.ARGUMENT_VALUE)
    roles[observed_start : observed_start + len("observed_state")] = int(TokenRole.ARGUMENT_VALUE)
    roles[consumer_start : consumer_start + len("plan")] = int(TokenRole.ARGUMENT_VALUE)

    local_pairs, cross_pairs = _state_binding_token_pairs(
        _CharacterTokenizer(), input_ids, labels, roles
    )

    assert len(local_pairs) == 1
    assert len(cross_pairs) == 1
    cross = cross_pairs[0]
    assert text[cross[0] : cross[1]] == "observed_state"
    assert text[cross[2] : cross[3]] == "observed_state"


def test_python_roles_ignore_trailing_chat_template_token() -> None:
    text = "result = tools.lookup(state=observed_state)<|im_end|>\n"
    observed = {
        text[start:end]: role
        for start, end, role in role_spans(text)
    }

    assert observed["tools.lookup"] == TokenRole.TOOL_NAME
    assert observed["observed_state"] == TokenRole.ARGUMENT_VALUE


def test_embedded_tool_menu_is_detected_before_generation() -> None:
    assert _prompt_has_embedded_tool_menu(
        [
            {"role": "system", "content": "Policy\nList of tools: []"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert not _prompt_has_embedded_tool_menu(
        [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "hello"},
        ]
    )
