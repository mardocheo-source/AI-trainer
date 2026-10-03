from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import Boolean3DConfig
from .token_roles import TokenRole


class Boolean3DProjector(nn.Module):
    """Projects hidden states from hidden dimension D into 3D canonical Boolean space [0, 1]^3."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 3, bias=True)
        # Initialize orthogonal weights for maximum initial dispersion
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.proj.weight.dtype:
            x = x.to(self.proj.weight.dtype)
        out = self.proj(x)
        return torch.sigmoid(out.float())


class Boolean3DLoss(nn.Module):
    """
    3D Geometric Boolean Latent Space v2.0 (3D-GBLS v2) Loss.
    
    Coordinates in 3D:
      u_1 (Action): 0 -> No-tool conversational refusal / chat; 1 -> Tool invocation.
      u_2 (Arity): 0 -> Single atomic call; 1 -> Multi-turn / parallel chained execution.
      u_3 (Syntax): 0 -> Free-form natural language dialogue; 1 -> Strict typed AST / Java / SQL.
    
    Higher-Order Continuous Boolean Operators:
      AND(a, b) = a * b
      OR(a, b) = a + b - a * b
      NOT(a) = 1.0 - a
      XOR(a, b) = a + b - 2.0 * a * b
      IMPLY(a, b) = 1.0 - a + a * b
    """

    def __init__(self, config: Boolean3DConfig) -> None:
        super().__init__()
        self.config = config
        self.dim_weights = nn.Parameter(
            torch.tensor(config.dimension_weights, dtype=torch.float32),
            requires_grad=False,
        )

    def resolve_target_coordinate(
        self,
        token_role: int,
        is_tool_query: bool,
        is_multiturn_or_parallel: bool,
        is_typed_syntax: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Derive the 3D continuous Boolean target point u* for a token."""
        action_val = 1.0 if is_tool_query else 0.0
        arity_val = 1.0 if is_multiturn_or_parallel else 0.0
        syntax_val = 1.0 if is_typed_syntax else 0.0

        if not is_tool_query:
            # 1. Pure Chatable / No-Tool Refusal: strictly pinned to (0, 0, 0)
            return torch.tensor([0.0, 0.0, 0.0], device=device)

        if token_role == int(TokenRole.ORDINARY):
            # Ordinary text inside a tool trajectory: intermediate thought or clarification
            if is_multiturn_or_parallel:
                # Clarification / intermediate conversational thought
                return torch.tensor([0.05, 0.85, 0.10], device=device)
            return torch.tensor([0.0, 0.0, 0.0], device=device)

        elif token_role == int(TokenRole.TOOL_NAME):
            # Tool name: high action anchor with multi-tool arity separation
            return torch.tensor([1.0, arity_val, 1.0], device=device)

        elif token_role in {int(TokenRole.ARGUMENT_KEY), int(TokenRole.ARGUMENT_VALUE)}:
            # Arguments: high syntax strictness
            return torch.tensor([1.0, arity_val, 1.0 if syntax_val > 0.5 else 0.88], device=device)

        elif token_role in {
            int(TokenRole.DELIMITER),
            int(TokenRole.CLOSING_DELIMITER),
        }:
            # Structural delimiters (<|tool_call_start|>, braces, brackets)
            return torch.tensor([1.0, arity_val, 0.92], device=device)

        return torch.tensor([action_val, arity_val, syntax_val], device=device)

    def forward(
        self,
        projector: Boolean3DProjector,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        roles: torch.Tensor | None,
        is_tool_query: Any | None = None,
        is_parallel: Any | None = None,
        is_multiturn: Any | None = None,
        is_typed: Any | None = None,
    ) -> torch.Tensor:
        """
        Computes 3D Boolean attractor loss + volumetric polytope repulsion + radial refusal anchor.
        """
        if not self.config.enabled or roles is None or labels is None:
            return hidden_states.new_zeros(())

        shifted_labels = labels[..., 1:]
        shifted_roles = roles[..., 1:]
        valid_mask = shifted_labels.ne(-100)

        if not valid_mask.any():
            return hidden_states.new_zeros(())

        selected_positions = valid_mask.nonzero(as_tuple=False)
        selected_hidden = hidden_states[..., :-1, :][valid_mask]
        selected_roles = shifted_roles[valid_mask]

        # Resolve Action independently for every conversation in the batch.
        # The previous caller always passed ``True`` and therefore made the
        # no-tool/refusal vertex and radial anchor unreachable.
        row_has_tool = (
            shifted_roles.ge(int(TokenRole.TOOL_NAME)) & valid_mask
        ).any(dim=-1)
        def row_flags(value: Any | None, fallback: torch.Tensor) -> torch.Tensor:
            if value is None:
                return fallback
            if torch.is_tensor(value):
                flat = value.to(device=fallback.device, dtype=torch.bool).reshape(-1)
                if flat.numel() != fallback.numel():
                    raise ValueError(
                        "Boolean3D capability target batch does not match hidden-state batch"
                    )
                return flat
            return torch.full_like(fallback, bool(value))

        row_has_tool = row_flags(is_tool_query, row_has_tool)
        row_is_parallel = row_flags(is_parallel, torch.zeros_like(row_has_tool))
        row_is_multiturn = row_flags(
            is_multiturn, torch.zeros_like(row_has_tool)
        ) | row_is_parallel
        row_is_typed = row_flags(is_typed, row_has_tool)

        if selected_hidden.numel() == 0:
            return hidden_states.new_zeros(())

        if selected_hidden.shape[0] > self.config.sample_tokens:
            indices = torch.linspace(
                0, selected_hidden.shape[0] - 1, steps=self.config.sample_tokens, device=selected_hidden.device
            ).long()
            selected_hidden = selected_hidden.index_select(0, indices)
            selected_roles = selected_roles.index_select(0, indices)
            selected_positions = selected_positions.index_select(0, indices)

        coords_3d = projector(selected_hidden)  # [N, 3]

        # 1. Attractor Loss: pull tokens toward their continuous Boolean target vertices
        target_coords = []
        selected_rows = selected_positions[:, 0]
        for r, row_index in zip(selected_roles, selected_rows, strict=True):
            target_coords.append(
                self.resolve_target_coordinate(
                    int(r.item()),
                    bool(row_has_tool[row_index].item()),
                    bool(row_is_multiturn[row_index].item()),
                    bool(row_is_typed[row_index].item()),
                    device=coords_3d.device,
                )
            )
        targets_tensor = torch.stack(target_coords, dim=0)  # [N, 3]

        dim_weights = self.dim_weights.to(coords_3d.device)
        diff_sq = ((coords_3d - targets_tensor) ** 2) * dim_weights.unsqueeze(0)
        attractor_loss = diff_sq.sum(dim=-1).mean()

        # 2. Polytope Margin Repulsion between Tool and Non-Tool Tokens
        polytope_loss = coords_3d.new_zeros(())
        if coords_3d.shape[0] >= 4:
            tool_mask = selected_roles.ge(int(TokenRole.TOOL_NAME))
            non_tool_mask = ~tool_mask
            if tool_mask.any() and non_tool_mask.any():
                tool_coords = coords_3d[tool_mask]
                non_tool_coords = coords_3d[non_tool_mask]
                dists = torch.cdist(tool_coords, non_tool_coords, p=2)
                margin_violations = F.relu(self.config.polytope_margin - dists)
                polytope_loss = (margin_violations ** 2).mean()

        # 3. Radial Refusal Anchor for Chatable / No-Tool Queries
        refusal_loss = coords_3d.new_zeros(())
        selected_is_tool = row_has_tool.index_select(0, selected_rows)
        refusal_coords = coords_3d[~selected_is_tool]
        if refusal_coords.numel():
            action_overflow = F.relu(
                refusal_coords[:, 0] - self.config.refusal_action_threshold
            )
            refusal_loss = (action_overflow ** 2).mean()

        # Optional asymmetric row-level barrier in the bounded 3D cube.
        asymmetric_barrier = coords_3d.new_zeros(())
        if self.config.call_margin > 0 or self.config.refuse_margin > 0:
            call_coords = coords_3d[selected_is_tool]
            if call_coords.numel() and self.config.call_margin > 0:
                call_distance = torch.linalg.vector_norm(call_coords, dim=-1)
                asymmetric_barrier = asymmetric_barrier + (
                    F.relu(self.config.call_margin - call_distance) ** 2
                ).mean()
            if refusal_coords.numel() and self.config.refuse_margin > 0:
                action_vertex = torch.ones(
                    3, device=refusal_coords.device, dtype=refusal_coords.dtype
                )
                refusal_distance = torch.linalg.vector_norm(
                    refusal_coords - action_vertex, dim=-1
                )
                asymmetric_barrier = asymmetric_barrier + float(
                    self.config.asymmetric_refusal_multiplier
                ) * (F.relu(self.config.refuse_margin - refusal_distance) ** 2).mean()

        total_loss = (
            attractor_loss
            + self.config.polytope_weight * polytope_loss
            + self.config.refusal_weight * refusal_loss
            + asymmetric_barrier
        )
        return self.config.weight * total_loss.to(hidden_states.dtype)
