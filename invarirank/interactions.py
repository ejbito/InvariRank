from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class InteractionResult:
    correction: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class RelationalCandidateInteraction(nn.Module):
    """Permutation-equivariant relational correction over invariant candidates."""

    def __init__(
        self,
        input_dim: int,
        *,
        interaction_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        score_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or interaction_dim < 1 or hidden_dim < 1:
            raise ValueError("Interaction dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if score_scale < 0:
            raise ValueError("score_scale must be non-negative.")

        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, interaction_dim)
        self.relation_features = _MLP(4 * interaction_dim + 1, hidden_dim, interaction_dim, dropout)
        self.relation_comparator = nn.Linear(interaction_dim, 1)
        self.relation_gate = _MLP(4 * interaction_dim, hidden_dim, 1, dropout)
        self.score_scale = nn.Parameter(torch.tensor(float(score_scale)))

    def forward(self, representations: torch.Tensor, base_scores: torch.Tensor) -> InteractionResult:
        if representations.ndim != 2:
            raise ValueError("Candidate representations must have shape [candidates, hidden_size].")
        if base_scores.ndim != 1 or base_scores.shape[0] != representations.shape[0]:
            raise ValueError("base_scores must contain one score per candidate representation.")
        if representations.shape[0] < 1:
            raise ValueError("Candidate interaction requires at least one candidate.")

        projected = self.projection(self.input_norm(representations))
        return self._relational(projected, base_scores)

    def _relational(self, values: torch.Tensor, base_scores: torch.Tensor) -> InteractionResult:
        count, width = values.shape
        if count == 1:
            zeros = values.new_zeros((1, 1))
            return InteractionResult(
                correction=base_scores.new_zeros(1),
                diagnostics={"relational_advantages": zeros, "relational_gates": zeros},
            )

        first = values[:, None, :].expand(count, count, width)
        second = values[None, :, :].expand(count, count, width)
        # Base log-probabilities are accumulated in float32, while this
        # network normally follows the backbone dtype (for example bfloat16).
        # Cast only the feature copy so the MLP input matches its parameters.
        score_difference = (base_scores[:, None] - base_scores[None, :]).to(values.dtype)
        relation_input = torch.cat(
            (first, second, first - second, first * second, score_difference.unsqueeze(-1)),
            dim=-1,
        )
        pair_scores = self.relation_comparator(self.relation_features(relation_input)).squeeze(-1)
        advantages = pair_scores - pair_scores.transpose(0, 1)

        context = values.mean(dim=0).view(1, 1, width).expand(count, count, width)
        gate_input = torch.cat((first + second, torch.abs(first - second), first * second, context), dim=-1)
        gate_logits = self.relation_gate(gate_input).squeeze(-1)
        # Symmetrize after the MLP so stochastic dropout cannot make the
        # (i, j) and (j, i) gates disagree during training.
        gate_logits = 0.5 * (gate_logits + gate_logits.transpose(0, 1))
        gates = torch.sigmoid(gate_logits)
        diagonal = torch.eye(count, dtype=torch.bool, device=values.device)
        gates = gates.masked_fill(diagonal, 0.0)
        advantages = advantages.masked_fill(diagonal, 0.0)
        correction = self.score_scale * torch.sum(gates * advantages, dim=-1) / float(count - 1)
        return InteractionResult(
            correction=correction,
            diagnostics={"relational_advantages": advantages, "relational_gates": gates},
        )


def build_candidate_interaction(config: Any, input_dim: int) -> RelationalCandidateInteraction | None:
    mode = str(getattr(config, "interaction", "none"))
    if mode == "none":
        return None
    if mode != "relational":
        raise ValueError(f"Unsupported candidate interaction mode: {mode}")
    return RelationalCandidateInteraction(
        input_dim,
        interaction_dim=int(getattr(config, "interaction_dim", 256)),
        hidden_dim=int(getattr(config, "interaction_hidden_dim", 512)),
        dropout=float(getattr(config, "interaction_dropout", 0.1)),
        score_scale=float(getattr(config, "interaction_score_scale", 0.1)),
    )


__all__ = ["InteractionResult", "RelationalCandidateInteraction", "build_candidate_interaction"]
