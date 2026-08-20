"""Conflict-aware router architectures (Stage 3 / M4).

Both routers map disagreement features phi(x) to a *correction* over teacher
log-weights.  Final mixture weights are ``softmax(correction(phi) + log_prior)``,
where ``log_prior`` is the fixed entropy-weighted baseline (see
``entropy_prior_log_weights`` in ``train_router.py``).  Final layers are
zero-initialised so an untrained router reproduces the entropy prior exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_HIDDEN_DIM = 16


def _zero_init_final_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


class LinearRouter(nn.Module):
    """Single linear layer mapping phi(x) to teacher correction logits."""

    def __init__(self, feature_dim: int, num_teachers: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_teachers)
        _zero_init_final_linear(self.linear)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.linear(phi)


class MLPRouter(nn.Module):
    """Small MLP mapping phi(x) to teacher correction logits."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_teachers: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_teachers)
        _zero_init_final_linear(self.fc2)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(phi)))
        return self.fc2(x)
