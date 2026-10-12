"""Privileged teacher policy for simulation training."""

import torch
from torch import nn

from terrain_carry_rl.actions import ACTION_DIM
from terrain_carry_rl.models.encoders import MLPEncoder
from terrain_carry_rl.schema import PRIVILEGED_OBSERVATION


class TeacherPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int = PRIVILEGED_OBSERVATION.dim,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (128, 128),
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = MLPEncoder(observation_dim, hidden_dims, feature_dim)
        self.action_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self, observation: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        action = torch.tanh(self.action_head(features))
        if return_features:
            return action, features
        return action
