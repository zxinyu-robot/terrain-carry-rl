"""Small dependency-light policy encoders."""

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], feature_dim: int) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, feature_dim)
        layers: list[nn.Module] = []
        for input_size, output_size in zip(dims[:-1], dims[1:], strict=True):
            layers.extend((nn.Linear(input_size, output_size), nn.ELU()))
        self.network = nn.Sequential(*layers)
        self.output_dim = feature_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class HistoryEncoder(nn.Module):
    """Encodes deployment history without simulator-only recurrent state."""

    def __init__(
        self,
        observation_dim: int,
        history_length: int,
        hidden_dims: tuple[int, ...],
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.history_length = history_length
        self.encoder = MLPEncoder(observation_dim * history_length, hidden_dims, feature_dim)
        self.output_dim = feature_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        expected = (self.history_length, self.observation_dim)
        if history.shape[-2:] != expected:
            raise ValueError(f"expected history shape (..., {expected[0]}, {expected[1]})")
        return self.encoder(history.flatten(start_dim=-2))
