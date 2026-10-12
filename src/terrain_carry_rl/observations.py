"""Observation packing and deployment history utilities."""

from collections.abc import Mapping

import torch

from terrain_carry_rl.schema import ObservationSchema


def pack_observation(values: Mapping[str, torch.Tensor], schema: ObservationSchema) -> torch.Tensor:
    """Pack named values in schema order and validate every field."""
    parts: list[torch.Tensor] = []
    for field in schema.fields:
        value = values[field.name]
        if value.shape[-1] != field.size:
            raise ValueError(f"{field.name}: expected {field.size}, got {value.shape[-1]}")
        parts.append(value)
    return torch.cat(parts, dim=-1)


class ObservationHistory:
    """Fixed-length oldest-to-newest history used by the deployable student."""

    def __init__(self, history_length: int, observation_dim: int) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.observation_dim = observation_dim
        self._buffer: torch.Tensor | None = None

    def reset(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate(observation)
        self._buffer = observation.unsqueeze(-2).repeat_interleave(self.history_length, dim=-2)
        return self._buffer.clone()

    def append(self, observation: torch.Tensor) -> torch.Tensor:
        self._validate(observation)
        if self._buffer is None:
            return self.reset(observation)
        self._buffer = torch.cat((self._buffer[..., 1:, :], observation.unsqueeze(-2)), dim=-2)
        return self._buffer.clone()

    def _validate(self, observation: torch.Tensor) -> None:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation dimension {self.observation_dim}, "
                f"got {observation.shape[-1]}"
            )
