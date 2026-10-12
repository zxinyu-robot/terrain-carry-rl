"""Deterministic Gaussian observation corruption."""

import torch


class ObservationNoise:
    def __init__(self, standard_deviation: float | torch.Tensor, seed: int = 0) -> None:
        self.standard_deviation = standard_deviation
        self.generator = torch.Generator().manual_seed(seed)

    def apply(self, observation: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            observation.shape,
            generator=self.generator,
            device=observation.device,
            dtype=observation.dtype,
        )
        scale = torch.as_tensor(
            self.standard_deviation, device=observation.device, dtype=observation.dtype
        )
        return observation + noise * scale
