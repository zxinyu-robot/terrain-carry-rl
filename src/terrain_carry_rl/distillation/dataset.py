"""Deterministic synthetic samples used only for tests and smoke runs."""

import torch
from torch.utils.data import Dataset

from terrain_carry_rl.schema import DEPLOYABLE_OBSERVATION, PRIVILEGED_OBSERVATION


class SyntheticDistillationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        size: int = 128,
        history_length: int = 4,
        seed: int = 0,
        deployable_dim: int = DEPLOYABLE_OBSERVATION.dim,
        privileged_dim: int = PRIVILEGED_OBSERVATION.dim,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.privileged = torch.randn(size, privileged_dim, generator=generator)
        current = self.privileged[:, :deployable_dim]
        history_noise = 0.03 * torch.randn(
            size, history_length, deployable_dim, generator=generator
        )
        self.history = current[:, None, :].expand(-1, history_length, -1) + history_noise

    def __len__(self) -> int:
        return self.privileged.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"privileged": self.privileged[index], "history": self.history[index]}
