"""Metric aggregation and explicit training checkpoint roundtrips."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class MetricAggregator:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def update(self, **metrics: float) -> None:
        for name, value in metrics.items():
            self._values[name].append(float(value))

    def compute(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(self._values.items()):
            array = np.asarray(values, dtype=np.float64)
            output[name] = {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
                "count": int(array.size),
            }
        return output


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    **metadata: Any,
) -> None:
    payload: dict[str, Any] = {"model": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("metadata", {})
