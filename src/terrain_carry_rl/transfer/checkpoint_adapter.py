"""Shape-aware checkpoint transfer for evolving observations and action heads."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class TransferReport:
    loaded: list[str]
    missing: list[str]
    unexpected: list[str]
    shape_mismatch: dict[str, dict[str, list[int]]]
    partially_loaded: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_dict(checkpoint: str | Path | dict[str, Any]) -> dict[str, torch.Tensor]:
    payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, (str, Path))
        else checkpoint
    )
    candidate = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(candidate, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    return candidate


def load_partial_checkpoint(
    model: nn.Module,
    checkpoint: str | Path | dict[str, Any],
    *,
    copy_intersection: bool = True,
) -> TransferReport:
    """Load exact tensors and optionally copy overlapping slices for expanded layers."""
    source = _state_dict(checkpoint)
    target = model.state_dict()
    loaded: list[str] = []
    partial: list[str] = []
    mismatches: dict[str, dict[str, list[int]]] = {}

    for key in target.keys() & source.keys():
        source_tensor = source[key]
        target_tensor = target[key]
        if not isinstance(source_tensor, torch.Tensor):
            continue
        if source_tensor.shape == target_tensor.shape:
            target[key] = source_tensor.to(dtype=target_tensor.dtype)
            loaded.append(key)
        else:
            mismatches[key] = {
                "checkpoint": list(source_tensor.shape),
                "model": list(target_tensor.shape),
            }
            if copy_intersection and source_tensor.ndim == target_tensor.ndim:
                slices = tuple(
                    slice(0, min(old_size, new_size))
                    for old_size, new_size in zip(
                        source_tensor.shape, target_tensor.shape, strict=True
                    )
                )
                expanded = target_tensor.clone()
                expanded[slices] = source_tensor[slices].to(dtype=target_tensor.dtype)
                target[key] = expanded
                partial.append(key)

    model.load_state_dict(target)
    return TransferReport(
        loaded=sorted(loaded),
        missing=sorted(target.keys() - source.keys()),
        unexpected=sorted(source.keys() - target.keys()),
        shape_mismatch=dict(sorted(mismatches.items())),
        partially_loaded=sorted(partial),
    )
