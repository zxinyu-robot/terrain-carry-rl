"""Canonical tightly coupled whole-body action contract."""

from dataclasses import dataclass

import torch

LEG_ACTION_DIM = 12
ARM_ACTION_DIM = 6
GRIPPER_ACTION_DIM = 1
ACTION_DIM = LEG_ACTION_DIM + ARM_ACTION_DIM + GRIPPER_ACTION_DIM


@dataclass(frozen=True)
class WholeBodyAction:
    leg: torch.Tensor
    arm: torch.Tensor
    gripper: torch.Tensor


def split_action(action: torch.Tensor) -> WholeBodyAction:
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} actions, got {action.shape[-1]}")
    return WholeBodyAction(action[..., :12], action[..., 12:18], action[..., 18:19])


def scale_action(
    normalized: torch.Tensor,
    leg_scale: float = 0.5,
    arm_scale: float = 0.35,
    gripper_scale: float = 1.0,
) -> torch.Tensor:
    """Clamp normalized policy output and apply group-specific command scales."""
    action = normalized.clamp(-1.0, 1.0)
    scales = action.new_tensor([leg_scale] * 12 + [arm_scale] * 6 + [gripper_scale])
    return action * scales
