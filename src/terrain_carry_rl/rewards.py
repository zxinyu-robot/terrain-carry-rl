"""Simulator-agnostic reward terms for downstream task integration."""

import torch


def tracking_reward(
    actual: torch.Tensor, target: torch.Tensor, sigma: float = 0.25
) -> torch.Tensor:
    error = (actual - target).square().sum(dim=-1)
    return torch.exp(-error / sigma)


def carry_stability_reward(
    object_tilt: torch.Tensor, object_acceleration: torch.Tensor
) -> torch.Tensor:
    return torch.exp(-object_tilt.square() - 0.1 * object_acceleration.square().sum(dim=-1))


def action_rate_penalty(action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    return (action - previous_action).square().sum(dim=-1)
