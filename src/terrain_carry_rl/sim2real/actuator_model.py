"""First-order actuator lag approximation for command-side testing."""

import torch


class ActuatorLag:
    def __init__(self, time_constant: float, time_step: float, motor_strength: float = 1.0) -> None:
        if time_constant < 0 or time_step <= 0:
            raise ValueError("time_constant must be non-negative and time_step positive")
        self.alpha = 1.0 if time_constant == 0 else time_step / (time_constant + time_step)
        self.motor_strength = motor_strength
        self.state: torch.Tensor | None = None

    def reset(self, state: torch.Tensor | None = None) -> None:
        self.state = None if state is None else state.clone()

    def step(self, command: torch.Tensor) -> torch.Tensor:
        target = command * self.motor_strength
        if self.state is None:
            self.state = torch.zeros_like(target)
        self.state = self.state + self.alpha * (target - self.state)
        return self.state.clone()
