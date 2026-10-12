"""Discrete observation and action latency queues."""

from collections import deque

import torch


class LatencyQueue:
    def __init__(self, delay_steps: int) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")
        self.delay_steps = delay_steps
        self._queue: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        self._queue.clear()

    def push(self, value: torch.Tensor) -> torch.Tensor:
        value = value.clone()
        if self.delay_steps == 0:
            return value
        if not self._queue:
            self._queue.extend(value.clone() for _ in range(self.delay_steps))
        self._queue.append(value)
        return self._queue.popleft()


class ObservationActionLatency:
    def __init__(self, observation_steps: int = 0, action_steps: int = 0) -> None:
        self.observations = LatencyQueue(observation_steps)
        self.actions = LatencyQueue(action_steps)

    def apply_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return self.observations.push(observation)

    def apply_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.actions.push(action)

    def reset(self) -> None:
        self.observations.reset()
        self.actions.reset()
