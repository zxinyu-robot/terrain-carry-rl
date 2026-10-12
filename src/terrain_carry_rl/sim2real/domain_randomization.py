"""Seeded dynamics randomization contracts for simulator adapters."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RandomizationRanges:
    mass_scale: tuple[float, float] = (0.8, 1.2)
    center_of_mass_offset: tuple[float, float] = (-0.02, 0.02)
    friction: tuple[float, float] = (0.4, 1.2)
    motor_strength: tuple[float, float] = (0.85, 1.15)
    observation_noise_scale: tuple[float, float] = (0.0, 0.03)


@dataclass(frozen=True)
class RandomizedDynamics:
    mass_scale: float
    center_of_mass_offset: tuple[float, float, float]
    friction: float
    motor_strength: float
    observation_noise_scale: float

    def to_dict(self) -> dict[str, float | tuple[float, float, float]]:
        return asdict(self)


class DomainRandomizer:
    def __init__(self, seed: int, ranges: RandomizationRanges | None = None) -> None:
        self.ranges = ranges or RandomizationRanges()
        self.generator = np.random.default_rng(seed)

    def sample(self) -> RandomizedDynamics:
        uniform = self.generator.uniform
        center = tuple(
            float(value) for value in uniform(*self.ranges.center_of_mass_offset, size=3)
        )
        return RandomizedDynamics(
            mass_scale=float(uniform(*self.ranges.mass_scale)),
            center_of_mass_offset=center,
            friction=float(uniform(*self.ranges.friction)),
            motor_strength=float(uniform(*self.ranges.motor_strength)),
            observation_noise_scale=float(uniform(*self.ranges.observation_noise_scale)),
        )
