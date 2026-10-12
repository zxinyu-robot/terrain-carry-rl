import torch

from terrain_carry_rl.sim2real import (
    ActuatorLag,
    DomainRandomizer,
    ObservationActionLatency,
    ObservationNoise,
)


def test_domain_randomization_and_noise_are_deterministic() -> None:
    assert DomainRandomizer(11).sample() == DomainRandomizer(11).sample()
    observation = torch.ones(4)
    assert torch.equal(
        ObservationNoise(0.1, seed=5).apply(observation),
        ObservationNoise(0.1, seed=5).apply(observation),
    )


def test_observation_and_action_latency() -> None:
    latency = ObservationActionLatency(observation_steps=1, action_steps=2)
    first = torch.tensor([1.0])
    assert torch.equal(latency.apply_observation(first), first)
    assert torch.equal(latency.apply_observation(torch.tensor([2.0])), first)
    assert torch.equal(latency.apply_action(first), first)
    assert torch.equal(latency.apply_action(torch.tensor([2.0])), first)
    assert torch.equal(latency.apply_action(torch.tensor([3.0])), first)
    assert torch.equal(latency.apply_action(torch.tensor([4.0])), torch.tensor([2.0]))


def test_actuator_lag_converges_without_overshoot() -> None:
    actuator = ActuatorLag(time_constant=0.1, time_step=0.02, motor_strength=0.8)
    outputs = [float(actuator.step(torch.ones(1))) for _ in range(40)]
    assert outputs == sorted(outputs)
    assert 0.7 < outputs[-1] <= 0.8
