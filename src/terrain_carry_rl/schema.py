"""Explicit observation contracts for simulation and deployment."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservationField:
    name: str
    size: int
    deployable: bool = True


@dataclass(frozen=True)
class ObservationSchema:
    fields: tuple[ObservationField, ...]

    @property
    def dim(self) -> int:
        return sum(field.size for field in self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def validate(self, observation: torch.Tensor) -> None:
        if observation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {observation.shape[-1]}")


DEPLOYABLE_OBSERVATION = ObservationSchema(
    (
        ObservationField("base_angular_velocity", 3),
        ObservationField("projected_gravity", 3),
        ObservationField("command", 4),
        ObservationField("leg_joint_position", 12),
        ObservationField("leg_joint_velocity", 12),
        ObservationField("arm_joint_position", 6),
        ObservationField("arm_joint_velocity", 6),
        ObservationField("gripper_position", 1),
        ObservationField("previous_action", 19),
    )
)

PRIVILEGED_OBSERVATION = ObservationSchema(
    DEPLOYABLE_OBSERVATION.fields
    + (
        ObservationField("base_linear_velocity", 3, False),
        ObservationField("terrain_heights", 32, False),
        ObservationField("object_pose", 7, False),
        ObservationField("object_velocity", 6, False),
        ObservationField("contact_forces", 16, False),
        ObservationField("randomized_dynamics", 10, False),
    )
)
