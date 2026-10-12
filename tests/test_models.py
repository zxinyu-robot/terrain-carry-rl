import torch

from terrain_carry_rl.actions import ACTION_DIM, scale_action, split_action
from terrain_carry_rl.models import StudentPolicy, TeacherPolicy
from terrain_carry_rl.observations import ObservationHistory
from terrain_carry_rl.schema import DEPLOYABLE_OBSERVATION, PRIVILEGED_OBSERVATION


def test_action_shape_and_split() -> None:
    action = scale_action(torch.full((3, ACTION_DIM), 2.0))
    parts = split_action(action)
    assert parts.leg.shape == (3, 12)
    assert parts.arm.shape == (3, 6)
    assert parts.gripper.shape == (3, 1)
    assert action.max() <= 1.0


def test_teacher_student_forward_with_features() -> None:
    teacher = TeacherPolicy(feature_dim=32)
    student = StudentPolicy(history_length=4, feature_dim=32)
    teacher_action, teacher_features = teacher(
        torch.randn(5, PRIVILEGED_OBSERVATION.dim), return_features=True
    )
    student_action, student_features = student(
        torch.randn(5, 4, DEPLOYABLE_OBSERVATION.dim), return_features=True
    )
    assert teacher_action.shape == student_action.shape == (5, ACTION_DIM)
    assert teacher_features.shape == student_features.shape == (5, 32)


def test_observation_history() -> None:
    history = ObservationHistory(3, 2)
    assert history.reset(torch.tensor([[1.0, 2.0]])).shape == (1, 3, 2)
    updated = history.append(torch.tensor([[3.0, 4.0]]))
    assert torch.equal(updated[0, -1], torch.tensor([3.0, 4.0]))
