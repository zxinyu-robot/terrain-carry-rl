"""Inspect policy contracts without requiring a simulator."""

import torch

from terrain_carry_rl.models import StudentPolicy, TeacherPolicy
from terrain_carry_rl.schema import DEPLOYABLE_OBSERVATION, PRIVILEGED_OBSERVATION

teacher = TeacherPolicy()
student = StudentPolicy(history_length=4)

teacher_action, teacher_features = teacher(
    torch.zeros(2, PRIVILEGED_OBSERVATION.dim), return_features=True
)
student_action, student_features = student(
    torch.zeros(2, 4, DEPLOYABLE_OBSERVATION.dim), return_features=True
)

print(
    {
        "teacher": (teacher_action.shape, teacher_features.shape),
        "student": (student_action.shape, student_features.shape),
    }
)
