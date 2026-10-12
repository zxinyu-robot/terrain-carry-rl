"""Minimal supervised distillation loop independent of a simulator."""

from collections.abc import Iterable

import torch
from torch import nn

from terrain_carry_rl.distillation.losses import DistillationLoss


class DistillationTrainer:
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        loss: DistillationLoss,
        learning_rate: float = 1e-3,
    ) -> None:
        self.teacher = teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.student = student
        self.loss = loss
        parameters = list(student.parameters()) + list(loss.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def train_epoch(self, batches: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
        self.student.train()
        totals = {"loss": 0.0, "action_loss": 0.0, "feature_loss": 0.0}
        count = 0
        for batch in batches:
            with torch.no_grad():
                teacher_action, teacher_features = self.teacher(
                    batch["privileged"], return_features=True
                )
            student_action, student_features = self.student(batch["history"], return_features=True)
            total, terms = self.loss(
                student_action, teacher_action, student_features, teacher_features
            )
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()
            totals["loss"] += float(total.detach())
            totals["action_loss"] += float(terms["action_loss"])
            totals["feature_loss"] += float(terms["feature_loss"])
            count += 1
        if count == 0:
            raise ValueError("at least one batch is required")
        return {name: value / count for name, value in totals.items()}
