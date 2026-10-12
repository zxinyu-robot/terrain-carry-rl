"""Teacher-to-student action and representation losses."""

import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(
        self,
        action_weight: float = 1.0,
        feature_weight: float = 0.25,
        uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.action_weight = action_weight
        self.feature_weight = feature_weight
        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting:
            self.log_variances = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("log_variances", None)

    def forward(
        self,
        student_action: torch.Tensor,
        teacher_action: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        action = F.mse_loss(student_action, teacher_action)
        feature = F.mse_loss(student_features, teacher_features.detach())
        weighted = torch.stack((self.action_weight * action, self.feature_weight * feature))
        if self.log_variances is not None:
            total = (torch.exp(-self.log_variances) * weighted + self.log_variances).sum()
        else:
            total = weighted.sum()
        return total, {"action_loss": action.detach(), "feature_loss": feature.detach()}
