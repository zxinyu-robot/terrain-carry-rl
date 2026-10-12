"""Backbone freezing and staged unfreezing."""

from dataclasses import dataclass

from torch import nn


@dataclass
class BackboneFreezeSchedule:
    unfreeze_epoch: int
    backbone_attribute: str = "backbone"

    def apply(self, model: nn.Module, epoch: int) -> bool:
        """Apply schedule and return whether the backbone is trainable."""
        trainable = epoch >= self.unfreeze_epoch
        backbone = getattr(model, self.backbone_attribute)
        for parameter in backbone.parameters():
            parameter.requires_grad = trainable
        return trainable
