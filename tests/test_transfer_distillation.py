import math

import torch
from torch.utils.data import DataLoader

from terrain_carry_rl.distillation import (
    DistillationLoss,
    DistillationTrainer,
    SyntheticDistillationDataset,
)
from terrain_carry_rl.models import StudentPolicy, TeacherPolicy
from terrain_carry_rl.transfer import BackboneFreezeSchedule, load_partial_checkpoint


def test_partial_transfer_copies_intersection_and_reports() -> None:
    source = TeacherPolicy(observation_dim=10, action_dim=8, hidden_dims=(16,), feature_dim=7)
    target = TeacherPolicy(observation_dim=12, action_dim=9, hidden_dims=(16,), feature_dim=7)
    checkpoint = source.state_dict()
    checkpoint["retired.weight"] = torch.ones(1)
    report = load_partial_checkpoint(target, checkpoint)
    assert "backbone.network.0.weight" in report.shape_mismatch
    assert "action_head.weight" in report.partially_loaded
    assert "retired.weight" in report.unexpected
    assert torch.equal(
        target.backbone.network[0].weight[:, :10],
        source.backbone.network[0].weight,
    )


def test_freeze_then_unfreeze_backbone() -> None:
    model = StudentPolicy(hidden_dims=(16,), feature_dim=8)
    schedule = BackboneFreezeSchedule(unfreeze_epoch=2)
    assert schedule.apply(model, 0) is False
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert schedule.apply(model, 2) is True
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_loss_is_finite_and_short_training_runs() -> None:
    teacher = TeacherPolicy(hidden_dims=(16,), feature_dim=8)
    student = StudentPolicy(history_length=2, hidden_dims=(16,), feature_dim=8)
    loss = DistillationLoss(uncertainty_weighting=True)
    trainer = DistillationTrainer(teacher, student, loss)
    dataset = SyntheticDistillationDataset(size=16, history_length=2, seed=2)
    metrics = trainer.train_epoch(DataLoader(dataset, batch_size=4))
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["loss"] >= 0
