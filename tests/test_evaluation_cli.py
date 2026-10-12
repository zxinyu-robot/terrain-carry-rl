import math

import torch

from terrain_carry_rl.cli import run_smoke
from terrain_carry_rl.evaluation import (
    MetricAggregator,
    load_training_checkpoint,
    save_training_checkpoint,
)
from terrain_carry_rl.models import StudentPolicy


def test_metric_aggregation() -> None:
    metrics = MetricAggregator()
    metrics.update(reward=1.0, success=0.0)
    metrics.update(reward=3.0, success=1.0)
    result = metrics.compute()
    assert result["reward"]["mean"] == 2.0
    assert result["reward"]["count"] == 2


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = StudentPolicy(history_length=2, hidden_dims=(8,), feature_dim=4)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, model, optimizer, epoch=3)
    restored = StudentPolicy(history_length=2, hidden_dims=(8,), feature_dim=4)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    metadata = load_training_checkpoint(path, restored, restored_optimizer)
    assert metadata == {"epoch": 3}
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_smoke_command_function() -> None:
    result = run_smoke(
        {
            "seed": 1,
            "samples": 8,
            "batch_size": 4,
            "epochs": 1,
            "history_length": 2,
            "learning_rate": 0.001,
            "feature_dim": 8,
        }
    )
    assert result["mode"] == "synthetic_smoke_test"
    assert math.isclose(result["checkpoint_roundtrip_max_error"], 0.0)
