"""Command-line entry point for simulator-free infrastructure checks."""

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from terrain_carry_rl.distillation import (
    DistillationLoss,
    DistillationTrainer,
    SyntheticDistillationDataset,
)
from terrain_carry_rl.evaluation import load_training_checkpoint, save_training_checkpoint
from terrain_carry_rl.models import StudentPolicy, TeacherPolicy


def _load_config(path: str | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "seed": 7,
        "samples": 64,
        "batch_size": 16,
        "epochs": 2,
        "history_length": 4,
        "learning_rate": 0.001,
        "feature_dim": 32,
    }
    if path is not None:
        defaults.update(json.loads(Path(path).read_text()))
    return defaults


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    history_length = int(config["history_length"])
    feature_dim = int(config["feature_dim"])
    teacher = TeacherPolicy(hidden_dims=(64,), feature_dim=feature_dim)
    student = StudentPolicy(
        history_length=history_length, hidden_dims=(64,), feature_dim=feature_dim
    )
    dataset = SyntheticDistillationDataset(
        size=int(config["samples"]), history_length=history_length, seed=seed
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    loss = DistillationLoss(uncertainty_weighting=True)
    trainer = DistillationTrainer(
        teacher, student, loss, learning_rate=float(config["learning_rate"])
    )
    epochs = [trainer.train_epoch(loader) for _ in range(int(config["epochs"]))]

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "student.pt"
        save_training_checkpoint(checkpoint, student, trainer.optimizer, epoch=len(epochs))
        restored = StudentPolicy(
            history_length=history_length, hidden_dims=(64,), feature_dim=feature_dim
        )
        metadata = load_training_checkpoint(checkpoint, restored)
        reference = dataset.history[:2]
        roundtrip_error = float((student(reference) - restored(reference)).abs().max())

    return {
        "mode": "synthetic_smoke_test",
        "samples": len(dataset),
        "epochs": epochs,
        "checkpoint_epoch": metadata["epoch"],
        "checkpoint_roundtrip_max_error": roundtrip_error,
        "disclaimer": "Synthetic data only; no Isaac Lab or robot validation was performed.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="terrain-carry-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run synthetic distillation smoke test")
    smoke.add_argument("--config", help="JSON config path")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(run_smoke(_load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
