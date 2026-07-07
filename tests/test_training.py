"""Unit tests for TrainingManager (dummy trainables, no ML stack)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.core import ModelCategory
from adaptivehb.exceptions import PipelineError
from adaptivehb.managers import (
    CheckpointManager,
    RegistryManager,
    StateManager,
    TrainingManager,
    TrainingPlan,
)


class SeqTrainable:
    """A trainable whose validation loss follows a fixed schedule."""

    def __init__(self, val_losses: list[float]) -> None:
        self._val_losses = val_losses
        self.epochs_seen: list[int] = []

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.epochs_seen.append(epoch)
        return {"train_loss": 1.0 / epoch}

    def validate(self, epoch: int) -> dict[str, float]:
        return {"val_loss": self._val_losses[epoch - 1]}

    def state_dict(self) -> dict[str, Any]:
        return {"seen": list(self.epochs_seen)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epochs_seen = list(state.get("seen", []))


@pytest.fixture()
def managers(
    framework_config: FrameworkConfig, tmp_path: Path
) -> tuple[TrainingManager, RegistryManager, StateManager, CheckpointManager]:
    checkpoints = CheckpointManager(framework_config, tmp_path)
    state = StateManager(framework_config, tmp_path)
    registry = RegistryManager(framework_config, tmp_path)
    for manager in (checkpoints, state, registry):
        manager.initialize()
    training = TrainingManager(framework_config, tmp_path, checkpoints, state, registry)
    training.initialize()
    return training, registry, state, checkpoints


def test_train_tracks_best_and_saves_checkpoint(managers) -> None:
    training, _, _, checkpoints = managers
    trainable = SeqTrainable([1.0, 0.8, 0.6, 0.4])
    result = training.train(trainable, TrainingPlan(name="unet", epochs=4))
    assert result.epochs_run == 4
    assert result.best_epoch == 4
    assert result.best_metric == pytest.approx(0.4)
    assert not result.stopped_early
    assert checkpoints.exists("unet", tag="best")
    assert len(result.history) == 4


def test_registration_on_completion(managers) -> None:
    training, registry, _, _ = managers
    trainable = SeqTrainable([0.9, 0.5])
    result = training.train(
        trainable,
        TrainingPlan(
            name="vit", epochs=2, category=ModelCategory.PREDICTION,
            task="hb", architecture="ViT",
        ),
    )
    assert result.registered_id == "HB_VIT_V001"
    assert registry.load_latest(ModelCategory.PREDICTION, "vit").metrics["val_loss"] == pytest.approx(0.5)


def test_state_marks_module_completed(managers) -> None:
    training, _, state, _ = managers
    training.train(SeqTrainable([0.5, 0.4]), TrainingPlan(name="unet", epochs=2))
    assert state.is_completed("unet")


def test_resume_continues_from_last_epoch(managers) -> None:
    training, _, _, _ = managers
    plan = TrainingPlan(name="unet", epochs=2)
    training.train(SeqTrainable([0.9, 0.8]), plan)

    resumed = SeqTrainable([0.9, 0.8, 0.7, 0.6])
    result = training.train(resumed, TrainingPlan(name="unet", epochs=4), resume=True)
    # Only epochs 3 and 4 run on the fresh trainable; state restored [1, 2].
    assert resumed.epochs_seen == [1, 2, 3, 4]
    assert result.epochs_run == 4
    assert result.best_epoch == 4


def test_early_stopping(managers) -> None:
    training, _, _, _ = managers
    trainable = SeqTrainable([1.0, 0.5, 0.6, 0.7, 0.8])
    result = training.train(
        trainable, TrainingPlan(name="unet", epochs=5, patience=2)
    )
    assert result.stopped_early
    assert result.best_epoch == 2
    assert result.epochs_run == 4  # stopped two epochs after the best


def test_missing_monitor_metric_raises(managers) -> None:
    training, _, _, _ = managers
    trainable = SeqTrainable([0.5])
    with pytest.raises(PipelineError):
        training.train(trainable, TrainingPlan(name="unet", epochs=1, monitor="dice"))


def test_requires_checkpoint_manager(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    training = TrainingManager(framework_config, tmp_path, checkpoints=None)
    with pytest.raises(PipelineError):
        training.train(SeqTrainable([0.5]), TrainingPlan(name="x", epochs=1))
