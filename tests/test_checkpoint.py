"""Unit tests for CheckpointManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.exceptions import CheckpointError
from adaptivehb.managers import CheckpointManager


@pytest.fixture()
def checkpoints(framework_config: FrameworkConfig, tmp_path: Path) -> CheckpointManager:
    manager = CheckpointManager(framework_config, base_dir=tmp_path)
    manager.initialize()
    return manager


def test_save_and_load_latest(checkpoints: CheckpointManager) -> None:
    payload = {"weights": [1, 2, 3], "optimizer": {"lr": 0.001}, "epoch": 5}
    metadata = {"epoch": 5, "metrics": {"val_mae": 0.5}, "seed": 42}
    checkpoints.save("unet", payload, metadata)

    loaded_payload, loaded_meta = checkpoints.load_latest("unet")
    assert loaded_payload == payload
    assert loaded_meta["epoch"] == 5
    assert "saved_at" in loaded_meta


def test_best_checkpoint(checkpoints: CheckpointManager) -> None:
    checkpoints.save("unet", {"epoch": 1}, {"val_mae": 0.9})
    checkpoints.save("unet", {"epoch": 7}, {"val_mae": 0.3}, is_best=True)

    latest, _ = checkpoints.load_latest("unet")
    best, best_meta = checkpoints.load_best("unet")
    assert latest["epoch"] == 7
    assert best["epoch"] == 7
    assert best_meta["is_best"] is True


def test_exists_and_list(checkpoints: CheckpointManager) -> None:
    assert not checkpoints.exists("unet")
    checkpoints.save("unet", {"epoch": 1}, {})
    checkpoints.save("segformer", {"epoch": 1}, {}, is_best=True)
    assert checkpoints.exists("unet")
    assert checkpoints.exists("segformer", tag="best")
    assert checkpoints.list_checkpoints() == ["segformer", "unet"]


def test_load_missing_raises(checkpoints: CheckpointManager) -> None:
    with pytest.raises(CheckpointError):
        checkpoints.load_latest("does_not_exist")
    with pytest.raises(CheckpointError):
        checkpoints.load_best("unet")  # never saved as best
