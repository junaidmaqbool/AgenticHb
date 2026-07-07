"""Unit tests for registry/checkpoint-backed model loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.managers import CheckpointManager
from adaptivehb.model_loading import load_weights_into
from adaptivehb.pipeline import HbPipeline
from adaptivehb.prediction import ReferencePredictionModel
from adaptivehb.prediction.manager import PredictionManager
from adaptivehb.segmentation.manager import SegmentationManager


# --------------------------------------------------------------------------- #
# load_weights_into
# --------------------------------------------------------------------------- #

def test_load_weights_round_trip(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    checkpoints = CheckpointManager(framework_config, tmp_path)
    checkpoints.initialize()
    # Save a checkpoint carrying a model_state (as TrainingManager does).
    trained = ReferencePredictionModel(name="hb_eye", tissue="eye")
    trained.build()
    trained.train_epoch(1)
    trained.train_epoch(2)
    checkpoints.save("hb_eye", {"epoch": 2, "model_state": trained.state_dict()}, {"epoch": 2}, is_best=True)

    fresh = ReferencePredictionModel(name="hb_eye", tissue="eye")
    fresh.build()
    loaded = load_weights_into(fresh, checkpoints, "hb_eye")
    assert loaded is True
    assert fresh.state_dict()["epochs_seen"] == [1, 2]


def test_load_weights_missing_checkpoint(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    checkpoints = CheckpointManager(framework_config, tmp_path)
    checkpoints.initialize()
    model = ReferencePredictionModel(name="hb_eye")
    model.build()
    assert load_weights_into(model, checkpoints, "does_not_exist") is False


# --------------------------------------------------------------------------- #
# Manager.load_trained
# --------------------------------------------------------------------------- #

def test_prediction_manager_load_trained(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    checkpoints = CheckpointManager(framework_config, tmp_path)
    checkpoints.initialize()
    trained = ReferencePredictionModel(name="hb_palm", tissue="palm")
    trained.build()
    trained.train_epoch(1)
    checkpoints.save("hb_palm", {"epoch": 1, "model_state": trained.state_dict()}, {"epoch": 1}, is_best=True)

    manager = PredictionManager(framework_config, tmp_path)
    manager.initialize()
    model = manager.load_trained("hb_palm", checkpoints, tissue="palm")
    assert model.is_built
    assert model.state_dict()["epochs_seen"] == [1]


def test_load_trained_without_checkpoint_returns_untrained(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    checkpoints = CheckpointManager(framework_config, tmp_path)
    checkpoints.initialize()
    manager = PredictionManager(framework_config, tmp_path)
    manager.initialize()
    model = manager.load_trained("hb_eye", checkpoints, tissue="eye")
    assert model.is_built  # untrained but usable


def test_segmentation_manager_load_trained(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    from adaptivehb.segmentation import ReferenceSegmentationModel

    checkpoints = CheckpointManager(framework_config, tmp_path)
    checkpoints.initialize()
    trained = ReferenceSegmentationModel(name="seg_unet")
    trained.build()
    trained.train_epoch(1)
    checkpoints.save("seg_unet", {"epoch": 1, "model_state": trained.state_dict()}, {"epoch": 1}, is_best=True)

    manager = SegmentationManager(framework_config, tmp_path)
    manager.initialize()
    model = manager.load_trained("seg_unet", checkpoints, architecture="unet")
    assert model.is_built
    assert model.state_dict()["epochs_seen"] == [1]


# --------------------------------------------------------------------------- #
# Pipeline integration: trained weights are loaded after training
# --------------------------------------------------------------------------- #

def test_pipeline_load_trained_after_training(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=6, seed=1)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()
    pipeline.train(epochs=3)
    pm = pipeline.manager
    model = pm.prediction.load_trained("hb_eye", pm.checkpoints, tissue="eye")
    # Training ran 3 epochs; the loaded weights reflect that (not a fresh model).
    assert model.state_dict()["epochs_seen"] == [1, 2, 3]
