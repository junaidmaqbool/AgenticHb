"""Unit tests for training operations (torch-free helpers + guarded loops)."""

from __future__ import annotations

import pytest

from adaptivehb import training_ops as ops
from adaptivehb.exceptions import ModelError


# --------------------------------------------------------------------------- #
# LossAccumulator (torch-free)
# --------------------------------------------------------------------------- #

def test_loss_accumulator_weighted_average() -> None:
    acc = ops.LossAccumulator()
    acc.add(1.0, 2)  # 2 samples at 1.0
    acc.add(4.0, 2)  # 2 samples at 4.0
    assert acc.average == pytest.approx(2.5)
    assert acc.count == 4


def test_loss_accumulator_empty() -> None:
    assert ops.LossAccumulator().average == 0.0


# --------------------------------------------------------------------------- #
# Name validation (torch-free)
# --------------------------------------------------------------------------- #

def test_supported_sets() -> None:
    assert "adamw" in ops.supported_optimizers()
    assert "mse" in ops.supported_losses("prediction")
    assert "dice_bce" in ops.supported_losses("segmentation")


def test_validate_optimizer() -> None:
    assert ops.validate_optimizer("AdamW") == "adamw"
    with pytest.raises(ModelError):
        ops.validate_optimizer("rmsprop_unsupported")


def test_validate_loss_by_task() -> None:
    assert ops.validate_loss("mse", "prediction") == "mse"
    with pytest.raises(ModelError):
        ops.validate_loss("dice_bce", "prediction")  # segmentation-only loss
    assert ops.validate_loss("dice_bce", "segmentation") == "dice_bce"


def test_resolve_device_is_str() -> None:
    assert ops.resolve_device() in {"cpu", "cuda"}


# --------------------------------------------------------------------------- #
# Guarded builders: validate first, then require torch
# --------------------------------------------------------------------------- #

def test_builders_validate_before_requiring_torch() -> None:
    # Invalid names raise regardless of torch availability.
    with pytest.raises(ModelError):
        ops.build_optimizer("bogus", [], 1e-3)
    with pytest.raises(ModelError):
        ops.build_regression_loss("bogus")


def test_builders_require_torch_when_absent() -> None:
    if ops.torch_available():
        pytest.skip("torch installed; builders return real objects")
    # Valid names, but torch missing -> a clear error.
    with pytest.raises(ModelError):
        ops.build_optimizer("adam", [], 1e-3)
    with pytest.raises(ModelError):
        ops.build_regression_loss("mse")
    with pytest.raises(ModelError):
        ops.build_segmentation_loss("dice_bce")


# --------------------------------------------------------------------------- #
# Torch-guarded integration (skipped without torch)
# --------------------------------------------------------------------------- #

def test_regression_epoch_runs_with_torch() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    module = nn.Linear(4, 1)
    optimizer = ops.build_optimizer("adam", module.parameters(), 1e-2)
    loss_fn = ops.build_regression_loss("mse")
    batch = (torch.randn(3, 4), torch.randn(3))
    metrics = ops.run_regression_epoch(module, [batch], loss_fn, optimizer, device="cpu")
    assert "loss" in metrics and "mae" in metrics
    assert metrics["loss"] >= 0.0


def test_torch_prediction_model_requires_data() -> None:
    pytest.importorskip("torch")
    from adaptivehb.prediction import build_prediction, is_registered

    if not is_registered("resnet"):
        pytest.skip("torch present but backbone not registered")
    model = build_prediction("resnet", tissue="eye")
    model.build()
    with pytest.raises(ModelError):
        model.train_epoch(1)  # no data attached
