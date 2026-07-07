"""Unit tests for the prediction subsystem (interface, factory, manager).

Real torch backbones are covered by a guarded test that is skipped when PyTorch
is absent; the torch-free reference regressor exercises the full interface and
the pipeline integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.core.types import ModelCategory
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.pipeline import HbPipeline
from adaptivehb.prediction import (
    PredictionConfig,
    PredictionManager,
    PredictionModel,
    ReferencePredictionModel,
    available_prediction,
    build_prediction,
    is_registered,
    register_prediction,
    torch_available,
)


def test_reference_is_always_registered() -> None:
    assert "reference" in available_prediction()


def test_build_returns_prediction_model() -> None:
    model = build_prediction("efficientnet", tissue="palm")
    model.build()
    assert isinstance(model, PredictionModel)
    assert model.is_built
    if not torch_available():
        assert isinstance(model, ReferencePredictionModel)
        assert model.name == "efficientnet"
        assert model.tissue == "palm"


def test_reference_model_interface_and_persistence(tmp_path: Path) -> None:
    model = build_prediction("reference", tissue="eye")
    model.build()
    train = model.train_epoch(1)
    val = model.validate(2)
    assert train["train_loss"] > val["val_loss"]  # loss decreases
    assert "val_mae" in val
    estimate = model.predict(None)
    assert isinstance(estimate, float)
    assert 0.0 < estimate < 25.0  # plausible g/dL range

    path = model.save(tmp_path / "pred.pkl")
    restored = ReferencePredictionModel()
    restored.load(path)
    assert restored.state_dict() == model.state_dict()


def test_prediction_config_parses(framework_config: FrameworkConfig) -> None:
    config = PredictionConfig.from_section(framework_config.section("prediction"))
    assert config.default_model == "efficientnet"
    assert config.architecture_for_tissue("eye") == "vit"
    assert config.architecture_for_tissue("unknown_tissue") == "efficientnet"
    assert config.epochs >= 1


def test_manager_builds_and_routes(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    manager = PredictionManager(framework_config, tmp_path)
    manager.initialize()
    assert "reference" in manager.available()
    assert manager.architecture_for_tissue("tongue") == "convnext"

    class _Plan:
        name = "hb_nail"
        architecture = "resnet"

    trainable = manager.build_trainable(_Plan())
    assert isinstance(trainable, PredictionModel)
    assert trainable.tissue == "nail"


def test_register_custom_model() -> None:
    @register_prediction("unit_test_pred")
    def _builder(name: str = "unit_test_pred", **kwargs: object) -> ReferencePredictionModel:
        return ReferencePredictionModel(name=name, **kwargs)

    assert is_registered("unit_test_pred")
    model = build_prediction("unit_test_pred")
    model.build()
    assert model.name == "unit_test_pred"


def test_pipeline_training_registers_prediction_models(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=6, seed=6)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()
    pipeline.train(epochs=2)
    pred_models = pipeline.manager.registry.find(ModelCategory.PREDICTION)
    assert len(pred_models) == 4  # eye, palm, tongue, nail


@pytest.mark.parametrize("architecture", ["resnet"])
def test_torch_backbone_forward(architecture: str) -> None:
    torch = pytest.importorskip("torch")
    if not is_registered(architecture):
        pytest.skip("torch present but backbone not registered")
    model = build_prediction(architecture, tissue="eye")
    model.build()
    estimate = model.predict(torch.randn(1, 3, 224, 224))
    assert isinstance(estimate, float)
