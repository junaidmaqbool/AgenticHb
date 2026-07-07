"""Unit tests for the segmentation subsystem (interface, factory, manager).

Real torch backends are covered by a guarded test that is skipped when PyTorch
is absent; the torch-free reference model exercises the full interface and the
pipeline integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.core.types import ModelCategory
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.pipeline import HbPipeline
from adaptivehb.segmentation import (
    ReferenceSegmentationModel,
    SegmentationConfig,
    SegmentationManager,
    SegmentationModel,
    available_segmentation,
    build_segmentation,
    is_registered,
    register_segmentation,
    torch_available,
)


def test_reference_is_always_registered() -> None:
    assert "reference" in available_segmentation()


def test_build_returns_segmentation_model() -> None:
    model = build_segmentation("unet")
    model.build()
    assert isinstance(model, SegmentationModel)
    assert model.is_built
    # Without torch, any name falls back to the reference implementation.
    if not torch_available():
        assert isinstance(model, ReferenceSegmentationModel)
        assert model.name == "unet"  # requested name is preserved


def test_reference_model_interface_and_persistence(tmp_path: Path) -> None:
    model = build_segmentation("reference")
    model.build()
    m1 = model.train_epoch(1)
    m2 = model.validate(2)
    assert m1["train_loss"] > model.validate(2)["val_loss"]  # loss decreases
    assert "val_dice" in m2
    assert model.predict(None) is not None

    path = model.save(tmp_path / "seg.pkl")
    restored = ReferenceSegmentationModel()
    restored.load(path)
    assert restored.state_dict() == model.state_dict()


def test_segmentation_config_parses(framework_config: FrameworkConfig) -> None:
    config = SegmentationConfig.from_section(framework_config.section("segmentation"))
    assert config.default_model == "unet"
    assert "unet" in config.available_models
    assert config.epochs >= 1


def test_manager_builds_and_lists(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    manager = SegmentationManager(framework_config, tmp_path)
    manager.initialize()
    assert "reference" in manager.available()
    model = manager.build("unet")
    assert model.is_built

    class _Plan:
        architecture = "deeplabv3plus"

    trainable = manager.build_trainable(_Plan())
    assert isinstance(trainable, SegmentationModel)


def test_register_custom_model() -> None:
    @register_segmentation("unit_test_seg")
    def _builder(name: str = "unit_test_seg", **kwargs: object) -> ReferenceSegmentationModel:
        return ReferenceSegmentationModel(name=name, **kwargs)

    assert is_registered("unit_test_seg")
    model = build_segmentation("unit_test_seg")
    model.build()
    assert model.name == "unit_test_seg"


def test_pipeline_training_registers_segmentation_models(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=6, seed=4)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()
    pipeline.train(epochs=2)
    seg_models = pipeline.manager.registry.find(ModelCategory.SEGMENTATION)
    assert len(seg_models) == 3  # unet, segformer, deeplabv3plus


@pytest.mark.parametrize("architecture", ["unet"])
def test_torch_backend_forward(architecture: str) -> None:
    torch = pytest.importorskip("torch")
    if not is_registered(architecture):
        pytest.skip("torch present but backend not registered")
    model = build_segmentation(architecture, num_classes=1, in_channels=3)
    model.build()
    output = model.predict(torch.randn(1, 3, 64, 64))
    assert output.shape[0] == 1
