"""Tests for a separate segmentation dataset source (Decision 039)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from adaptivehb.config import ConfigLoader, FrameworkConfig
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.dataset.config import DatasetConfig
from adaptivehb.dataset.manager import DatasetManager
from adaptivehb.pipeline import HbPipeline


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

def test_segmentation_source_parsed() -> None:
    cfg = DatasetConfig.from_section(
        {"dataset": {"images_dir": "img", "segmentation_source": {"root": "/seg", "masks_dir": "m"}}}
    )
    assert cfg.segmentation_root == "/seg"
    assert cfg.segmentation_masks_dir == "m"
    assert cfg.segmentation_images_dir == "img"  # defaults to the main images_dir
    assert cfg.segmentation_metadata_file is None


def test_no_segmentation_source_is_none() -> None:
    cfg = DatasetConfig.from_section({"dataset": {}})
    assert cfg.segmentation_root is None


# --------------------------------------------------------------------------- #
# DatasetManager: metadata-optional source
# --------------------------------------------------------------------------- #

def test_metadata_optional_dataset_without_csv(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    root = tmp_path / "seg"
    generate_synthetic_dataset(root, num_patients=6, seed=1, tissues=["left_eye", "palm"])
    shutil.rmtree(root / "metadata")  # mask-only: no labels CSV

    manager = DatasetManager(framework_config, base_dir=tmp_path, dataset_root=root, metadata_optional=True)
    manager.initialize()
    # We need the dataset config to look for these tissue folders.
    # (framework_config default tissues include eye/palm/... — left_eye/palm here.)
    samples = manager.samples()  # builds the index without a CSV
    # Patient-level split works from image-derived patient IDs (no metadata).
    splits = manager.split()
    assert sum(len(v) for v in splits.values()) == 6  # all patients assigned
    assert manager.load_metadata().patient_ids == []  # empty, but no error


def test_missing_csv_without_optional_raises(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    from adaptivehb.exceptions import DatasetError

    root = tmp_path / "seg"
    generate_synthetic_dataset(root, num_patients=4, seed=1, tissues=["palm"])
    shutil.rmtree(root / "metadata")
    manager = DatasetManager(framework_config, base_dir=tmp_path, dataset_root=root)  # not optional
    manager.initialize()
    with pytest.raises(DatasetError):
        manager.load_metadata()


# --------------------------------------------------------------------------- #
# Pipeline: two distinct dataset sources
# --------------------------------------------------------------------------- #

@pytest.fixture()
def two_datasets(tmp_path: Path) -> tuple[Path, Path]:
    seg = tmp_path / "seg_dataset"
    hb = tmp_path / "hb_dataset"
    generate_synthetic_dataset(seg, num_patients=8, seed=1, tissues=["left_eye", "right_eye", "palm"])
    shutil.rmtree(seg / "metadata")  # mask-only segmentation source
    generate_synthetic_dataset(hb, num_patients=12, seed=2, tissues=["left_eye", "right_eye", "palm"])
    return seg, hb


def _config_with_seg_root(configs_dir: Path, seg_root: Path) -> FrameworkConfig:
    cfg = ConfigLoader(configs_dir).load()
    ds = cfg.section("dataset")["dataset"]
    ds["tissues"] = ["left_eye", "right_eye", "palm"]
    ds["segmentation_source"] = {"root": str(seg_root)}
    cfg.section("segmentation")["segmentation"]["available_models"] = ["unet"]
    cfg.section("prediction")["prediction"]["available_models"] = ["efficientnet"]
    cfg.section("prediction")["prediction"]["default_model"] = "efficientnet"
    return cfg


def test_pipeline_uses_distinct_segmentation_dataset(
    configs_dir: Path, tmp_path: Path, two_datasets: tuple[Path, Path]
) -> None:
    seg, hb = two_datasets
    cfg = _config_with_seg_root(configs_dir, seg)
    pipeline = HbPipeline(cfg, base_dir=tmp_path, dataset_root=hb).initialize()
    pm = pipeline.manager
    assert pm.segmentation_dataset is not pm.dataset
    assert pm.segmentation_dataset.root.name == "seg_dataset"
    assert pm.dataset.root.name == "hb_dataset"
    # Segmentation source has no CSV but still yields samples and a split.
    assert len(pm.segmentation_dataset.load()) > 0
    assert sum(len(v) for v in pm.segmentation_dataset.split().values()) == 8
    assert sum(len(v) for v in pm.dataset.split().values()) == 12


def test_pipeline_without_seg_root_reuses_main(configs_dir: Path, tmp_path: Path) -> None:
    hb = tmp_path / "hb"
    generate_synthetic_dataset(hb, num_patients=6, seed=3, tissues=["palm"])
    cfg = ConfigLoader(configs_dir).load()
    cfg.section("dataset")["dataset"]["tissues"] = ["palm"]
    pipeline = HbPipeline(cfg, base_dir=tmp_path, dataset_root=hb).initialize()
    # No segmentation_source -> the segmentation dataset IS the main dataset.
    assert pipeline.manager.segmentation_dataset is pipeline.manager.dataset


def test_experiment_runs_with_separate_seg_dataset(
    configs_dir: Path, tmp_path: Path, two_datasets: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTIVEHB_FORCE_REFERENCE", "1")  # torch-free reference models
    seg, hb = two_datasets
    cfg = _config_with_seg_root(configs_dir, seg)
    pipeline = HbPipeline(cfg, base_dir=tmp_path, dataset_root=hb)
    result = pipeline.experiment("multi_ds", epochs=2)
    assert result.experiment_id
    assert result.report_paths  # a report was archived
