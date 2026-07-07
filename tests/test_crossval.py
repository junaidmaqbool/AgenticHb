"""Tests for patient-level k-fold cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptivehb.crossval import CrossValidationRunner
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.dataset.manager import DatasetManager
from adaptivehb.dataset.splitting import k_fold_split
from adaptivehb.exceptions import DatasetError, PipelineError
from adaptivehb.pipeline import HbPipeline


# --------------------------------------------------------------------------- #
# k_fold_split
# --------------------------------------------------------------------------- #

def test_k_fold_split_partitions_every_patient_once() -> None:
    patients = [f"P{i:02d}" for i in range(12)]
    folds = k_fold_split(patients, 4, seed=0)
    assert len(folds) == 4
    test_union = sorted(pid for fold in folds for pid in fold["test"])
    assert test_union == sorted(patients)  # each patient in exactly one test fold


def test_k_fold_split_is_patient_level_no_leakage() -> None:
    patients = [f"P{i:02d}" for i in range(10)]
    for fold in k_fold_split(patients, 5, seed=1):
        assert set(fold["train"]).isdisjoint(fold["test"])


def test_k_fold_split_balanced_and_deterministic() -> None:
    patients = [f"P{i:02d}" for i in range(12)]
    a = k_fold_split(patients, 3, seed=7)
    b = k_fold_split(patients, 3, seed=7)
    assert a == b  # deterministic for a given seed
    sizes = [len(fold["test"]) for fold in a]
    assert max(sizes) - min(sizes) <= 1  # balanced


def test_k_fold_split_validation_slice() -> None:
    patients = [f"P{i:02d}" for i in range(20)]
    fold = k_fold_split(patients, 4, seed=0, val_fraction=0.25)[0]
    assert fold["validation"]
    assert set(fold["validation"]).isdisjoint(fold["test"])
    assert set(fold["validation"]).isdisjoint(fold["train"])


def test_k_fold_split_rejects_bad_k() -> None:
    patients = [f"P{i}" for i in range(4)]
    with pytest.raises(DatasetError):
        k_fold_split(patients, 1, seed=0)
    with pytest.raises(DatasetError):
        k_fold_split(patients, 5, seed=0)  # k > number of patients


# --------------------------------------------------------------------------- #
# DatasetManager pinning
# --------------------------------------------------------------------------- #

def test_apply_split_survives_internal_split(framework_config, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=9, seed=2)
    manager = DatasetManager(framework_config, base_dir=tmp_path, dataset_root=root)
    manager.initialize()
    folds = k_fold_split(manager.load_metadata().patient_ids, 3, seed=0)
    manager.apply_split(folds[0])
    pinned = {k: len(v) for k, v in manager.split().items()}  # split() must honor the pin
    assert pinned == {k: len(v) for k, v in folds[0].items()}
    assert manager.samples("test")  # test patients are tagged
    manager.clear_pinned_split()
    # After clearing, split() draws a fresh (config-ratio) split again.
    assert set(manager.split()) == {"train", "validation", "test"}


# --------------------------------------------------------------------------- #
# CrossValidationRunner
# --------------------------------------------------------------------------- #

@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=12, seed=3)
    return root


def test_cross_validation_runs_all_folds(framework_config, tmp_path: Path, dataset_root: Path) -> None:
    runner = CrossValidationRunner(
        framework_config, base_dir=tmp_path, dataset_root=dataset_root, folds=3, seed=0
    )
    result = runner.run("cv_demo", epochs=2)
    assert result.k == 3
    assert len(result.per_fold) == 3
    # Each fold recorded its own experiment and test-patient count.
    assert all(f["experiment_id"] for f in result.per_fold)
    assert sum(f["num_test_patients"] for f in result.per_fold) == 12


def test_cross_validation_aggregates_metrics(framework_config, tmp_path: Path, dataset_root: Path) -> None:
    runner = CrossValidationRunner(
        framework_config, base_dir=tmp_path, dataset_root=dataset_root, folds=3, seed=0
    )
    result = runner.run("cv_agg", epochs=2)
    agg = result.aggregate
    assert "mae" in agg["metrics"]
    for stat in ("mean", "std", "min", "max"):
        assert stat in agg["metrics"]["mae"]
    assert agg["num_folds"] == 3
    assert 0 <= agg["folds_adaptive_better"] <= 3


def test_cross_validation_archives_outputs(framework_config, tmp_path: Path, dataset_root: Path) -> None:
    runner = CrossValidationRunner(
        framework_config, base_dir=tmp_path, dataset_root=dataset_root, folds=3, seed=0
    )
    result = runner.run("cv_files", epochs=2)
    root = Path(result.root)
    assert (root / "cv_summary.json").is_file()
    assert (root / "cv_metrics.csv").is_file()
    assert (root / "cv_report.md").is_file()
    assert sorted(p.name for p in root.glob("fold_*")) == ["fold_0", "fold_1", "fold_2"]
    summary = json.loads((root / "cv_summary.json").read_text())
    assert summary["k"] == 3 and len(summary["per_fold"]) == 3


def test_cross_validation_rejects_too_many_folds(framework_config, tmp_path: Path) -> None:
    root = tmp_path / "tiny"
    generate_synthetic_dataset(root, num_patients=3, seed=1)
    runner = CrossValidationRunner(
        framework_config, base_dir=tmp_path, dataset_root=root, folds=5, seed=0
    )
    with pytest.raises(PipelineError):
        runner.run("cv_toosmall", epochs=1)


def test_pipeline_cross_validate_facade(framework_config, tmp_path: Path, dataset_root: Path) -> None:
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=dataset_root)
    result = pipeline.cross_validate("facade_cv", folds=3, epochs=2)
    assert result.k == 3
    assert result.report_path.endswith("cv_report.md")
