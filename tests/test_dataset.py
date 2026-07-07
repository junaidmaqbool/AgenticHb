"""Unit tests for the DatasetManager, validator, and statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.dataset import (
    DatasetManager,
    Sample,
    Severity,
    generate_synthetic_dataset,
)
from adaptivehb.dataset.config import DatasetConfig
from adaptivehb.dataset.metadata import MetadataTable
from adaptivehb.dataset.validation import DatasetValidator

_COLUMNS = ["Patient_ID", "Hemoglobin", "Age", "Gender"]


def _table(rows: list[dict[str, str]], columns: list[str] | None = None) -> MetadataTable:
    return MetadataTable(rows, columns or _COLUMNS, "Patient_ID")


def _image_sample(pid: str, tissue: str = "eye", mask: bool = True) -> Sample:
    return Sample(
        patient_id=pid,
        tissue=tissue,
        image_path=f"/tmp/{pid}_{tissue}.png",
        mask_path=f"/tmp/{pid}_{tissue}_mask.png" if mask else None,
    )


# --------------------------------------------------------------------------- #
# Validator (no filesystem needed)
# --------------------------------------------------------------------------- #

def test_valid_dataset_has_no_errors() -> None:
    rows = [{"Patient_ID": "P1", "Hemoglobin": "12.5", "Age": "30", "Gender": "F"}]
    report = DatasetValidator(DatasetConfig()).validate(_table(rows), [_image_sample("P1")])
    assert report.is_valid
    assert report.num_masks == 1


def test_missing_mandatory_column_is_error() -> None:
    rows = [{"Patient_ID": "P1", "Hemoglobin": "12.5"}]
    report = DatasetValidator(DatasetConfig()).validate(
        _table(rows, ["Patient_ID", "Hemoglobin"]), [_image_sample("P1")]
    )
    assert not report.is_valid
    assert any(i.code == "missing_mandatory_columns" for i in report.errors)


def test_duplicate_patient_ids_is_error() -> None:
    rows = [
        {"Patient_ID": "P1", "Hemoglobin": "12", "Age": "30", "Gender": "F"},
        {"Patient_ID": "P1", "Hemoglobin": "13", "Age": "31", "Gender": "M"},
    ]
    report = DatasetValidator(DatasetConfig()).validate(_table(rows), [_image_sample("P1")])
    assert any(i.code == "duplicate_patient_ids" for i in report.errors)


def test_missing_and_invalid_hb_are_errors() -> None:
    rows = [
        {"Patient_ID": "P1", "Hemoglobin": "", "Age": "30", "Gender": "F"},
        {"Patient_ID": "P2", "Hemoglobin": "abc", "Age": "31", "Gender": "M"},
    ]
    report = DatasetValidator(DatasetConfig()).validate(
        _table(rows), [_image_sample("P1"), _image_sample("P2")]
    )
    codes = {i.code for i in report.errors}
    assert "missing_hemoglobin" in codes
    assert "invalid_hemoglobin" in codes


def test_orphan_and_missing_images_are_warnings() -> None:
    rows = [{"Patient_ID": "P1", "Hemoglobin": "12", "Age": "30", "Gender": "F"}]
    # P1 has metadata but no image; P9 has an image but no metadata.
    report = DatasetValidator(DatasetConfig()).validate(_table(rows), [_image_sample("P9")])
    codes = {i.code for i in report.warnings}
    assert "orphan_images" in codes
    assert "patients_without_images" in codes
    assert report.is_valid  # warnings do not invalidate


def test_missing_masks_is_warning() -> None:
    rows = [{"Patient_ID": "P1", "Hemoglobin": "12", "Age": "30", "Gender": "F"}]
    report = DatasetValidator(DatasetConfig()).validate(
        _table(rows), [_image_sample("P1", mask=False)]
    )
    assert any(i.code == "missing_masks" for i in report.warnings)
    assert report.is_valid  # a missing mask is only a warning


# --------------------------------------------------------------------------- #
# DatasetManager end-to-end on a synthetic dataset
# --------------------------------------------------------------------------- #

@pytest.fixture()
def dataset_manager(framework_config: FrameworkConfig, tmp_path: Path) -> DatasetManager:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=10, seed=1)
    manager = DatasetManager(framework_config, base_dir=tmp_path, dataset_root=root)
    manager.initialize()
    return manager


def test_index_covers_all_images(dataset_manager: DatasetManager) -> None:
    index = dataset_manager.load()
    # 10 patients * (eye2 + palm2 + tongue1 + nail2) = 70 samples.
    assert len(index) == 70
    assert all(s.mask_path for s in index)
    assert all(s.hb is not None for s in index)


def test_manager_validation_passes(dataset_manager: DatasetManager) -> None:
    report = dataset_manager.validate()
    assert report.is_valid
    assert report.num_patients == 10
    assert report.num_images == 70


def test_split_tags_samples_without_leakage(dataset_manager: DatasetManager) -> None:
    dataset_manager.split()
    train = dataset_manager.samples("train")
    test = dataset_manager.samples("test")
    train_patients = {s.patient_id for s in train}
    test_patients = {s.patient_id for s in test}
    assert train_patients.isdisjoint(test_patients)
    assert all(s.split is not None for s in dataset_manager.samples())


def test_statistics_are_computed(dataset_manager: DatasetManager) -> None:
    stats = dataset_manager.statistics()
    assert stats.num_patients == 10
    assert stats.images_per_tissue["eye"] == 20
    assert set(stats.hb) == {"count", "min", "max", "mean", "std"}
    assert sum(stats.gender_distribution.values()) == 10


def test_summary_and_export(dataset_manager: DatasetManager) -> None:
    dataset_manager.split()
    summary = dataset_manager.summary()
    assert summary["valid"] is True
    assert sum(summary["split_sizes"].values()) == 10

    report = dataset_manager.validate()
    path = dataset_manager.export_report(report)
    assert path.is_file()
