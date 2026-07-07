"""Unit tests for the data-driven pipeline modes on a synthetic dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.exceptions import PipelineError
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.pipeline import HbPipeline

# 3 segmentation architectures + 4 prediction tissues = 7 models per run.
_MODELS_PER_RUN = 7


@pytest.fixture()
def pipeline(framework_config: FrameworkConfig, tmp_path: Path) -> HbPipeline:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=8, seed=2)
    return HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()


def test_training_trains_and_registers_all_models(pipeline: HbPipeline) -> None:
    result = pipeline.train(epochs=2)
    assert set(result["jobs"].values()) == {"completed"}
    assert len(result["segmentation"]) == 3
    assert len(result["prediction"]) == 4
    assert result["registry"]["total"] == _MODELS_PER_RUN
    # Every trained model received a registry ID.
    assert all(v["registered_id"] for v in result["segmentation"].values())


def test_evaluation_computes_hb_metrics(pipeline: HbPipeline) -> None:
    pipeline.train(epochs=2)
    result = pipeline.evaluate()
    assert result["num_samples"] >= 1
    assert "mae" in result["metrics"] and "rmse" in result["metrics"]
    assert "clinical" in result["metrics"]
    assert set(result["reports"]) >= {"json", "csv"}


def test_evaluation_without_models_raises(pipeline: HbPipeline) -> None:
    with pytest.raises(PipelineError):
        pipeline.evaluate()


def test_inference_produces_predictions(pipeline: HbPipeline) -> None:
    pipeline.train(epochs=1)
    result = pipeline.predict()
    assert result["num_predictions"] >= 1
    first = result["predictions"][0]
    assert isinstance(first["predicted_hb"], float)
    assert "true_hb" in first


def test_resume_registers_new_versions(pipeline: HbPipeline) -> None:
    pipeline.train(epochs=2)
    result = pipeline.resume(epochs=3)
    assert set(result["jobs"].values()) == {"completed"}
    # A second training run creates new versions without overwriting the first.
    assert result["registry"]["total"] == 2 * _MODELS_PER_RUN


def test_training_on_invalid_dataset_raises(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    root = tmp_path / "bad_ds"
    generate_synthetic_dataset(root, num_patients=5, seed=1)
    _blank_hemoglobin(root / "metadata" / "patients.csv")
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()
    with pytest.raises(PipelineError):
        pipeline.train(epochs=1)


def _blank_hemoglobin(csv_path: Path) -> None:
    """Rewrite the metadata CSV with all Hemoglobin values blanked."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for row in rows:
        row["Hemoglobin"] = ""
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
