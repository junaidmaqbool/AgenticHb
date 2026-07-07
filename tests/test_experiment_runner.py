"""Unit tests for the ExperimentRunner (baseline vs adaptive, archived outputs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.experiment import ExperimentResult, ExperimentRunner
from adaptivehb.pipeline import HbPipeline
from adaptivehb.reporting import figures_available


@pytest.fixture()
def pipeline(framework_config: FrameworkConfig, tmp_path: Path) -> HbPipeline:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=10, seed=2)
    return HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root)


def test_experiment_runs_and_archives(pipeline: HbPipeline) -> None:
    result = ExperimentRunner(pipeline).run("unit_experiment", epochs=2)
    assert isinstance(result, ExperimentResult)
    assert result.experiment_id.startswith("unit_experiment_")

    root = Path(result.root)
    assert root.is_dir()
    # Archived outputs.
    assert (root / "metrics" / "adaptive_metrics.json").is_file()
    assert (root / "metrics" / "comparison.json").is_file()
    assert (root / "predictions" / "predictions.csv").is_file()
    assert Path(result.summary_path).is_file()


def test_experiment_metrics_and_comparison(pipeline: HbPipeline) -> None:
    result = ExperimentRunner(pipeline).run("cmp", epochs=1)
    assert "mae" in result.metrics and "rmse" in result.metrics
    comparison = result.comparison
    assert set(comparison) >= {"metric", "baseline", "adaptive", "improvement", "adaptive_better"}
    assert comparison["metric"] == "mae"
    # Baseline and adaptive are evaluated on the same test patients.
    assert comparison["baseline"] >= 0.0 and comparison["adaptive"] >= 0.0


def test_experiment_tables_and_figures(pipeline: HbPipeline) -> None:
    result = ExperimentRunner(pipeline).run("assets", epochs=1)
    assert Path(result.tables["csv"]).is_file()
    if figures_available():
        assert "comparison" in result.figures and result.figures["comparison"]
        assert all(Path(p).is_file() for paths in result.figures.values() for p in paths)
    else:
        assert result.figures == {}


def test_facade_experiment_method(pipeline: HbPipeline) -> None:
    result = pipeline.experiment("via_facade", epochs=1)
    assert isinstance(result, ExperimentResult)
    assert "metrics" in result.to_dict()
    assert result.to_dict()["comparison"]["metric"] == "mae"
