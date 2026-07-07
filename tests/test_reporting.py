"""Unit tests for the reporting subsystem (figures, tables, manager)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.exceptions import ReportingError
from adaptivehb.pipeline import HbPipeline
from adaptivehb.reporting import (
    FigureGenerator,
    ReportingManager,
    excel_available,
    export_table_csv,
    export_table_excel,
    figures_available,
    flatten_metrics,
)

_TRUE = [12.0, 13.0, 11.0, 14.0, 10.5, 12.8]
_PRED = [12.2, 12.7, 11.6, 13.4, 10.9, 12.5]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def test_figures_available_is_bool() -> None:
    assert isinstance(figures_available(), bool)


def test_figure_generation_writes_files(tmp_path: Path) -> None:
    if not figures_available():
        pytest.skip("matplotlib not installed")
    gen = FigureGenerator(("png", "pdf"))
    scatter = gen.scatter(_TRUE, _PRED, tmp_path / "scatter.png")
    bland = gen.bland_altman(_TRUE, _PRED, tmp_path / "ba.png")
    residuals = gen.residual_hist(_TRUE, _PRED, tmp_path / "res.png")
    assert {p.suffix for p in scatter} == {".png", ".pdf"}
    assert all(p.is_file() for p in scatter + bland + residuals)


def test_training_curve_and_comparison(tmp_path: Path) -> None:
    if not figures_available():
        pytest.skip("matplotlib not installed")
    gen = FigureGenerator(("png",))
    history = [{"train_loss": 1.0, "val_loss": 1.1}, {"train_loss": 0.4, "val_loss": 0.6}]
    curve = gen.training_curve(history, tmp_path / "curve.png")
    bars = gen.model_comparison({"unet": 0.4, "segformer": 0.5}, tmp_path / "cmp.png")
    assert curve[0].is_file() and bars[0].is_file()


def test_figure_bad_inputs_raise(tmp_path: Path) -> None:
    if not figures_available():
        pytest.skip("matplotlib not installed")
    gen = FigureGenerator(("png",))
    with pytest.raises(ReportingError):
        gen.scatter([], [], tmp_path / "x.png")
    with pytest.raises(ReportingError):
        gen.training_curve([{"other": 1.0}], tmp_path / "y.png")
    with pytest.raises(ReportingError):
        gen.model_comparison({}, tmp_path / "z.png")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def test_flatten_metrics_nested() -> None:
    rows = flatten_metrics({"mae": 0.3, "clinical": {"within": {"0.5": 0.8}}})
    by_name = {r["metric"]: r["value"] for r in rows}
    assert by_name["mae"] == 0.3
    assert by_name["clinical.within.0.5"] == 0.8


def test_export_csv(tmp_path: Path) -> None:
    rows = flatten_metrics({"mae": 0.3, "rmse": 0.4})
    path = export_table_csv(rows, tmp_path / "t.csv")
    assert path.is_file()
    assert "mae" in path.read_text()


def test_export_excel_availability(tmp_path: Path) -> None:
    rows = [{"metric": "mae", "value": 0.3}]
    if excel_available():
        assert export_table_excel(rows, tmp_path / "t.xlsx").is_file()
    else:
        with pytest.raises(ReportingError):
            export_table_excel(rows, tmp_path / "t.xlsx")


# --------------------------------------------------------------------------- #
# Manager + pipeline integration
# --------------------------------------------------------------------------- #

@pytest.fixture()
def reporting(framework_config: FrameworkConfig, tmp_path: Path) -> ReportingManager:
    manager = ReportingManager(framework_config, base_dir=tmp_path)
    manager.initialize()
    return manager


def test_manager_available_and_table(reporting: ReportingManager) -> None:
    availability = reporting.available()
    assert set(availability) == {"figures", "excel"}
    tables = reporting.export_metrics_table("hb", {"mae": 0.3, "clinical": {"within": {"0.5": 0.9}}})
    assert Path(tables["csv"]).is_file()


def test_manager_generate_figures(reporting: ReportingManager) -> None:
    figures = reporting.generate_evaluation_figures(_TRUE, _PRED, subdir="demo")
    if figures_available():
        assert "scatter" in figures and figures["scatter"]
    else:
        assert figures == {}


def test_pipeline_evaluation_emits_assets(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=6, seed=2)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()
    pipeline.train(epochs=2)
    result = pipeline.evaluate()
    assert "figures" in result and isinstance(result["figures"], dict)
    assert "csv" in result["tables"]
