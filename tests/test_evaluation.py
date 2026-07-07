"""Unit tests for the evaluation subsystem (metrics, manager, report)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.evaluation import EvaluationConfig, EvaluationManager, EvaluationReport
from adaptivehb.evaluation import metrics as M
from adaptivehb.exceptions import EvaluationError

# A monotonic, close-but-imperfect set of predictions.
_TRUE = [12.0, 13.0, 11.0, 14.0, 10.0]
_PRED = [12.2, 12.8, 11.5, 13.5, 10.5]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_perfect_predictions() -> None:
    assert M.mae(_TRUE, _TRUE) == 0.0
    assert M.rmse(_TRUE, _TRUE) == 0.0
    assert M.r2_score(_TRUE, _TRUE) == pytest.approx(1.0)
    assert M.pearson(_TRUE, _TRUE) == pytest.approx(1.0)


def test_regression_metric_values() -> None:
    metrics = M.regression_metrics(_TRUE, _PRED)
    assert metrics["mae"] == pytest.approx(0.38)
    assert metrics["rmse"] == pytest.approx(0.4074, abs=1e-3)
    assert 0.9 < metrics["pearson"] <= 1.0
    assert metrics["spearman"] == pytest.approx(1.0)  # order preserved


def test_constant_predictions_are_handled() -> None:
    # Zero-variance predictions must not raise; correlations degrade to 0.
    assert M.pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0
    assert M.spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0
    assert M.r2_score([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0  # zero-variance truth


def test_bland_altman_and_within_band() -> None:
    ba = M.bland_altman(_TRUE, _PRED)
    assert ba["loa_lower"] < ba["bias"] < ba["loa_upper"]
    assert M.within_band(_TRUE, _PRED, 0.5) == 1.0
    assert 0.0 <= M.within_band(_TRUE, _PRED, 0.1) <= 1.0


def test_classification_metrics() -> None:
    # Anemia threshold 12.0: true positives are values < 12.
    result = M.classification_metrics(_TRUE, _PRED, 12.0)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert set(result) == {"accuracy", "precision", "recall", "f1"}


def test_expected_calibration_error() -> None:
    ece = M.expected_calibration_error([0.9, 0.8, 0.6, 0.4], [True, True, False, False])
    assert 0.0 <= ece <= 1.0


def test_metrics_reject_bad_inputs() -> None:
    with pytest.raises(EvaluationError):
        M.mae([1.0, 2.0], [1.0])
    with pytest.raises(EvaluationError):
        M.mae([], [])


# --------------------------------------------------------------------------- #
# EvaluationManager + report
# --------------------------------------------------------------------------- #

@pytest.fixture()
def manager(framework_config: FrameworkConfig, tmp_path: Path) -> EvaluationManager:
    em = EvaluationManager(framework_config, base_dir=tmp_path)
    em.initialize()
    return em


def test_manager_evaluate_bundle(manager: EvaluationManager) -> None:
    report = manager.evaluate(_TRUE, _PRED, name="hb")
    assert isinstance(report, EvaluationReport)
    assert "mae" in report.metrics
    assert "clinical" in report.metrics
    assert "classification" in report.metrics
    assert "within" in report.metrics["clinical"]


def test_manager_export_writes_files(manager: EvaluationManager) -> None:
    rows = [{"patient_id": "P1", "true_hb": 12.0, "predicted_hb": 12.5}]
    report = manager.evaluate(_TRUE, _PRED, name="hb", per_sample_rows=rows)
    paths = manager.export(report)
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()


def test_manager_compare_baseline_vs_adaptive(manager: EvaluationManager) -> None:
    baseline = manager.evaluate(_TRUE, [12.5] * len(_TRUE), name="baseline")
    adaptive = manager.evaluate(_TRUE, _PRED, name="adaptive")
    comparison = manager.compare(baseline, adaptive, metric="mae")
    assert comparison["baseline"] >= comparison["adaptive"]  # adaptive has lower MAE here
    assert comparison["adaptive_better"] is True


def test_evaluation_config_parses(framework_config: FrameworkConfig) -> None:
    config = EvaluationConfig.from_section(framework_config.section("evaluation"))
    assert "mae" in config.regression_metrics
    assert config.within_thresholds == (0.5, 1.0)
    assert config.anemia_threshold > 0
