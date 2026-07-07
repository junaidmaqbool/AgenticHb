"""Tests for paired significance testing (evaluation.significance)."""

from __future__ import annotations

import math

import pytest

from adaptivehb.evaluation import significance as S
from adaptivehb.evaluation.config import EvaluationConfig
from adaptivehb.evaluation.manager import EvaluationManager
from adaptivehb.exceptions import EvaluationError


# --------------------------------------------------------------------------- #
# Paired difference construction
# --------------------------------------------------------------------------- #

def test_paired_abs_error_diffs_positive_when_adaptive_closer() -> None:
    y_true = [10.0, 12.0, 14.0]
    baseline = [11.0, 13.0, 15.0]  # off by 1.0
    adaptive = [10.2, 12.2, 14.2]  # off by 0.2
    diffs = S.paired_absolute_error_diffs(y_true, baseline, adaptive)
    assert all(d == pytest.approx(0.8) for d in diffs)


def test_paired_abs_error_diffs_length_mismatch_raises() -> None:
    with pytest.raises(EvaluationError):
        S.paired_absolute_error_diffs([1.0, 2.0], [1.0], [1.0, 2.0])


def test_paired_abs_error_diffs_empty_raises() -> None:
    with pytest.raises(EvaluationError):
        S.paired_absolute_error_diffs([], [], [])


# --------------------------------------------------------------------------- #
# Paired t-test — validated against known reference values
# --------------------------------------------------------------------------- #

def test_student_t_cdf_matches_reference() -> None:
    # Two-sided p for t=2.0, df=10 is ~0.0734 (scipy.stats.t.sf*2).
    assert S._student_t_sf_two_sided(2.0, 10) == pytest.approx(0.0734, abs=1e-3)
    # t=1.0, df=1 gives exactly 0.5.
    assert S._student_t_sf_two_sided(1.0, 1) == pytest.approx(0.5, abs=1e-6)


def test_paired_t_test_matches_reference() -> None:
    diffs = [0.5, 1.0, 1.5, 0.0, 2.0, 1.0, 0.5, 1.5]
    result = S.paired_t_test(diffs)
    # scipy.stats.ttest_1samp(diffs, 0): t≈4.3205, p≈0.00348.
    assert result["t_statistic"] == pytest.approx(4.3205, abs=1e-3)
    assert result["p_value"] == pytest.approx(0.00348, abs=1e-4)
    assert result["df"] == 7.0


def test_paired_t_test_degenerate_inputs() -> None:
    assert S.paired_t_test([1.0])["p_value"] == 1.0
    # Consistent non-zero effect (zero variance) -> strong evidence.
    assert S.paired_t_test([2.0, 2.0, 2.0])["p_value"] == 0.0
    # No effect (all zero) -> no evidence.
    assert S.paired_t_test([0.0, 0.0, 0.0])["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank
# --------------------------------------------------------------------------- #

def test_wilcoxon_drops_zeros_and_reports_n() -> None:
    diffs = [0.5, 1.0, -0.3, 0.0, 2.0, 1.0]
    result = S.wilcoxon_signed_rank(diffs)
    assert result["n"] == 5.0  # the zero is dropped
    assert 0.0 <= result["p_value"] <= 1.0


def test_wilcoxon_all_zero_is_nonsignificant() -> None:
    result = S.wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert result["p_value"] == 1.0


def test_wilcoxon_strong_one_sided_effect() -> None:
    # All positive differences -> W=0, small p.
    result = S.wilcoxon_signed_rank([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert result["statistic"] == 0.0
    assert result["p_value"] < 0.05


# --------------------------------------------------------------------------- #
# Bootstrap CI + effect size
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_is_reproducible_and_brackets_mean() -> None:
    diffs = [0.5, 1.0, 1.5, 0.0, 2.0, 1.0, 0.5, 1.5]
    a = S.bootstrap_ci_mean(diffs, iterations=2000, seed=42)
    b = S.bootstrap_ci_mean(diffs, iterations=2000, seed=42)
    assert a == b  # deterministic given the seed
    assert a["ci_lower"] <= a["mean"] <= a["ci_upper"]


def test_bootstrap_ci_single_sample_collapses() -> None:
    result = S.bootstrap_ci_mean([1.5])
    assert result["ci_lower"] == result["ci_upper"] == result["mean"] == 1.5


def test_cohens_d_paired() -> None:
    diffs = [0.5, 1.0, 1.5, 0.0, 2.0, 1.0, 0.5, 1.5]
    # mean/sd of the sample (ddof=1): ~1.5275.
    assert S.cohens_d_paired(diffs) == pytest.approx(1.5275, abs=1e-3)
    assert S.cohens_d_paired([1.0]) == 0.0


# --------------------------------------------------------------------------- #
# Bundle + manager wiring
# --------------------------------------------------------------------------- #

def test_compare_significance_bundle_shape() -> None:
    y_true = [12.0, 13.0, 11.0, 14.0, 10.0, 13.5, 12.5, 11.5]
    baseline = [12.9, 13.9, 11.9, 14.9, 10.9, 14.4, 13.4, 12.4]
    adaptive = [12.1, 13.1, 11.05, 14.2, 10.05, 13.7, 12.55, 11.6]
    bundle = S.compare_significance(y_true, baseline, adaptive, bootstrap_iterations=500, seed=1)
    assert bundle["n_pairs"] == 8
    assert bundle["mean_abs_error_diff"] > 0  # adaptive closer on average
    assert set(bundle["paired_t_test"]) >= {"t_statistic", "p_value", "df"}
    assert "ci_lower" in bundle["bootstrap_ci"]
    assert isinstance(bundle["significant_at"], bool)


def test_manager_compare_attaches_significance(framework_config) -> None:
    manager = EvaluationManager(framework_config)
    manager.initialize()
    y_true = [12.0, 13.0, 11.0, 14.0, 10.0, 13.5, 12.5, 11.5]
    baseline_pred = [12.9, 13.9, 11.9, 14.9, 10.9, 14.4, 13.4, 12.4]
    adaptive_pred = [12.1, 13.1, 11.05, 14.2, 10.05, 13.7, 12.55, 11.6]
    base = manager.evaluate(y_true, baseline_pred, name="baseline")
    adapt = manager.evaluate(y_true, adaptive_pred, name="adaptive")
    comparison = manager.compare(
        base, adapt, metric="mae",
        y_true=y_true, baseline_pred=baseline_pred, adaptive_pred=adaptive_pred,
    )
    assert "significance" in comparison
    assert comparison["significance"]["n_pairs"] == 8


def test_manager_compare_without_arrays_has_no_significance(framework_config) -> None:
    manager = EvaluationManager(framework_config)
    manager.initialize()
    y_true = [12.0, 13.0, 11.0]
    base = manager.evaluate(y_true, [12.5, 13.5, 11.5], name="baseline")
    adapt = manager.evaluate(y_true, [12.1, 13.1, 11.1], name="adaptive")
    comparison = manager.compare(base, adapt, metric="mae")
    assert "significance" not in comparison  # backward compatible


def test_significance_can_be_disabled_via_config(framework_config) -> None:
    manager = EvaluationManager(framework_config)
    manager.initialize()
    # Force-disable through the typed config.
    disabled = EvaluationConfig(**{**manager.evaluation_config.__dict__, "significance_enabled": False})
    manager._eval_config = disabled  # type: ignore[attr-defined]
    y_true = [12.0, 13.0, 11.0]
    base = manager.evaluate(y_true, [12.5, 13.5, 11.5], name="baseline")
    adapt = manager.evaluate(y_true, [12.1, 13.1, 11.1], name="adaptive")
    comparison = manager.compare(
        base, adapt, metric="mae",
        y_true=y_true, baseline_pred=[12.5, 13.5, 11.5], adaptive_pred=[12.1, 13.1, 11.1],
    )
    assert "significance" not in comparison
