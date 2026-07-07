"""Paired statistical significance testing for baseline-vs-adaptive comparison.

The project's core scientific claim is that the adaptive (agent-fused) pipeline
improves on a static baseline (Decision 008, EXPERIMENT_SPEC Ch.14). A point
difference in MAE is not publishable on its own — a high-impact journal expects a
*paired* significance test, a confidence interval, and an effect size, because
baseline and adaptive predictions are made on the **same** patients (Decision 031).

This module provides those, computed on the per-sample absolute-error differences

    d_i = |baseline_i - true_i| - |adaptive_i - true_i|          (positive => adaptive better)

with pure standard-library math (no numpy/scipy), so the evaluation subsystem
stays torch-free and importable everywhere. Provided tests:

* paired two-sided Student t-test (t distribution CDF via the regularized
  incomplete beta function),
* Wilcoxon signed-rank test (normal approximation with tie/zero handling),
* a bootstrap confidence interval for the mean paired difference, and
* Cohen's d for paired samples (effect size).

Degenerate inputs (fewer than two pairs, zero variance, all-zero differences) are
handled gracefully rather than raising, so an experiment never crashes on an easy
or tiny split.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import fmean

from adaptivehb.exceptions import EvaluationError

Numbers = Sequence[float]


# --------------------------------------------------------------------------- #
# Paired error differences
# --------------------------------------------------------------------------- #

def paired_absolute_error_diffs(
    y_true: Numbers, baseline_pred: Numbers, adaptive_pred: Numbers
) -> list[float]:
    """Return per-sample absolute-error differences (baseline minus adaptive).

    A positive value means the adaptive pipeline was closer to the truth on that
    sample. This is the paired quantity every test in this module operates on.

    Args:
        y_true: Ground-truth hemoglobin values.
        baseline_pred: Baseline predictions (same order/length as ``y_true``).
        adaptive_pred: Adaptive predictions (same order/length as ``y_true``).

    Returns:
        The list of paired differences ``|baseline - true| - |adaptive - true|``.

    Raises:
        EvaluationError: If the three sequences differ in length or are empty.
    """
    if not (len(y_true) == len(baseline_pred) == len(adaptive_pred)):
        raise EvaluationError("y_true, baseline_pred, and adaptive_pred must have equal length.")
    if not y_true:
        raise EvaluationError("Cannot compute paired differences on empty inputs.")
    return [
        abs(b - t) - abs(a - t)
        for t, b, a in zip(y_true, baseline_pred, adaptive_pred)
    ]


# --------------------------------------------------------------------------- #
# Paired Student t-test
# --------------------------------------------------------------------------- #

def paired_t_test(diffs: Numbers) -> dict[str, float]:
    """Two-sided paired Student t-test on the differences ``diffs``.

    Tests H0: mean difference == 0. Returns the t statistic, degrees of freedom,
    and the two-sided p-value. With fewer than two samples or zero variance the
    p-value is 1.0 (no evidence against H0) and the t statistic is 0.0.
    """
    n = len(diffs)
    if n < 2:
        return {"t_statistic": 0.0, "df": 0.0, "p_value": 1.0, "n": float(n)}
    mean_d = fmean(diffs)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d <= 0.0:
        # No variance: a non-zero mean is a perfectly consistent effect; a zero
        # mean is no effect. Report a degenerate-but-honest result.
        p = 0.0 if mean_d != 0.0 else 1.0
        return {"t_statistic": math.inf if mean_d != 0 else 0.0, "df": float(n - 1),
                "p_value": p, "n": float(n)}
    se = math.sqrt(var_d / n)
    t = mean_d / se
    df = n - 1
    p = _student_t_sf_two_sided(t, df)
    return {"t_statistic": t, "df": float(df), "p_value": p, "n": float(n)}


def _student_t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided survival probability for Student's t with ``df`` d.o.f."""
    x = df / (df + t * t)
    # P(|T| >= |t|) = I_x(df/2, 1/2)
    return _reg_incomplete_beta(x, df / 2.0, 0.5)


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank test (normal approximation)
# --------------------------------------------------------------------------- #

def wilcoxon_signed_rank(diffs: Numbers) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test (normal approximation).

    A non-parametric paired test on the differences. Zero differences are dropped
    (Wilcoxon convention); tied magnitudes receive average ranks and the variance
    is tie-corrected. Returns the W statistic, the z score, the p-value, and the
    number of non-zero pairs used. With fewer than two non-zero pairs the p-value
    is 1.0.
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n < 2:
        return {"statistic": 0.0, "z": 0.0, "p_value": 1.0, "n": float(n)}
    magnitudes = [abs(d) for d in nonzero]
    ranks = _average_ranks(magnitudes)
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    tie_correction = _tie_correction(magnitudes)
    var_w = (n * (n + 1) * (2 * n + 1) - tie_correction) / 24.0
    if var_w <= 0.0:
        return {"statistic": w, "z": 0.0, "p_value": 1.0, "n": float(n)}
    # Continuity correction toward the mean.
    z = (w - mean_w + 0.5) / math.sqrt(var_w) if w < mean_w else (w - mean_w - 0.5) / math.sqrt(var_w)
    p = 2.0 * _standard_normal_sf(abs(z))
    return {"statistic": w, "z": z, "p_value": min(1.0, p), "n": float(n)}


def _tie_correction(magnitudes: Numbers) -> float:
    """Sum of (t^3 - t) over groups of tied magnitudes (Wilcoxon variance term)."""
    counts: dict[float, int] = {}
    for m in magnitudes:
        counts[m] = counts.get(m, 0) + 1
    return float(sum(c**3 - c for c in counts.values() if c > 1))


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval
# --------------------------------------------------------------------------- #

def bootstrap_ci_mean(
    diffs: Numbers, *, iterations: int = 5000, confidence: float = 0.95, seed: int = 0
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for the mean paired difference.

    Args:
        diffs: Paired differences.
        iterations: Number of bootstrap resamples.
        confidence: Two-sided confidence level (e.g. 0.95).
        seed: Seed for the deterministic resampler (reproducibility).

    Returns:
        The observed ``mean`` and the ``ci_lower``/``ci_upper`` bounds at the
        requested ``confidence``. With fewer than two samples the interval
        collapses to the point estimate.
    """
    n = len(diffs)
    mean_d = fmean(diffs) if diffs else 0.0
    if n < 2:
        return {"mean": mean_d, "ci_lower": mean_d, "ci_upper": mean_d,
                "confidence": confidence, "iterations": 0.0}
    rng = random.Random(seed)
    values = list(diffs)
    means = []
    for _ in range(iterations):
        resample = (values[rng.randrange(n)] for _ in range(n))
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = _percentile(means, alpha)
    upper = _percentile(means, 1.0 - alpha)
    return {"mean": mean_d, "ci_lower": lower, "ci_upper": upper,
            "confidence": confidence, "iterations": float(iterations)}


# --------------------------------------------------------------------------- #
# Effect size
# --------------------------------------------------------------------------- #

def cohens_d_paired(diffs: Numbers) -> float:
    """Cohen's d for paired samples (mean difference / SD of differences).

    Returns 0.0 with fewer than two samples or zero variance.
    """
    n = len(diffs)
    if n < 2:
        return 0.0
    mean_d = fmean(diffs)
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    return mean_d / sd if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #

def compare_significance(
    y_true: Numbers,
    baseline_pred: Numbers,
    adaptive_pred: Numbers,
    *,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """Full paired-significance bundle for baseline vs adaptive predictions.

    Computes the paired absolute-error differences and runs the paired t-test,
    the Wilcoxon signed-rank test, a bootstrap CI for the mean difference, and
    Cohen's d. All values are keyed for direct JSON archival in the experiment
    directory (EXPERIMENT_SPEC Ch.14).

    Returns:
        A mapping with ``n_pairs``, ``mean_abs_error_diff`` (positive => adaptive
        better on average), ``paired_t_test``, ``wilcoxon``, ``bootstrap_ci``,
        ``cohens_d``, and a convenience ``significant_at`` flag (t-test p < 1-conf).
    """
    diffs = paired_absolute_error_diffs(y_true, baseline_pred, adaptive_pred)
    t_test = paired_t_test(diffs)
    bundle = {
        "n_pairs": len(diffs),
        "mean_abs_error_diff": fmean(diffs),
        "paired_t_test": t_test,
        "wilcoxon": wilcoxon_signed_rank(diffs),
        "bootstrap_ci": bootstrap_ci_mean(
            diffs, iterations=bootstrap_iterations, confidence=confidence, seed=seed
        ),
        "cohens_d": cohens_d_paired(diffs),
        "significant_at": bool(t_test["p_value"] < (1.0 - confidence)),
        "alpha": 1.0 - confidence,
    }
    return bundle


# --------------------------------------------------------------------------- #
# Numerical helpers (stdlib only)
# --------------------------------------------------------------------------- #

def _standard_normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def _average_ranks(values: Numbers) -> list[float]:
    """Return fractional (average) ranks, so ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) (Numerical Recipes).

    Used for the Student t distribution tail. Returns a probability in [0, 1].
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(math.log(x) * a + math.log(1.0 - x) * b + ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_cf(x, a, b) / a
    return 1.0 - bt * _beta_cf(1.0 - x, b, a) / b


def _beta_cf(x: float, a: float, b: float) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    tiny = 1e-30
    max_iter = 300
    eps = 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


__all__ = [
    "paired_absolute_error_diffs",
    "paired_t_test",
    "wilcoxon_signed_rank",
    "bootstrap_ci_mean",
    "cohens_d_paired",
    "compare_significance",
]
