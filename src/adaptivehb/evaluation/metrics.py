"""Evaluation metrics for hemoglobin estimation (EXPERIMENT_SPEC Ch.10, Ch.13-17).

Pure standard-library implementations (no numpy/scipy required) so evaluation is
importable and testable without the ML stack. Covers regression, clinical
agreement (Bland-Altman, within-bands), classification (anemia thresholding), and
calibration (expected calibration error). Degenerate inputs (e.g. zero-variance
predictions) are handled gracefully rather than raising.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean, pstdev

from adaptivehb.exceptions import EvaluationError

Numbers = Sequence[float]


def _validate(y_true: Numbers, y_pred: Numbers) -> None:
    if len(y_true) != len(y_pred):
        raise EvaluationError("y_true and y_pred must have equal length.")
    if not y_true:
        raise EvaluationError("Cannot compute metrics on empty inputs.")


# -- regression -------------------------------------------------------------- #


def mae(y_true: Numbers, y_pred: Numbers) -> float:
    """Mean absolute error."""
    _validate(y_true, y_pred)
    return fmean(abs(p - t) for t, p in zip(y_true, y_pred))


def rmse(y_true: Numbers, y_pred: Numbers) -> float:
    """Root mean squared error."""
    _validate(y_true, y_pred)
    return math.sqrt(fmean((p - t) ** 2 for t, p in zip(y_true, y_pred)))


def mean_bias(y_true: Numbers, y_pred: Numbers) -> float:
    """Mean signed error (prediction minus truth)."""
    _validate(y_true, y_pred)
    return fmean(p - t for t, p in zip(y_true, y_pred))


def r2_score(y_true: Numbers, y_pred: Numbers) -> float:
    """Coefficient of determination (0.0 when the truth has zero variance)."""
    _validate(y_true, y_pred)
    mean_t = fmean(y_true)
    ss_tot = sum((t - mean_t) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def pearson(x: Numbers, y: Numbers) -> float:
    """Pearson correlation (0.0 when either series has zero variance)."""
    _validate(x, y)
    mx, my = fmean(x), fmean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def spearman(x: Numbers, y: Numbers) -> float:
    """Spearman rank correlation."""
    _validate(x, y)
    return pearson(_average_ranks(x), _average_ranks(y))


def regression_metrics(y_true: Numbers, y_pred: Numbers) -> dict[str, float]:
    """Return the standard regression metric bundle."""
    _validate(y_true, y_pred)
    residuals = [p - t for t, p in zip(y_true, y_pred)]
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "mean_bias": fmean(residuals),
        "std_diff": pstdev(residuals) if len(residuals) > 1 else 0.0,
    }


# -- clinical ---------------------------------------------------------------- #


def bland_altman(y_true: Numbers, y_pred: Numbers) -> dict[str, float]:
    """Bland-Altman bias and 95% limits of agreement."""
    _validate(y_true, y_pred)
    diffs = [p - t for t, p in zip(y_true, y_pred)]
    bias = fmean(diffs)
    sd = pstdev(diffs) if len(diffs) > 1 else 0.0
    return {"bias": bias, "sd": sd, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd}


def within_band(y_true: Numbers, y_pred: Numbers, band: float) -> float:
    """Fraction of predictions within ``band`` g/dL of the truth."""
    _validate(y_true, y_pred)
    hits = sum(1 for t, p in zip(y_true, y_pred) if abs(p - t) <= band)
    return hits / len(y_true)


def clinical_metrics(y_true: Numbers, y_pred: Numbers, bands: Sequence[float]) -> dict[str, object]:
    """Return Bland-Altman plus within-band fractions for each band."""
    return {
        "bland_altman": bland_altman(y_true, y_pred),
        "within": {f"{band}": within_band(y_true, y_pred, float(band)) for band in bands},
    }


# -- classification (anemia thresholding) ------------------------------------ #


def classification_metrics(y_true: Numbers, y_pred: Numbers, threshold: float) -> dict[str, float]:
    """Anemia classification metrics (positive = below ``threshold``)."""
    _validate(y_true, y_pred)
    true_pos_flags = [t < threshold for t in y_true]
    pred_pos_flags = [p < threshold for p in y_pred]
    tp = sum(1 for a, b in zip(true_pos_flags, pred_pos_flags) if a and b)
    fp = sum(1 for a, b in zip(true_pos_flags, pred_pos_flags) if not a and b)
    fn = sum(1 for a, b in zip(true_pos_flags, pred_pos_flags) if a and not b)
    tn = sum(1 for a, b in zip(true_pos_flags, pred_pos_flags) if not a and not b)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -- calibration ------------------------------------------------------------- #


def expected_calibration_error(
    confidences: Numbers, correct: Sequence[bool], bins: int = 10
) -> float:
    """Expected calibration error over ``bins`` equal-width confidence bins."""
    if len(confidences) != len(correct):
        raise EvaluationError("confidences and correct must have equal length.")
    if not confidences:
        raise EvaluationError("Cannot compute calibration on empty inputs.")
    total = len(confidences)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (c, hit)
            for c, hit in zip(confidences, correct)
            if (low < c <= high) or (index == 0 and c <= high)
        ]
        if not members:
            continue
        avg_conf = fmean(c for c, _ in members)
        accuracy = fmean(1.0 if hit else 0.0 for _, hit in members)
        ece += (len(members) / total) * abs(avg_conf - accuracy)
    return ece


# -- helpers ----------------------------------------------------------------- #


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


__all__ = [
    "mae",
    "rmse",
    "mean_bias",
    "r2_score",
    "pearson",
    "spearman",
    "regression_metrics",
    "bland_altman",
    "within_band",
    "clinical_metrics",
    "classification_metrics",
    "expected_calibration_error",
]
