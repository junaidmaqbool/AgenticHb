"""Tests for segmentation evaluation metrics."""

from __future__ import annotations

import pytest

from adaptivehb.exceptions import SegmentationError
from adaptivehb.segmentation import metrics as M


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #

def test_confusion_matrix_counts() -> None:
    pred = [[1, 1], [0, 0]]
    target = [[1, 0], [0, 0]]
    cm = M.confusion_matrix(pred, target, 2)
    # rows = target, cols = pred. target0 pixels: 3 (two predicted 0, one predicted 1);
    # target1 pixels: 1 (predicted 1).
    assert cm == [[2, 1], [0, 1]]


def test_confusion_matrix_size_mismatch_raises() -> None:
    with pytest.raises(SegmentationError):
        M.confusion_matrix([[1, 0]], [[1, 0, 1]], 2)


def test_confusion_matrix_bad_num_classes() -> None:
    with pytest.raises(SegmentationError):
        M.confusion_matrix([[0]], [[0]], 0)


# --------------------------------------------------------------------------- #
# Core metrics against hand-computed values
# --------------------------------------------------------------------------- #

def test_perfect_prediction_scores_one() -> None:
    m = [[1, 1], [0, 0]]
    assert M.iou_score(m, m, 2) == 1.0
    assert M.dice_score(m, m, 2) == 1.0
    assert M.pixel_accuracy(m, m, 2) == 1.0


def test_partial_prediction_known_values() -> None:
    pred = [[1, 1], [0, 0]]
    target = [[1, 0], [0, 0]]
    b = M.segmentation_metrics(pred, target, 2)
    # class0 IoU = 2/3, class1 IoU = 1/2 -> mean 0.5833; pixel acc 3/4.
    assert b["per_class_iou"][0] == pytest.approx(2 / 3)
    assert b["per_class_iou"][1] == pytest.approx(0.5)
    assert b["mean_iou"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert b["pixel_accuracy"] == pytest.approx(0.75)
    # Dice: class0 = 0.8, class1 = 2/3.
    assert b["mean_dice"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_disjoint_prediction_zero_iou() -> None:
    pred = [[1, 1], [1, 1]]
    target = [[0, 0], [0, 0]]
    b = M.segmentation_metrics(pred, target, 2)
    # No overlap on either class -> IoU 0 for both present classes.
    assert b["mean_iou"] == 0.0
    assert b["pixel_accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# Absent classes, ignore_index
# --------------------------------------------------------------------------- #

def test_absent_class_excluded_from_mean() -> None:
    m = [[0, 1], [1, 0]]  # only classes 0 and 1 used
    b = M.segmentation_metrics(m, m, 3)  # class 2 never appears
    assert b["per_class_iou"][2] is None
    assert b["mean_iou"] == 1.0  # perfect on the present classes


def test_ignore_index_skips_pixels() -> None:
    pred = [[1, 1], [0, 0]]
    target = [[1, 255], [0, 0]]
    # The (0,1) pixel is ignored; the remaining 3 are all correct.
    assert M.pixel_accuracy(pred, target, 2, ignore_index=255) == 1.0


def test_all_ignored_is_zero_not_error() -> None:
    b = M.segmentation_metrics([[0]], [[255]], 2, ignore_index=255)
    assert b["mean_iou"] == 0.0
    assert b["pixel_accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# Accumulator
# --------------------------------------------------------------------------- #

def test_accumulator_matches_single_pass() -> None:
    acc = M.SegmentationMetrics(2)
    acc.update([[1, 0]], [[1, 0]])
    acc.update([[0, 1]], [[0, 0]])
    combined = acc.compute()
    # Equivalent to scoring the concatenation [1,0,0,1] vs [1,0,0,0].
    single = M.segmentation_metrics([1, 0, 0, 1], [1, 0, 0, 0], 2)
    assert combined["mean_iou"] == pytest.approx(single["mean_iou"])
    assert combined["pixel_accuracy"] == pytest.approx(single["pixel_accuracy"])


def test_accumulator_reset() -> None:
    acc = M.SegmentationMetrics(2)
    acc.update([[1, 1]], [[1, 1]])
    acc.reset()
    assert acc.confusion == [[0, 0], [0, 0]]
    assert acc.compute()["pixel_accuracy"] == 0.0


def test_accumulator_bad_num_classes() -> None:
    with pytest.raises(SegmentationError):
        M.SegmentationMetrics(0)


# --------------------------------------------------------------------------- #
# Multi-class + binarize + numpy-optional
# --------------------------------------------------------------------------- #

def test_multiclass_frequency_weighted_iou() -> None:
    # 3-class perfect prediction -> all aggregate metrics 1.0.
    m = [[0, 1, 2], [2, 1, 0]]
    b = M.segmentation_metrics(m, m, 3)
    assert b["mean_iou"] == 1.0
    assert b["frequency_weighted_iou"] == pytest.approx(1.0)
    assert b["mean_dice"] == 1.0


def test_binarize_threshold() -> None:
    assert M.binarize([[0.2, 0.9], [0.5, 0.49]], 0.5) == [[0, 1], [1, 0]]


def test_accepts_numpy_arrays_if_available() -> None:
    np = pytest.importorskip("numpy")
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [0, 0]])
    b = M.segmentation_metrics(pred, target, 2)
    assert b["pixel_accuracy"] == pytest.approx(0.75)
