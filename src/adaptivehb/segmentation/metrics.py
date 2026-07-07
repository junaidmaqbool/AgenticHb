"""Segmentation evaluation metrics (Decision 038).

Standard-library implementations of the region-overlap metrics used to evaluate
tissue segmentation — intersection-over-union (IoU / Jaccard), Dice (F1), pixel
accuracy, mean per-class accuracy, and frequency-weighted IoU — so segmentation can
be scored without numpy, torch, or any imaging dependency (mirroring the torch-free
design of the Hb regression metrics in ``adaptivehb.evaluation.metrics``).

All metrics are computed from a class confusion matrix, so binary and multi-class
masks are handled uniformly and a whole dataset can be scored incrementally through
the :class:`SegmentationMetrics` accumulator (``update`` per mask pair, ``compute``
at the end). Masks may be nested Python sequences or numpy arrays; probability maps
can be thresholded to labels with :func:`binarize`. A class that is absent from both
the prediction and the ground truth is excluded from the mean metrics (rather than
counted as a perfect or zero score), following common convention; degenerate inputs
never raise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from adaptivehb.exceptions import SegmentationError

Mask = Any  # nested sequence of ints, or a numpy array


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #

def _flatten_labels(mask: Mask) -> list[int]:
    """Flatten a mask (numpy array or nested sequence) into a list of int labels."""
    # numpy array (duck-typed to avoid a hard numpy dependency).
    if hasattr(mask, "ravel") and hasattr(mask, "tolist"):
        return [int(v) for v in mask.ravel().tolist()]
    flat: list[int] = []

    def _recurse(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for element in item:
                _recurse(element)
        else:
            flat.append(int(item))

    _recurse(mask)
    return flat


def confusion_matrix(
    pred: Mask, target: Mask, num_classes: int, *, ignore_index: int | None = None
) -> list[list[int]]:
    """Accumulate a ``num_classes × num_classes`` confusion matrix.

    Rows are ground-truth classes, columns are predicted classes. Pixels whose
    target equals ``ignore_index`` are skipped, as are pixels whose label falls
    outside ``[0, num_classes)``.

    Raises:
        SegmentationError: If the prediction and target have different sizes or
            ``num_classes < 1``.
    """
    if num_classes < 1:
        raise SegmentationError("num_classes must be >= 1.")
    p = _flatten_labels(pred)
    t = _flatten_labels(target)
    if len(p) != len(t):
        raise SegmentationError(
            f"Prediction and target sizes differ ({len(p)} vs {len(t)})."
        )
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pi, ti in zip(p, t):
        if ignore_index is not None and ti == ignore_index:
            continue
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            matrix[ti][pi] += 1
    return matrix


# --------------------------------------------------------------------------- #
# Metrics from a confusion matrix
# --------------------------------------------------------------------------- #

def _row_sum(matrix: list[list[int]], c: int) -> int:
    return sum(matrix[c])


def _col_sum(matrix: list[list[int]], c: int) -> int:
    return sum(row[c] for row in matrix)


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, Any]:
    """Compute the segmentation metric bundle from a confusion matrix.

    Returns per-class IoU/Dice/accuracy (with ``None`` for classes absent from
    both prediction and truth) and the aggregate ``mean_iou``, ``mean_dice``,
    ``pixel_accuracy``, ``mean_accuracy`` (mean per-class recall), and
    ``frequency_weighted_iou``.
    """
    num_classes = len(matrix)
    total = sum(sum(row) for row in matrix)
    per_class_iou: list[float | None] = []
    per_class_dice: list[float | None] = []
    per_class_acc: list[float | None] = []
    correct = 0

    for c in range(num_classes):
        tp = matrix[c][c]
        correct += tp
        row = _row_sum(matrix, c)  # ground-truth pixels of class c
        col = _col_sum(matrix, c)  # predicted pixels of class c
        union = row + col - tp
        present = (row + col) > 0
        per_class_iou.append(tp / union if union > 0 else (None if not present else 1.0))
        per_class_dice.append((2 * tp) / (row + col) if (row + col) > 0 else None)
        per_class_acc.append(tp / row if row > 0 else None)

    iou_values = [v for v in per_class_iou if v is not None]
    dice_values = [v for v in per_class_dice if v is not None]
    acc_values = [v for v in per_class_acc if v is not None]

    # Frequency-weighted IoU: weight each present class by its ground-truth frequency.
    fw_iou = 0.0
    if total > 0:
        for c in range(num_classes):
            iou_c = per_class_iou[c]
            if iou_c is not None:
                fw_iou += (_row_sum(matrix, c) / total) * iou_c

    return {
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "per_class_accuracy": per_class_acc,
        "mean_iou": sum(iou_values) / len(iou_values) if iou_values else 0.0,
        "mean_dice": sum(dice_values) / len(dice_values) if dice_values else 0.0,
        "pixel_accuracy": correct / total if total > 0 else 0.0,
        "mean_accuracy": sum(acc_values) / len(acc_values) if acc_values else 0.0,
        "frequency_weighted_iou": fw_iou,
        "num_classes": num_classes,
    }


# --------------------------------------------------------------------------- #
# One-shot convenience functions
# --------------------------------------------------------------------------- #

def iou_score(
    pred: Mask, target: Mask, num_classes: int = 2, *, ignore_index: int | None = None
) -> float:
    """Mean intersection-over-union over the classes present in ``pred``/``target``."""
    return metrics_from_confusion(
        confusion_matrix(pred, target, num_classes, ignore_index=ignore_index)
    )["mean_iou"]


def dice_score(
    pred: Mask, target: Mask, num_classes: int = 2, *, ignore_index: int | None = None
) -> float:
    """Mean Dice (F1) over the classes present in ``pred``/``target``."""
    return metrics_from_confusion(
        confusion_matrix(pred, target, num_classes, ignore_index=ignore_index)
    )["mean_dice"]


def pixel_accuracy(
    pred: Mask, target: Mask, num_classes: int = 2, *, ignore_index: int | None = None
) -> float:
    """Fraction of pixels classified correctly."""
    return metrics_from_confusion(
        confusion_matrix(pred, target, num_classes, ignore_index=ignore_index)
    )["pixel_accuracy"]


def segmentation_metrics(
    pred: Mask, target: Mask, num_classes: int = 2, *, ignore_index: int | None = None
) -> dict[str, Any]:
    """Full segmentation metric bundle for a single prediction/target pair."""
    return metrics_from_confusion(
        confusion_matrix(pred, target, num_classes, ignore_index=ignore_index)
    )


# --------------------------------------------------------------------------- #
# Accumulator (dataset-level scoring)
# --------------------------------------------------------------------------- #

class SegmentationMetrics:
    """Incremental segmentation scorer over many mask pairs.

    Accumulates a single confusion matrix across ``update`` calls so a whole
    dataset (or epoch) is scored in one pass, then :meth:`compute` returns the
    metric bundle. This mirrors how segmentation is evaluated batch-by-batch.
    """

    def __init__(self, num_classes: int = 2, *, ignore_index: int | None = None) -> None:
        """Initialize the accumulator.

        Args:
            num_classes: Number of segmentation classes (2 for binary).
            ignore_index: Optional target label to exclude from scoring.
        """
        if num_classes < 1:
            raise SegmentationError("num_classes must be >= 1.")
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._matrix = [[0] * num_classes for _ in range(num_classes)]

    @property
    def num_classes(self) -> int:
        """Number of segmentation classes."""
        return self._num_classes

    def update(self, pred: Mask, target: Mask) -> SegmentationMetrics:
        """Accumulate one prediction/target mask pair. Returns ``self``."""
        batch = confusion_matrix(
            pred, target, self._num_classes, ignore_index=self._ignore_index
        )
        for i in range(self._num_classes):
            row = self._matrix[i]
            batch_row = batch[i]
            for j in range(self._num_classes):
                row[j] += batch_row[j]
        return self

    def compute(self) -> dict[str, Any]:
        """Return the accumulated metric bundle."""
        return metrics_from_confusion(self._matrix)

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self._matrix = [[0] * self._num_classes for _ in range(self._num_classes)]

    @property
    def confusion(self) -> list[list[int]]:
        """A copy of the accumulated confusion matrix."""
        return [row[:] for row in self._matrix]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def binarize(prob: Mask, threshold: float = 0.5) -> list[Any]:
    """Threshold a probability map into a 0/1 label mask (nested lists preserved)."""
    def _apply(item: Any) -> Any:
        if isinstance(item, (list, tuple)):
            return [_apply(e) for e in item]
        if hasattr(item, "ravel"):  # numpy sub-array
            return [_apply(e) for e in item.tolist()]
        return 1 if float(item) >= threshold else 0

    if hasattr(prob, "tolist") and not isinstance(prob, (list, tuple)):
        prob = prob.tolist()
    return _apply(prob)


__all__ = [
    "confusion_matrix",
    "metrics_from_confusion",
    "segmentation_metrics",
    "iou_score",
    "dice_score",
    "pixel_accuracy",
    "SegmentationMetrics",
    "binarize",
]
