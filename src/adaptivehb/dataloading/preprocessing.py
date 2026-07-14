"""Agentic data preprocessing: HB-range filtering and balanced bin sampling.

Clinical hemoglobin data is bell-shaped around ~12-13 g/dL, so a model sees far
more "average" samples than peripherally anaemic ones and learns to *predict the
mean* — the failure mode behind an MAE that equals the label range. This module
provides two torch-free, fully testable primitives that break that plateau:

* :func:`filter_hb_range` — restrict samples to a clinically meaningful window
  (e.g. 6-14 g/dL), dropping outliers and unlabelled samples.
* :func:`balanced_sample_weights` — assign each training sample a weight of
  ``1 / count(its Hb bin)`` so every bin contributes equally per epoch when fed
  to a ``WeightedRandomSampler``.

Both are configured from the ``dataset.preprocessing`` section (nothing is
hardcoded) via :class:`PreprocessingSpec`. Keeping the logic dependency-free
mirrors the rest of the dataloading bridge (Decision 025): it imports and is
testable without torch, numpy, or a vision stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from adaptivehb.dataset.schema import Sample


@dataclass(frozen=True)
class HbFilterSpec:
    """Clinical Hb-range filter settings."""

    enabled: bool = True
    min: float = 6.0
    max: float = 14.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HbFilterSpec":
        """Build from a raw mapping (missing keys fall back to defaults)."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            min=float(data.get("min", 6.0)),
            max=float(data.get("max", 14.0)),
        )


@dataclass(frozen=True)
class BalancedSamplingSpec:
    """Balanced bin-oversampling settings (train split only)."""

    enabled: bool = True
    bins: int = 10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BalancedSamplingSpec":
        """Build from a raw mapping (missing keys fall back to defaults)."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            bins=max(int(data.get("bins", 10)), 1),
        )


@dataclass(frozen=True)
class PreprocessingSpec:
    """Typed view of the ``dataset.preprocessing`` sub-configuration."""

    hb_filter: HbFilterSpec = HbFilterSpec()
    balanced_sampling: BalancedSamplingSpec = BalancedSamplingSpec()

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> "PreprocessingSpec":
        """Build from the ``dataset`` section (accepts the full or inner mapping)."""
        if "dataset" in section and isinstance(section["dataset"], Mapping):
            section = section["dataset"]
        pre = dict(section.get("preprocessing", {}))
        return cls(
            hb_filter=HbFilterSpec.from_dict(pre.get("hb_filter", {})),
            balanced_sampling=BalancedSamplingSpec.from_dict(pre.get("balanced_sampling", {})),
        )


def filter_hb_range(
    samples: Sequence[Sample], hb_min: float, hb_max: float
) -> list[Sample]:
    """Return only labelled samples whose Hb lies within ``[hb_min, hb_max]``.

    Samples without an Hb label are dropped (they cannot be range-checked and are
    unusable for supervised regression training).

    Args:
        samples: Samples to filter.
        hb_min: Inclusive lower Hb bound (g/dL).
        hb_max: Inclusive upper Hb bound (g/dL).

    Returns:
        A new list of samples inside the range, preserving input order.
    """
    return [
        s
        for s in samples
        if s.hb is not None and hb_min <= float(s.hb) <= hb_max
    ]


def _bin_index(value: float, hb_min: float, hb_max: float, n_bins: int) -> int:
    """Map an Hb value onto ``[0, n_bins)`` over ``[hb_min, hb_max]``."""
    if hb_max <= hb_min:
        return 0
    frac = (value - hb_min) / (hb_max - hb_min)
    idx = int(frac * n_bins)
    return min(max(idx, 0), n_bins - 1)


def balanced_sample_weights(
    samples: Sequence[Sample],
    *,
    n_bins: int = 10,
    hb_min: float | None = None,
    hb_max: float | None = None,
) -> list[float]:
    """Compute inverse-frequency sampling weights over Hb bins.

    Each sample receives ``1 / count(its bin)`` so that, under a
    ``WeightedRandomSampler``, every bin is expected to contribute equally per
    epoch. This oversamples rare (severely anaemic / borderline) samples and
    breaks the "predict the mean" plateau.

    Args:
        samples: Samples to weight (should already be Hb-filtered).
        n_bins: Number of equal-width Hb bins.
        hb_min: Lower bin edge; defaults to the minimum observed Hb.
        hb_max: Upper bin edge; defaults to the maximum observed Hb.

    Returns:
        One positive weight per input sample, in order. Samples without an Hb
        label receive weight ``0.0`` (they are never drawn).
    """
    labelled = [float(s.hb) for s in samples if s.hb is not None]
    if not labelled:
        return [0.0] * len(samples)

    lo = float(hb_min) if hb_min is not None else min(labelled)
    hi = float(hb_max) if hb_max is not None else max(labelled)

    counts = [0] * n_bins
    for value in labelled:
        counts[_bin_index(value, lo, hi, n_bins)] += 1

    weights: list[float] = []
    for s in samples:
        if s.hb is None:
            weights.append(0.0)
            continue
        count = counts[_bin_index(float(s.hb), lo, hi, n_bins)]
        weights.append(1.0 / count if count else 0.0)
    return weights


__all__ = [
    "HbFilterSpec",
    "BalancedSamplingSpec",
    "PreprocessingSpec",
    "filter_hb_range",
    "balanced_sample_weights",
]
