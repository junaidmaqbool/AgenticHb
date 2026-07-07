"""Automatic dataset statistics (DATASET_SPEC Ch.15).

Computes patient/image counts, per-tissue coverage, and label/demographic
distributions using only the standard library.
"""

from __future__ import annotations

import statistics as stats

from adaptivehb.dataset.config import DatasetConfig
from adaptivehb.dataset.metadata import MetadataTable
from adaptivehb.dataset.schema import DatasetStatistics, Sample


def compute_statistics(
    config: DatasetConfig, metadata: MetadataTable, samples: list[Sample]
) -> DatasetStatistics:
    """Compute dataset statistics from metadata and samples.

    Args:
        config: Typed dataset configuration.
        metadata: The loaded metadata table.
        samples: The scanned image samples.

    Returns:
        A populated :class:`DatasetStatistics`.
    """
    result = DatasetStatistics(
        num_patients=len(metadata.patient_ids),
        num_images=len(samples),
        num_masks=sum(1 for s in samples if s.mask_path),
    )

    for tissue in config.tissues:
        tissue_samples = [s for s in samples if s.tissue == tissue]
        result.images_per_tissue[tissue] = len(tissue_samples)
        result.patients_per_tissue[tissue] = len({s.patient_id for s in tissue_samples})

    result.hb = _numeric_summary(metadata, config.metadata.target_column)
    result.missing_hb = _count_missing(metadata, config.metadata.target_column)
    result.age = _numeric_summary(metadata, "Age")
    result.gender_distribution = _categorical_counts(metadata, "Gender")
    return result


def _numeric_summary(metadata: MetadataTable, column: str) -> dict[str, float]:
    """Return count/min/max/mean/std for a numeric column (empty if absent)."""
    if column not in metadata.columns:
        return {}
    values: list[float] = []
    for row in metadata.rows:
        parsed = _to_float(row.get(column))
        if parsed is not None:
            values.append(parsed)
    if not values:
        return {}
    summary = {
        "count": float(len(values)),
        "min": min(values),
        "max": max(values),
        "mean": stats.fmean(values),
    }
    summary["std"] = stats.pstdev(values) if len(values) > 1 else 0.0
    return summary


def _categorical_counts(metadata: MetadataTable, column: str) -> dict[str, int]:
    """Return a value → count mapping for a categorical column."""
    if column not in metadata.columns:
        return {}
    counts: dict[str, int] = {}
    for row in metadata.rows:
        value = (row.get(column) or "").strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_missing(metadata: MetadataTable, column: str) -> int:
    """Count rows whose value in ``column`` is missing or blank."""
    if column not in metadata.columns:
        return len(metadata.rows)
    return sum(1 for row in metadata.rows if not (row.get(column) or "").strip())


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


__all__ = ["compute_statistics"]
