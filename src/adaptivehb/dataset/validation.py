"""Dataset validation (DATASET_SPEC Ch.12, Ch.21).

Produces a :class:`ValidationReport` summarising dataset health. Critical
problems (missing mandatory columns, duplicate patient IDs, missing/invalid
hemoglobin labels) are recorded as errors and make the dataset invalid; lesser
problems (orphan images, patients without images, missing masks) are warnings.
"""

from __future__ import annotations

from adaptivehb.dataset.config import DatasetConfig
from adaptivehb.dataset.metadata import MetadataTable
from adaptivehb.dataset.schema import Sample, Severity, ValidationReport


class DatasetValidator:
    """Runs configurable integrity checks over metadata and samples."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the validator.

        Args:
            config: Typed dataset configuration.
        """
        self._config = config

    def validate(self, metadata: MetadataTable, samples: list[Sample]) -> ValidationReport:
        """Validate a loaded dataset.

        Args:
            metadata: The loaded metadata table.
            samples: The scanned image samples.

        Returns:
            A populated :class:`ValidationReport`.
        """
        report = ValidationReport(
            num_patients=len(metadata.patient_ids),
            num_images=len(samples),
            num_masks=sum(1 for s in samples if s.mask_path),
        )
        self._check_columns(metadata, report)
        self._check_duplicates(metadata, report)
        self._check_labels(metadata, report)
        self._check_image_metadata_linkage(metadata, samples, report)
        self._check_masks(samples, report)
        return report

    # -- individual checks -------------------------------------------------

    def _check_columns(self, metadata: MetadataTable, report: ValidationReport) -> None:
        missing = metadata.has_columns(self._config.metadata.mandatory_columns)
        if missing:
            report.add(
                "missing_mandatory_columns",
                Severity.ERROR,
                f"Metadata is missing mandatory columns: {missing}.",
                count=len(missing),
            )

    def _check_duplicates(self, metadata: MetadataTable, report: ValidationReport) -> None:
        id_col = self._config.metadata.patient_id_column
        seen: set[str] = set()
        duplicates = 0
        for row in metadata.rows:
            pid = (row.get(id_col) or "").strip()
            if not pid:
                continue
            if pid in seen:
                duplicates += 1
            seen.add(pid)
        if duplicates:
            report.add(
                "duplicate_patient_ids",
                Severity.ERROR,
                f"Found {duplicates} duplicate patient ID row(s).",
                count=duplicates,
            )

    def _check_labels(self, metadata: MetadataTable, report: ValidationReport) -> None:
        target = self._config.metadata.target_column
        if target not in metadata.columns:
            return  # already reported by column check
        missing = 0
        invalid = 0
        for row in metadata.rows:
            raw = (row.get(target) or "").strip()
            if not raw:
                missing += 1
                continue
            if _to_float(raw) is None:
                invalid += 1
        if missing:
            report.add(
                "missing_hemoglobin",
                Severity.ERROR,
                f"{missing} record(s) have a missing {target} value.",
                count=missing,
            )
        if invalid:
            report.add(
                "invalid_hemoglobin",
                Severity.ERROR,
                f"{invalid} record(s) have a non-numeric {target} value.",
                count=invalid,
            )

    def _check_image_metadata_linkage(
        self, metadata: MetadataTable, samples: list[Sample], report: ValidationReport
    ) -> None:
        known = set(metadata.patient_ids)
        with_images = {s.patient_id for s in samples}

        orphans = sorted({s.patient_id for s in samples if s.patient_id not in known})
        if orphans:
            report.add(
                "orphan_images",
                Severity.WARNING,
                f"{len(orphans)} patient(s) have images but no metadata: {orphans[:5]}...",
                count=len(orphans),
            )

        no_images = sorted(known - with_images)
        if no_images:
            report.add(
                "patients_without_images",
                Severity.WARNING,
                f"{len(no_images)} patient(s) have metadata but no images.",
                count=len(no_images),
            )

    def _check_masks(self, samples: list[Sample], report: ValidationReport) -> None:
        missing_masks = sum(1 for s in samples if not s.mask_path)
        if missing_masks:
            report.add(
                "missing_masks",
                Severity.WARNING,
                f"{missing_masks} image(s) have no corresponding mask.",
                count=missing_masks,
            )


def _to_float(value: str) -> float | None:
    """Parse a float, returning ``None`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DatasetValidator"]
