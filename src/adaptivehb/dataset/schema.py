"""Standardized dataset records: samples, validation reports, and statistics.

Every module in the framework consumes the same :class:`Sample` structure
(DATASET_SPEC Ch.20), and dataset health is reported through
:class:`ValidationReport` (DATASET_SPEC Ch.12) and :class:`DatasetStatistics`
(DATASET_SPEC Ch.15).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# Canonical split names.
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLIT_NAMES = (TRAIN, VALIDATION, TEST)


class Severity(str, Enum):
    """Severity of a validation issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Sample:
    """A single dataset sample (one image and its associated metadata).

    Attributes:
        patient_id: Owning patient identifier.
        tissue: Tissue class (e.g. ``eye``, ``palm``, ``tongue``, ``nail``).
        image_path: Absolute path to the image file.
        mask_path: Absolute path to the segmentation mask, if present.
        hb: Hemoglobin value for the patient, if available.
        metadata: Raw metadata row for the patient.
        split: Assigned split name (``train``/``validation``/``test``), if any.
    """

    patient_id: str
    tissue: str
    image_path: str
    mask_path: str | None = None
    hb: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    split: str | None = None

    def with_split(self, split: str) -> Sample:
        """Return a copy of this sample assigned to ``split``."""
        return Sample(
            patient_id=self.patient_id,
            tissue=self.tissue,
            image_path=self.image_path,
            mask_path=self.mask_path,
            hb=self.hb,
            metadata=self.metadata,
            split=split,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sample to a dictionary."""
        return asdict(self)


@dataclass
class ValidationIssue:
    """A single problem found during dataset validation."""

    code: str
    severity: Severity
    message: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize the issue to a dictionary."""
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ValidationReport:
    """Outcome of dataset validation.

    The dataset is considered valid when it contains no ``ERROR`` issues;
    warnings are informative and do not block training.
    """

    num_patients: int = 0
    num_images: int = 0
    num_masks: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, severity: Severity, message: str, count: int = 1) -> None:
        """Append an issue to the report."""
        self.issues.append(ValidationIssue(code, severity, message, count))

    @property
    def errors(self) -> list[ValidationIssue]:
        """All error-severity issues."""
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """All warning-severity issues."""
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """True when there are no error-severity issues."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "num_patients": self.num_patients,
            "num_images": self.num_images,
            "num_masks": self.num_masks,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class DatasetStatistics:
    """Automatically computed dataset statistics (DATASET_SPEC Ch.15)."""

    num_patients: int = 0
    num_images: int = 0
    num_masks: int = 0
    images_per_tissue: dict[str, int] = field(default_factory=dict)
    patients_per_tissue: dict[str, int] = field(default_factory=dict)
    hb: dict[str, float] = field(default_factory=dict)
    age: dict[str, float] = field(default_factory=dict)
    gender_distribution: dict[str, int] = field(default_factory=dict)
    missing_hb: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the statistics to a dictionary."""
        return asdict(self)


__all__ = [
    "TRAIN",
    "VALIDATION",
    "TEST",
    "SPLIT_NAMES",
    "Severity",
    "Sample",
    "ValidationIssue",
    "ValidationReport",
    "DatasetStatistics",
]
