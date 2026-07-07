"""Dataset subsystem: configurable loading, validation, splitting, statistics.

The DatasetManager is the single access point for dataset files; every other
module consumes the standardized :class:`Sample` structure.
"""

from adaptivehb.dataset.config import (
    DatasetConfig,
    ImageSpec,
    MetadataSpec,
    SplitSpec,
)
from adaptivehb.dataset.manager import DatasetManager
from adaptivehb.dataset.metadata import MetadataTable
from adaptivehb.dataset.schema import (
    DatasetStatistics,
    Sample,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from adaptivehb.dataset.splitting import invert_split, patient_level_split
from adaptivehb.dataset.synthetic import generate_synthetic_dataset

__all__ = [
    "DatasetManager",
    "DatasetConfig",
    "ImageSpec",
    "MetadataSpec",
    "SplitSpec",
    "MetadataTable",
    "Sample",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "DatasetStatistics",
    "patient_level_split",
    "invert_split",
    "generate_synthetic_dataset",
]
