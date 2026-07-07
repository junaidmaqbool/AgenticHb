"""Shared enumerations and record types used across the infrastructure layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from adaptivehb.core.utils import utcnow_iso
from adaptivehb.version import __version__


class ModelCategory(str, Enum):
    """Registry categories. Each category is versioned independently."""

    SEGMENTATION = "segmentation"
    PREDICTION = "prediction"
    DECISION_MODULE = "decision_module"
    FUSION = "fusion"
    CONFIDENCE = "confidence"


class ModelStatus(str, Enum):
    """Lifecycle status of a registered model."""

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class PipelineMode(str, Enum):
    """Supported pipeline execution modes (PIPELINE_SPEC Ch.7)."""

    BUILD = "build"
    TRAINING = "training"
    RESUME = "resume"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    DEPLOYMENT = "deployment"


class JobStatus(str, Enum):
    """Status of a pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ModelRecord:
    """A single registered model version (MODEL_REGISTRY_SPEC Ch.7).

    The registry stores records (metadata + checkpoint reference), never model
    weights. ``unique_id`` and ``version`` are assigned by the RegistryManager
    at registration time and must never change afterwards.
    """

    name: str
    category: ModelCategory
    task: str
    architecture: str
    version: int = 1
    unique_id: str = ""
    author: str = "Junaid"
    created_at: str = field(default_factory=utcnow_iso)
    dataset_version: str | None = None
    config_version: str | None = None
    framework_version: str = __version__
    checkpoint_path: str | None = None
    input_resolution: list[int] | None = None
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    metrics: dict[str, float] = field(default_factory=dict)
    inference_ms: float | None = None
    model_size_mb: float | None = None
    hardware: str | None = None
    seed: int | None = None
    training_time_s: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a JSON-friendly dictionary."""
        data = asdict(self)
        data["category"] = self.category.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRecord:
        """Reconstruct a record from a stored dictionary."""
        payload = dict(data)
        payload["category"] = ModelCategory(payload["category"])
        payload["status"] = ModelStatus(payload["status"])
        return cls(**payload)


__all__ = [
    "ModelCategory",
    "ModelStatus",
    "PipelineMode",
    "JobStatus",
    "ModelRecord",
]
