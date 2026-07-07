"""Central exception hierarchy for the AdaptiveHb framework.

All framework-specific errors derive from :class:`AdaptiveHbError` so that
callers can catch every framework error with a single ``except`` clause while
still being able to distinguish specific failure modes.
"""

from __future__ import annotations


class AdaptiveHbError(Exception):
    """Base class for all AdaptiveHb framework errors."""


class ConfigError(AdaptiveHbError):
    """Raised when configuration loading or validation fails."""


class DatasetError(AdaptiveHbError):
    """Raised when dataset loading or validation fails."""


class RegistryError(AdaptiveHbError):
    """Raised when a model-registry operation fails."""


class CheckpointError(AdaptiveHbError):
    """Raised when checkpoint saving, loading, or recovery fails."""


class StateError(AdaptiveHbError):
    """Raised when pipeline-state persistence or recovery fails."""


class PipelineError(AdaptiveHbError):
    """Raised when pipeline orchestration fails."""


class AgentError(AdaptiveHbError):
    """Raised when an adaptive decision module fails."""


class SegmentationError(AdaptiveHbError):
    """Raised when a segmentation model or its dependencies fail."""


class ModelError(AdaptiveHbError):
    """Raised when a model build or inference operation fails."""


class EvaluationError(AdaptiveHbError):
    """Raised when an evaluation or metric computation fails."""


class DeploymentError(AdaptiveHbError):
    """Raised when a deployment or serving operation fails."""


class ReportingError(AdaptiveHbError):
    """Raised when figure or table generation fails."""


__all__ = [
    "AdaptiveHbError",
    "ConfigError",
    "DatasetError",
    "RegistryError",
    "CheckpointError",
    "StateError",
    "PipelineError",
    "AgentError",
    "SegmentationError",
    "ModelError",
    "EvaluationError",
    "DeploymentError",
    "ReportingError",
]
