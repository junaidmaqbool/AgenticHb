"""Core layer: shared interfaces, types, and utilities for the framework."""

from adaptivehb.core.interfaces import BaseAgent, BaseManager, BaseModel
from adaptivehb.core.types import (
    JobStatus,
    ModelCategory,
    ModelRecord,
    ModelStatus,
    PipelineMode,
)
from adaptivehb.core.utils import (
    ensure_dir,
    read_json,
    set_global_seed,
    timestamp_slug,
    utcnow_iso,
    write_json,
)

__all__ = [
    "BaseManager",
    "BaseModel",
    "BaseAgent",
    "ModelCategory",
    "ModelStatus",
    "PipelineMode",
    "JobStatus",
    "ModelRecord",
    "ensure_dir",
    "read_json",
    "write_json",
    "set_global_seed",
    "timestamp_slug",
    "utcnow_iso",
]
