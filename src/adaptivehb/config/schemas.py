"""Typed configuration schemas consumed directly by Phase 1 infrastructure.

Only the ``project`` and ``logging`` configurations are strongly typed here,
because they are required by the packaging, logging, and reproducibility layers
that already exist. Schemas for the remaining configuration files
(dataset, segmentation, prediction, agents, evaluation, deployment, registry)
are added in the phases that implement their consuming managers; until then
those sections are loaded as validated raw mappings (see ``ConfigLoader``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from adaptivehb.config.base import require_keys

_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware and data-loading settings."""

    device: str = "auto"
    num_workers: int = 4
    mixed_precision: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareConfig:
        """Build a :class:`HardwareConfig` from a raw mapping."""
        return cls(
            device=str(data.get("device", "auto")),
            num_workers=int(data.get("num_workers", 4)),
            mixed_precision=bool(data.get("mixed_precision", True)),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime behaviour settings."""

    resume: bool = True
    checkpoint_frequency: int = 1
    tensorboard: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeConfig:
        """Build a :class:`RuntimeConfig` from a raw mapping."""
        return cls(
            resume=bool(data.get("resume", True)),
            checkpoint_frequency=int(data.get("checkpoint_frequency", 1)),
            tensorboard=bool(data.get("tensorboard", True)),
        )


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations for framework outputs.

    Paths are stored verbatim from configuration; resolution against a root
    directory is performed by the consumer, never hardcoded in source.
    """

    output_root: str = "outputs"
    checkpoints: str = "checkpoints"
    weights: str = "weights"
    registry: str = "registry"
    logs: str = "logs"
    results: str = "results"
    figures: str = "figures"
    reports: str = "reports"
    tensorboard: str = "tensorboard"
    cache: str = "cache"
    experiments: str = "experiments"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PathsConfig:
        """Build a :class:`PathsConfig` from a raw mapping, keeping defaults."""
        defaults = cls()
        return cls(
            output_root=str(data.get("output_root", defaults.output_root)),
            checkpoints=str(data.get("checkpoints", defaults.checkpoints)),
            weights=str(data.get("weights", defaults.weights)),
            registry=str(data.get("registry", defaults.registry)),
            logs=str(data.get("logs", defaults.logs)),
            results=str(data.get("results", defaults.results)),
            figures=str(data.get("figures", defaults.figures)),
            reports=str(data.get("reports", defaults.reports)),
            tensorboard=str(data.get("tensorboard", defaults.tensorboard)),
            cache=str(data.get("cache", defaults.cache)),
            experiments=str(data.get("experiments", defaults.experiments)),
        )


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration (project.yaml)."""

    name: str
    experiment_name: str
    version: str
    seed: int
    deterministic: bool
    hardware: HardwareConfig
    runtime: RuntimeConfig
    paths: PathsConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectConfig:
        """Build a :class:`ProjectConfig` from a parsed project.yaml mapping."""
        require_keys(data, ["project", "hardware", "paths"], "project.yaml")
        project = data["project"]
        require_keys(project, ["name", "seed"], "project.yaml:project")
        return cls(
            name=str(project["name"]),
            experiment_name=str(project.get("experiment_name", "default_experiment")),
            version=str(project.get("version", "0.0.0")),
            seed=int(project["seed"]),
            deterministic=bool(project.get("deterministic", True)),
            hardware=HardwareConfig.from_dict(data.get("hardware", {})),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {})),
            paths=PathsConfig.from_dict(data.get("paths", {})),
        )


@dataclass(frozen=True)
class RotationConfig:
    """Rotating-file-handler settings."""

    enabled: bool = True
    max_bytes: int = 10_485_760
    backup_count: int = 5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RotationConfig:
        """Build a :class:`RotationConfig` from a raw mapping."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_bytes=int(data.get("max_bytes", 10_485_760)),
            backup_count=int(data.get("backup_count", 5)),
        )


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration (logging.yaml)."""

    level: str = "INFO"
    console: bool = True
    file: bool = True
    log_dir: str = "logs"
    filename: str = "adaptivehb.log"
    format: str = _DEFAULT_LOG_FORMAT
    datefmt: str = _DEFAULT_DATE_FORMAT
    rotation: RotationConfig = field(default_factory=RotationConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoggingConfig:
        """Build a :class:`LoggingConfig` from a parsed logging.yaml mapping."""
        require_keys(data, ["logging"], "logging.yaml")
        log = data["logging"]
        return cls(
            level=str(log.get("level", "INFO")).upper(),
            console=bool(log.get("console", True)),
            file=bool(log.get("file", True)),
            log_dir=str(log.get("log_dir", "logs")),
            filename=str(log.get("filename", "adaptivehb.log")),
            format=str(log.get("format", _DEFAULT_LOG_FORMAT)),
            datefmt=str(log.get("datefmt", _DEFAULT_DATE_FORMAT)),
            rotation=RotationConfig.from_dict(log.get("rotation", {})),
        )


__all__ = [
    "HardwareConfig",
    "RuntimeConfig",
    "PathsConfig",
    "ProjectConfig",
    "RotationConfig",
    "LoggingConfig",
]
