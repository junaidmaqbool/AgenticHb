"""Typed configuration for the deployment subsystem (deployment.yaml)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeploymentConfig:
    """Typed view of the ``deployment`` configuration section."""

    target: str = "fastapi"
    host: str = "127.0.0.1"
    port: int = 8000
    load_from_registry: bool = True
    status_filter: tuple[str, ...] = ("stable", "production")
    generate_pdf: bool = True
    include_confidence: bool = True

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> DeploymentConfig:
        """Build a :class:`DeploymentConfig` from the parsed section."""
        if "deployment" in section and isinstance(section["deployment"], Mapping):
            section = section["deployment"]
        models = dict(section.get("models", {}))
        report = dict(section.get("report", {}))
        return cls(
            target=str(section.get("target", "fastapi")),
            host=str(section.get("host", "127.0.0.1")),
            port=int(section.get("port", 8000)),
            load_from_registry=bool(models.get("load_from_registry", True)),
            status_filter=tuple(str(s) for s in models.get("status_filter", ["stable", "production"])),
            generate_pdf=bool(report.get("generate_pdf", True)),
            include_confidence=bool(report.get("include_confidence", True)),
        )


__all__ = ["DeploymentConfig"]
