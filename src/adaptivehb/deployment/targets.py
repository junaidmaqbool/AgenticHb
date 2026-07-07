"""Deployment targets — transport wrappers around the inference service.

The service is transport-agnostic; a target adapts it to a delivery mechanism
(desktop callable, FastAPI, Gradio, Streamlit). Web targets import their optional
dependencies lazily and raise a clear :class:`DeploymentError` if the extra is
not installed, so importing this module never requires web frameworks
(Decision 024). The desktop target has no extra dependencies.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any

from adaptivehb.deployment.config import DeploymentConfig
from adaptivehb.deployment.service import HbInferenceService
from adaptivehb.exceptions import DeploymentError

# Web targets and the module that must be importable to use them.
_WEB_REQUIREMENTS: dict[str, str] = {
    "fastapi": "fastapi",
    "gradio": "gradio",
    "streamlit": "streamlit",
}


class DeploymentTarget(ABC):
    """Base class for a deployment transport."""

    name: str = "base"

    def __init__(self, service: HbInferenceService, config: DeploymentConfig) -> None:
        """Store the service and configuration."""
        self._service = service
        self._config = config

    @property
    def service(self) -> HbInferenceService:
        """The wrapped inference service."""
        return self._service

    @abstractmethod
    def launch(self) -> dict[str, Any]:
        """Launch (or prepare) the target and return a readiness summary."""

    def describe(self) -> dict[str, Any]:
        """Return a static description of the target."""
        return {"target": self.name, "host": self._config.host, "port": self._config.port}


class DesktopTarget(DeploymentTarget):
    """A dependency-free local target exposing a bound predict callable."""

    name = "desktop"

    def launch(self) -> dict[str, Any]:
        """Load models and expose an in-process predict function."""
        readiness = self._service.load()
        return {"target": self.name, "ready": True, "predict": self._service.predict, **readiness}


class _WebTarget(DeploymentTarget):
    """Base for web targets that require an optional dependency."""

    requirement: str = ""

    def launch(self) -> dict[str, Any]:
        """Verify the optional dependency, then build/serve the app."""
        if importlib.util.find_spec(self.requirement) is None:
            raise DeploymentError(
                f"The '{self.name}' target requires '{self.requirement}'. "
                f"Install it (optional deployment extra) to launch this target."
            )
        # Real server launch is performed in the deployment environment; the
        # framework confirms readiness without blocking on a server here.
        self._service.load()
        return {"target": self.name, "ready": True, **self.describe()}


class FastAPITarget(_WebTarget):
    """FastAPI deployment target."""

    name = "fastapi"
    requirement = "fastapi"


class GradioTarget(_WebTarget):
    """Gradio deployment target."""

    name = "gradio"
    requirement = "gradio"


class StreamlitTarget(_WebTarget):
    """Streamlit deployment target."""

    name = "streamlit"
    requirement = "streamlit"


_TARGETS: dict[str, type[DeploymentTarget]] = {
    "desktop": DesktopTarget,
    "fastapi": FastAPITarget,
    "gradio": GradioTarget,
    "streamlit": StreamlitTarget,
}


def build_target(name: str, service: HbInferenceService, config: DeploymentConfig) -> DeploymentTarget:
    """Construct a deployment target by name.

    Args:
        name: Target name (desktop/fastapi/gradio/streamlit).
        service: The inference service to wrap.
        config: Deployment configuration.

    Raises:
        DeploymentError: If the target name is unknown.
    """
    key = name.lower()
    if key not in _TARGETS:
        raise DeploymentError(f"Unknown deployment target: {name!r}. Available: {sorted(_TARGETS)}.")
    return _TARGETS[key](service, config)


def available_targets() -> list[str]:
    """Return targets usable in this environment (desktop plus installed web)."""
    usable = ["desktop"]
    usable += [name for name, req in _WEB_REQUIREMENTS.items() if importlib.util.find_spec(req)]
    return usable


__all__ = [
    "DeploymentTarget",
    "DesktopTarget",
    "FastAPITarget",
    "GradioTarget",
    "StreamlitTarget",
    "build_target",
    "available_targets",
]
