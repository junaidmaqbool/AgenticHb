"""DeploymentManager — turns trained models into a usable clinical service.

Deployment requires no retraining, no manual checkpoint selection, and no code
modification (PIPELINE_SPEC Ch.17): it loads registry-approved models, serves
predictions through the adaptive workflow, generates clinical reports, and can
launch a chosen deployment target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import ensure_dir
from adaptivehb.deployment.config import DeploymentConfig
from adaptivehb.deployment.report import ClinicalReport
from adaptivehb.deployment.service import HbInferenceService
from adaptivehb.deployment.targets import available_targets, build_target


class DeploymentManager(BaseManager):
    """Loads models, serves clinical reports, and launches deployment targets."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the deployment manager and its inference service."""
        super().__init__(config, base_dir)
        self._dep_config = DeploymentConfig.from_section(config.section("deployment"))
        self._service = HbInferenceService(config, base_dir)
        self._reports_root = self._base_dir / config.project.paths.results / "deployment"

    @property
    def deployment_config(self) -> DeploymentConfig:
        """The typed deployment configuration."""
        return self._dep_config

    @property
    def service(self) -> HbInferenceService:
        """The underlying inference service."""
        return self._service

    def _on_initialize(self) -> None:
        ensure_dir(self._reports_root)

    def available_targets(self) -> list[str]:
        """Return deployment targets usable in this environment."""
        return available_targets()

    def load(self) -> dict[str, Any]:
        """Load registry-approved models. Returns a readiness summary."""
        return self._service.load()

    def predict(self, patient: dict[str, Any]) -> ClinicalReport:
        """Produce a clinical report for one patient (loads models if needed)."""
        return self._service.predict(patient)

    def launch(self, target: str | None = None) -> dict[str, Any]:
        """Launch (or prepare) a deployment target.

        Args:
            target: Target name; defaults to the configured target.

        Returns:
            A readiness summary from the target.
        """
        name = target or self._dep_config.target
        built = build_target(name, self._service, self._dep_config)
        self._log.info("Launching deployment target '%s'.", name)
        return built.launch()

    def export_report(self, report: ClinicalReport) -> dict[str, str]:
        """Export a clinical report as JSON and human-readable text."""
        ensure_dir(self._reports_root)
        stem = self._reports_root / f"report_{report.patient_id}"
        return {
            "json": str(report.export_json(stem.with_suffix(".json"))),
            "text": str(report.export_text(stem.with_suffix(".txt"))),
        }


__all__ = ["DeploymentManager"]
