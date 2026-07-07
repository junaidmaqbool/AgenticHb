"""HbInferenceService — the load-only prediction engine for deployment.

Deployment loads registry-approved models and configuration, accepts patient
input, runs the adaptive inference workflow, and returns a clinical report — with
no retraining (PIPELINE_SPEC Ch.17, Charter §27). The service is transport-
agnostic; deployment targets (desktop/FastAPI/…) wrap it.

It composes its own PredictionManager and AgentManager (each independent and
config-driven), and consults the RegistryManager to confirm trained models are
available before serving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptivehb.agents.manager import AgentManager
from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.types import ModelCategory
from adaptivehb.deployment.config import DeploymentConfig
from adaptivehb.deployment.report import ClinicalReport
from adaptivehb.exceptions import DeploymentError
from adaptivehb.logging import get_logger
from adaptivehb.managers.registry import RegistryManager
from adaptivehb.prediction.manager import PredictionManager


class HbInferenceService:
    """Loads models once and serves clinical hemoglobin reports."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the service (models are loaded lazily via :meth:`load`)."""
        self._config = config
        self._base_dir = Path(base_dir)
        self._log = get_logger("deployment.service")
        self._registry = RegistryManager(config, base_dir)
        self._prediction = PredictionManager(config, base_dir)
        self._agents = AgentManager(config, base_dir)
        self._dep_config = DeploymentConfig.from_section(config.section("deployment"))
        self._models: dict[str, Any] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        """Whether models have been loaded."""
        return self._loaded

    @property
    def deployment_config(self) -> DeploymentConfig:
        """The typed deployment configuration."""
        return self._dep_config

    def load(self) -> dict[str, Any]:
        """Load registry-confirmed models and build the serving stack.

        Returns:
            A readiness summary (model count, tissues).

        Raises:
            DeploymentError: If no prediction models are registered (nothing to
                deploy without training first).
        """
        self._registry.initialize()
        self._prediction.initialize()
        self._agents.initialize()

        records = self._registry.find(ModelCategory.PREDICTION)
        if not records:
            raise DeploymentError("No prediction models registered; train before deploying.")

        tissues = list(self._prediction.prediction_config.tissue_models) or ["eye", "palm", "tongue", "nail"]
        self._models = {tissue: self._prediction.build(tissue=tissue) for tissue in tissues}
        self._loaded = True
        self._log.info("Deployment loaded %d model(s) for tissues %s.", len(records), tissues)
        return {"num_models": len(records), "tissues": list(self._models)}

    def predict(self, patient: dict[str, Any]) -> ClinicalReport:
        """Produce a clinical report for one patient.

        Args:
            patient: ``{"patient_id", "tissues": {tissue: {features...}}, "metadata"}``.
                Missing per-tissue predictions are filled by the loaded models;
                missing quality/ROI features fall back to acceptable defaults.

        Returns:
            A :class:`ClinicalReport`.
        """
        if not self._loaded:
            self.load()

        patient_id = str(patient.get("patient_id", "unknown"))
        tissues_in = {t: dict(f) for t, f in dict(patient.get("tissues", {})).items()}
        if not tissues_in:
            tissues_in = {tissue: {} for tissue in self._models}

        for tissue, features in tissues_in.items():
            features.setdefault("quality", 0.8)
            features.setdefault("roi_iou", 0.75)
            if "pred_hb" not in features and tissue in self._models:
                features["pred_hb"] = float(self._models[tissue].predict(None))
            features.setdefault("pred_confidence", 0.8)

        pred_config = self._prediction.prediction_config
        context = {
            "patient_id": patient_id,
            "tissues": tissues_in,
            "available_segmentation": list(
                self._config.section("segmentation")["segmentation"].get("available_models", [])
            ),
            "default_prediction_model": pred_config.default_model,
            "tissue_models": dict(pred_config.tissue_models),
            "metadata": dict(patient.get("metadata", {})),
        }
        result = self._agents.run_workflow(context)
        return ClinicalReport.from_workflow(
            patient_id, result, include_confidence=self._dep_config.include_confidence
        )


__all__ = ["HbInferenceService"]
