"""Unit tests for the deployment subsystem (service, targets, manager, report)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.core.types import ModelCategory, ModelRecord, ModelStatus
from adaptivehb.deployment import (
    ClinicalReport,
    DeploymentConfig,
    DeploymentManager,
    HbInferenceService,
    available_targets,
    build_target,
)
from adaptivehb.deployment.targets import DesktopTarget
from adaptivehb.exceptions import DeploymentError
from adaptivehb.managers.registry import RegistryManager


def _register_prediction_model(config: FrameworkConfig, base_dir: Path) -> None:
    registry = RegistryManager(config, base_dir)
    registry.initialize()
    registry.register(
        ModelRecord(
            name="hb_eye",
            category=ModelCategory.PREDICTION,
            task="hb_estimation",
            architecture="vit",
            metrics={"val_loss": 0.4},
            status=ModelStatus.STABLE,
        )
    )


_PATIENT = {
    "patient_id": "P0001",
    "tissues": {
        "eye": {"quality": 0.9, "roi_iou": 0.85},
        "palm": {"quality": 0.7, "roi_iou": 0.75},
    },
}


# --------------------------------------------------------------------------- #
# Config, targets, report
# --------------------------------------------------------------------------- #

def test_deployment_config_parses(framework_config: FrameworkConfig) -> None:
    config = DeploymentConfig.from_section(framework_config.section("deployment"))
    assert config.target == "fastapi"
    assert config.port == 8000
    assert "stable" in config.status_filter


def test_available_targets_includes_desktop() -> None:
    assert "desktop" in available_targets()


def test_build_unknown_target_raises(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    service = HbInferenceService(framework_config, tmp_path)
    config = DeploymentConfig.from_section(framework_config.section("deployment"))
    with pytest.raises(DeploymentError):
        build_target("nonexistent", service, config)


def test_web_target_without_dependency_raises(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    service = HbInferenceService(framework_config, tmp_path)
    config = DeploymentConfig.from_section(framework_config.section("deployment"))
    target = build_target("fastapi", service, config)
    # FastAPI is not installed in the framework-only environment.
    with pytest.raises(DeploymentError):
        target.launch()


def test_clinical_report_text_and_export(tmp_path: Path) -> None:
    report = ClinicalReport(
        patient_id="P1", predicted_hb=12.5, confidence=0.9, interval=0.4,
        recommendation="reliable", selected_tissues=["eye"],
    )
    text = report.to_text()
    assert "12.50 g/dL" in text
    assert "P1" in text
    json_path = report.export_json(tmp_path / "r.json")
    text_path = report.export_text(tmp_path / "r.txt")
    assert json_path.is_file() and text_path.is_file()


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

def test_service_requires_registered_models(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    service = HbInferenceService(framework_config, tmp_path)
    with pytest.raises(DeploymentError):
        service.load()


def test_service_loads_and_predicts(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    _register_prediction_model(framework_config, tmp_path)
    service = HbInferenceService(framework_config, tmp_path)
    info = service.load()
    assert info["num_models"] >= 1
    report = service.predict(_PATIENT)
    assert isinstance(report, ClinicalReport)
    assert report.predicted_hb is not None
    assert report.selected_tissues  # the agent workflow selected tissues


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #

@pytest.fixture()
def deployment(framework_config: FrameworkConfig, tmp_path: Path) -> DeploymentManager:
    _register_prediction_model(framework_config, tmp_path)
    manager = DeploymentManager(framework_config, tmp_path)
    manager.initialize()
    return manager


def test_manager_predict_and_export(deployment: DeploymentManager) -> None:
    report = deployment.predict(_PATIENT)
    paths = deployment.export_report(report)
    assert Path(paths["json"]).is_file()
    assert Path(paths["text"]).is_file()


def test_manager_launch_desktop(deployment: DeploymentManager) -> None:
    result = deployment.launch("desktop")
    assert result["target"] == "desktop"
    assert result["ready"] is True
    assert callable(result["predict"])


def test_manager_available_targets(deployment: DeploymentManager) -> None:
    assert "desktop" in deployment.available_targets()
