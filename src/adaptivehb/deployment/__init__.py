"""Deployment subsystem.

Load-only serving of the trained framework: a transport-agnostic inference
service, a clinical report, guarded deployment targets (desktop plus optional
FastAPI/Gradio/Streamlit), and the DeploymentManager. No retraining occurs at
deployment time.
"""

from adaptivehb.deployment.config import DeploymentConfig
from adaptivehb.deployment.manager import DeploymentManager
from adaptivehb.deployment.report import ClinicalReport
from adaptivehb.deployment.service import HbInferenceService
from adaptivehb.deployment.targets import available_targets, build_target

__all__ = [
    "DeploymentManager",
    "DeploymentConfig",
    "ClinicalReport",
    "HbInferenceService",
    "build_target",
    "available_targets",
]
