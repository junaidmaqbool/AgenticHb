"""Unit tests for the adaptive decision framework (agents + controller + manager)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.agents import (
    AgentManager,
    AgentsConfig,
    ConfidenceAgent,
    FusionAgent,
    PredictionRoutingAgent,
    QualityAssessmentAgent,
    ROIVerificationAgent,
    SegmentationSelectionAgent,
    TissueSelectionAgent,
    WorkflowController,
)
from adaptivehb.config import FrameworkConfig


def _context() -> dict:
    return {
        "available_segmentation": ["unet", "segformer", "deeplabv3plus"],
        "default_prediction_model": "efficientnet",
        "tissue_models": {"eye": "vit", "palm": "efficientnet", "tongue": "convnext", "nail": "resnet"},
        "tissues": {
            "eye": {"quality": 0.9, "roi_iou": 0.85, "pred_hb": 12.4, "pred_confidence": 0.9},
            "palm": {"quality": 0.7, "roi_iou": 0.75, "pred_hb": 12.9, "pred_confidence": 0.8},
            "tongue": {"quality": 0.4, "roi_iou": 0.5, "pred_hb": 10.0, "pred_confidence": 0.3},
            "nail": {"quality": 0.65, "roi_iou": 0.55, "pred_hb": 13.1, "pred_confidence": 0.6},
        },
    }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_agents_config_order_and_enable() -> None:
    config = AgentsConfig.from_section(
        {"quality_assessment": {"enabled": True}, "fusion": {"enabled": False},
         "workflow_controller": {"log_decisions": False}}
    )
    order = config.enabled_in_order()
    assert order[0] == "quality_assessment"
    assert "fusion" not in order
    assert config.log_decisions is False


# --------------------------------------------------------------------------- #
# Perception agents
# --------------------------------------------------------------------------- #

def test_quality_agent_accepts_and_rejects() -> None:
    agent = QualityAssessmentAgent("quality_assessment", {"thresholds": {"min_quality": 0.5}})
    decision = agent.predict(_context())
    assert set(decision.outputs["accepted_tissues"]) == {"eye", "palm", "nail"}
    assert decision.outputs["rejected_tissues"] == ["tongue"]
    assert decision.outputs["reacquire"] is False


def test_quality_agent_recommends_reacquire() -> None:
    agent = QualityAssessmentAgent("quality_assessment", {"thresholds": {"min_quality": 0.99}})
    decision = agent.predict(_context())
    assert decision.outputs["reacquire"] is True


def test_roi_agent_verifies_by_iou() -> None:
    ctx = _context()
    ctx["accepted_tissues"] = ["eye", "palm", "nail"]
    agent = ROIVerificationAgent("roi_verification", {"thresholds": {"min_iou": 0.6}})
    decision = agent.predict(ctx)
    assert set(decision.outputs["verified_tissues"]) == {"eye", "palm"}
    assert "nail" in decision.outputs["rejected_rois"]


# --------------------------------------------------------------------------- #
# Decision agents
# --------------------------------------------------------------------------- #

def test_segmentation_selection_by_quality() -> None:
    ctx = _context()
    ctx["verified_tissues"] = ["eye", "palm"]  # mean quality 0.8 -> unet
    decision = SegmentationSelectionAgent("segmentation_selection", {}).predict(ctx)
    assert decision.outputs["selected_segmentation"] == "unet"


def test_tissue_selection_ranks_and_caps() -> None:
    ctx = _context()
    ctx["verified_tissues"] = ["palm", "eye"]
    decision = TissueSelectionAgent("tissue_selection", {"max_tissues": 1}).predict(ctx)
    # eye has the highest quality+roi score, so it ranks first and is the sole pick.
    assert decision.outputs["selected_tissues"] == ["eye"]


def test_prediction_routing_uses_tissue_models() -> None:
    ctx = _context()
    ctx["selected_tissues"] = ["eye", "palm"]
    decision = PredictionRoutingAgent("prediction_routing", {}).predict(ctx)
    assert decision.outputs["prediction_routing"] == {"eye": "vit", "palm": "efficientnet"}


# --------------------------------------------------------------------------- #
# Clinical agents
# --------------------------------------------------------------------------- #

def test_fusion_confidence_weighted() -> None:
    ctx = _context()
    ctx["selected_tissues"] = ["eye", "palm"]
    decision = FusionAgent("fusion", {"method": "confidence_weighted"}).predict(ctx)
    # Weighted toward eye (higher confidence): between 12.4 and 12.9, closer to 12.4.
    assert 12.4 <= decision.outputs["final_hb"] <= 12.9
    assert sum(decision.outputs["fusion_weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_fusion_handles_no_predictions() -> None:
    decision = FusionAgent("fusion", {}).predict({"tissues": {}, "selected_tissues": []})
    assert decision.outputs["final_hb"] is None


def test_confidence_agent_recommendation() -> None:
    ctx = _context()
    ctx["selected_tissues"] = ["eye", "palm"]
    ctx["final_hb"] = 12.6
    decision = ConfidenceAgent("confidence", {"thresholds": {"min_confidence": 0.8}}).predict(ctx)
    assert 0.0 <= decision.outputs["confidence"] <= 1.0
    assert decision.outputs["recommendation"] in {"reliable", "review_recommended"}


# --------------------------------------------------------------------------- #
# Base agent behaviour
# --------------------------------------------------------------------------- #

def test_agent_persistence_and_noop_train(tmp_path: Path) -> None:
    agent = QualityAssessmentAgent("quality_assessment", {"thresholds": {"min_quality": 0.7}})
    assert agent.train() is None  # deterministic agents do not train
    agent.save(tmp_path / "agent.pkl")
    restored = QualityAssessmentAgent("quality_assessment", {})
    restored.load(tmp_path / "agent.pkl")
    assert restored.threshold("min_quality", 0.5) == 0.7


# --------------------------------------------------------------------------- #
# Controller + manager
# --------------------------------------------------------------------------- #

def test_controller_runs_full_workflow() -> None:
    agents = [
        QualityAssessmentAgent("quality_assessment", {"thresholds": {"min_quality": 0.5}}),
        ROIVerificationAgent("roi_verification", {"thresholds": {"min_iou": 0.6}}),
        SegmentationSelectionAgent("segmentation_selection", {}),
        TissueSelectionAgent("tissue_selection", {}),
        PredictionRoutingAgent("prediction_routing", {}),
        FusionAgent("fusion", {"method": "confidence_weighted"}),
        ConfidenceAgent("confidence", {}),
    ]
    result = WorkflowController(agents, log_decisions=False).run(_context())
    assert len(result.decisions) == 7
    assert result.final_hb is not None
    assert result.context["verified_tissues"] == ["eye", "palm"]
    assert result.recommendation in {"reliable", "review_recommended"}


def test_agent_manager_runs_workflow(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    manager = AgentManager(framework_config, tmp_path)
    manager.initialize()
    assert manager.enabled_agents()[0] == "quality_assessment"
    assert len(manager.agents) == 7
    result = manager.run_workflow(_context())
    assert result.final_hb is not None
    assert len(result.decisions) == 7


def test_disabled_agent_is_excluded(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    # Build a config view with fusion disabled and confirm it is not constructed.
    section = {"agents": {**framework_config.section("agents")["agents"]}}
    section["agents"]["fusion"] = {**section["agents"].get("fusion", {}), "enabled": False}

    class _Cfg:
        def section(self, name: str) -> dict:
            return section

        project = framework_config.project
        logging = framework_config.logging

    manager = AgentManager(_Cfg(), tmp_path)  # type: ignore[arg-type]
    manager.initialize()
    assert "fusion" not in manager.agents
