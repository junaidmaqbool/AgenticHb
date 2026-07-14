"""Adaptive decision framework — the project's primary scientific contribution.

Lightweight, deterministic decision agents organized in three layers (perception,
decision, clinical) coordinated by a deterministic Workflow Controller. Agents
share a common interface, communicate only through the controller, and are
configurable/enable-able via ``agents.yaml``. Learned policies can replace
deterministic ones later without changing the interface (Decision 005).
"""

from adaptivehb.agents.base import Agent
from adaptivehb.agents.clinical import ConfidenceAgent, FusionAgent
from adaptivehb.agents.config import AGENT_ORDER, AgentsConfig
from adaptivehb.agents.controller import WorkflowController
from adaptivehb.agents.decision import (
    PredictionRoutingAgent,
    SegmentationSelectionAgent,
    TissueSelectionAgent,
)
from adaptivehb.agents.manager import AgentManager
from adaptivehb.agents.perception import QualityAssessmentAgent, ROIVerificationAgent
from adaptivehb.agents.preprocessing import PreprocessingAgent
from adaptivehb.agents.schema import AgentDecision, WorkflowResult

__all__ = [
    "Agent",
    "AgentDecision",
    "WorkflowResult",
    "AgentsConfig",
    "AGENT_ORDER",
    "WorkflowController",
    "AgentManager",
    "PreprocessingAgent",
    "QualityAssessmentAgent",
    "ROIVerificationAgent",
    "SegmentationSelectionAgent",
    "TissueSelectionAgent",
    "PredictionRoutingAgent",
    "FusionAgent",
    "ConfidenceAgent",
]
