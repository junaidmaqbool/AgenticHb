"""AgentManager — builds and coordinates the adaptive decision agents.

Single responsibility: construct the enabled agents from configuration, hold the
Workflow Controller, and run the adaptive workflow. Fusion and confidence are
agents inside this manager, not separate managers (Decision 012).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptivehb.agents.base import Agent
from adaptivehb.agents.clinical import ConfidenceAgent, FusionAgent
from adaptivehb.agents.config import AGENT_ORDER, AgentsConfig
from adaptivehb.agents.controller import WorkflowController
from adaptivehb.agents.decision import (
    PredictionRoutingAgent,
    SegmentationSelectionAgent,
    TissueSelectionAgent,
)
from adaptivehb.agents.perception import QualityAssessmentAgent, ROIVerificationAgent
from adaptivehb.agents.schema import WorkflowResult
from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager

# Config key -> agent class.
_AGENT_CLASSES: dict[str, type[Agent]] = {
    "quality_assessment": QualityAssessmentAgent,
    "roi_verification": ROIVerificationAgent,
    "segmentation_selection": SegmentationSelectionAgent,
    "tissue_selection": TissueSelectionAgent,
    "prediction_routing": PredictionRoutingAgent,
    "fusion": FusionAgent,
    "confidence": ConfidenceAgent,
}


class AgentManager(BaseManager):
    """Builds the adaptive agents and runs the deterministic workflow."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the agent manager (agents built lazily at initialize())."""
        super().__init__(config, base_dir)
        self._agents_config = AgentsConfig.from_section(config.section("agents"))
        self._agents: dict[str, Agent] = {}
        self._controller: WorkflowController | None = None

    @property
    def agents_config(self) -> AgentsConfig:
        """The typed agents configuration."""
        return self._agents_config

    @property
    def agents(self) -> dict[str, Agent]:
        """The enabled agents, keyed by name."""
        return self._agents

    def enabled_agents(self) -> list[str]:
        """Return the enabled agent names in execution order."""
        return self._agents_config.enabled_in_order()

    @property
    def controller(self) -> WorkflowController:
        """The workflow controller (available after initialize())."""
        if self._controller is None:
            raise RuntimeError("AgentManager.initialize() must be called first.")
        return self._controller

    def _on_initialize(self) -> None:
        self._agents = {}
        for name in AGENT_ORDER:
            if name not in _AGENT_CLASSES or not self._agents_config.is_enabled(name):
                continue
            agent = _AGENT_CLASSES[name](name=name, config=self._agents_config.for_agent(name))
            agent.initialize()
            self._agents[name] = agent
        self._controller = WorkflowController(
            list(self._agents.values()),
            log_decisions=self._agents_config.log_decisions,
            logger=self._log,
        )
        self._log.info("AgentManager initialized: %s.", list(self._agents))

    def run_workflow(self, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Run the adaptive decision workflow over ``context``."""
        return self.controller.run(context)


__all__ = ["AgentManager"]
