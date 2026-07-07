"""Workflow Controller — deterministic orchestrator of the adaptive agents.

Unlike the agents, the controller makes no clinical decisions: it initializes
modules, schedules execution in dependency order, passes each agent's structured
outputs to the next via a shared context, logs decisions, and assembles the final
result (AGENT_SPECIFICATION Ch.21-22). Adaptive behavior belongs only to the
agents; the controller remains deterministic.
"""

from __future__ import annotations

from typing import Any

from adaptivehb.agents.base import Agent
from adaptivehb.agents.schema import AgentDecision, WorkflowResult


class WorkflowController:
    """Runs an ordered sequence of agents over a shared context."""

    def __init__(self, agents: list[Agent], *, log_decisions: bool = True, logger: Any = None) -> None:
        """Initialize the controller.

        Args:
            agents: Enabled agents in execution order.
            log_decisions: Whether to log every agent decision.
            logger: Optional logger.
        """
        self._agents = agents
        self._log_decisions = log_decisions
        self._log = logger

    @property
    def agents(self) -> list[Agent]:
        """The ordered agents this controller runs."""
        return self._agents

    def run(self, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute the adaptive workflow.

        Args:
            context: Initial context (per-tissue features, routing hints, etc.).
                It is copied; the original is not mutated.

        Returns:
            A :class:`WorkflowResult` with the fused estimate, confidence,
            recommendation, final context, and the ordered decisions.
        """
        working: dict[str, Any] = dict(context or {})
        decisions: list[AgentDecision] = []

        for agent in self._agents:
            decision = agent.predict(working)
            decisions.append(decision)
            # Agents communicate only via the controller: merge structured
            # outputs into the shared context for downstream agents.
            working.update(decision.outputs)
            if self._log_decisions and self._log is not None:
                self._log.info("Agent %s: %s", agent.name, decision.reason)

        return WorkflowResult(
            final_hb=working.get("final_hb"),
            confidence=working.get("confidence"),
            recommendation=working.get("recommendation"),
            context=working,
            decisions=decisions,
        )


__all__ = ["WorkflowController"]
