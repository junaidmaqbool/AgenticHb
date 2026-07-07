"""Structured outputs for the adaptive decision framework.

Agents never return arbitrary data (AGENT_SPECIFICATION Ch.9): each returns an
:class:`AgentDecision` with a named-outputs mapping and an interpretable reason.
The Workflow Controller aggregates decisions into a :class:`WorkflowResult`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentDecision:
    """A single agent's structured, interpretable decision."""

    agent: str
    outputs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision to a dictionary."""
        return asdict(self)


@dataclass
class WorkflowResult:
    """Aggregated outcome of an adaptive-workflow run."""

    final_hb: float | None = None
    confidence: float | None = None
    recommendation: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    decisions: list[AgentDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result (decisions expanded to dicts)."""
        return {
            "final_hb": self.final_hb,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "decisions": [d.to_dict() for d in self.decisions],
        }


__all__ = ["AgentDecision", "WorkflowResult"]
