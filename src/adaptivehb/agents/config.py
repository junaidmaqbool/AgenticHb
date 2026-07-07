"""Typed configuration for the adaptive decision framework (agents.yaml)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Canonical execution order of the agents (AGENT_SPECIFICATION Ch.22).
AGENT_ORDER: tuple[str, ...] = (
    "quality_assessment",
    "roi_verification",
    "segmentation_selection",
    "tissue_selection",
    "prediction_routing",
    "fusion",
    "confidence",
)


@dataclass(frozen=True)
class AgentsConfig:
    """Typed view of the ``agents`` configuration section."""

    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_decisions: bool = True

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> AgentsConfig:
        """Build an :class:`AgentsConfig` from the parsed section.

        Accepts either the full ``{"agents": {...}}`` mapping or the inner
        mapping directly. The ``workflow_controller`` sub-mapping is treated as
        controller settings, not an agent.
        """
        if "agents" in section and isinstance(section["agents"], Mapping):
            section = section["agents"]
        controller = dict(section.get("workflow_controller", {}))
        agents = {
            str(name): dict(spec)
            for name, spec in section.items()
            if name != "workflow_controller" and isinstance(spec, Mapping)
        }
        return cls(agents=agents, log_decisions=bool(controller.get("log_decisions", True)))

    def for_agent(self, name: str) -> dict[str, Any]:
        """Return the raw configuration for an agent (empty if absent)."""
        return self.agents.get(name, {})

    def is_enabled(self, name: str) -> bool:
        """Whether an agent is enabled (default true when present/absent)."""
        return bool(self.for_agent(name).get("enabled", True))

    def enabled_in_order(self) -> list[str]:
        """Return enabled agent names in canonical execution order."""
        return [name for name in AGENT_ORDER if self.is_enabled(name)]


__all__ = ["AgentsConfig", "AGENT_ORDER"]
