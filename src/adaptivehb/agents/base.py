"""Common agent base for the adaptive decision framework.

Every adaptive module follows the same architecture and exposes the same
interface (AGENT_SPECIFICATION Ch.4-5): ``initialize``/``train``/``predict``/
``evaluate``/``save``/``load``/``reset``/``shutdown``. Agents are lightweight and
deterministic in this phase (no LLMs, no external APIs); learned policies can
replace deterministic ones later without changing the interface (Decision 005).

Agents never communicate directly — the Workflow Controller passes their
structured outputs (AGENT_SPECIFICATION Ch.7). Each agent reads named fields from
a shared context mapping and returns an :class:`AgentDecision`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from adaptivehb.agents.schema import AgentDecision
from adaptivehb.core.interfaces import BaseAgent
from adaptivehb.core.utils import ensure_dir
from adaptivehb.logging import get_logger


class Agent(BaseAgent):
    """Base class for deterministic (and future learned) decision agents."""

    def __init__(self, name: str, config: dict[str, Any] | None = None, logger: Any = None) -> None:
        """Initialize the agent.

        Args:
            name: Agent name (its configuration key, e.g. ``"quality_assessment"``).
            config: The agent's raw configuration sub-mapping.
            logger: Optional logger; a namespaced logger is created when omitted.
        """
        self.name = name
        self._config = dict(config or {})
        self._log = logger or get_logger(f"agent.{name}")
        self._initialized = False

    # -- properties --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether this agent is enabled."""
        return bool(self._config.get("enabled", True))

    @property
    def policy(self) -> str:
        """The agent's decision policy (``deterministic`` by default)."""
        return str(self._config.get("policy", "deterministic"))

    @property
    def config(self) -> dict[str, Any]:
        """The agent's raw configuration."""
        return self._config

    def threshold(self, key: str, default: float) -> float:
        """Return a named threshold from config (``thresholds.<key>``)."""
        return float(self._config.get("thresholds", {}).get(key, default))

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the agent for use (idempotent)."""
        self._initialized = True

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Deterministic agents do not train; returns ``None``."""
        return None

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate the agent; deterministic agents return an empty report."""
        return {}

    def reset(self) -> None:
        """Reset transient agent state (no-op for deterministic agents)."""

    def shutdown(self) -> None:
        """Release resources held by the agent."""
        self._initialized = False

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist agent state via pickle."""
        destination = Path(path)
        ensure_dir(destination.parent)
        with destination.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Restore agent state from a pickle file."""
        with Path(path).open("rb") as handle:
            self.load_state_dict(pickle.load(handle))

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the agent."""
        return {"name": self.name, "config": self._config}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the agent snapshot."""
        self.name = state.get("name", self.name)
        self._config = dict(state.get("config", self._config))

    # -- decision ----------------------------------------------------------

    def predict(self, context: dict[str, Any]) -> AgentDecision:  # type: ignore[override]
        """Produce a decision from the shared context. Implemented by subclasses."""
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _tissues(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Return the per-tissue candidate features from the context."""
        return dict(context.get("tissues", {}))

    def _decision(self, outputs: dict[str, Any], reason: str = "", confidence: float | None = None) -> AgentDecision:
        return AgentDecision(agent=self.name, outputs=outputs, reason=reason, confidence=confidence)


__all__ = ["Agent"]
