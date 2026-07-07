"""Common base interfaces shared across the framework.

These contracts keep managers, models, and agents uniform and replaceable
(PROJECT_DESIGN_SPECIFICATION Ch.20, AGENT_SPECIFICATION Ch.5). ``BaseManager``
provides shared plumbing (config, resolved base directory, logger, lifecycle
hooks). ``BaseModel`` and ``BaseAgent`` are abstract contracts implemented by
later phases; they are declared now so infrastructure can depend on the
interface rather than concrete classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.logging import get_logger


class BaseManager:
    """Base class for every framework manager.

    Managers coordinate a single responsibility and never call one another
    directly (all coordination flows through the pipeline). This base supplies
    the configuration, a resolved base directory, a namespaced logger, and
    idempotent ``initialize``/``shutdown`` lifecycle hooks.
    """

    def __init__(
        self,
        config: FrameworkConfig,
        base_dir: str | Path = ".",
        logger: Any = None,
    ) -> None:
        """Initialize the manager.

        Args:
            config: The validated framework configuration.
            base_dir: Root directory against which output paths are resolved.
            logger: Optional logger; a namespaced framework logger is created
                when omitted.
        """
        self._config = config
        self._base_dir = Path(base_dir)
        self._log = logger or get_logger(self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the manager's name (its class name)."""
        return type(self).__name__

    @property
    def config(self) -> FrameworkConfig:
        """Return the framework configuration."""
        return self._config

    @property
    def base_dir(self) -> Path:
        """Return the resolved base directory."""
        return self._base_dir

    @property
    def logger(self) -> Any:
        """Return the manager's logger."""
        return self._log

    @property
    def initialized(self) -> bool:
        """Return whether :meth:`initialize` has completed."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the manager (idempotent). Subclasses override the hook."""
        if self._initialized:
            return
        self._on_initialize()
        self._initialized = True
        self._log.debug("%s initialized.", self.name)

    def shutdown(self) -> None:
        """Release resources held by the manager. Subclasses override the hook."""
        self._on_shutdown()
        self._initialized = False
        self._log.debug("%s shut down.", self.name)

    def _on_initialize(self) -> None:
        """Subclass initialization hook. Default: no-op."""

    def _on_shutdown(self) -> None:
        """Subclass shutdown hook. Default: no-op."""


class BaseModel(ABC):
    """Abstract contract for trainable models (segmentation, prediction)."""

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train the model."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference."""

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the model and return metrics."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model weights."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model weights."""


class BaseAgent(ABC):
    """Abstract contract for adaptive decision modules (AGENT_SPECIFICATION)."""

    @abstractmethod
    def initialize(self) -> None:
        """Prepare the agent for use."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Produce a decision from the agent's declared inputs."""

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train the agent (deterministic agents may treat this as a no-op)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the agent's state."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore the agent's state."""


__all__ = ["BaseManager", "BaseModel", "BaseAgent"]
