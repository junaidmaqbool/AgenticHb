"""HbPipeline — the single public entry point of the framework.

The notebook (or any client) interacts only with :class:`HbPipeline`, never with
individual managers (PIPELINE_SPEC Ch.2, Decision 001). The facade loads and
validates configuration, configures logging, and delegates every operation to
the internal :class:`~adaptivehb.managers.pipeline.PipelineManager` via a small,
stable public API (PIPELINE_SPEC Ch.23).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from adaptivehb.config.loader import ConfigLoader, FrameworkConfig
from adaptivehb.core.types import PipelineMode
from adaptivehb.logging import setup_logging
from adaptivehb.managers.pipeline import PipelineManager


class HbPipeline:
    """Public facade coordinating the entire framework."""

    def __init__(
        self,
        config: FrameworkConfig,
        base_dir: str | Path = ".",
        dataset_root: str | Path | None = None,
        trainable_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize the pipeline from a validated configuration.

        Prefer :meth:`from_config_dir` to build directly from a config directory.

        Args:
            config: A validated framework configuration.
            base_dir: Root directory for all framework outputs.
            dataset_root: Explicit dataset root (overrides config).
            trainable_factory: Optional model factory (defaults to dummy models).
        """
        self._config = config
        self._base_dir = Path(base_dir)
        self._manager = PipelineManager(
            config, base_dir, dataset_root=dataset_root, trainable_factory=trainable_factory
        )
        self._initialized = False

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str | Path = "configs",
        base_dir: str | Path = ".",
        dataset_root: str | Path | None = None,
    ) -> HbPipeline:
        """Build a pipeline by loading configuration from a directory.

        Args:
            config_dir: Directory containing the nine ``*.yaml`` config files.
            base_dir: Root directory for all framework outputs.
            dataset_root: Explicit dataset root (overrides config).

        Returns:
            An un-initialized :class:`HbPipeline`.
        """
        config = ConfigLoader(config_dir).load()
        return cls(config, base_dir, dataset_root=dataset_root)

    @property
    def config(self) -> FrameworkConfig:
        """Return the loaded framework configuration."""
        return self._config

    @property
    def manager(self) -> PipelineManager:
        """Return the internal pipeline manager."""
        return self._manager

    def initialize(self) -> HbPipeline:
        """Configure logging and initialize all managers (idempotent)."""
        if not self._initialized:
            setup_logging(self._config.logging, root_dir=self._base_dir)
            self._manager.initialize()
            self._initialized = True
        return self

    def run(
        self, mode: PipelineMode | str = PipelineMode.BUILD, *, epochs: int | None = None
    ) -> dict[str, Any]:
        """Initialize if needed and execute the requested mode."""
        self.initialize()
        return self._manager.run(mode, epochs=epochs)

    # -- convenience API (PIPELINE_SPEC Ch.23) -----------------------------

    def build(self) -> dict[str, Any]:
        """Run BUILD mode: validate the framework with dummy self-checks."""
        return self.run(PipelineMode.BUILD)

    def train(self, *, epochs: int | None = None) -> dict[str, Any]:
        """Run TRAINING mode: validate → split → train models → register."""
        return self.run(PipelineMode.TRAINING, epochs=epochs)

    def resume(self, *, epochs: int | None = None) -> dict[str, Any]:
        """Run RESUME mode: continue training from checkpoints."""
        return self.run(PipelineMode.RESUME, epochs=epochs)

    def evaluate(self) -> dict[str, Any]:
        """Run EVALUATION mode over registered models."""
        return self.run(PipelineMode.EVALUATION)

    def test(self) -> dict[str, Any]:
        """Alias for evaluation on the held-out test split."""
        return self.run(PipelineMode.EVALUATION)

    def predict(self) -> dict[str, Any]:
        """Run INFERENCE mode over held-out samples."""
        return self.run(PipelineMode.INFERENCE)

    def deploy(self) -> dict[str, Any]:
        """Run DEPLOYMENT mode (available once deployment exists, Phase 9)."""
        return self.run(PipelineMode.DEPLOYMENT)

    def experiment(self, name: str = "experiment", *, epochs: int | None = None) -> Any:
        """Run a full baseline-vs-adaptive experiment and archive its outputs.

        Trains the models, evaluates the static baseline against the adaptive
        pipeline on the test split, generates figures/tables, and writes
        everything into a fresh experiment directory (EXPERIMENT_SPEC).
        """
        from adaptivehb.experiment import ExperimentRunner

        return ExperimentRunner(self).run(name, epochs=epochs)

    def cross_validate(
        self, name: str = "cross_validation", *, folds: int = 5, epochs: int | None = None
    ) -> Any:
        """Run patient-level k-fold cross-validation and archive aggregated results.

        Each fold is an isolated experiment (own directory) trained on the fold's
        training patients and evaluated on its held-out test patients; per-fold
        metrics and comparisons are aggregated into mean/std summaries
        (EXPERIMENT_SPEC; Decision 035).
        """
        from adaptivehb.crossval import CrossValidationRunner

        runner = CrossValidationRunner(
            self._config,
            base_dir=self._base_dir,
            dataset_root=self._manager.dataset.root,
            folds=folds,
            seed=self._config.project.seed,
        )
        return runner.run(name, epochs=epochs)

    def shutdown(self) -> None:
        """Shut down all managers."""
        self._manager.shutdown()
        self._initialized = False


__all__ = ["HbPipeline"]
