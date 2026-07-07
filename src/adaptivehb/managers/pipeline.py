"""PipelineManager — internal orchestrator of the framework.

The PipelineManager owns the infrastructure managers plus the domain managers
(Dataset, Segmentation, Prediction, Agents, Evaluation, Deployment), initializes
them in the documented order (PIPELINE_SPEC Ch.10), seeds the RNGs for
reproducibility, and dispatches execution by mode (PIPELINE_SPEC Ch.7). All six
modes are implemented: BUILD, TRAINING, RESUME, EVALUATION, INFERENCE, DEPLOYMENT.

Models are produced by an injectable trainable factory. The default factory
dispatches by category: SEGMENTATION plans are built by the SegmentationManager,
PREDICTION plans by the PredictionManager; other categories use dummy models
until their phases land (Decision 020). ``HbPipeline`` (in
:mod:`adaptivehb.pipeline`) is the thin public facade over this class
(Decision 012).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from adaptivehb.agents.manager import AgentManager
from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.types import ModelCategory, PipelineMode
from adaptivehb.core.utils import set_global_seed
from adaptivehb.dataset.manager import DatasetManager
from adaptivehb.deployment.manager import DeploymentManager
from adaptivehb.evaluation.manager import EvaluationManager
from adaptivehb.exceptions import PipelineError
from adaptivehb.managers import pipeline_modes
from adaptivehb.managers.checkpoint import CheckpointManager
from adaptivehb.managers.experiment import ExperimentManager
from adaptivehb.managers.jobs import Job, JobQueue
from adaptivehb.managers.registry import RegistryManager
from adaptivehb.managers.state import StateManager
from adaptivehb.managers.training import TrainingManager
from adaptivehb.models.dummy import make_dummy_factory
from adaptivehb.prediction.manager import PredictionManager
from adaptivehb.reporting.manager import ReportingManager
from adaptivehb.segmentation.manager import SegmentationManager


class PipelineManager(BaseManager):
    """Coordinates every manager and dispatches by execution mode."""

    def __init__(
        self,
        config: FrameworkConfig,
        base_dir: str | Path = ".",
        dataset_root: str | Path | None = None,
        trainable_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize the pipeline manager and its sub-managers."""
        super().__init__(config, base_dir)
        self.registry = RegistryManager(config, base_dir)
        self.state = StateManager(config, base_dir)
        self.checkpoints = CheckpointManager(config, base_dir)
        self.experiments = ExperimentManager(config, base_dir)
        self.dataset = DatasetManager(config, base_dir, dataset_root=dataset_root)
        # Optional SEPARATE segmentation dataset (its own images + masks, no labels
        # required). Falls back to the main dataset when not configured (Decision 039).
        self.segmentation_dataset = self._build_segmentation_dataset(config, base_dir)
        self.segmentation = SegmentationManager(config, base_dir)
        self.prediction = PredictionManager(config, base_dir)
        self.agents = AgentManager(config, base_dir)
        self.evaluation = EvaluationManager(config, base_dir)
        self.reporting = ReportingManager(config, base_dir)
        self.deployment = DeploymentManager(config, base_dir)
        self.training = TrainingManager(
            config, base_dir, self.checkpoints, self.state, self.registry
        )
        self._trainable_factory = trainable_factory or self._default_trainable_factory()

    @property
    def trainable_factory(self) -> Callable[[Any], Any]:
        """Factory mapping a training plan to a trainable model."""
        return self._trainable_factory

    def _default_trainable_factory(self) -> Callable[[Any], Any]:
        """Build the category-dispatching default factory."""
        dummy = make_dummy_factory()

        def _factory(plan: Any) -> Any:
            category = getattr(plan, "category", None)
            if category is ModelCategory.SEGMENTATION:
                return self.segmentation.build_trainable(plan)
            if category is ModelCategory.PREDICTION:
                return self.prediction.build_trainable(plan)
            return dummy(plan)

        return _factory

    def _build_segmentation_dataset(
        self, config: FrameworkConfig, base_dir: str | Path
    ) -> DatasetManager:
        """Return a distinct DatasetManager for the segmentation source, or the
        main dataset when no separate segmentation root is configured."""
        ds_config = self.dataset.dataset_config
        seg_root = ds_config.segmentation_root
        if not seg_root:
            return self.dataset
        return DatasetManager(
            config,
            base_dir,
            dataset_root=seg_root,
            images_dir=ds_config.segmentation_images_dir,
            masks_dir=ds_config.segmentation_masks_dir,
            metadata_file=ds_config.segmentation_metadata_file,
            metadata_optional=True,
        )

    def _on_initialize(self) -> None:
        set_global_seed(
            self._config.project.seed, deterministic=self._config.project.deterministic
        )
        # Documented order: infrastructure, dataset, then domain managers.
        for manager in (
            self.registry,
            self.state,
            self.checkpoints,
            self.experiments,
            self.dataset,
            *( (self.segmentation_dataset,) if self.segmentation_dataset is not self.dataset else () ),
            self.segmentation,
            self.prediction,
            self.agents,
            self.evaluation,
            self.reporting,
            self.deployment,
            self.training,
        ):
            manager.initialize()
        self._log.info("PipelineManager initialized (seed=%d).", self._config.project.seed)

    def run(
        self, mode: PipelineMode | str = PipelineMode.BUILD, *, epochs: int | None = None
    ) -> dict[str, Any]:
        """Execute the pipeline in the requested mode."""
        if not self._initialized:
            raise PipelineError("PipelineManager.initialize() must be called first.")
        resolved = self._coerce_mode(mode)

        if resolved is PipelineMode.BUILD:
            return self._run_build()
        if resolved is PipelineMode.TRAINING:
            return pipeline_modes.run_training(self, epochs=epochs)
        if resolved is PipelineMode.RESUME:
            return pipeline_modes.run_training(self, epochs=epochs, resume=True)
        if resolved is PipelineMode.EVALUATION:
            return pipeline_modes.run_evaluation(self)
        if resolved is PipelineMode.INFERENCE:
            return pipeline_modes.run_inference(self)
        if resolved is PipelineMode.DEPLOYMENT:
            return pipeline_modes.run_deployment(self)
        raise PipelineError(f"Pipeline mode '{resolved.value}' is not implemented.")

    def submit(self, jobs: list[Job], *, resume: bool = False) -> dict[str, str]:
        """Run a job sequence through the queue with optional resume."""
        queue = JobQueue(self._log)
        for job in jobs:
            queue.add(job)
        is_completed: Callable[[str], bool] | None = (
            self.state.is_completed if resume else None
        )
        on_complete: Callable[[str], None] | None = (
            self.state.mark_completed if resume else None
        )
        return queue.run(is_completed=is_completed, on_complete=on_complete)

    # -- BUILD mode --------------------------------------------------------

    def _run_build(self) -> dict[str, Any]:
        """Validate every manager via a dependency-ordered self-check queue."""
        queue = JobQueue(self._log)
        queue.add(Job("check_registry", self.registry.report))
        queue.add(Job("check_state", self.state.snapshot))
        queue.add(Job("check_checkpoints", self.checkpoints.list_checkpoints))
        queue.add(Job("check_segmentation", self.segmentation.available))
        queue.add(Job("check_prediction", self.prediction.available))
        queue.add(Job("check_agents", self.agents.enabled_agents))
        queue.add(Job("check_evaluation", lambda: list(self.evaluation.evaluation_config.regression_metrics)))
        queue.add(Job("check_reporting", lambda: sorted(self.reporting.available())))
        queue.add(Job("check_deployment", self.deployment.available_targets))
        queue.add(
            Job(
                "check_experiments",
                self.experiments.list_experiments,
                depends_on=["check_registry"],
            )
        )
        statuses = queue.run()
        self.state.update(current_phase="build", status="completed")
        self._log.info("Build mode completed: %s.", statuses)
        return {"mode": PipelineMode.BUILD.value, "jobs": statuses}

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _coerce_mode(mode: PipelineMode | str) -> PipelineMode:
        if isinstance(mode, PipelineMode):
            return mode
        try:
            return PipelineMode(mode)
        except ValueError as exc:
            raise PipelineError(f"Unknown pipeline mode: {mode!r}") from exc

    def _on_shutdown(self) -> None:
        for manager in (
            self.training,
            self.deployment,
            self.reporting,
            self.evaluation,
            self.agents,
            self.prediction,
            self.segmentation,
            self.dataset,
            self.experiments,
            self.checkpoints,
            self.state,
            self.registry,
        ):
            manager.shutdown()


__all__ = ["PipelineManager"]
