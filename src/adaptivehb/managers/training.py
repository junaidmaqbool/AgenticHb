"""TrainingManager — a generic, resumable training loop.

The manager drives any object satisfying the :class:`Trainable` protocol through
an epoch loop, wiring together the CheckpointManager (latest/best checkpoints),
the StateManager (progress + resume), and, on success, the RegistryManager
(automatic registration). It is model-agnostic and torch-free: concrete
segmentation/prediction models (later phases) only need to implement the
``Trainable`` methods. Training is decoupled from experiments (Decision 007) and
supports checkpoint recovery (Decision 009).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.types import ModelCategory, ModelRecord, ModelStatus
from adaptivehb.exceptions import PipelineError
from adaptivehb.managers.checkpoint import CheckpointManager
from adaptivehb.managers.registry import RegistryManager
from adaptivehb.managers.state import StateManager


@runtime_checkable
class Trainable(Protocol):
    """Contract a model must satisfy to be driven by the TrainingManager."""

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return its metrics."""

    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation for an epoch and return its metrics."""

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of trainable state."""

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trainable state from a snapshot."""


@dataclass
class TrainingPlan:
    """Declarative training configuration for a single model.

    Attributes:
        name: Checkpoint/registry name for the model.
        epochs: Total number of epochs to run.
        monitor: Metric key used to select the best checkpoint.
        direction: ``"min"`` (lower is better) or ``"max"``.
        patience: Early-stopping patience in epochs (``None`` disables it).
        checkpoint_frequency: Save a ``latest`` checkpoint every N epochs.
        category: Registry category; when set (with the fields below), the model
            is registered after training.
        task: Task label for the registry record.
        architecture: Architecture label for the registry record.
    """

    name: str
    epochs: int
    monitor: str = "val_loss"
    direction: str = "min"
    patience: int | None = None
    checkpoint_frequency: int = 1
    category: ModelCategory | None = None
    task: str = ""
    architecture: str = ""


@dataclass
class TrainingResult:
    """Outcome of a training run."""

    name: str
    epochs_run: int
    best_epoch: int
    best_metric: float
    stopped_early: bool
    registered_id: str | None
    history: list[dict[str, float]] = field(default_factory=list)


class TrainingManager(BaseManager):
    """Runs resumable training loops and integrates the infrastructure managers."""

    def __init__(
        self,
        config: FrameworkConfig,
        base_dir: str | Path = ".",
        checkpoints: CheckpointManager | None = None,
        state: StateManager | None = None,
        registry: RegistryManager | None = None,
    ) -> None:
        """Initialize the training manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory (used to record checkpoint references).
            checkpoints: Checkpoint manager (required to save/resume).
            state: Optional state manager for progress tracking.
            registry: Optional registry manager for automatic registration.
        """
        super().__init__(config, base_dir)
        self._checkpoints = checkpoints
        self._state = state
        self._registry = registry

    def train(
        self, trainable: Trainable, plan: TrainingPlan, *, resume: bool = False
    ) -> TrainingResult:
        """Train a model to completion (or early stop) and register it.

        Args:
            trainable: Object implementing the :class:`Trainable` protocol.
            plan: Training configuration.
            resume: When true, resume from the model's ``latest`` checkpoint if
                one exists.

        Returns:
            A :class:`TrainingResult`.

        Raises:
            PipelineError: If no CheckpointManager is configured, the monitored
                metric is absent, or the plan is invalid.
        """
        if self._checkpoints is None:
            raise PipelineError("TrainingManager requires a CheckpointManager.")
        if plan.direction not in {"min", "max"}:
            raise PipelineError(f"Invalid direction: {plan.direction!r}.")

        start_epoch, best_metric, best_epoch, history = self._resume_or_start(plan, trainable, resume)
        wait = 0
        stopped_early = False
        epochs_run = start_epoch - 1

        for epoch in range(start_epoch, plan.epochs + 1):
            metrics = dict(trainable.train_epoch(epoch))
            metrics.update(trainable.validate(epoch))
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            epochs_run = epoch

            if plan.monitor not in metrics:
                raise PipelineError(
                    f"Monitored metric {plan.monitor!r} not produced at epoch {epoch}."
                )
            monitored = metrics[plan.monitor]
            is_best = best_metric is None or self._is_better(
                monitored, best_metric, plan.direction
            )
            if is_best:
                best_metric, best_epoch, wait = monitored, epoch, 0
            else:
                wait += 1

            self._record_state(plan.name, epoch)
            if is_best or epoch % max(plan.checkpoint_frequency, 1) == 0:
                self._save_checkpoint(plan, trainable, epoch, metrics, best_metric, best_epoch, history, is_best)

            if plan.patience is not None and wait >= plan.patience:
                stopped_early = True
                self._log.info("Early stopping at epoch %d (patience=%d).", epoch, plan.patience)
                break

        registered_id = self._register(plan, best_metric)
        if self._state is not None:
            self._state.mark_completed(plan.name)
        self._log.info(
            "Training '%s' done: best %s=%.6g @epoch %d.",
            plan.name, plan.monitor, best_metric if best_metric is not None else float("nan"), best_epoch,
        )
        return TrainingResult(
            name=plan.name,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_metric=best_metric if best_metric is not None else float("nan"),
            stopped_early=stopped_early,
            registered_id=registered_id,
            history=history,
        )

    # -- internals ---------------------------------------------------------

    def _resume_or_start(
        self, plan: TrainingPlan, trainable: Trainable, resume: bool
    ) -> tuple[int, float | None, int, list[dict[str, float]]]:
        assert self._checkpoints is not None
        if resume and self._checkpoints.exists(plan.name):
            payload, meta = self._checkpoints.load_latest(plan.name)
            trainable.load_state_dict(payload.get("model_state", {}))
            start_epoch = int(payload.get("epoch", 0)) + 1
            best_metric = meta.get("best_metric")
            best_epoch = int(meta.get("best_epoch", 0))
            history = list(payload.get("history", []))
            self._log.info("Resuming '%s' from epoch %d.", plan.name, start_epoch)
            return start_epoch, best_metric, best_epoch, history
        return 1, None, 0, []

    def _save_checkpoint(
        self,
        plan: TrainingPlan,
        trainable: Trainable,
        epoch: int,
        metrics: dict[str, float],
        best_metric: float | None,
        best_epoch: int,
        history: list[dict[str, float]],
        is_best: bool,
    ) -> None:
        assert self._checkpoints is not None
        payload = {"epoch": epoch, "model_state": trainable.state_dict(), "history": history}
        meta = {
            "epoch": epoch,
            "metrics": metrics,
            "monitor": plan.monitor,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        self._checkpoints.save(plan.name, payload, meta, is_best=is_best)

    def _record_state(self, name: str, epoch: int) -> None:
        if self._state is not None:
            self._state.update(current_module=name, current_epoch=epoch, status="training")

    def _register(self, plan: TrainingPlan, best_metric: float | None) -> str | None:
        if self._registry is None or plan.category is None or best_metric is None:
            return None
        checkpoint_dir = self._base_dir / self._config.project.paths.checkpoints / plan.name
        record = ModelRecord(
            name=plan.name,
            category=plan.category,
            task=plan.task or plan.name,
            architecture=plan.architecture or plan.name,
            checkpoint_path=str(checkpoint_dir),
            metrics={plan.monitor: best_metric},
            status=ModelStatus.EXPERIMENTAL,
            seed=self._config.project.seed,
        )
        return self._registry.register(record).unique_id

    @staticmethod
    def _is_better(candidate: float, best: float, direction: str) -> bool:
        return candidate < best if direction == "min" else candidate > best


__all__ = ["TrainingManager", "TrainingPlan", "TrainingResult", "Trainable"]
