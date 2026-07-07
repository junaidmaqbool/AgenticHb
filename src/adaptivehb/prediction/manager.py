"""PredictionManager — builds and provides per-tissue Hb prediction models.

Single responsibility: construct interchangeable prediction models by name (or
per-tissue routing) from configuration, and expose them as trainables for the
TrainingManager. It never trains directly — the generic TrainingManager drives
the epoch loop and the RegistryManager records results (Decision 012 roster).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.prediction.base import PredictionModel
from adaptivehb.prediction.config import PredictionConfig
from adaptivehb.model_loading import load_weights_into
from adaptivehb.prediction.registry import available_prediction, build_prediction

# Import for registration side effects (registers torch builders if present).
from adaptivehb.prediction import torch_models as _torch_models  # noqa: F401
from adaptivehb.prediction import reference as _reference  # noqa: F401


class PredictionManager(BaseManager):
    """Creates hemoglobin prediction models from configuration via the factory."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the prediction manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory (for resolving checkpoint paths).
        """
        super().__init__(config, base_dir)
        self._pred_config = PredictionConfig.from_section(config.section("prediction"))

    @property
    def prediction_config(self) -> PredictionConfig:
        """The typed prediction configuration."""
        return self._pred_config

    def available(self) -> list[str]:
        """Return the prediction architectures available in this environment."""
        return available_prediction()

    def architecture_for_tissue(self, tissue: str) -> str:
        """Return the configured architecture for a tissue (or the default)."""
        return self._pred_config.architecture_for_tissue(tissue)

    def build(
        self, architecture: str | None = None, tissue: str | None = None, **kwargs: Any
    ) -> PredictionModel:
        """Build (and construct) a prediction model.

        Args:
            architecture: Architecture name; defaults to the configured default
                (or the tissue-specific model when ``tissue`` is given).
            tissue: Optional tissue class; selects the tissue-specific model when
                ``architecture`` is not provided.
            **kwargs: Forwarded to the model builder.

        Returns:
            A built :class:`PredictionModel`.
        """
        name = architecture or (
            self.architecture_for_tissue(tissue) if tissue else self._pred_config.default_model
        )
        model = build_prediction(name, tissue=tissue, config=self._pred_config, **kwargs)
        model.build()
        self._log.info(
            "Built prediction model '%s' (tissue=%s, %s).", name, tissue, type(model).__name__
        )
        return model

    def build_trainable(self, plan: Any) -> PredictionModel:
        """Build a prediction model for a training plan (Trainable-compatible).

        The plan's ``architecture`` selects the backbone; its ``name`` (e.g.
        ``"hb_eye"``) yields the tissue when it follows the ``hb_<tissue>``
        convention.
        """
        architecture = getattr(plan, "architecture", None)
        plan_name = getattr(plan, "name", "")
        tissue = plan_name[3:] if plan_name.startswith("hb_") else None
        return self.build(architecture=architecture, tissue=tissue)

    def load_trained(
        self,
        name: str,
        checkpoints: Any,
        *,
        tissue: str | None = None,
        architecture: str | None = None,
        prefer: str = "best",
    ) -> PredictionModel:
        """Build a prediction model and load its trained weights from a checkpoint.

        Args:
            name: Checkpoint/model name (e.g. ``"hb_eye"``).
            checkpoints: The CheckpointManager holding trained weights.
            tissue: Optional tissue (drives architecture routing).
            architecture: Explicit architecture (overrides tissue routing).
            prefer: Preferred checkpoint tag.

        Returns:
            A built model with trained weights loaded when available; otherwise an
            untrained model (loading is best-effort).
        """
        model = self.build(architecture=architecture, tissue=tissue)
        load_weights_into(model, checkpoints, name, prefer=prefer, logger=self._log)
        return model


__all__ = ["PredictionManager"]
