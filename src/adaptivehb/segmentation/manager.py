"""SegmentationManager — builds and provides segmentation models.

Single responsibility: construct interchangeable segmentation models by name
from configuration, and expose them as trainables for the TrainingManager. It
never trains directly — the generic TrainingManager drives the epoch loop and
the RegistryManager records results (Decision 012 roster).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.segmentation.base import SegmentationModel
from adaptivehb.segmentation.config import SegmentationConfig
from adaptivehb.model_loading import load_weights_into
from adaptivehb.segmentation.registry import available_segmentation, build_segmentation

# Import for its registration side effects (registers torch builders if present).
from adaptivehb.segmentation import torch_models as _torch_models  # noqa: F401
from adaptivehb.segmentation import reference as _reference  # noqa: F401


class SegmentationManager(BaseManager):
    """Creates segmentation models from configuration via the factory."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the segmentation manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory (for resolving checkpoint paths).
        """
        super().__init__(config, base_dir)
        self._seg_config = SegmentationConfig.from_section(config.section("segmentation"))

    @property
    def segmentation_config(self) -> SegmentationConfig:
        """The typed segmentation configuration."""
        return self._seg_config

    def available(self) -> list[str]:
        """Return the segmentation architectures available in this environment."""
        return available_segmentation()

    def build(self, architecture: str | None = None, **kwargs: Any) -> SegmentationModel:
        """Build (and construct) a segmentation model by architecture name.

        Args:
            architecture: Name from the config's ``available_models``; defaults
                to the configured ``default_model``.
            **kwargs: Forwarded to the model builder.

        Returns:
            A built :class:`SegmentationModel`.
        """
        name = architecture or self._seg_config.default_model
        model = build_segmentation(name, config=self._seg_config, **kwargs)
        model.build()
        self._log.info("Built segmentation model '%s' (%s).", name, type(model).__name__)
        return model

    def load_trained(
        self, name: str, checkpoints: Any, *, architecture: str | None = None, prefer: str = "best"
    ) -> SegmentationModel:
        """Build a segmentation model and load its trained weights from a checkpoint.

        Returns a built model with trained weights when available; otherwise an
        untrained model (loading is best-effort).
        """
        model = self.build(architecture)
        load_weights_into(model, checkpoints, name, prefer=prefer, logger=self._log)
        return model

    def build_trainable(self, plan: Any) -> SegmentationModel:
        """Build a segmentation model for a training plan (Trainable-compatible).

        Args:
            plan: A TrainingPlan; its ``architecture`` selects the model.

        Returns:
            A built :class:`SegmentationModel` ready for the TrainingManager.
        """
        architecture = getattr(plan, "architecture", None) or self._seg_config.default_model
        return self.build(architecture)


__all__ = ["SegmentationManager"]
