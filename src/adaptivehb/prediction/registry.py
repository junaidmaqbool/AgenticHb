"""Prediction model factory (Registry / Factory pattern).

Prediction backbones register a builder under a name; the pipeline and adaptive
framework construct models by name without importing concrete classes, so new
models are added without modifying the PipelineManager (Decision 004).

Graceful degradation mirrors segmentation (Decision 021): real (torch) backbones
register only when torch is importable; when a requested name has no registered
builder, the factory falls back to the torch-free
:class:`ReferencePredictionModel`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from adaptivehb.prediction.base import PredictionModel

PredictionBuilder = Callable[..., PredictionModel]

_BUILDERS: dict[str, PredictionBuilder] = {}


def register_prediction(name: str) -> Callable[[PredictionBuilder], PredictionBuilder]:
    """Decorator registering a prediction builder under ``name``."""

    def _decorator(builder: PredictionBuilder) -> PredictionBuilder:
        _BUILDERS[name.lower()] = builder
        return builder

    return _decorator


def available_prediction() -> list[str]:
    """Return the sorted names of all registered prediction builders."""
    return sorted(_BUILDERS)


def is_registered(name: str) -> bool:
    """Whether a real builder is registered for ``name``."""
    return name.lower() in _BUILDERS


def build_prediction(name: str, **kwargs: Any) -> PredictionModel:
    """Construct a prediction model by name (falls back to the reference model).

    Args:
        name: Architecture name.
        **kwargs: Forwarded to the builder (e.g. ``tissue``, ``config``).

    Returns:
        A :class:`PredictionModel` instance (not yet built).
    """
    key = name.lower()
    if key in _BUILDERS:
        return _BUILDERS[key](name=name, **kwargs)
    from adaptivehb.prediction.reference import ReferencePredictionModel

    return ReferencePredictionModel(name=name, **kwargs)


__all__ = [
    "register_prediction",
    "available_prediction",
    "is_registered",
    "build_prediction",
    "PredictionBuilder",
]
