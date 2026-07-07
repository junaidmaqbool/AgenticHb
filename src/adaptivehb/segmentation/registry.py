"""Segmentation model factory (Registry / Factory pattern).

Segmentation architectures register a builder under a name; the pipeline and the
adaptive framework construct models by name without importing concrete classes,
so new segmentation models are added without modifying the PipelineManager
(IMPLEMENTATION_ROADMAP Phase 5).

Graceful degradation: real (torch) backends register themselves only when torch
is importable. When a requested name has no registered builder (e.g. torch is
absent in a framework-only environment), the factory falls back to the
torch-free :class:`ReferenceSegmentationModel` so the framework still runs
(Decision 021).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from adaptivehb.segmentation.base import SegmentationModel

SegmentationBuilder = Callable[..., SegmentationModel]

_BUILDERS: dict[str, SegmentationBuilder] = {}


def register_segmentation(name: str) -> Callable[[SegmentationBuilder], SegmentationBuilder]:
    """Decorator registering a segmentation builder under ``name``.

    Args:
        name: Case-insensitive architecture name.

    Returns:
        The decorator.
    """

    def _decorator(builder: SegmentationBuilder) -> SegmentationBuilder:
        _BUILDERS[name.lower()] = builder
        return builder

    return _decorator


def available_segmentation() -> list[str]:
    """Return the sorted names of all registered segmentation builders."""
    return sorted(_BUILDERS)


def is_registered(name: str) -> bool:
    """Whether a real builder is registered for ``name``."""
    return name.lower() in _BUILDERS


def build_segmentation(name: str, **kwargs: Any) -> SegmentationModel:
    """Construct a segmentation model by name.

    Falls back to the torch-free reference model when no real builder is
    registered for ``name`` (keeps the framework runnable without torch).

    Args:
        name: Architecture name.
        **kwargs: Forwarded to the builder (e.g. ``num_classes``, ``config``).

    Returns:
        A :class:`SegmentationModel` instance (not yet built).
    """
    from adaptivehb.segmentation.reference import ReferenceSegmentationModel

    # Explicit override: force the torch-free reference backend (e.g. for the
    # synthetic smoke test) regardless of whether torch builders are registered.
    if os.environ.get("ADAPTIVEHB_FORCE_REFERENCE"):
        return ReferenceSegmentationModel(name=name, **kwargs)
    key = name.lower()
    if key in _BUILDERS:
        return _BUILDERS[key](name=name, **kwargs)
    # Fallback: torch-free reference implementation under the requested name.
    return ReferenceSegmentationModel(name=name, **kwargs)


__all__ = [
    "register_segmentation",
    "available_segmentation",
    "is_registered",
    "build_segmentation",
    "SegmentationBuilder",
]
