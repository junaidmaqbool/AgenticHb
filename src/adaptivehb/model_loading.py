"""Checkpoint-backed loading of trained model weights.

Bridges the CheckpointManager (which stores each model's ``model_state`` under
its name) to a freshly built model, so evaluation, inference, and experiments use
the *trained* weights rather than an untrained/reference model (Decision 029). No
model is ever loaded from a hardcoded path — the checkpoint is located by name via
the CheckpointManager (MODEL_REGISTRY_SPEC Ch.15).
"""

from __future__ import annotations

from typing import Any


def load_weights_into(
    model: Any, checkpoints: Any, name: str, *, prefer: str = "best", logger: Any = None
) -> bool:
    """Load a model's trained weights from the checkpoint store, if present.

    Args:
        model: A built model exposing ``load_state_dict``.
        checkpoints: The CheckpointManager.
        name: Checkpoint/model name (e.g. ``"hb_eye"``, ``"seg_unet"``).
        prefer: Preferred checkpoint tag (``"best"`` then ``"latest"``).
        logger: Optional logger for status messages.

    Returns:
        True if weights were loaded; False when no checkpoint exists (the model is
        left untrained rather than raising, so callers degrade gracefully).
    """
    if checkpoints.exists(name, tag=prefer):
        tag = prefer
    elif checkpoints.exists(name, tag="latest"):
        tag = "latest"
    else:
        if logger is not None:
            logger.warning("No checkpoint for %r; using an untrained model.", name)
        return False

    payload, meta = checkpoints.load_best(name) if tag == "best" else checkpoints.load_latest(name)
    state = payload.get("model_state")
    if state is None:
        if logger is not None:
            logger.warning("Checkpoint %r has no model_state; using an untrained model.", name)
        return False
    model.load_state_dict(state)
    if logger is not None:
        logger.info("Loaded %r weights from %s checkpoint (epoch=%s).", name, tag, meta.get("epoch"))
    return True


__all__ = ["load_weights_into"]
