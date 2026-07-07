"""Decision-layer agents (AGENT_SPECIFICATION Ch.16-18).

These agents choose the optimal computational pathway: which segmentation model
to run, which tissues to analyze, and which prediction model handles each tissue.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any

from adaptivehb.agents.base import Agent
from adaptivehb.agents.schema import AgentDecision


class SegmentationSelectionAgent(Agent):
    """Select a segmentation model based on input characteristics."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Choose a segmentation model from the available set by mean quality.

        Reads ``available_segmentation`` and per-tissue ``quality``; writes
        ``selected_segmentation`` and ``avg_quality``.
        """
        available = list(context.get("available_segmentation", ["unet", "segformer", "deeplabv3plus"]))
        tissues = self._tissues(context)
        candidates = context.get("verified_tissues") or context.get("accepted_tissues") or list(tissues)
        qualities = [float(tissues.get(t, {}).get("quality", 0.0)) for t in candidates]
        avg_quality = fmean(qualities) if qualities else 0.0

        # Deterministic policy: higher quality favours a lighter model.
        if avg_quality >= 0.8:
            preferred = "unet"
        elif avg_quality >= 0.6:
            preferred = "deeplabv3plus"
        else:
            preferred = "segformer"
        selected = preferred if preferred in available else (available[0] if available else preferred)
        return self._decision(
            {"selected_segmentation": selected, "avg_quality": round(avg_quality, 3)},
            reason=f"avg_quality={avg_quality:.2f} → {selected}",
        )


class TissueSelectionAgent(Agent):
    """Select which available tissues to analyze, in priority order."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Rank verified tissues by quality+ROI and select up to ``max_tissues``.

        Reads ``verified_tissues`` (or accepted/all) and per-tissue metrics;
        writes ``selected_tissues`` and ``tissue_priority``.
        """
        tissues = self._tissues(context)
        candidates = context.get("verified_tissues") or context.get("accepted_tissues") or list(tissues)

        def _score(tissue: str) -> float:
            data = tissues.get(tissue, {})
            return float(data.get("quality", 0.0)) + float(data.get("roi_iou", 0.0))

        ranked = sorted(candidates, key=_score, reverse=True)
        max_tissues = int(self._config.get("max_tissues", len(ranked) or 1))
        selected = ranked[: max(max_tissues, 1)] if ranked else []
        return self._decision(
            {"selected_tissues": selected, "tissue_priority": ranked},
            reason=f"selected {len(selected)}/{len(candidates)} tissue(s)",
        )


class PredictionRoutingAgent(Agent):
    """Assign a prediction model to each selected tissue."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Route each selected tissue to a prediction model.

        Reads ``selected_tissues``, ``tissue_models`` (mapping) and
        ``default_prediction_model``; writes ``prediction_routing``.
        """
        tissues = self._tissues(context)
        selected = context.get("selected_tissues") or list(tissues)
        tissue_models = dict(context.get("tissue_models", {}))
        default_model = str(context.get("default_prediction_model", "efficientnet"))
        routing = {tissue: tissue_models.get(tissue, default_model) for tissue in selected}
        return self._decision(
            {"prediction_routing": routing},
            reason=f"routed {len(routing)} tissue(s)",
        )


__all__ = [
    "SegmentationSelectionAgent",
    "TissueSelectionAgent",
    "PredictionRoutingAgent",
]
