"""Clinical-output agents (AGENT_SPECIFICATION Ch.19-20).

These agents produce the final, reliable estimate: dynamic fusion of per-tissue
predictions and a calibrated confidence with a clinical recommendation.
"""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Any

from adaptivehb.agents.base import Agent
from adaptivehb.agents.schema import AgentDecision


class FusionAgent(Agent):
    """Combine per-tissue hemoglobin predictions into one estimate."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Fuse predictions using the configured method.

        Reads ``selected_tissues`` and per-tissue ``pred_hb``/``pred_confidence``;
        writes ``final_hb`` and ``fusion_weights``. Unlike static averaging,
        confidence-weighted fusion favours more reliable tissues.
        """
        method = str(self._config.get("method", "confidence_weighted"))
        tissues = self._tissues(context)
        selected = context.get("selected_tissues") or list(tissues)
        contributions = [
            (t, float(tissues[t]["pred_hb"]), float(tissues[t].get("pred_confidence", 1.0)))
            for t in selected
            if "pred_hb" in tissues.get(t, {})
        ]
        if not contributions:
            return self._decision({"final_hb": None, "fusion_weights": {}}, reason="no predictions to fuse")

        if method == "confidence_weighted":
            total = sum(conf for _, _, conf in contributions) or 1.0
            final = sum(value * conf for _, value, conf in contributions) / total
            weights = {t: round(conf / total, 4) for t, _, conf in contributions}
        else:  # simple mean
            final = fmean(value for _, value, _ in contributions)
            weights = {t: round(1.0 / len(contributions), 4) for t, _, _ in contributions}

        return self._decision(
            {"final_hb": round(final, 2), "fusion_weights": weights, "fusion_method": method},
            reason=f"fused {len(contributions)} tissue(s) via {method}",
        )


class ConfidenceAgent(Agent):
    """Estimate prediction confidence, an interval, and a recommendation."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Derive confidence from prediction agreement and quality.

        Reads ``selected_tissues``, per-tissue ``pred_hb``/``pred_confidence``/
        ``quality`` and ``final_hb``; writes ``confidence``, ``interval`` and a
        clinical ``recommendation``.
        """
        tissues = self._tissues(context)
        selected = context.get("selected_tissues") or list(tissues)
        predictions = [float(tissues[t]["pred_hb"]) for t in selected if "pred_hb" in tissues.get(t, {})]
        confidences = [float(tissues[t].get("pred_confidence", 0.5)) for t in selected if t in tissues]
        qualities = [float(tissues[t].get("quality", 0.5)) for t in selected if t in tissues]

        if not predictions:
            return self._decision(
                {"confidence": 0.0, "interval": None, "recommendation": "insufficient_data"},
                reason="no predictions available",
            )

        disagreement = pstdev(predictions) if len(predictions) > 1 else 0.0
        base = fmean(confidences) if confidences else 0.5
        quality_factor = fmean(qualities) if qualities else 0.5
        # Higher agreement (low disagreement) and quality raise confidence.
        confidence = max(0.0, min(1.0, base * quality_factor / (1.0 + disagreement)))
        interval = round(0.5 + disagreement, 2)
        threshold = self.threshold("min_confidence", 0.8)
        recommendation = "reliable" if confidence >= threshold else "review_recommended"
        return self._decision(
            {
                "confidence": round(confidence, 3),
                "interval": interval,
                "recommendation": recommendation,
            },
            reason=f"confidence={confidence:.2f}, disagreement={disagreement:.2f}",
            confidence=round(confidence, 3),
        )


__all__ = ["FusionAgent", "ConfidenceAgent"]
