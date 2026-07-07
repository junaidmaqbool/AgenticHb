"""Perception-layer agents (AGENT_SPECIFICATION Ch.14-15).

These agents understand input suitability before any prediction: image quality
and segmentation-ROI validity. Poor inputs are flagged so they do not propagate.
"""

from __future__ import annotations

from typing import Any

from adaptivehb.agents.base import Agent
from adaptivehb.agents.schema import AgentDecision


class QualityAssessmentAgent(Agent):
    """Decide whether each tissue image is good enough to process."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Accept tissues whose quality meets the threshold.

        Reads ``tissues[*].quality``; writes ``accepted_tissues``,
        ``rejected_tissues`` and a ``reacquire`` recommendation.
        """
        min_quality = self.threshold("min_quality", 0.5)
        tissues = self._tissues(context)
        accepted = [t for t, d in tissues.items() if float(d.get("quality", 0.0)) >= min_quality]
        rejected = [t for t in tissues if t not in accepted]
        reacquire = len(accepted) == 0 and len(tissues) > 0
        reason = (
            "all tissues below quality threshold; recommend reacquisition"
            if reacquire
            else f"{len(accepted)}/{len(tissues)} tissue(s) passed quality"
        )
        return self._decision(
            {"accepted_tissues": accepted, "rejected_tissues": rejected, "reacquire": reacquire},
            reason=reason,
        )


class ROIVerificationAgent(Agent):
    """Verify that segmented regions of interest are usable for estimation."""

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Keep quality-accepted tissues whose ROI IoU meets the threshold.

        Reads ``accepted_tissues`` (or all tissues) and ``tissues[*].roi_iou``;
        writes ``verified_tissues`` and ``rejected_rois``.
        """
        min_iou = self.threshold("min_iou", 0.6)
        tissues = self._tissues(context)
        candidates = context.get("accepted_tissues") or list(tissues)
        verified = [t for t in candidates if float(tissues.get(t, {}).get("roi_iou", 0.0)) >= min_iou]
        rejected = [t for t in candidates if t not in verified]
        return self._decision(
            {"verified_tissues": verified, "rejected_rois": rejected},
            reason=f"{len(verified)}/{len(candidates)} ROI(s) verified",
        )


__all__ = ["QualityAssessmentAgent", "ROIVerificationAgent"]
