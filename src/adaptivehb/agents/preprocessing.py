"""Preprocessing-layer agent: clinical Hb-range gating + balanced sampling.

The :class:`PreprocessingAgent` is the agentic front door to the training-data
bridge. Before any model sees a sample it decides *which* samples are in scope
(a clinically meaningful Hb window, e.g. 6-14 g/dL) and *how often* each should
be drawn (inverse-frequency bin weights that break the "predict the mean"
plateau). The heavy lifting lives in torch-free primitives
(:mod:`adaptivehb.dataloading.preprocessing`); this agent wraps them behind the
common :class:`~adaptivehb.agents.base.Agent` interface so it is independently
configurable (``agents.yaml``), independently testable, and can be swapped for a
learned policy later without changing callers (Decision 005).

Numeric knobs (range bounds, bin count) come from ``dataset.preprocessing`` via a
:class:`~adaptivehb.dataloading.preprocessing.PreprocessingSpec`; the ``enabled``/
``policy`` flags come from ``agents.yaml``. Nothing is hardcoded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from adaptivehb.agents.base import Agent
from adaptivehb.agents.schema import AgentDecision
from adaptivehb.dataloading.preprocessing import (
    PreprocessingSpec,
    balanced_sample_weights,
    filter_hb_range,
)
from adaptivehb.dataset.schema import Sample


class PreprocessingAgent(Agent):
    """Gate samples to a clinical Hb range and balance them across Hb bins."""

    def __init__(
        self,
        name: str = "preprocessing",
        config: dict[str, Any] | None = None,
        *,
        spec: PreprocessingSpec | None = None,
        logger: Any = None,
    ) -> None:
        """Initialize the agent.

        Args:
            name: Agent name / config key.
            config: The agent's ``agents.yaml`` sub-mapping (``enabled``/``policy``).
            spec: The data-preprocessing spec (Hb-range + balanced-sampling knobs),
                normally built from the ``dataset`` config. Defaults are used when
                omitted so the agent is safe to construct in isolation.
            logger: Optional logger.
        """
        super().__init__(name, config=config, logger=logger)
        self._spec = spec or PreprocessingSpec()

    @property
    def spec(self) -> PreprocessingSpec:
        """The active preprocessing specification."""
        return self._spec

    def filter_samples(self, samples: Sequence[Sample]) -> list[Sample]:
        """Return samples inside the configured Hb range (pass-through if disabled)."""
        hb = self._spec.hb_filter
        if not (self.enabled and hb.enabled):
            return list(samples)
        return filter_hb_range(samples, hb.min, hb.max)

    def sample_weights(self, samples: Sequence[Sample]) -> list[float] | None:
        """Return per-sample balanced weights, or ``None`` when balancing is off.

        The bins span the configured Hb window so weighting is stable across
        splits (not sensitive to a split's observed min/max).
        """
        bal = self._spec.balanced_sampling
        if not (self.enabled and bal.enabled):
            return None
        hb = self._spec.hb_filter
        return balanced_sample_weights(
            samples,
            n_bins=bal.bins,
            hb_min=hb.min if hb.enabled else None,
            hb_max=hb.max if hb.enabled else None,
        )

    def predict(self, context: dict[str, Any]) -> AgentDecision:
        """Produce a preprocessing plan for the samples in ``context``.

        Reads ``context['samples']`` (a sequence of :class:`Sample`); writes
        ``kept_samples`` (count), ``dropped_samples`` (count), ``sample_weights``
        (or ``None``), and the effective ``hb_range``.
        """
        samples = list(context.get("samples", []))
        kept = self.filter_samples(samples)
        weights = self.sample_weights(kept)
        hb = self._spec.hb_filter
        return self._decision(
            {
                "kept_samples": len(kept),
                "dropped_samples": len(samples) - len(kept),
                "sample_weights": weights,
                "hb_range": [hb.min, hb.max] if hb.enabled else None,
                "balanced": weights is not None,
            },
            reason=(
                f"kept {len(kept)}/{len(samples)} sample(s) in Hb "
                f"[{hb.min}, {hb.max}]"
                + ("; balanced across bins" if weights is not None else "")
            ),
        )


__all__ = ["PreprocessingAgent"]
