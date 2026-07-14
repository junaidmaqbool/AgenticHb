"""Tests for agentic preprocessing: Hb-range filtering + balanced sampling.

Torch-free: exercises the pure primitives and the PreprocessingAgent wrapper
without the ML stack, matching the rest of the dataloading-bridge test strategy.
"""

from __future__ import annotations

from adaptivehb.agents.preprocessing import PreprocessingAgent
from adaptivehb.dataloading.preprocessing import (
    BalancedSamplingSpec,
    HbFilterSpec,
    PreprocessingSpec,
    balanced_sample_weights,
    filter_hb_range,
)
from adaptivehb.dataset.schema import Sample


def _sample(pid: str, hb: float | None, tissue: str = "eye") -> Sample:
    return Sample(patient_id=pid, tissue=tissue, image_path=f"/img/{pid}.jpg", hb=hb)


def _samples(hbs: list[float | None]) -> list[Sample]:
    return [_sample(str(i), hb) for i, hb in enumerate(hbs)]


# -- filter_hb_range -------------------------------------------------------- #


def test_filter_keeps_in_range_and_drops_outliers():
    samples = _samples([5.9, 6.0, 10.0, 14.0, 14.1, 20.0])
    kept = filter_hb_range(samples, 6.0, 14.0)
    assert [s.hb for s in kept] == [6.0, 10.0, 14.0]


def test_filter_drops_unlabelled_samples():
    samples = _samples([None, 8.0, None, 12.0])
    kept = filter_hb_range(samples, 6.0, 14.0)
    assert [s.hb for s in kept] == [8.0, 12.0]


def test_filter_preserves_order():
    samples = _samples([13.0, 7.0, 9.0])
    kept = filter_hb_range(samples, 6.0, 14.0)
    assert [s.patient_id for s in kept] == ["0", "1", "2"]


# -- balanced_sample_weights ------------------------------------------------ #


def test_balanced_weights_upweight_rare_bin():
    # Nine samples near 12, one rare anaemic sample near 7.
    samples = _samples([7.0] + [12.0] * 9)
    weights = balanced_sample_weights(samples, n_bins=10, hb_min=6.0, hb_max=14.0)
    # The lone rare sample sits alone in its bin -> weight 1.0; the dense bin
    # gets 1/9 each. Rare sample must be weighted far higher.
    assert weights[0] == 1.0
    assert all(abs(w - 1 / 9) < 1e-9 for w in weights[1:])
    assert weights[0] > weights[1]


def test_balanced_weights_equalise_expected_draws_per_bin():
    samples = _samples([7.0, 7.5] + [12.0] * 8)  # bin A: 2, bin B: 8
    weights = balanced_sample_weights(samples, n_bins=10, hb_min=6.0, hb_max=14.0)
    # Total weight of each bin should be equal (2 * 1/2 == 8 * 1/8 == 1.0).
    assert abs(sum(weights[:2]) - sum(weights[2:])) < 1e-9


def test_balanced_weights_length_matches_and_unlabelled_zero():
    samples = _samples([8.0, None, 12.0])
    weights = balanced_sample_weights(samples, n_bins=5, hb_min=6.0, hb_max=14.0)
    assert len(weights) == 3
    assert weights[1] == 0.0


def test_balanced_weights_empty():
    assert balanced_sample_weights([], n_bins=5) == []


# -- PreprocessingSpec parsing --------------------------------------------- #


def test_spec_from_section_reads_dataset_config():
    section = {
        "dataset": {
            "preprocessing": {
                "hb_filter": {"enabled": True, "min": 6.0, "max": 14.0},
                "balanced_sampling": {"enabled": True, "bins": 8},
            }
        }
    }
    spec = PreprocessingSpec.from_section(section)
    assert spec.hb_filter == HbFilterSpec(True, 6.0, 14.0)
    assert spec.balanced_sampling == BalancedSamplingSpec(True, 8)


def test_spec_defaults_when_missing():
    spec = PreprocessingSpec.from_section({"dataset": {}})
    assert spec.hb_filter.min == 6.0 and spec.hb_filter.max == 14.0
    assert spec.balanced_sampling.bins == 10


# -- PreprocessingAgent ----------------------------------------------------- #


def _agent(enabled=True, hb=(True, 6.0, 14.0), bal=(True, 10)) -> PreprocessingAgent:
    spec = PreprocessingSpec(
        hb_filter=HbFilterSpec(*hb),
        balanced_sampling=BalancedSamplingSpec(*bal),
    )
    return PreprocessingAgent("preprocessing", config={"enabled": enabled}, spec=spec)


def test_agent_filters_and_weights():
    agent = _agent()
    samples = _samples([5.0, 7.0, 12.0, 15.0])
    kept = agent.filter_samples(samples)
    assert [s.hb for s in kept] == [7.0, 12.0]
    weights = agent.sample_weights(kept)
    assert weights is not None and len(weights) == 2


def test_agent_disabled_is_passthrough():
    agent = _agent(enabled=False)
    samples = _samples([5.0, 20.0])
    assert agent.filter_samples(samples) == samples
    assert agent.sample_weights(samples) is None


def test_agent_balancing_off_returns_none_weights():
    agent = _agent(bal=(False, 10))
    assert agent.sample_weights(_samples([7.0, 12.0])) is None


def test_agent_predict_reports_plan():
    agent = _agent()
    decision = agent.predict({"samples": _samples([5.0, 7.0, 12.0])})
    assert decision.agent == "preprocessing"
    assert decision.outputs["kept_samples"] == 2
    assert decision.outputs["dropped_samples"] == 1
    assert decision.outputs["hb_range"] == [6.0, 14.0]
    assert decision.outputs["balanced"] is True
