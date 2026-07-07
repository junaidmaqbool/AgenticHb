"""Typed configuration for the evaluation subsystem (evaluation.yaml)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Default anemia threshold (g/dL) used for classification metrics. Configurable
# via evaluation.yaml; a single generic cutoff is used unless overridden.
_DEFAULT_ANEMIA_THRESHOLD = 12.0

# Defaults for paired significance testing of the baseline-vs-adaptive comparison
# (Decision 031). Overridable via the ``significance`` subsection of evaluation.yaml.
_DEFAULT_BOOTSTRAP_ITERATIONS = 5000
_DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    """Typed view of the ``evaluation`` configuration section."""

    regression_metrics: tuple[str, ...] = ("mae", "rmse", "r2", "pearson", "spearman")
    classification_metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
    calibration_metrics: tuple[str, ...] = ("calibration_error",)
    bland_altman: bool = True
    within_thresholds: tuple[float, ...] = (0.5, 1.0)
    anemia_threshold: float = _DEFAULT_ANEMIA_THRESHOLD
    export_csv: bool = True
    export_json: bool = True
    compare_against_baseline: bool = True
    significance_enabled: bool = True
    bootstrap_iterations: int = _DEFAULT_BOOTSTRAP_ITERATIONS
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> EvaluationConfig:
        """Build an :class:`EvaluationConfig` from the parsed section."""
        if "evaluation" in section and isinstance(section["evaluation"], Mapping):
            section = section["evaluation"]
        clinical = dict(section.get("clinical", {}))
        outputs = dict(section.get("outputs", {}))
        significance = dict(section.get("significance", {}))
        return cls(
            regression_metrics=tuple(
                str(m) for m in section.get("regression_metrics", ["mae", "rmse", "r2", "pearson", "spearman"])
            ),
            classification_metrics=tuple(
                str(m) for m in section.get("classification_metrics", ["accuracy", "precision", "recall", "f1"])
            ),
            calibration_metrics=tuple(str(m) for m in section.get("calibration_metrics", ["calibration_error"])),
            bland_altman=bool(clinical.get("bland_altman", True)),
            within_thresholds=tuple(float(b) for b in clinical.get("within_thresholds", [0.5, 1.0])),
            anemia_threshold=float(section.get("anemia_threshold", _DEFAULT_ANEMIA_THRESHOLD)),
            export_csv=bool(outputs.get("csv", True)),
            export_json=bool(outputs.get("json", True)),
            compare_against_baseline=bool(section.get("compare_against_baseline", True)),
            significance_enabled=bool(significance.get("enabled", True)),
            bootstrap_iterations=int(
                significance.get("bootstrap_iterations", _DEFAULT_BOOTSTRAP_ITERATIONS)
            ),
            confidence_level=float(significance.get("confidence_level", _DEFAULT_CONFIDENCE_LEVEL)),
        )


__all__ = ["EvaluationConfig"]
