"""EvaluationManager — computes metrics and generates reports.

Single responsibility: turn predictions-versus-ground-truth into the standard
metric bundle (regression, clinical agreement, classification), compare
pipelines (baseline vs adaptive), and export CSV/JSON reports
(EXPERIMENT_SPEC Ch.10, Ch.13-19). Torch-free and numpy-optional.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import ensure_dir
from adaptivehb.evaluation import metrics as M
from adaptivehb.evaluation import significance
from adaptivehb.evaluation.config import EvaluationConfig
from adaptivehb.evaluation.report import EvaluationReport


class EvaluationManager(BaseManager):
    """Computes evaluation metrics and produces publication-ready reports."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the evaluation manager."""
        super().__init__(config, base_dir)
        self._eval_config = EvaluationConfig.from_section(config.section("evaluation"))
        self._results_root = self._base_dir / config.project.paths.results / "evaluation"

    @property
    def evaluation_config(self) -> EvaluationConfig:
        """The typed evaluation configuration."""
        return self._eval_config

    def _on_initialize(self) -> None:
        ensure_dir(self._results_root)

    def evaluate(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        *,
        name: str = "model",
        per_sample_rows: list[dict[str, Any]] | None = None,
    ) -> EvaluationReport:
        """Compute the full metric bundle for one set of predictions.

        Args:
            y_true: Ground-truth hemoglobin values.
            y_pred: Predicted hemoglobin values.
            name: Report name (used in exported filenames).
            per_sample_rows: Optional per-sample rows for the CSV export.

        Returns:
            A populated :class:`EvaluationReport`.
        """
        metrics: dict[str, Any] = M.regression_metrics(y_true, y_pred)
        metrics["clinical"] = M.clinical_metrics(y_true, y_pred, self._eval_config.within_thresholds)
        metrics["classification"] = M.classification_metrics(
            y_true, y_pred, self._eval_config.anemia_threshold
        )
        self._log.info("Evaluated '%s': MAE=%.4f, RMSE=%.4f.", name, metrics["mae"], metrics["rmse"])
        return EvaluationReport(name=name, metrics=metrics, per_sample=per_sample_rows or [])

    def compare(
        self,
        baseline: EvaluationReport,
        adaptive: EvaluationReport,
        *,
        metric: str = "mae",
        lower_is_better: bool = True,
        y_true: Sequence[float] | None = None,
        baseline_pred: Sequence[float] | None = None,
        adaptive_pred: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Compare a baseline and an adaptive report on one metric.

        Returns a dictionary with both values, the improvement, and whether the
        adaptive pipeline is better (EXPERIMENT_SPEC Ch.14). When the paired
        per-sample predictions (``y_true``, ``baseline_pred``, ``adaptive_pred``)
        are supplied, a ``significance`` block (paired t-test, Wilcoxon
        signed-rank, bootstrap CI, and Cohen's d on the paired absolute-error
        differences) is attached so the comparison is publication-ready
        (Decision 031). Without them the point comparison is returned unchanged,
        preserving backward compatibility.
        """
        base_value = float(baseline.metrics[metric])
        adaptive_value = float(adaptive.metrics[metric])
        improvement = base_value - adaptive_value if lower_is_better else adaptive_value - base_value
        result: dict[str, Any] = {
            "metric": metric,
            "baseline": base_value,
            "adaptive": adaptive_value,
            "improvement": improvement,
            "adaptive_better": improvement > 0,
        }
        if (
            self._eval_config.significance_enabled
            and y_true is not None
            and baseline_pred is not None
            and adaptive_pred is not None
        ):
            result["significance"] = significance.compare_significance(
                y_true,
                baseline_pred,
                adaptive_pred,
                bootstrap_iterations=self._eval_config.bootstrap_iterations,
                confidence=self._eval_config.confidence_level,
                seed=self._config.project.seed,
            )
        return result

    def export(self, report: EvaluationReport, subdir: str | None = None) -> dict[str, str]:
        """Export a report's metrics (JSON) and per-sample rows (CSV).

        Returns:
            Mapping with the written ``json``/``csv`` paths (present per config).
        """
        out_dir = ensure_dir(self._results_root / subdir) if subdir else ensure_dir(self._results_root)
        written: dict[str, str] = {}
        if self._eval_config.export_json:
            written["json"] = str(report.export_json(out_dir / f"{report.name}_metrics.json"))
        if self._eval_config.export_csv:
            written["csv"] = str(report.export_csv(out_dir / f"{report.name}_predictions.csv"))
        return written


__all__ = ["EvaluationManager"]
