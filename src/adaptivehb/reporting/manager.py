"""ReportingManager — generates publication figures and tables.

Ties the figure generator and table exporters to the evaluation configuration and
the experiment output directories (EXPERIMENT_SPEC Ch.19-20). Reporting is
optional: when matplotlib/openpyxl are absent, figure/Excel generation is skipped
gracefully rather than failing the pipeline (Decision 027).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import ensure_dir
from adaptivehb.reporting.figures import FigureGenerator, figures_available
from adaptivehb.reporting.tables import (
    excel_available,
    export_table_csv,
    export_table_excel,
    flatten_metrics,
)


class ReportingManager(BaseManager):
    """Produces figures and tables from evaluation results."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the reporting manager."""
        super().__init__(config, base_dir)
        outputs = config.section("evaluation")["evaluation"].get("outputs", {})
        self._formats = tuple(str(f) for f in outputs.get("figure_formats", ["png", "pdf"]))
        paths = config.project.paths
        self._figures_root = self._base_dir / paths.figures
        self._reports_root = self._base_dir / paths.reports

    @property
    def figure_formats(self) -> tuple[str, ...]:
        """Figure output formats from the evaluation config."""
        return self._formats

    def _on_initialize(self) -> None:
        ensure_dir(self._figures_root)
        ensure_dir(self._reports_root)

    def available(self) -> dict[str, bool]:
        """Report which reporting backends are installed."""
        return {"figures": figures_available(), "excel": excel_available()}

    def generate_evaluation_figures(
        self, y_true: Sequence[float], y_pred: Sequence[float], *, subdir: str | None = None
    ) -> dict[str, list[str]]:
        """Generate scatter, Bland-Altman, and residual figures.

        Returns a mapping of figure name to written paths (empty when matplotlib
        is unavailable or there is no data).
        """
        if not figures_available() or not y_true:
            return {}
        out_dir = ensure_dir(self._figures_root / subdir) if subdir else ensure_dir(self._figures_root)
        generator = FigureGenerator(self._formats)
        figures = {
            "scatter": generator.scatter(y_true, y_pred, out_dir / "scatter.png"),
            "bland_altman": generator.bland_altman(y_true, y_pred, out_dir / "bland_altman.png"),
            "residuals": generator.residual_hist(y_true, y_pred, out_dir / "residuals.png"),
        }
        return {name: [str(p) for p in paths] for name, paths in figures.items()}

    def generate_training_curve(
        self, history: list[dict[str, float]], name: str, *, subdir: str | None = None
    ) -> list[str]:
        """Generate a training-curve figure from an epoch-metric history."""
        if not figures_available() or not history:
            return []
        out_dir = ensure_dir(self._figures_root / subdir) if subdir else ensure_dir(self._figures_root)
        generator = FigureGenerator(self._formats)
        return [str(p) for p in generator.training_curve(history, out_dir / f"{name}_curve.png")]

    def export_metrics_table(self, name: str, metrics: dict[str, Any]) -> dict[str, str]:
        """Export a flattened metrics table as CSV (and Excel when available)."""
        rows = flatten_metrics(metrics)
        ensure_dir(self._reports_root)
        result = {"csv": str(export_table_csv(rows, self._reports_root / f"{name}_table.csv"))}
        if excel_available():
            result["excel"] = str(export_table_excel(rows, self._reports_root / f"{name}_table.xlsx"))
        return result


__all__ = ["ReportingManager"]
