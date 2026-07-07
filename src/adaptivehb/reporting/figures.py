"""Publication-quality figure generation (matplotlib-guarded).

Generates the standard evaluation figures (scatter, Bland-Altman, residuals,
training curves, model comparison) from evaluation results, saving each in one or
more formats (PNG/PDF). Matplotlib is an optional dependency: the module imports
cleanly without it, and :class:`FigureGenerator` raises a clear
:class:`ReportingError` if asked to render when matplotlib is absent
(Decision 027).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from adaptivehb.core.utils import ensure_dir
from adaptivehb.exceptions import ReportingError

Numbers = Sequence[float]


def figures_available() -> bool:
    """Whether matplotlib is installed."""
    return importlib.util.find_spec("matplotlib") is not None


class FigureGenerator:
    """Renders and saves evaluation figures using matplotlib (Agg backend)."""

    def __init__(self, formats: Sequence[str] = ("png",)) -> None:
        """Initialize the generator.

        Args:
            formats: Output formats to write for each figure (e.g. png, pdf).
        """
        self._formats = tuple(str(f).lower().lstrip(".") for f in formats) or ("png",)
        self._plt: Any = None

    @property
    def formats(self) -> tuple[str, ...]:
        """The output formats this generator writes."""
        return self._formats

    def _pyplot(self) -> Any:
        if self._plt is None:
            if not figures_available():
                raise ReportingError("matplotlib is required to generate figures (install the extra).")
            import matplotlib

            matplotlib.use("Agg")  # headless-safe
            import matplotlib.pyplot as plt

            self._plt = plt
        return self._plt

    def _save(self, fig: Any, path: str | Path) -> list[Path]:
        stem = Path(path).with_suffix("")
        ensure_dir(stem.parent)
        written: list[Path] = []
        for fmt in self._formats:
            out = stem.with_suffix(f".{fmt}")
            fig.savefig(out, bbox_inches="tight", dpi=150)
            written.append(out)
        self._pyplot().close(fig)
        return written

    @staticmethod
    def _require_pairs(y_true: Numbers, y_pred: Numbers) -> None:
        if len(y_true) != len(y_pred) or not y_true:
            raise ReportingError("Figure requires non-empty, equal-length inputs.")

    # -- figures -----------------------------------------------------------

    def scatter(self, y_true: Numbers, y_pred: Numbers, path: str | Path,
                title: str = "Predicted vs True Hb (g/dL)") -> list[Path]:
        """Predicted-vs-true scatter with an identity line."""
        self._require_pairs(y_true, y_pred)
        plt = self._pyplot()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
        low = min(min(y_true), min(y_pred))
        high = max(max(y_true), max(y_pred))
        ax.plot([low, high], [low, high], "r--", linewidth=1, label="identity")
        ax.set_xlabel("True Hb (g/dL)")
        ax.set_ylabel("Predicted Hb (g/dL)")
        ax.set_title(title)
        ax.legend(loc="best")
        return self._save(fig, path)

    def bland_altman(self, y_true: Numbers, y_pred: Numbers, path: str | Path) -> list[Path]:
        """Bland-Altman agreement plot with bias and 95% limits of agreement."""
        self._require_pairs(y_true, y_pred)
        plt = self._pyplot()
        means = [(t + p) / 2 for t, p in zip(y_true, y_pred)]
        diffs = [p - t for t, p in zip(y_true, y_pred)]
        bias = fmean(diffs)
        sd = pstdev(diffs) if len(diffs) > 1 else 0.0
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(means, diffs, alpha=0.6, edgecolor="k", linewidth=0.3)
        ax.axhline(bias, color="b", label=f"bias={bias:.2f}")
        ax.axhline(bias + 1.96 * sd, color="grey", linestyle="--", label="+1.96 SD")
        ax.axhline(bias - 1.96 * sd, color="grey", linestyle="--", label="-1.96 SD")
        ax.set_xlabel("Mean of true & predicted (g/dL)")
        ax.set_ylabel("Difference (pred - true)")
        ax.set_title("Bland-Altman")
        ax.legend(loc="best")
        return self._save(fig, path)

    def residual_hist(self, y_true: Numbers, y_pred: Numbers, path: str | Path) -> list[Path]:
        """Histogram of residuals (predicted minus true)."""
        self._require_pairs(y_true, y_pred)
        plt = self._pyplot()
        residuals = [p - t for t, p in zip(y_true, y_pred)]
        bins = min(20, max(5, len(residuals) // 2))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residuals, bins=bins, edgecolor="k")
        ax.set_xlabel("Residual (g/dL)")
        ax.set_ylabel("Count")
        ax.set_title("Residual distribution")
        return self._save(fig, path)

    def training_curve(self, history: list[dict[str, float]], path: str | Path,
                       keys: Sequence[str] = ("train_loss", "val_loss")) -> list[Path]:
        """Plot one or more metric series across epochs."""
        if not history:
            raise ReportingError("Training curve requires a non-empty history.")
        plt = self._pyplot()
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for key in keys:
            series = [row[key] for row in history if key in row]
            if series:
                ax.plot(range(1, len(series) + 1), series, marker="o", label=key)
                plotted = True
        if not plotted:
            raise ReportingError(f"None of {list(keys)} present in the history.")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training curves")
        ax.legend(loc="best")
        return self._save(fig, path)

    def model_comparison(self, scores: dict[str, float], path: str | Path,
                         metric_name: str = "MAE") -> list[Path]:
        """Bar chart comparing a metric across models/pipelines."""
        if not scores:
            raise ReportingError("Model comparison requires at least one score.")
        plt = self._pyplot()
        names = list(scores)
        values = [scores[name] for name in names]
        fig, ax = plt.subplots(figsize=(max(4, len(names)), 4))
        ax.bar(names, values, edgecolor="k")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Model comparison ({metric_name})")
        fig.autofmt_xdate(rotation=30)
        return self._save(fig, path)


__all__ = ["FigureGenerator", "figures_available"]
