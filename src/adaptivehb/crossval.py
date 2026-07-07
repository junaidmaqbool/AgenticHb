"""Patient-level k-fold cross-validation (Decision 035).

A single held-out test split is statistically fragile on the modest datasets
typical of non-invasive hemoglobin studies; reviewers expect cross-validation with
aggregated metrics (mean ± std) and per-fold significance. This module runs
patient-level *k*-fold cross-validation by reusing the existing
:class:`~adaptivehb.experiment.ExperimentRunner`: each fold is a fully isolated
experiment (its own ``base_dir``, so checkpoints/registry/experiments never bleed
between folds) trained on that fold's training patients and evaluated on its
held-out test patients. Per-fold metrics and baseline-vs-adaptive comparisons are
then aggregated and archived (``cv_summary.json``, ``cv_metrics.csv``,
``cv_report.md``).

The harness is torch-free and runs end-to-end on the reference models; with the
real PyTorch backbones and a dataset it produces genuine cross-validated metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.utils import ensure_dir, write_json
from adaptivehb.dataset.manager import DatasetManager
from adaptivehb.dataset.splitting import k_fold_split
from adaptivehb.exceptions import PipelineError
from adaptivehb.logging import get_logger
from adaptivehb.reporting.tables import export_table_csv

if TYPE_CHECKING:  # pragma: no cover
    from adaptivehb.experiment import ExperimentResult

# Scalar adaptive-metric keys aggregated across folds.
_AGGREGATED_METRICS: tuple[str, ...] = (
    "mae", "rmse", "r2", "pearson", "spearman", "mean_bias",
)


@dataclass
class CrossValidationResult:
    """Outcome of a k-fold cross-validation run."""

    name: str
    root: str
    k: int
    per_fold: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    summary_path: str = ""
    report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {
            "name": self.name,
            "root": self.root,
            "k": self.k,
            "per_fold": self.per_fold,
            "aggregate": self.aggregate,
            "summary_path": self.summary_path,
            "report_path": self.report_path,
        }


class CrossValidationRunner:
    """Runs and archives patient-level k-fold cross-validation."""

    def __init__(
        self,
        config: FrameworkConfig,
        *,
        base_dir: str | Path = ".",
        dataset_root: str | Path | None = None,
        folds: int = 5,
        seed: int | None = None,
    ) -> None:
        """Initialize the cross-validation runner.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory; CV outputs live under
                ``base_dir/cross_validation/<name>``.
            dataset_root: Explicit dataset root (overrides config).
            folds: Number of folds (``k >= 2``).
            seed: Fold-shuffle seed; defaults to the project seed.
        """
        self._config = config
        self._base_dir = Path(base_dir)
        self._dataset_root = dataset_root
        self._folds = folds
        self._seed = seed if seed is not None else config.project.seed
        self._log = get_logger("CrossValidationRunner")

    def run(self, name: str = "cross_validation", *, epochs: int | None = None) -> CrossValidationResult:
        """Execute k-fold cross-validation and archive the aggregated outputs.

        Args:
            name: Run name (a directory is created for it).
            epochs: Optional training-epochs override per fold.

        Returns:
            A :class:`CrossValidationResult`.

        Raises:
            PipelineError: If the dataset has too few patients for ``k`` folds.
        """
        # Reuse ExperimentRunner per fold; imported here to avoid a cycle.
        from adaptivehb.experiment import ExperimentRunner
        from adaptivehb.pipeline import HbPipeline

        patient_ids = self._patient_ids()
        if self._folds > len(patient_ids):
            raise PipelineError(
                f"Cannot run {self._folds}-fold CV with only {len(patient_ids)} patients."
            )
        fold_splits = k_fold_split(patient_ids, self._folds, seed=self._seed)

        cv_root = ensure_dir(self._base_dir / "cross_validation" / name)
        per_fold: list[dict[str, Any]] = []
        for index, fold_split in enumerate(fold_splits):
            self._log.info("Cross-validation fold %d/%d.", index + 1, self._folds)
            fold_dir = cv_root / f"fold_{index}"
            pipeline = HbPipeline(self._config, base_dir=fold_dir, dataset_root=self._dataset_root)
            pipeline.initialize()
            pipeline.manager.dataset.apply_split(fold_split)  # pin this fold
            result = ExperimentRunner(pipeline).run(f"{name}_fold{index}", epochs=epochs)
            per_fold.append(self._fold_record(index, fold_split, result))
            pipeline.shutdown()

        aggregate = self._aggregate(per_fold)
        summary = {
            "name": name,
            "k": self._folds,
            "seed": self._seed,
            "aggregate": aggregate,
            "per_fold": per_fold,
        }
        summary_path = write_json(cv_root / "cv_summary.json", summary)
        export_table_csv(self._fold_rows(per_fold), cv_root / "cv_metrics.csv")
        report_path = self._write_report(cv_root, name, aggregate, per_fold)

        return CrossValidationResult(
            name=name,
            root=str(cv_root),
            k=self._folds,
            per_fold=per_fold,
            aggregate=aggregate,
            summary_path=str(summary_path),
            report_path=str(report_path),
        )

    # -- internals ---------------------------------------------------------

    def _patient_ids(self) -> list[str]:
        """Load the dataset's patient IDs (via a throwaway DatasetManager)."""
        dataset = DatasetManager(self._config, base_dir=self._base_dir, dataset_root=self._dataset_root)
        dataset.initialize()
        return list(dataset.load_metadata().patient_ids)

    @staticmethod
    def _fold_record(index: int, fold_split: dict[str, list[str]], result: ExperimentResult) -> dict[str, Any]:
        comparison = result.comparison or {}
        significance = comparison.get("significance", {}) if isinstance(comparison, dict) else {}
        return {
            "fold": index,
            "experiment_id": result.experiment_id,
            "num_test_patients": len(fold_split.get("test", [])),
            "metrics": {k: result.metrics.get(k) for k in _AGGREGATED_METRICS if k in result.metrics},
            "baseline_mae": comparison.get("baseline"),
            "adaptive_mae": comparison.get("adaptive"),
            "improvement": comparison.get("improvement"),
            "adaptive_better": comparison.get("adaptive_better"),
            "p_value": (significance.get("paired_t_test") or {}).get("p_value"),
            "cohens_d": significance.get("cohens_d"),
        }

    @staticmethod
    def _aggregate(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate per-fold scalar metrics into mean/std/min/max summaries."""
        metrics: dict[str, dict[str, float]] = {}
        for key in _AGGREGATED_METRICS:
            values = [f["metrics"].get(key) for f in per_fold if isinstance(f["metrics"].get(key), (int, float))]
            if values:
                metrics[key] = {
                    "mean": fmean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
        improvements = [f["improvement"] for f in per_fold if isinstance(f["improvement"], (int, float))]
        baselines = [f["baseline_mae"] for f in per_fold if isinstance(f["baseline_mae"], (int, float))]
        adaptives = [f["adaptive_mae"] for f in per_fold if isinstance(f["adaptive_mae"], (int, float))]
        return {
            "metrics": metrics,
            "baseline_mae_mean": fmean(baselines) if baselines else None,
            "adaptive_mae_mean": fmean(adaptives) if adaptives else None,
            "improvement_mean": fmean(improvements) if improvements else None,
            "improvement_std": pstdev(improvements) if len(improvements) > 1 else 0.0,
            "folds_adaptive_better": sum(1 for f in per_fold if f.get("adaptive_better")),
            "num_folds": len(per_fold),
        }

    @staticmethod
    def _fold_rows(per_fold: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for f in per_fold:
            row = {"fold": f["fold"], "num_test_patients": f["num_test_patients"],
                   "baseline_mae": f["baseline_mae"], "adaptive_mae": f["adaptive_mae"],
                   "improvement": f["improvement"], "p_value": f["p_value"]}
            row.update({f"metric_{k}": v for k, v in f["metrics"].items()})
            rows.append(row)
        return rows

    def _write_report(self, cv_root: Path, name: str, aggregate: dict[str, Any],
                      per_fold: list[dict[str, Any]]) -> Path:
        lines: list[str] = []
        add = lines.append
        add(f"# Cross-Validation Report — {name}")
        add("")
        add(f"- **Folds (k):** {self._folds}")
        add(f"- **Seed:** {self._seed}")
        add(f"- **Folds where adaptive beat baseline:** "
            f"{aggregate['folds_adaptive_better']}/{aggregate['num_folds']}")
        add("")
        add("## Aggregated metrics (mean ± std over folds)")
        add("")
        add("| Metric | Mean | Std | Min | Max |")
        add("| --- | --- | --- | --- | --- |")
        for key, stats in aggregate["metrics"].items():
            add(f"| {key} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | "
                f"{_fmt(stats['min'])} | {_fmt(stats['max'])} |")
        add("")
        add("## Baseline vs adaptive (MAE)")
        add("")
        add(f"- Baseline mean: {_fmt(aggregate['baseline_mae_mean'])}")
        add(f"- Adaptive mean: {_fmt(aggregate['adaptive_mae_mean'])}")
        add(f"- Improvement: {_fmt(aggregate['improvement_mean'])} ± "
            f"{_fmt(aggregate['improvement_std'])}")
        add("")
        add("## Per-fold results")
        add("")
        add("| Fold | Test patients | Baseline MAE | Adaptive MAE | Improvement | t-test p |")
        add("| --- | --- | --- | --- | --- | --- |")
        for f in per_fold:
            add(f"| {f['fold']} | {f['num_test_patients']} | {_fmt(f['baseline_mae'])} | "
                f"{_fmt(f['adaptive_mae'])} | {_fmt(f['improvement'])} | {_fmt(f['p_value'])} |")
        add("")
        report_path = cv_root / "cv_report.md"
        report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return report_path


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "n/a" if value is None else str(value)


__all__ = ["CrossValidationRunner", "CrossValidationResult"]
