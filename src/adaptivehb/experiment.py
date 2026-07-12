"""ExperimentRunner — a reproducible end-to-end experiment.

Composes the framework into one archived experiment (EXPERIMENT_SPEC): create an
immutable experiment directory, train the models, then evaluate a **static
baseline** against the **adaptive** (agent-fused) pipeline on the held-out test
split, comparing them on the same metrics and conditions (Decision 008,
EXPERIMENT_SPEC Ch.13-14). Metrics, the comparison, per-sample predictions,
figures, and a summary are written into the experiment directory.

This runs end-to-end on reference models (no torch required); with the real
backbones + a dataset it becomes a genuine scientific experiment (Decision 028).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adaptivehb.core.types import PipelineMode
from adaptivehb.core.utils import ensure_dir, write_json
from adaptivehb.exceptions import PipelineError
from adaptivehb.provenance import build_manifest, write_manifest
from adaptivehb.reporting.experiment_report import (
    ExperimentReportData,
    write_experiment_report,
)
from adaptivehb.reporting.figures import FigureGenerator, figures_available
from adaptivehb.reporting.tables import (
    excel_available,
    export_table_csv,
    export_table_excel,
    flatten_metrics,
)

if TYPE_CHECKING:  # pragma: no cover
    from adaptivehb.pipeline import HbPipeline


@dataclass
class ExperimentResult:
    """Outcome of a full experiment run."""

    experiment_id: str
    root: str
    metrics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, list[str]] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    report_paths: dict[str, str] = field(default_factory=dict)
    summary_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {
            "experiment_id": self.experiment_id,
            "root": self.root,
            "metrics": self.metrics,
            "comparison": self.comparison,
            "figures": self.figures,
            "tables": self.tables,
            "provenance": self.provenance,
            "report_paths": self.report_paths,
            "summary_path": self.summary_path,
        }


class ExperimentRunner:
    """Runs and archives a baseline-vs-adaptive experiment."""

    def __init__(self, pipeline: HbPipeline) -> None:
        """Initialize with an :class:`~adaptivehb.pipeline.HbPipeline`."""
        self._pipeline = pipeline
        self._pm = pipeline.manager

    def run(self, name: str = "experiment", *, epochs: int | None = None) -> ExperimentResult:
        """Execute the experiment and archive its outputs.

        Args:
            name: Experiment name (a unique directory is created for it).
            epochs: Optional training epochs override.

        Returns:
            An :class:`ExperimentResult`.
        """
        self._pipeline.initialize()
        pm = self._pm

        experiment = pm.experiments.create(name, config_snapshot=self._config_snapshot())
        provenance = build_manifest(self._pipeline, extra={"experiment_id": experiment.experiment_id})
        write_manifest(provenance, experiment.path_for("configuration") / "provenance.json")
        pm.run(PipelineMode.TRAINING, epochs=epochs)

        y_true, baseline_pred, adaptive_pred, rows = self._baseline_vs_adaptive()
        if not y_true:
            raise PipelineError("No labelled test samples for the experiment.")

        baseline_report = pm.evaluation.evaluate(y_true, baseline_pred, name="baseline")
        adaptive_report = pm.evaluation.evaluate(y_true, adaptive_pred, name="adaptive", per_sample_rows=rows)
        comparison = pm.evaluation.compare(
            baseline_report,
            adaptive_report,
            metric="mae",
            y_true=y_true,
            baseline_pred=baseline_pred,
            adaptive_pred=adaptive_pred,
        )

        self._write_reports(experiment, baseline_report, adaptive_report, comparison, rows)
        tables = self._write_tables(experiment, adaptive_report.metrics)
        figures = self._write_figures(experiment, y_true, adaptive_pred, comparison)
        report_paths = self._write_report(
            experiment, name, adaptive_report.metrics, comparison, provenance, figures, len(y_true)
        )

        summary = {
            "experiment_id": experiment.experiment_id,
            "name": name,
            "num_test_patients": len(y_true),
            "registered_models": pm.registry.report(),
            "metrics": adaptive_report.metrics,
            "comparison": comparison,
            "figures": figures,
            "tables": tables,
            "provenance": provenance,
            "report_paths": report_paths,
        }
        summary_path = pm.experiments.save_summary(experiment, summary)
        pm.state.update(current_phase="experiment", status="completed")
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            root=str(experiment.root),
            metrics=adaptive_report.metrics,
            comparison=comparison,
            figures=figures,
            tables=tables,
            provenance=provenance,
            report_paths=report_paths,
            summary_path=str(summary_path),
        )

    # -- internals ---------------------------------------------------------

    def _config_snapshot(self) -> dict[str, Any]:
        project = self._pipeline.config.project
        return {
            "project": project.name,
            "seed": project.seed,
            "dataset_root": str(self._pm.dataset.root),
        }

    def _baseline_vs_adaptive(self) -> tuple[list[float], list[float], list[float], list[dict[str, Any]]]:
        pm = self._pm
        dataset = pm.dataset
        if not dataset.samples("test"):
            dataset.split()
        test_samples = dataset.samples("test") or dataset.samples()

        patients: dict[str, dict[str, Any]] = {}
        for sample in test_samples:
            if sample.hb is None:
                continue
            record = patients.setdefault(
                sample.patient_id, {"hb": sample.hb, "tissue_samples": {}}
            )
            record["tissue_samples"].setdefault(sample.tissue, []).append(sample)
        if not patients:
            return [], [], [], []

        pred_config = pm.prediction.prediction_config
        # Trained per-tissue models loaded from the checkpoint store (Decision 029).
        tissue_models = {
            tissue: pm.prediction.load_trained(f"hb_{tissue}", pm.checkpoints, tissue=tissue)
            for tissue in dataset.dataset_config.tissues
        }
        available_seg = list(pm.config.section("segmentation")["segmentation"].get("available_models", []))

        y_true: list[float] = []
        baseline: list[float] = []
        adaptive: list[float] = []
        rows: list[dict[str, Any]] = []
        for patient_id, info in patients.items():
            y_true.append(info["hb"])
            tissue_preds = self._predict_tissues(tissue_models, info["tissue_samples"])
            # Static baseline: unweighted mean over all available tissues (no agents).
            baseline_hb = round(sum(tissue_preds.values()) / len(tissue_preds), 4) if tissue_preds else 0.0
            baseline.append(baseline_hb)

            # Adaptive: the agent workflow selects tissues and fuses by confidence.
            tissues_ctx = {
                tissue: {"quality": 0.8, "roi_iou": 0.75, "pred_hb": value, "pred_confidence": 0.8}
                for tissue, value in tissue_preds.items()
            }
            context = {
                "patient_id": patient_id,
                "tissues": tissues_ctx,
                "available_segmentation": available_seg,
                "default_prediction_model": pred_config.default_model,
                "tissue_models": dict(pred_config.tissue_models),
            }
            result = pm.agents.run_workflow(context)
            adaptive_hb = result.final_hb if result.final_hb is not None else baseline_hb
            adaptive.append(adaptive_hb)
            rows.append({
                "patient_id": patient_id, "true_hb": info["hb"],
                "baseline_hb": baseline_hb, "adaptive_hb": adaptive_hb,
            })
        return y_true, baseline, adaptive, rows

    def _predict_tissues(
        self,
        tissue_models: dict[str, Any],
        tissue_samples: dict[str, list[Any]],
    ) -> dict[str, float]:
        """Estimate Hb per tissue for one patient from that tissue's samples.

        Each tissue's estimate is the mean prediction over the patient's images
        for that tissue (real images are decoded and fed for learned backbones;
        image-independent reference models return their constant estimate). A
        tissue whose inference fails (e.g. a missing/corrupt image) is logged and
        skipped rather than aborting the whole experiment, so one bad sample
        never wastes a long run.
        """
        predictions: dict[str, float] = {}
        for tissue, samples in tissue_samples.items():
            model = tissue_models.get(tissue)
            if model is None:
                continue
            try:
                values = self._pm.prediction.predict_samples(model, samples)
            except Exception as error:  # noqa: BLE001 - degrade gracefully, keep the run alive
                self._pm.logger.warning(
                    "Skipping tissue %r during evaluation: %s", tissue, error
                )
                continue
            if values:
                predictions[tissue] = sum(values) / len(values)
        return predictions

    def _write_reports(self, experiment: Any, baseline: Any, adaptive: Any,
                       comparison: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        metrics_dir = ensure_dir(experiment.path_for("metrics"))
        write_json(metrics_dir / "baseline_metrics.json", baseline.to_dict())
        write_json(metrics_dir / "adaptive_metrics.json", adaptive.to_dict())
        write_json(metrics_dir / "comparison.json", comparison)
        export_table_csv(rows, experiment.path_for("predictions") / "predictions.csv")

    def _write_tables(self, experiment: Any, metrics: dict[str, Any]) -> dict[str, str]:
        rows = flatten_metrics(metrics)
        csv_dir = ensure_dir(experiment.path_for("csv"))
        tables = {"csv": str(export_table_csv(rows, csv_dir / "adaptive_metrics_table.csv"))}
        if excel_available():
            tables["excel"] = str(export_table_excel(rows, experiment.path_for("excel") / "adaptive_metrics_table.xlsx"))
        return tables

    def _write_report(self, experiment: Any, name: str, metrics: dict[str, Any],
                      comparison: dict[str, Any], provenance: dict[str, Any],
                      figures: dict[str, list[str]], num_test_patients: int) -> dict[str, str]:
        data = ExperimentReportData(
            name=name,
            experiment_id=experiment.experiment_id,
            metrics=metrics,
            comparison=comparison,
            provenance=provenance,
            figures=figures,
            num_test_patients=num_test_patients,
        )
        return write_experiment_report(data, experiment.path_for("reports"))

    def _write_figures(self, experiment: Any, y_true: list[float], adaptive_pred: list[float],
                       comparison: dict[str, Any]) -> dict[str, list[str]]:
        if not figures_available():
            return {}
        out_dir = ensure_dir(experiment.path_for("figures"))
        generator = FigureGenerator(self._pm.reporting.figure_formats)
        figures = {
            "scatter": generator.scatter(y_true, adaptive_pred, out_dir / "adaptive_scatter.png"),
            "bland_altman": generator.bland_altman(y_true, adaptive_pred, out_dir / "adaptive_bland_altman.png"),
            "comparison": generator.model_comparison(
                {"baseline": comparison["baseline"], "adaptive": comparison["adaptive"]},
                out_dir / "baseline_vs_adaptive.png", metric_name="MAE",
            ),
        }
        return {name: [str(p) for p in paths] for name, paths in figures.items()}


__all__ = ["ExperimentRunner", "ExperimentResult"]
