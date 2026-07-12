"""Data-driven pipeline mode runners (PIPELINE_SPEC Ch.13-16).

Each mode is expressed as a dependency-ordered job sequence over the managers
owned by the PipelineManager. Segmentation and prediction models are trained via
the injected trainable factory (dummy models until Phases 5-6), so the whole
data flow — validate → split → train → register → evaluate → infer — is
exercised end-to-end without real networks.

These are free functions taking the PipelineManager to keep ``pipeline.py``
focused on ownership and dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adaptivehb.core.types import ModelCategory, PipelineMode
from adaptivehb.managers.jobs import Job, JobQueue
from adaptivehb.managers.training import TrainingPlan
from adaptivehb.exceptions import PipelineError

if TYPE_CHECKING:  # pragma: no cover
    from adaptivehb.managers.pipeline import PipelineManager


def run_training(pm: PipelineManager, *, epochs: int | None = None, resume: bool = False) -> dict[str, Any]:
    """Run the training pipeline: validate → split → train → register.

    Args:
        pm: The owning pipeline manager.
        epochs: Optional override for the number of training epochs per model.
        resume: When true, each model resumes from its latest checkpoint.

    Returns:
        A result dictionary with per-job statuses and registered models.

    Raises:
        PipelineError: If dataset validation fails or a job errors.
    """
    dataset = pm.dataset

    def _validate() -> dict[str, Any]:
        report = dataset.validate()
        if not report.is_valid:
            raise PipelineError(
                f"Dataset is invalid: {[issue.code for issue in report.errors]}"
            )
        return report.to_dict()

    def _split() -> dict[str, int]:
        sizes = {name: len(ids) for name, ids in dataset.split().items()}
        seg_dataset = pm.segmentation_dataset
        if seg_dataset is not dataset:
            seg_sizes = {f"seg_{name}": len(ids) for name, ids in seg_dataset.split().items()}
            sizes.update(seg_sizes)
        return sizes

    def _train_segmentation() -> dict[str, Any]:
        return _train_group(pm, _segmentation_plans(pm, epochs), resume)

    def _train_prediction() -> dict[str, Any]:
        return _train_group(pm, _prediction_plans(pm, epochs), resume)

    queue = JobQueue(pm.logger)
    queue.add(Job("validate_dataset", _validate))
    queue.add(Job("split_dataset", _split, depends_on=["validate_dataset"]))
    queue.add(Job("train_segmentation", _train_segmentation, depends_on=["split_dataset"]))
    queue.add(Job("train_prediction", _train_prediction, depends_on=["train_segmentation"]))
    statuses = queue.run()

    mode = PipelineMode.RESUME if resume else PipelineMode.TRAINING
    pm.state.update(current_phase=mode.value, status="completed")
    return {
        "mode": mode.value,
        "jobs": statuses,
        "segmentation": queue.jobs["train_segmentation"].result,
        "prediction": queue.jobs["train_prediction"].result,
        "registry": pm.registry.report(),
    }


def run_evaluation(pm: PipelineManager) -> dict[str, Any]:
    """Compute hemoglobin metrics for the test split and export reports.

    Predictions come from the registered prediction models applied to the
    held-out patients; metrics (regression + clinical + classification) are
    computed by the EvaluationManager and exported as CSV/JSON
    (EXPERIMENT_SPEC Ch.10, Ch.15).

    Raises:
        PipelineError: If no prediction models are registered or there are no
            labelled test samples.
    """
    records = pm.registry.find(ModelCategory.PREDICTION)
    if not records:
        raise PipelineError("No prediction models registered; run training first.")
    y_true, y_pred, rows = _evaluation_pairs(pm)
    if not y_true:
        raise PipelineError("No test samples with hemoglobin labels to evaluate.")

    report = pm.evaluation.evaluate(y_true, y_pred, name="hb_prediction", per_sample_rows=rows)
    reports = pm.evaluation.export(report)
    figures = pm.reporting.generate_evaluation_figures(y_true, y_pred, subdir="hb_prediction")
    tables = pm.reporting.export_metrics_table("hb_prediction", report.metrics)
    pm.state.update(current_phase=PipelineMode.EVALUATION.value, status="completed")
    return {
        "mode": PipelineMode.EVALUATION.value,
        "num_models": len(records),
        "num_samples": len(y_true),
        "metrics": report.metrics,
        "reports": reports,
        "figures": figures,
        "tables": tables,
    }


def _evaluation_pairs(pm: PipelineManager) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    """Build per-patient (true, predicted) hemoglobin pairs for the test split."""
    dataset = pm.dataset
    if not dataset.samples("test"):
        dataset.split()
    test_samples = dataset.samples("test") or dataset.samples()

    by_patient: dict[str, dict[str, Any]] = {}
    for sample in test_samples:
        if sample.hb is None:
            continue
        record = by_patient.setdefault(sample.patient_id, {"hb": sample.hb, "tissue_samples": {}})
        record["tissue_samples"].setdefault(sample.tissue, []).append(sample)
    if not by_patient:
        return [], [], []

    # Predictions come from the trained per-tissue models loaded from the
    # checkpoint store (falls back to untrained where no checkpoint exists).
    tissue_models = {
        tissue: pm.prediction.load_trained(f"hb_{tissue}", pm.checkpoints, tissue=tissue)
        for tissue in dataset.dataset_config.tissues
    }
    y_true: list[float] = []
    y_pred: list[float] = []
    rows: list[dict[str, Any]] = []
    for patient_id, info in by_patient.items():
        tissue_estimates = _predict_patient_tissues(pm, tissue_models, info["tissue_samples"])
        estimate = (
            round(sum(tissue_estimates) / len(tissue_estimates), 4) if tissue_estimates else 0.0
        )
        y_true.append(info["hb"])
        y_pred.append(estimate)
        rows.append({"patient_id": patient_id, "true_hb": info["hb"], "predicted_hb": estimate})
    return y_true, y_pred, rows


def _predict_patient_tissues(
    pm: PipelineManager,
    tissue_models: dict[str, Any],
    tissue_samples: dict[str, list[Any]],
) -> list[float]:
    """Mean Hb estimate per available tissue for one patient (feeds real images).

    Learned backbones receive decoded test images via
    :meth:`PredictionManager.predict_samples`; image-independent reference models
    return their constant estimate. Tissues whose inference fails are logged and
    skipped so a single bad sample cannot abort evaluation.
    """
    estimates: list[float] = []
    for tissue, samples in tissue_samples.items():
        model = tissue_models.get(tissue)
        if model is None:
            continue
        try:
            values = pm.prediction.predict_samples(model, samples)
        except Exception as error:  # noqa: BLE001 - degrade gracefully
            pm.logger.warning("Skipping tissue %r during evaluation: %s", tissue, error)
            continue
        if values:
            estimates.append(sum(values) / len(values))
    return estimates


def run_inference(pm: PipelineManager, *, limit: int = 5) -> dict[str, Any]:
    """Produce dummy Hb predictions for held-out samples.

    The placeholder predictor returns the mean training-split hemoglobin; real
    inference (segmentation → routing → prediction → fusion) arrives in later
    phases.

    Raises:
        PipelineError: If the dataset yields no samples.
    """
    dataset = pm.dataset
    if not dataset.samples("test"):
        dataset.split()
    test_samples = dataset.samples("test") or dataset.samples()
    if not test_samples:
        raise PipelineError("No samples available for inference.")

    train_hb = [s.hb for s in dataset.samples("train") if s.hb is not None]
    if not train_hb:
        train_hb = [s.hb for s in dataset.samples() if s.hb is not None]
    predicted = round(sum(train_hb) / len(train_hb), 2) if train_hb else 0.0

    predictions = [
        {
            "patient_id": s.patient_id,
            "tissue": s.tissue,
            "predicted_hb": predicted,
            "true_hb": s.hb,
        }
        for s in test_samples[:limit]
    ]
    pm.state.update(current_phase=PipelineMode.INFERENCE.value, status="completed")
    return {
        "mode": PipelineMode.INFERENCE.value,
        "num_predictions": len(predictions),
        "predictions": predictions,
    }


def run_deployment(pm: PipelineManager) -> dict[str, Any]:
    """Load registry-approved models and produce a demo clinical report.

    Deployment performs no retraining (PIPELINE_SPEC Ch.17): it loads the trained
    models, runs the adaptive workflow on a demo patient, and exports a clinical
    report. Actual server launch is available via ``pm.deployment.launch()``.

    Raises:
        PipelineError: If no prediction models are registered.
    """
    records = pm.registry.find(ModelCategory.PREDICTION)
    if not records:
        raise PipelineError("No prediction models registered; run training first.")

    pm.deployment.load()
    report = pm.deployment.predict(_demo_patient(pm))
    reports = pm.deployment.export_report(report)
    pm.state.update(current_phase=PipelineMode.DEPLOYMENT.value, status="completed")
    return {
        "mode": PipelineMode.DEPLOYMENT.value,
        "target": pm.deployment.deployment_config.target,
        "num_models": len(records),
        "ready": True,
        "report": report.to_dict(),
        "reports": reports,
        "available_targets": pm.deployment.available_targets(),
    }


def _demo_patient(pm: PipelineManager) -> dict[str, Any]:
    """Build a demo patient (one held-out patient's tissues) for deployment."""
    dataset = pm.dataset
    if not dataset.samples("test"):
        dataset.split()
    test_samples = dataset.samples("test") or dataset.samples()
    if test_samples:
        patient_id = test_samples[0].patient_id
        tissues = sorted({s.tissue for s in test_samples if s.patient_id == patient_id})
    else:
        patient_id, tissues = "demo", list(dataset.dataset_config.tissues)
    return {"patient_id": patient_id, "tissues": {t: {} for t in tissues}}


# -- plan construction ------------------------------------------------------- #


def _segmentation_plans(pm: PipelineManager, epochs: int | None) -> list[TrainingPlan]:
    section = pm.config.section("segmentation")["segmentation"]
    default_epochs = int(section.get("training", {}).get("epochs", 1))
    n_epochs = epochs if epochs is not None else default_epochs
    return [
        TrainingPlan(
            name=f"seg_{model}",
            epochs=n_epochs,
            category=ModelCategory.SEGMENTATION,
            task="segmentation",
            architecture=str(model),
        )
        for model in section.get("available_models", [])
    ]


def _prediction_plans(pm: PipelineManager, epochs: int | None) -> list[TrainingPlan]:
    section = pm.config.section("prediction")["prediction"]
    default_epochs = int(section.get("training", {}).get("epochs", 1))
    n_epochs = epochs if epochs is not None else default_epochs
    default_arch = str(section.get("default_model", "efficientnet"))
    tissue_models = dict(section.get("tissue_models", {}))
    return [
        TrainingPlan(
            name=f"hb_{tissue}",
            epochs=n_epochs,
            category=ModelCategory.PREDICTION,
            task="hb_estimation",
            architecture=str(tissue_models.get(tissue, default_arch)),
        )
        for tissue in pm.dataset.dataset_config.tissues
    ]


def _train_group(
    pm: PipelineManager, plans: list[TrainingPlan], resume: bool
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for plan in plans:
        trainable = pm.trainable_factory(plan)
        _attach_dataloaders(pm, trainable, plan)
        result = pm.training.train(trainable, plan, resume=resume)
        results[plan.name] = {
            "registered_id": result.registered_id,
            "best_metric": result.best_metric,
            "epochs_run": result.epochs_run,
        }
    return results


def _attach_dataloaders(pm: PipelineManager, trainable: Any, plan: TrainingPlan) -> None:
    """Build and attach train/val dataloaders to a data-driven (torch) model.

    Reference models don't accept data (they train deterministically), so this is
    a no-op for them. For torch models it builds dataloaders from the dataset
    splits (per tissue for prediction) with the configured transforms/batch size.
    """
    if not hasattr(trainable, "attach_data"):
        return
    from adaptivehb.dataloading import TransformSpec, build_dataloader, build_transform

    task = "segmentation" if plan.category is ModelCategory.SEGMENTATION else "prediction"
    dataset = pm.segmentation_dataset if task == "segmentation" else pm.dataset
    if not dataset.samples("train"):
        dataset.split()

    train_samples = list(dataset.samples("train"))
    val_samples = list(dataset.samples("validation")) or list(train_samples)
    if task == "prediction" and plan.name.startswith("hb_"):
        tissue = plan.name[len("hb_"):]
        train_samples = [s for s in train_samples if s.tissue == tissue]
        val_samples = [s for s in val_samples if s.tissue == tissue]

    spec = TransformSpec.from_section(pm.config.section("dataset"))
    batch_size = _training_batch_size(pm, task)
    train_loader = build_dataloader(
        train_samples, batch_size=batch_size, task=task, shuffle=True,
        transform=build_transform(spec, training=True),
    )
    val_loader = build_dataloader(
        val_samples, batch_size=batch_size, task=task, shuffle=False,
        transform=build_transform(spec, training=False),
    )
    trainable.attach_data(train_loader, val_loader)


def _training_batch_size(pm: PipelineManager, task: str) -> int:
    """Read the configured training batch size for a task."""
    section = "segmentation" if task == "segmentation" else "prediction"
    training = pm.config.section(section)[section].get("training", {})
    return int(training.get("batch_size", 8))


__all__ = ["run_training", "run_evaluation", "run_inference", "run_deployment"]
