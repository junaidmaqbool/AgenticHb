# CHANGELOG.md

Version: 1.0

Status: Living Development Log

Project

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Purpose

This document records the chronological history of the project.

Unlike PROJECT_STATE.md, which describes the current status, the CHANGELOG records every important modification made to the repository.

It should be updated after every coding session.

---

# Version Format

Major.Minor.Patch

Example

v0.1.0

Major

Architectural changes

Minor

New functionality

Patch

Bug fixes

---

# Changelog

---

## v0.1.0

Date

July 2026

Status

Project Initialization

Completed

- Project Charter
- Project Design Specification
- Implementation Roadmap
- Pipeline Specification
- Dataset Specification
- Agent Specification
- Model Registry Specification
- Experiment Specification
- Claude Development Guide
- Development Rules
- Project Index

Notes

Initial documentation completed.

---

## v0.2.0

Date

2026-07-06

Status

Repository Construction (Phase 1) — Complete

Completed

- Canonical repository scaffold (`src/` layout package `adaptivehb`, `configs/`,
  `tests/`, and runtime output directories with `.gitkeep`).
- Packaging: `pyproject.toml` (setuptools, src layout, ruff/black/mypy/pytest
  config), pinned `requirements.txt`, `README.md`, `LICENSE` (Apache-2.0),
  `.gitignore`.
- Nine configuration files under `configs/` (project, dataset, segmentation,
  prediction, agents, evaluation, deployment, registry, logging).
- Typed configuration subsystem `adaptivehb.config` (`ConfigLoader`,
  `FrameworkConfig`, `ProjectConfig`, `LoggingConfig`, plus nested schemas) with
  per-file required-key validation.
- Logging subsystem `adaptivehb.logging` (`setup_logging`, `get_logger`) with
  console + rotating-file handlers, idempotent setup.
- Exception hierarchy `adaptivehb.exceptions` (`AdaptiveHbError` + subclasses).
- Smoke test suite (`tests/test_smoke.py`) — 11 tests passing.

Files Added

- Package: `src/adaptivehb/{__init__,version,exceptions}.py`,
  `src/adaptivehb/config/{__init__,base,schemas,loader}.py`,
  `src/adaptivehb/logging/{__init__,setup}.py`
- Configs: `configs/*.yaml` (9)
- Packaging: `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE`, `.gitignore`
- Tests: `tests/{__init__,conftest,test_smoke}.py`

Files Modified

- `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- No machine-learning code (Phase 1 is infrastructure only, per roadmap).
- Git not initialized from the sandbox (mount limitation); initialize natively.
- Provisional decisions 011–015 recorded in DECISION_LOG.md.

---

## v0.3.0

Date

2026-07-06

Status

Core Infrastructure (Phase 2) — Foundation Complete

Completed

- Core layer `adaptivehb.core`: `BaseManager`, `BaseModel`, `BaseAgent`
  interfaces; shared types (`ModelRecord`, `ModelCategory`, `ModelStatus`,
  `PipelineMode`, `JobStatus`); utilities (atomic JSON I/O, reproducible
  seeding, timestamps).
- `RegistryManager` (JSON backend): auto-versioning, stable unique IDs,
  discovery (`find`/`load_latest`/`load_best`), status updates, reporting, and
  automatic backups.
- `StateManager`: `pipeline_state.json` persistence, completed-module tracking,
  and cross-run recovery.
- `CheckpointManager`: `latest`/`best` checkpoints with a torch-free pickle
  payload plus a JSON metadata sidecar.
- `ExperimentManager`: immutable, uniquely-named experiment directories with the
  standard subfolder layout and config/environment snapshots.
- Unit tests for every component; full suite at 39 passing.

Files Added

- `src/adaptivehb/core/{__init__,interfaces,types,utils}.py`
- `src/adaptivehb/managers/{__init__,registry,state,checkpoint,experiment}.py`
- `tests/{test_core,test_registry,test_state,test_checkpoint,test_experiment}.py`
- `conftest.py` (root; shared `framework_config` fixture)

Files Modified

- `tests/conftest.py`, `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`,
  `docs/DECISION_LOG.md`

Notes

- No ML code; managers are exercised on dummy data only.
- TrainingManager, PipelineManager, and the HbPipeline facade are deferred to the
  next milestone (Decision 016).

---

## v0.3.1

Date

2026-07-06

Status

Core Infrastructure (Phase 2) — Orchestration Layer Complete

Completed

- Job engine (`adaptivehb.managers.jobs`): `Job` + `JobQueue` with topological
  dependency resolution, cycle and missing-dependency detection, resume-skip
  support, and fail-fast error propagation.
- `TrainingManager`: a generic, resumable training loop over any `Trainable`,
  wiring CheckpointManager (latest/best), StateManager (progress + resume), and
  RegistryManager (automatic registration); best-metric tracking and early
  stopping included.
- `PipelineManager`: owns and initializes the managers in the documented order,
  seeds the RNGs, and dispatches by mode; BUILD mode is fully implemented as a
  dependency-ordered self-check. Modes depending on later phases raise
  informative errors.
- `HbPipeline` (`adaptivehb.pipeline`): the single public facade
  (build/train/resume/evaluate/test/predict/deploy/shutdown).

Files Added

- `src/adaptivehb/managers/{jobs,training,pipeline}.py`
- `src/adaptivehb/pipeline.py`
- `tests/{test_jobs,test_training,test_pipeline}.py`

Files Modified

- `src/adaptivehb/managers/__init__.py`, `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- No ML code; TrainingManager is exercised on dummy trainables.
- The infrastructure phase (Phase 2) is now complete; DatasetManager (Phase 3)
  is next.

---

## v0.4.0

Date

2026-07-06

Status

Dataset Module (Phase 3) — Complete

Completed

- Typed `DatasetConfig` (`adaptivehb.dataset.config`) parsed from
  `configs/dataset.yaml` (paths, tissues, image spec, metadata columns, split
  ratios with sum validation).
- `MetadataTable` (`adaptivehb.dataset.metadata`): stdlib-csv metadata loading
  with patient-ID indexing — no pandas dependency.
- `DatasetValidator` (+ `ValidationReport`/`ValidationIssue`): mandatory-column,
  duplicate-ID, missing/invalid-hemoglobin checks (errors); orphan-image,
  patient-without-image, missing-mask checks (warnings).
- Patient-level `patient_level_split`: deterministic, seeded, leakage-free.
- `compute_statistics` (+ `DatasetStatistics`): per-tissue counts, Hb/age
  summaries, gender distribution, missing counts.
- Standardized `Sample` schema consumed across the framework.
- `DatasetManager`: single access point (load/validate/split/samples/statistics/
  summary/export) resolving all paths from config.
- `generate_synthetic_dataset`: reusable, spec-conformant synthetic dataset for
  tests, BUILD-mode, and examples.

Files Added

- `src/adaptivehb/dataset/{__init__,config,schema,metadata,validation,splitting,
  statistics,manager,synthetic}.py`
- `tests/{test_dataset,test_dataset_split}.py`

Files Modified

- `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- Dependency-light core (stdlib only); image decoding/preprocessing/augmentation
  deferred to the training data pipeline (Phase 5+) — Decision 019.
- 78 tests passing.

---

## v0.5.0

Date

2026-07-06

Status

Pipeline Framework (Phase 4) — Data-driven Modes Complete

Completed

- `adaptivehb.models` package with `DummyTrainable` + `make_dummy_factory`, so
  the pipeline runs end-to-end before real models exist (Phases 5-6).
- `adaptivehb.managers.pipeline_modes`: TRAINING (validate → split → train
  segmentation + per-tissue prediction → auto-register), RESUME (per-model
  checkpoint resume), EVALUATION (surface registered-model metrics), and
  INFERENCE (dummy Hb predictions over held-out samples) as dependency-ordered
  job sequences.
- `PipelineManager` now owns the `DatasetManager` and an injectable trainable
  factory, initializes managers in the documented order, and dispatches all
  modes; only DEPLOYMENT remains deferred (Phase 9).
- `HbPipeline` data-driven API activated: `train(epochs=)`, `resume(epochs=)`,
  `evaluate`, `test`, `predict`; `from_config_dir`/constructor accept
  `dataset_root`.

Files Added

- `src/adaptivehb/models/{__init__,dummy}.py`
- `src/adaptivehb/managers/pipeline_modes.py`
- `tests/test_pipeline_modes.py`

Files Modified

- `src/adaptivehb/managers/pipeline.py`, `src/adaptivehb/pipeline.py`,
  `tests/test_pipeline.py`, `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`,
  `docs/DECISION_LOG.md`

Notes

- Still no real ML; segmentation/prediction training uses dummy trainables via
  the injected factory (Decision 020). End-to-end run registers 3 segmentation +
  4 prediction models.
- 84 tests passing.

---

## v0.6.0

Date

2026-07-06

Status

Segmentation Framework (Phase 5) — Complete (torch-optional)

Completed

- `SegmentationModel` interface (`adaptivehb.segmentation.base`) implementing the
  Trainable contract plus `build`/`predict`/`save`/`load`, so segmentation models
  are trained by the generic TrainingManager and are interchangeable.
- Typed `SegmentationConfig` parsed from `configs/segmentation.yaml`.
- Name-keyed model factory (`register_segmentation`/`build_segmentation`/
  `available_segmentation`) with graceful fallback to the reference model when a
  requested torch backend is unavailable (Decision 021).
- Torch-free `ReferenceSegmentationModel` (deterministic) — keeps the framework
  runnable and testable without the ML stack.
- Guarded real backends (`torch_models`): from-scratch UNet, DeepLabV3+ (via
  torchvision), SegFormer (via segmentation_models_pytorch); registered only
  when torch is importable.
- `SegmentationManager`: builds models from config and exposes them as trainables.
- Pipeline wiring: the default trainable factory now dispatches SEGMENTATION
  plans to the SegmentationManager (prediction stays dummy until Phase 6); BUILD
  mode gained a segmentation self-check.

Files Added

- `src/adaptivehb/segmentation/{__init__,base,config,registry,reference,
  torch_models,manager}.py`
- `tests/test_segmentation.py`

Files Modified

- `src/adaptivehb/exceptions.py` (SegmentationError, ModelError),
  `src/adaptivehb/managers/pipeline.py`, `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- Real training loops require image tensors and run in the experiment phase;
  the framework is verified with the reference model (Decision 021).
- 91 tests passing (1 torch backend test skipped without torch).

---

## v0.7.0

Date

2026-07-06

Status

Prediction Framework (Phase 6) — Complete (torch-optional)

Completed

- `PredictionModel` interface (`adaptivehb.prediction.base`) implementing the
  Trainable contract plus `build`/`predict`(regression)/`save`/`load` — per-tissue
  models are interchangeable and trained by the generic TrainingManager
  (Decision 004).
- Typed `PredictionConfig` parsed from `configs/prediction.yaml` (models, per-tissue
  routing, training hyperparameters, input spec, metadata-fusion flags), with
  `architecture_for_tissue()`.
- Name-keyed model factory (`register_prediction`/`build_prediction`/
  `available_prediction`) with graceful fallback to the reference regressor when a
  torch backbone is unavailable (Decision 021).
- Torch-free `ReferencePredictionModel` (deterministic; returns a plausible g/dL
  estimate) — keeps the framework runnable/testable without the ML stack.
- Guarded real backbones (`torch_models`): EfficientNet/ResNet/DenseNet/ViT/
  ConvNeXt adapted to single-output regression heads via torchvision; registered
  only when torch is importable.
- `PredictionManager`: builds models by architecture or per-tissue routing and
  exposes them as trainables (derives the tissue from the ``hb_<tissue>`` plan name).
- Pipeline wiring: the default trainable factory now dispatches PREDICTION plans
  to the PredictionManager; BUILD mode gained a prediction self-check.

Files Added

- `src/adaptivehb/prediction/{__init__,base,config,registry,reference,
  torch_models,manager}.py`
- `tests/test_prediction.py`

Files Modified

- `src/adaptivehb/managers/pipeline.py`, `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`

Notes

- Real training loops need image tensors and run in the experiment phase;
  verified now with the reference regressor (Decision 021).
- 98 tests passing (2 torch backbone tests skipped without torch).

---

## v0.8.0

Date

2026-07-06

Status

Adaptive Decision Framework (Phase 7) — Complete (deterministic policies)

Completed

- Common `Agent` base (`adaptivehb.agents.base`) implementing the BaseAgent
  contract plus evaluate/reset/shutdown/save/load; deterministic ``train`` no-op;
  structured `AgentDecision` outputs (AGENT_SPEC Ch.4-9).
- Typed `AgentsConfig` parsed from `configs/agents.yaml` with canonical execution
  order and enable/disable per agent.
- Seven agents across three layers:
  perception — `QualityAssessmentAgent`, `ROIVerificationAgent`;
  decision — `SegmentationSelectionAgent`, `TissueSelectionAgent`,
  `PredictionRoutingAgent`;
  clinical — `FusionAgent` (confidence-weighted), `ConfidenceAgent` (agreement +
  quality → confidence, interval, recommendation).
- Deterministic `WorkflowController` that threads each agent's structured outputs
  through a shared context (agents never communicate directly, AGENT_SPEC Ch.7)
  and assembles a `WorkflowResult`.
- `AgentManager`: builds the enabled agents from config and runs the workflow.
- Pipeline wiring: PipelineManager owns the AgentManager and BUILD mode gained an
  agent self-check.

Files Added

- `src/adaptivehb/agents/{__init__,schema,config,base,perception,decision,
  clinical,controller,manager}.py`
- `tests/test_agents.py`

Files Modified

- `src/adaptivehb/managers/pipeline.py`, `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- Agents operate on structured intermediate features (quality/ROI/prediction),
  not raw images, matching EXPERIMENT_SPEC Ch.12 (Decision 022). Learned policies
  can replace deterministic ones behind the same interface.
- 112 tests passing (2 torch tests skipped without torch).

---

## v0.9.0

Date

2026-07-06

Status

Evaluation Framework (Phase 8) — Complete

Completed

- `adaptivehb.evaluation.metrics`: torch-free, stdlib-only metrics — regression
  (MAE, RMSE, R², Pearson, Spearman, mean bias, std of differences), clinical
  (Bland-Altman bias + 95% limits of agreement, within ±band fractions),
  classification (anemia thresholding → accuracy/precision/recall/F1), and
  calibration (expected calibration error). Degenerate inputs handled (zero
  variance → correlation 0.0) instead of raising.
- Typed `EvaluationConfig` parsed from `configs/evaluation.yaml` (metric lists,
  within-bands, anemia threshold, output toggles, baseline-comparison flag).
- `EvaluationReport` with CSV (per-sample) and JSON (metrics) export.
- `EvaluationManager`: `evaluate` (full bundle), `compare` (baseline vs adaptive),
  `export`.
- Real EVALUATION mode: replaces the placeholder that surfaced registry metrics —
  now builds per-patient (true, predicted) pairs from the test split via the
  registered prediction models, computes the metric bundle, and exports reports.
- Wired into the pipeline (BUILD self-check for evaluation metrics).

Files Added

- `src/adaptivehb/evaluation/{__init__,metrics,config,report,manager}.py`
- `tests/test_evaluation.py`

Files Modified

- `src/adaptivehb/exceptions.py` (EvaluationError),
  `src/adaptivehb/managers/pipeline.py`,
  `src/adaptivehb/managers/pipeline_modes.py`, `tests/test_pipeline_modes.py`,
  `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- Metrics are computed by the framework now; with reference (constant) predictions
  correlations are 0 by construction and improve once real models produce varying
  predictions (Decision 023).
- 123 tests passing (2 torch tests skipped without torch).

---

## v1.0.0

Date

2026-07-06

Status

Deployment Framework (Phase 9) — Complete → FRAMEWORK (STAGE A) COMPLETE

Completed

- Typed `DeploymentConfig` parsed from `configs/deployment.yaml`.
- `ClinicalReport`: patient-level hemoglobin report with JSON + human-readable
  text export.
- `HbInferenceService`: load-only serving engine — confirms registered prediction
  models, builds them, runs the adaptive workflow on patient input, and returns a
  `ClinicalReport`. No retraining (PIPELINE_SPEC Ch.17, Charter §27).
- Deployment targets (`adaptivehb.deployment.targets`): dependency-free
  `DesktopTarget` (exposes a bound predict callable) plus guarded
  FastAPI/Gradio/Streamlit targets that raise a clear error when the optional
  extra is absent; `build_target` / `available_targets` (Decision 024).
- `DeploymentManager`: load / predict / export-report / launch; wired into the
  pipeline with a BUILD self-check.
- Real DEPLOYMENT mode: loads registry models, produces a demo clinical report,
  and exports it. All six pipeline modes are now substantive.

Files Added

- `src/adaptivehb/deployment/{__init__,config,report,service,targets,manager}.py`
- `tests/test_deployment.py`

Files Modified

- `src/adaptivehb/exceptions.py` (DeploymentError),
  `src/adaptivehb/managers/pipeline.py`,
  `src/adaptivehb/managers/pipeline_modes.py`, `tests/test_pipeline.py`,
  `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- This completes Stage A (Framework Development, Phases 1-9). Stage B
  (Experiments) begins with the Sample→DataLoader bridge and real training.
- 133 tests passing (2 torch tests skipped without torch).

---

## v1.1.0

Date

2026-07-06

Status

Stage B — Training-data bridge (Sample → batches / DataLoader)

Completed

- New `adaptivehb.dataloading` package: the training-data bridge that turns
  standardized `Sample` records into batches and (guarded) tensors for the real
  training loops.
- `Batch` + `iter_batches` / `tissue_batches` / `batches_for_split`: dependency-
  free batching and hemoglobin-label extraction; unlabelled samples are skipped
  for training but can be retained for inference; drop-last and per-tissue
  grouping supported.
- `ImageDecoder` (`decoding`): decodes image files to RGB arrays via OpenCV or
  Pillow when installed, and raises a clear `DatasetError` otherwise; real
  decoding verified against OpenCV.
- `TransformSpec` + `build_transform` (`transforms`): preprocessing/augmentation
  parsed from `dataset.yaml`; builds an Albumentations pipeline when available
  (augmentation on the training split only) and returns `None` otherwise.
- `build_dataloader` (`torch_loader`): a guarded PyTorch `DataLoader` adapter
  yielding `(image, target)` tensors (target = Hb for prediction, mask for
  segmentation); raises a clear error when torch is absent.

Files Added

- `src/adaptivehb/dataloading/{__init__,batch,decoding,transforms,torch_loader}.py`
- `tests/test_dataloading.py`

Files Modified

- `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- All ML/vision dependencies are optional (Decision 025); batching/label logic is
  fully torch-free and tested. The real train_epoch/validate loops that consume
  the DataLoader are the next milestone.
- 145 tests passing (2 torch tests skipped without torch).

---

## v1.2.0

Date

2026-07-06

Status

Stage B — Real training loops (torch backbones train over the DataLoader)

Completed

- New `adaptivehb.training_ops` module shared by segmentation and prediction:
  - Torch-free, tested: `LossAccumulator` (sample-weighted running average),
    optimizer/loss name validation (`validate_optimizer`/`validate_loss`,
    `supported_optimizers`/`supported_losses`), and `resolve_device`.
  - Guarded builders: `build_optimizer` (Adam/AdamW/SGD), `build_scheduler`
    (cosine/step/none), `build_regression_loss` (MSE/MAE/SmoothL1), and
    `build_segmentation_loss` (Dice / BCE / Dice-BCE) — all validate the name
    first and raise a clear error when torch is absent.
  - Shared epoch loops `run_regression_epoch` / `run_segmentation_epoch`.
- `TorchPredictionModel` and `TorchSegmentationModel` now implement real
  `train_epoch`/`validate` over the attached DataLoader: device-aware, lazily
  building the config-driven optimizer/scheduler/loss, stepping the scheduler,
  and returning loss + MAE (prediction) / loss + Dice (segmentation). They raise
  a clear error when no data is attached.
- Pipeline TRAINING (`pipeline_modes._attach_dataloaders`) builds per-plan
  train/val dataloaders — per tissue for prediction, all samples for
  segmentation — with the configured transforms and batch size, and attaches
  them to torch models. Reference (torch-free) models are unaffected.

Files Added

- `src/adaptivehb/training_ops.py`
- `tests/test_training_ops.py`

Files Modified

- `src/adaptivehb/prediction/torch_models.py`,
  `src/adaptivehb/segmentation/torch_models.py`,
  `src/adaptivehb/managers/pipeline_modes.py`,
  `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- The real train loops are implemented to standard and unit-guarded; they execute
  only where torch is installed (their integration tests skip otherwise). The
  torch-free helpers are fully tested (Decision 026).
- 153 tests passing (4 torch tests skipped without torch).

---

## v1.3.0

Date

2026-07-06

Status

Stage B — Publication assets (figures + tables)

Completed

- New `adaptivehb.reporting` package generating publication-quality assets from
  evaluation results, with all plotting/Excel dependencies optional (Decision 027).
- `FigureGenerator` (matplotlib, headless Agg backend): predicted-vs-true scatter
  with identity line, Bland-Altman agreement (bias + 95% LoA), residual histogram,
  training curves, and model-comparison bars; each figure saved in the configured
  formats (PNG + PDF). Bad inputs raise a clear `ReportingError`.
- Table export (`tables`): `flatten_metrics` turns nested metric bundles into flat
  rows; `export_table_csv` (stdlib) and `export_table_excel` (openpyxl, guarded).
- `ReportingManager`: reads figure formats + output dirs from config, reports
  backend availability, and degrades gracefully when matplotlib/openpyxl are
  absent.
- Pipeline EVALUATION now emits scatter/Bland-Altman/residual figures and a
  CSV/Excel metrics table alongside the existing JSON/CSV reports; BUILD gained a
  reporting self-check.

Files Added

- `src/adaptivehb/reporting/{__init__,figures,tables,manager}.py`
- `tests/test_reporting.py`

Files Modified

- `src/adaptivehb/exceptions.py` (ReportingError),
  `src/adaptivehb/managers/pipeline.py`,
  `src/adaptivehb/managers/pipeline_modes.py`,
  `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Notes

- matplotlib/pandas/openpyxl are present in the dev sandbox, so figure and table
  generation are fully tested here (real PNG/PDF/XLSX output).
- 163 tests passing (4 torch tests skipped without torch).

---

## v1.14.0

Date

2026-07-07

Status

Stage B — Segmentation evaluation metrics (unblocks Paper 1)

Completed

- New `adaptivehb.segmentation.metrics` module (stdlib-only, torch-free,
  numpy-optional): confusion-matrix-based IoU/Jaccard, Dice/F1, pixel accuracy,
  mean per-class accuracy, and frequency-weighted IoU (Decision 038).
  - One-shot `iou_score` / `dice_score` / `pixel_accuracy` / `segmentation_metrics`.
  - `SegmentationMetrics` accumulator (`update`/`compute`/`reset`/`confusion`) for
    dataset-level scoring in one pass.
  - `binarize(prob, threshold)` for probability maps; `confusion_matrix` and
    `metrics_from_confusion` primitives.
  - Accepts nested sequences or numpy arrays; honors `ignore_index`; excludes
    classes absent from both prediction and truth from the means; degenerate inputs
    never raise.
- Exported from `adaptivehb.segmentation`.

Files Added

- `src/adaptivehb/segmentation/metrics.py`
- `tests/test_segmentation_metrics.py`

Files Modified

- `src/adaptivehb/segmentation/__init__.py`, `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Tests

- 244 passed, 4 skipped (torch not installed). 15 new tests in
  `tests/test_segmentation_metrics.py`: confusion-matrix counts, size/num_classes
  errors, perfect/partial/disjoint IoU-Dice against hand-computed values,
  absent-class exclusion, `ignore_index`, accumulator equivalence + reset,
  multi-class frequency-weighted IoU, `binarize`, and numpy-array input.

Notes

- Boundary/contour metrics and a segmentation-evaluation pipeline mode (needs torch
  inference) remain future work.

---

## v1.13.0

Date

2026-07-07

Status

Stage B — Paper 2 manuscript draft (prediction-model benchmark)

Completed

- New `paper/paper2_prediction_models.md`: full draft of Paper 2, "Deep Learning
  Models for Multi-Tissue Non-Invasive Hemoglobin Estimation" (Decision 037) — a
  benchmark of five backbones (EfficientNet, ResNet, DenseNet, ViT, ConvNeXt) for
  per-tissue Hb regression across eye/palm/tongue/nail. Problem formulation, model
  descriptions, training protocol (from `prediction.yaml`), metrics, paired
  statistics, and reproducibility complete; Results templated with `[[RESULT: …]]`
  placeholders.
- `paper/paper2_prediction_models.docx`: pandoc-rendered Word copy.
- `paper/README.md`: added the Paper 2 results-to-file mapping and file listing.

Files Added

- `paper/paper2_prediction_models.md`, `paper/paper2_prediction_models.docx`

Files Modified

- `paper/README.md`, `docs/PAPER_PLAN.md` (Paper 2 -> draft in progress),
  `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Tests

- No framework source changed; suite unchanged at 229 passed, 4 skipped.

Notes

- Results/discussion remain templated until the backbones are trained on the study
  dataset; the full backbone×tissue matrix requires training each backbone per
  tissue.

---

## v1.12.0

Date

2026-07-07

Status

Stage B — Paper 3 manuscript draft (parallel with the pending real run)

Completed

- New top-level `paper/` directory holding the Paper 3 manuscript draft (Decision
  036): "An Adaptive Multi-Agent Decision Framework for Non-Invasive Hemoglobin
  Estimation" (primary thesis paper; PAPER_PLAN Paper 3).
- `paper/paper3_adaptive_framework.md`: full draft — abstract, introduction, related
  work, methods (framework, segmentation, prediction, the seven agents across three
  layers + workflow controller, baseline-vs-adaptive protocol, paired statistics,
  reproducibility), experimental setup, a Results section templated with
  `[[RESULT: …]]` placeholders keyed to archived outputs, discussion, limitations,
  and conclusion.
- `paper/references.bib`: starter bibliography (architectures, statistics; domain
  citations marked TODO).
- `paper/README.md`: maps every results placeholder to its source file
  (`comparison.json`, `cv_summary.json`, figures, `provenance.json`, …).
- `paper/paper3_adaptive_framework.docx`: pandoc-rendered Word copy for
  reading/editing.

Files Added

- `paper/paper3_adaptive_framework.md`, `paper/references.bib`, `paper/README.md`,
  `paper/paper3_adaptive_framework.docx`

Files Modified

- `docs/PAPER_PLAN.md` (Paper 3 -> draft in progress), `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Tests

- No framework source changed; suite unchanged at 229 passed, 4 skipped.

Notes

- Results/discussion remain templated until the models are trained on the study
  dataset; the per-run report (`reports/experiment_report.md`) feeds the Results
  section directly.

---

## v1.11.0

Date

2026-07-07

Status

Stage B — Patient-level k-fold cross-validation

Completed

- New `adaptivehb.crossval` module: `CrossValidationRunner` runs patient-level
  k-fold cross-validation by reusing `ExperimentRunner` per fold in isolated
  directories, then aggregates per-fold metrics (mean/std/min/max) and
  baseline-vs-adaptive comparisons and archives `cv_summary.json`,
  `cv_metrics.csv`, and `cv_report.md` (Decision 035).
- New `dataset.splitting.k_fold_split(patient_ids, k, seed=, val_fraction=)`:
  balanced, deterministic, patient-level folds (every patient in exactly one test
  fold; no leakage).
- `DatasetManager` gains `apply_split()` / `clear_pinned_split()`, and `split()`
  now honors a pinned split (refactored through a shared `_retag` helper) so an
  externally-controlled fold assignment survives internal `split()` calls.
- `HbPipeline.cross_validate(name, folds=, epochs=)` facade and an `adaptivehb
  crossval --folds K` CLI subcommand.

Files Added

- `src/adaptivehb/crossval.py`
- `tests/test_crossval.py`

Files Modified

- `src/adaptivehb/dataset/splitting.py`, `src/adaptivehb/dataset/manager.py`,
  `src/adaptivehb/pipeline.py`, `src/adaptivehb/cli.py`, `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Tests

- 229 passed, 4 skipped (torch not installed). 11 new tests in
  `tests/test_crossval.py`: fold partition/leakage/balance/determinism/validation
  slice/error handling, split pinning surviving `split()`, CV end-to-end (all folds
  run, metrics aggregated, outputs archived), too-many-folds rejection, and the
  facade.

Notes

- Each fold is isolated (own `base_dir`) so checkpoints/registry never bleed across
  folds. On the reference models folds still show baseline == adaptive; real
  divergence needs the torch backbones + a dataset.

---

## v1.10.0

Date

2026-07-07

Status

Stage B — Publication-ready experiment report generator

Completed

- New `adaptivehb.reporting.experiment_report` module (stdlib-only, torch-free)
  that renders a consolidated experiment report in two formats from a single
  `ExperimentReportData` (Decision 034):
  - `render_markdown()` — `experiment_report.md` with a plain-language results
    statement (MAE change + significance verdict + Cohen's d effect size), a
    baseline-vs-adaptive table, a significance table, an adaptive-metrics table
    (incl. within-band clinical fractions), a reproducibility block, and
    relative figure links.
  - `render_html()` — a self-contained `experiment_report.html` with the figures
    embedded via `<img>`.
  - `write_experiment_report()` — writes both into a directory.
- `ExperimentRunner` now generates the report into `<experiment>/reports/` after
  the figures, and records the paths in the summary and in a new
  `ExperimentResult.report_paths` field (additive, backward compatible).
- Robust to missing pieces: absent comparison/significance/provenance/figures each
  degrade to a clear note rather than raising; degenerate (n<2) significance yields
  a "not evaluable" statement.

Files Added

- `src/adaptivehb/reporting/experiment_report.py`
- `tests/test_experiment_report.py`

Files Modified

- `src/adaptivehb/experiment.py`, `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`,
  `docs/DECISION_LOG.md`

Tests

- 218 passed, 4 skipped (torch not installed). 9 new tests in
  `tests/test_experiment_report.py`: Markdown sections, significant vs degenerate
  significance statements, provenance/metrics rendering, relative figure links,
  well-formed HTML with embedded figures, empty-data robustness, dual-format
  writing, and end-to-end archival by an experiment.

---

## v1.9.0

Date

2026-07-07

Status

Stage B — Runnable experiment notebooks

Completed

- New `notebooks/smoke_synthetic.ipynb`: runs the full pipeline **torch-free** on
  a generated synthetic dataset (reference models) and displays the
  baseline-vs-adaptive comparison with paired significance, the reproducibility
  provenance manifest, and the figures. Runs anywhere Python 3.11+ is available.
- New `notebooks/train_pipeline.ipynb` (the notebook named in PROJECT_MANIFEST):
  installs the `ml` extra, points at a real (or synthetic) dataset, trains the
  real PyTorch backbones via `HbPipeline.experiment`, and shows the archived
  metrics, comparison + significance, provenance, and publication figures. A
  terminal equivalent (`adaptivehb experiment …`) is documented inline.
- New `notebooks/README.md` describing both notebooks and their requirements.
- Both notebooks are thin facade callers (no framework logic, nothing hardcoded)
  and self-locate the repo root, so they run with or without `pip install`
  (Decision 033).

Files Added

- `notebooks/smoke_synthetic.ipynb`, `notebooks/train_pipeline.ipynb`,
  `notebooks/README.md`

Files Modified

- `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/CHANGELOG.md`,
  `docs/DECISION_LOG.md`

Tests / Verification

- `smoke_synthetic.ipynb` executed end-to-end headlessly with `nbclient`
  (torch-free) — all cells ran with no errors and archived a real experiment.
- `train_pipeline.ipynb` validated with `nbformat` and every code cell compiled
  (IPython magics stripped); it needs torch, so it is not executed here.
- Framework suite unchanged: 209 passed, 4 skipped (no source files modified).

Notes

- Notebooks were generated programmatically with `nbformat` (valid-by-construction
  JSON), not hand-edited.

---

## v1.8.0

Date

2026-07-06

Status

Stage B — Experiment reproducibility provenance manifest

Completed

- New `adaptivehb.provenance` module (stdlib-only, torch-free) assembling a
  self-describing reproducibility manifest (Decision 032):
  - `collect_environment()` — Python/platform details plus installed versions of
    the tracked scientific packages (absent packages omitted).
  - `git_revision()` — reads the current commit/branch straight from `.git`
    (HEAD, loose refs, and packed-refs), no `git` executable required; returns
    `None` outside a repo.
  - `config_fingerprint()` / `dataset_fingerprint()` — SHA-256 content hashes of
    the full configuration and of the dataset roster (patients, samples,
    per-tissue and per-split counts, and a `(patient_id, tissue, filename, hb)`
    hash).
  - `build_manifest()` / `write_manifest()` — assemble and persist the manifest.
- `ExperimentRunner` now builds the manifest at experiment creation and writes it
  to `<experiment>/configuration/provenance.json`; the manifest is also mirrored
  in the experiment summary and in a new `provenance` field on `ExperimentResult`
  (additive, backward compatible).

Files Added

- `src/adaptivehb/provenance.py`
- `tests/test_provenance.py`

Files Modified

- `src/adaptivehb/experiment.py`, `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`,
  `docs/DECISION_LOG.md`

Tests

- 209 passed, 4 skipped (torch not installed). 11 new tests in
  `tests/test_provenance.py`: environment shape/omission, git parsing
  (no-repo/branch/detached/packed-refs via a synthetic `.git`), deterministic
  config/dataset fingerprints, manifest shape, and end-to-end archival of
  `provenance.json` by an experiment.

Notes

- Git resolves to `None` in the dev sandbox (no `.git` on the mount) — the
  graceful-degradation path; it populates once run natively in the git checkout.

---

## v1.7.0

Date

2026-07-06

Status

Stage B — Paired statistical significance for the baseline-vs-adaptive comparison

Completed

- New `adaptivehb.evaluation.significance` module (stdlib-only, torch/numpy-free):
  paired two-sided Student t-test (t-tail via the regularized incomplete beta
  function), Wilcoxon signed-rank test (normal approximation, tie/zero handling),
  percentile bootstrap CI for the mean paired difference (seeded/reproducible), and
  Cohen's d for paired samples, bundled by `compare_significance(...)` over the
  per-sample absolute-error differences (Decision 031).
- `EvaluationManager.compare` accepts optional paired arrays (`y_true`,
  `baseline_pred`, `adaptive_pred`) and attaches a `significance` block when they
  are supplied; the point comparison is unchanged otherwise (backward compatible).
- `ExperimentRunner` passes the paired arrays, so the archived `comparison.json`
  now carries the p-values, confidence interval, and effect size.
- Config-driven via a new `significance` subsection in `evaluation.yaml`
  (`enabled`, `bootstrap_iterations`, `confidence_level`), surfaced on
  `EvaluationConfig`.

Files Added

- `src/adaptivehb/evaluation/significance.py`
- `tests/test_significance.py`

Files Modified

- `src/adaptivehb/evaluation/manager.py`, `src/adaptivehb/evaluation/config.py`,
  `src/adaptivehb/evaluation/__init__.py`, `src/adaptivehb/experiment.py`,
  `configs/evaluation.yaml`, `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`,
  `docs/DECISION_LOG.md`

Tests

- 198 passed, 4 skipped (torch not installed). 16 new tests in
  `tests/test_significance.py` validate the t-CDF and paired t-test against known
  reference values, Wilcoxon zero/tie handling and one-sided effect, bootstrap CI
  reproducibility/bracketing, Cohen's d, the bundle shape, manager wiring
  (attach/omit), and the config on/off switch.

Notes

- Numerics verified against scipy reference values (t=2,df=10 -> p~=0.0734;
  ttest_1samp -> t~=4.3205, p~=0.00348) without depending on scipy. On tiny splits
  (n<2 pairs) the tests degrade gracefully to p=1.0 rather than raising.

---

## v1.6.0

Date

2026-07-06

Status

Stage B — Command-line interface (reproducible entry point for every pipeline mode)

Completed

- New `adaptivehb.cli` module: a thin, config-driven argparse CLI over the
  `HbPipeline` facade (Decision 030). Subcommands map one-to-one to pipeline
  modes: `build`, `train`, `resume`, `evaluate`, `predict`, `deploy`,
  `experiment`.
- Global options supply everything from arguments/config (no hardcoding):
  `--config-dir`, `--base-dir`, `--dataset-root`, `--quiet`; `--epochs` on
  training-like modes; `--name` on `experiment`. `--version` prints the package
  version.
- Dispatch builds `HbPipeline.from_config_dir(...)`, calls the matching public
  method, shuts the pipeline down, and prints the result as a JSON summary
  (`ExperimentResult.to_dict()` for experiments). Framework errors
  (`AdaptiveHbError`) exit 1 with a stderr message; usage errors exit 2.
- Heavy ML imports stay lazy, so the CLI imports and tests without torch.
- New `adaptivehb.__main__` enables `python -m adaptivehb`; `[project.scripts]`
  registers the `adaptivehb` console script.

Files Added

- `src/adaptivehb/cli.py`
- `src/adaptivehb/__main__.py`
- `tests/test_cli.py`

Files Modified

- `pyproject.toml` (added `[project.scripts]`), `CURRENT_TASK.yaml`,
  `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Tests

- 182 passed, 4 skipped (torch not installed). 9 new tests in `tests/test_cli.py`
  (parser exposes all modes, command required, `--epochs` scoping, experiment
  name default/override, `--version`, train/build/experiment dispatch, `--quiet`,
  and framework-error exit code).

Notes

- Enables launching a real experiment reproducibly from one command in the
  torch/GPU environment: `adaptivehb experiment --dataset-root <data> --epochs N`.

---

## v1.5.0

Date

2026-07-06

Status

Stage B — Registry-backed model loading (evaluate/experiment use trained weights)

Completed

- New `adaptivehb.model_loading` module: `load_weights_into(model, checkpoints,
  name, prefer="best")` loads a model's trained `model_state` from the checkpoint
  store, located by name via the CheckpointManager (prefers `best`, falls back to
  `latest`). No checkpoint path is hardcoded (Decision 029, MODEL_REGISTRY_SPEC
  Ch.15). Missing checkpoints degrade to an untrained model with a warning.
- `PredictionManager.load_trained(name, checkpoints, tissue=, architecture=,
  prefer=)` and `SegmentationManager.load_trained(name, checkpoints,
  architecture=, prefer=)`: build a model and load its trained weights in one call.
- Evaluation (`managers.pipeline_modes._evaluation_pairs`) now scores **trained**
  per-tissue models instead of freshly built reference models.
- `ExperimentRunner._baseline_vs_adaptive` rewired: per-tissue predictions come
  from `load_trained` models; the static **baseline** is their unweighted mean and
  the **adaptive** result stays agent-fused (both consume the same trained models).

Files Added

- `src/adaptivehb/model_loading.py`
- `tests/test_model_loading.py`

Files Modified

- `src/adaptivehb/prediction/manager.py`, `src/adaptivehb/segmentation/manager.py`,
  `src/adaptivehb/managers/pipeline_modes.py`, `src/adaptivehb/experiment.py`,
  `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`

Tests

- 173 passed, 4 skipped (torch not installed). 6 new tests in
  `tests/test_model_loading.py` (weight round-trip, missing-checkpoint fallback,
  prediction/segmentation `load_trained`, untrained fallback, train->load
  integration).

Notes

- With the torch-free reference backend, reference predictions remain constant
  regardless of loaded state, so baseline == adaptive here; the loading machinery
  is verified and real divergence follows from the torch backbones + a real
  dataset.

---

## v1.4.0

Date

2026-07-06

Status

Stage B — Experiment orchestration (baseline vs adaptive)

Completed

- New `adaptivehb.experiment` module: `ExperimentRunner` runs a full, reproducible
  experiment and archives it (EXPERIMENT_SPEC).
- `ExperimentRunner.run(name, epochs=)`: creates an immutable experiment
  directory (via ExperimentManager), trains the models, then builds a **static
  baseline** and the **adaptive** (agent-fused) predictions on the held-out test
  split, evaluates both under identical conditions, and compares them on MAE
  (Decision 008, EXPERIMENT_SPEC Ch.13-14).
- Archives into the experiment directory: baseline + adaptive metrics JSON, the
  comparison, per-sample predictions CSV, a flattened CSV/Excel metrics table, and
  scatter/Bland-Altman/baseline-vs-adaptive figures, plus a summary.
- `HbPipeline.experiment(name=, epochs=)` public facade (lazy import to avoid a
  circular dependency).

Files Added

- `src/adaptivehb/experiment.py`
- `tests/test_experiment_runner.py`

Files Modified

- `src/adaptivehb/pipeline.py`, `CURRENT_TASK.yaml`, `docs/PROJECT_STATE.md`,
  `docs/DECISION_LOG.md`

Notes

- Runs end-to-end on reference models here (verified); with constant reference
  predictions baseline == adaptive (improvement 0.0). The comparison mechanism is
  real and differentiates once trained models produce varying predictions
  (Decision 028).
- 167 tests passing (4 torch tests skipped without torch).

---

# Update Rules

Update after every coding session.

Never delete previous entries.

Entries should be chronological.

Describe only meaningful changes.

Avoid documenting trivial edits.

---

# End of CHANGELOG.md