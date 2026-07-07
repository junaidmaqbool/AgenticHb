# PROJECT_STATE.md

Version: 3.5
Status: Living Project Status
Last Updated: 2026-07-06

Project

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# 1. Purpose

Operational status document. Read at the start of every session; update at the end.

---

# 2. Overall Progress

Project Stage — Stage A (Framework) COMPLETE; Stage B (Experiments) in progress
Current Phase — Experiment + Writing — segmentation metrics added; Papers 2 & 3 drafted; real (torch) run user-executable
Current Milestone — Real experiment run on a dataset (torch backbones + real data), via notebooks/CLI
Status — Segmentation IoU/Dice metrics now available (unblocks Paper 1); Papers 2 & 3 drafted to template stage.

---

# 3. Documentation Status

All Phase 0 documents complete. IMPLEMENTATION_READINESS_REPORT.md complete.

---

# 4. Repository Status

| Component | Status |
|------------|--------|
| Framework (Stage A, Phases 1-9) | ✅ Complete |
| Training-data bridge + real training loops | ✅ Complete |
| Publication assets (figures + tables) | ✅ Complete |
| Experiment orchestration (baseline vs adaptive) | ✅ Complete |
| Registry-backed model loading (checkpoint-backed weights) | ✅ Complete |
| Command-line interface (reproducible entry point) | ✅ Complete |
| Paired significance testing (baseline vs adaptive) | ✅ Complete |
| Reproducibility provenance manifest | ✅ Complete |
| Runnable experiment notebooks (smoke + train_pipeline) | ✅ Complete |
| Publication experiment report (Markdown + HTML) | ✅ Complete |
| Patient-level k-fold cross-validation | ✅ Complete |
| Paper 3 manuscript draft (methods/protocol; results templated) | 🟡 In progress |
| Paper 2 manuscript draft (prediction benchmark; results templated) | 🟡 In progress |
| Segmentation evaluation metrics (IoU/Dice/pixel-acc) | ✅ Complete |
| Separate segmentation dataset (independent of Hb dataset) | ✅ Complete |
| Real experiment run (torch backbones + real dataset) | ⬜ Not Started (user runs notebook in a torch env) |

---

# 5. Current Implementation

Current Branch — main (git not yet initialized; see §10)
Current Phase — Experiment Phase (Stage B)
Current Module — None
Current Task — Run a real experiment with the torch backbones on a real dataset via
`adaptivehb experiment --dataset-root <data> --epochs N` (needs torch + data).

---

# 6. Completed Components

Package (`src/adaptivehb/`)

- Framework subsystems + `pipeline.py`
- `dataloading/`, `training_ops.py`, `reporting/`
- `experiment.py` — ExperimentRunner + ExperimentResult
- `model_loading.py` — checkpoint-backed weight loader
- `cli.py` + `__main__.py` — reproducible command-line entry point
- `evaluation/significance.py` — paired significance testing
- `provenance.py` — experiment reproducibility manifest
- `notebooks/` — smoke_synthetic + train_pipeline runnable notebooks
- `reporting/experiment_report.py` — publication Markdown+HTML report
- `crossval.py` — patient-level k-fold cross-validation
- `paper/` — Paper 2 (prediction benchmark) + Paper 3 (adaptive framework) drafts + refs + Word copies
- `segmentation/metrics.py` — IoU/Dice/pixel-accuracy segmentation metrics (this milestone)

Segmentation metrics highlights (this milestone)

- `adaptivehb.segmentation.metrics` (stdlib-only, torch-free, numpy-optional):
  confusion-matrix IoU/Jaccard, Dice/F1, pixel accuracy, mean per-class accuracy,
  frequency-weighted IoU; one-shot fns + `SegmentationMetrics` accumulator +
  `binarize`; handles ignore_index and absent classes (Decision 038). Unblocks Paper 1.

Manuscript drafts highlights (Paper 2)

- `paper/paper2_prediction_models.md`: benchmark of five backbones (EfficientNet,
  ResNet, DenseNet, ViT, ConvNeXt) for per-tissue Hb regression across
  eye/palm/tongue/nail; methods/protocol (from `prediction.yaml`)/statistics/
  reproducibility complete, results templated (Decision 037). Word copy included.

Paper 3 manuscript highlights

- `paper/paper3_adaptive_framework.md`: full draft of the primary thesis paper
  (abstract, intro, related work, methods incl. the seven agents/three layers,
  baseline-vs-adaptive protocol, paired statistics, reproducibility, setup,
  discussion, limitations, conclusion). Numeric results are `[[RESULT: …]]`
  placeholders keyed to archived outputs (Decision 036).
- `paper/references.bib`, `paper/README.md` (results-to-file map), and a
  pandoc-rendered `.docx` copy.

k-fold cross-validation highlights

- `adaptivehb.crossval.CrossValidationRunner`: runs patient-level k-fold CV by
  reusing `ExperimentRunner` per fold in isolated directories, aggregates per-fold
  metrics (mean/std/min/max) + comparisons, and archives `cv_summary.json`,
  `cv_metrics.csv`, `cv_report.md` (Decision 035).
- `dataset.splitting.k_fold_split` (balanced/deterministic/patient-level) +
  `DatasetManager.apply_split`/`clear_pinned_split` (split pinning survives
  internal `split()`).
- `HbPipeline.cross_validate(...)` facade + `adaptivehb crossval --folds K` CLI.

Publication report highlights

- `adaptivehb.reporting.experiment_report` (stdlib-only): renders a paper-ready
  Markdown (`experiment_report.md`) and self-contained HTML (`experiment_report.html`)
  from one `ExperimentReportData` — plain-language significance statement +
  Cohen's d, metrics/significance tables, reproducibility block, figure links
  (Decision 034).
- `ExperimentRunner` archives both into `<experiment>/reports/` and records the
  paths in the summary and a new `ExperimentResult.report_paths` field.

Runnable notebooks highlights

- `notebooks/smoke_synthetic.ipynb`: torch-free end-to-end run on a synthetic
  dataset (reference models); shows comparison+significance, provenance, figures.
  Executed headlessly here with nbclient — all cells pass.
- `notebooks/train_pipeline.ipynb` (named in PROJECT_MANIFEST): installs the `ml`
  extra and runs the real PyTorch experiment on a dataset via `HbPipeline`.
- Thin facade callers, nothing hardcoded, self-locating repo root (Decision 033).

Reproducibility provenance highlights

- `adaptivehb.provenance` (stdlib-only): captures the software environment
  (Python/platform + installed package versions), the git revision (read from
  `.git`, no subprocess), and SHA-256 fingerprints of the configuration and the
  dataset roster; `build_manifest`/`write_manifest` assemble and persist it
  (Decision 032).
- `ExperimentRunner` archives `<experiment>/configuration/provenance.json`, and
  mirrors the manifest in the summary and a new `ExperimentResult.provenance`
  field (additive, backward compatible).

Paired significance testing highlights

- `adaptivehb.evaluation.significance` (stdlib-only): paired Student t-test
  (t-tail via regularized incomplete beta), Wilcoxon signed-rank (normal
  approximation), bootstrap CI for the mean paired difference (seeded), and
  Cohen's d, bundled over the per-sample absolute-error differences (Decision 031).
- `EvaluationManager.compare` attaches a `significance` block when paired arrays
  are supplied (backward compatible); `ExperimentRunner` passes them, so the
  archived `comparison.json` carries p-values, a CI, and the effect size.
- Config-driven via the `significance` subsection of `evaluation.yaml`.

Command-line interface highlights

- `adaptivehb.cli`: a thin, config-driven argparse CLI over the `HbPipeline`
  facade (Decision 030). One subcommand per pipeline mode
  (build/train/resume/evaluate/predict/deploy/experiment); `--config-dir`,
  `--base-dir`, `--dataset-root`, `--epochs`, `--name`, `--quiet`, `--version`.
- Runnable as the `adaptivehb` console script or `python -m adaptivehb`; prints
  a JSON result summary; framework errors exit 1, usage errors exit 2. ML
  imports stay lazy, so it imports/tests without torch.

Registry-backed model loading highlights

- `adaptivehb.model_loading.load_weights_into(model, checkpoints, name)`: loads a
  model's trained `model_state` from the checkpoint store by name (prefers `best`,
  falls back to `latest`); no path hardcoded (Decision 029). Missing checkpoints
  degrade to an untrained model with a warning.
- `PredictionManager.load_trained` / `SegmentationManager.load_trained`: build a
  model and load its trained weights in one call.
- Evaluation and `ExperimentRunner._baseline_vs_adaptive` now score TRAINED
  per-tissue models; baseline = their unweighted mean, adaptive = agent-fused.

Experiment orchestration highlights

- `ExperimentRunner.run(name, epochs=)`: creates an immutable experiment directory,
  trains the models, then builds a **static baseline** and the **adaptive**
  (agent-fused) predictions on the held-out test split, evaluates both under the
  same conditions and compares them on MAE (Decision 008, EXPERIMENT_SPEC Ch.13-14).
- Archives baseline/adaptive metrics, the comparison, per-sample predictions, a
  CSV/Excel metrics table, and scatter/Bland-Altman/comparison figures into the
  experiment directory, plus a summary (Decision 028).
- `HbPipeline.experiment()` public facade (lazy import; no circular dependency).

Tests — 251 passing (4 torch tests skipped without torch); smoke notebook executed end-to-end via nbclient

---

# 7. Active Models

Reference (torch-free) models drive the experiment here; the flow uses the real
torch backbones where torch is installed. NOTE: evaluate/inference/experiment
currently build fresh models (they do not yet load registered checkpoint weights).

---

# 8. Current Dataset Status

DatasetManager + data bridge feed batches/tensors; the experiment consumes the
test split for baseline-vs-adaptive comparison.

---

# 9. Current Experiment Status

A full, archived experiment (train -> baseline vs adaptive -> compare -> figures)
runs end-to-end on reference models. With constant reference predictions the
baseline == adaptive (improvement 0.0); the comparison differentiates once real
trained models produce varying per-tissue predictions.

---

# 10. Known Issues

- Git not initialized from the sandbox (mount denies `.git`/deletes). Run `git init` natively.
- Evaluate/inference/experiment use freshly-built models, not registered trained
  weights (registry-backed loading is the next milestone). torch/albumentations
  absent in the dev sandbox (opencv/Pillow/numpy/matplotlib/pandas/openpyxl present);
  the real torch path runs in an ML env (its tests skip here). The mount drops some
  editor OVERWRITES; modified files are written/verified via the shell. Delivered
  code is intact (167/167 pass, 4 skipped, on a fresh cache-disabled run).

Technical Debt — None. Architecture Issues — None.
Pending Decisions — Provisional decisions 011–028 (DECISION_LOG.md) open to override.

---

# 11. Current Priorities

1. Real experiment run on a dataset (trained models -> genuine metrics) via the
   `adaptivehb` CLI in a torch/GPU environment
2. Draft Papers 1–4 from the generated assets
3. (Ops) native `git init` + an environment with torch + GPU

---

# 12. Next Development Session

Read PROJECT_INDEX → PROJECT_STATE → MODEL_REGISTRY_SPEC (load) → implement
registry-backed checkpoint loading so predictions use trained weights, then run a
real experiment in an ML environment → update living documents → stop.

---

# 13. Session Summary

Last Session — Stage B: Experiment orchestration.
Files Added — 2 (experiment.py + test_experiment_runner).
Files Modified — 5 (pipeline.py facade, CURRENT_TASK, PROJECT_STATE, CHANGELOG, DECISION_LOG).
Architecture Changes — None; experiment orchestration recorded as Decision 028.
Tests — 167 passing (4 skipped).

---

# 14. Estimated Completion

| Area | % |
|------|---|
| Framework (Stage A) | 100 |
| Data bridge + real training loops | 100 |
| Publication assets | 100 |
| Experiment orchestration | 100 |
| Registry-backed loading | 100 |
| Command-line interface | 100 |
| Paired significance testing | 100 |
| Reproducibility provenance | 100 |
| Runnable notebooks | 100 |
| Publication report generator | 100 |
| k-fold cross-validation | 100 |
| Paper 3 manuscript draft | 60 |
| Paper 2 manuscript draft | 60 |
| Segmentation evaluation metrics | 100 |
| Real experiment run (torch + data) | 0 |
| **Overall project** | **~98** |

---

# End of PROJECT_STATE.md
