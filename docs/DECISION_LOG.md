# DECISION_LOG.md

Version: 1.0

Status: Living Design History

Last Updated: July 2026

Project

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# 1. Purpose

DECISION_LOG.md records every important architectural, methodological, and implementation decision made during the project.

Its purpose is to

• Explain why decisions were made

• Preserve architectural reasoning

• Support future development

• Assist thesis and paper writing

• Reduce repeated discussions

This document should only record significant decisions.

Minor implementation details should not be included.

---

# 2. Decision Template

Every decision should follow the same format.

Decision ID

Date

Category

Decision

Reason

Alternatives Considered

Advantages

Disadvantages

Impact

Status

Notes

---

# Decision 001

Decision ID

001

Date

July 2026

Category

Architecture

Decision

Use a single Pipeline (HbPipeline) as the only public interface.

Reason

Centralized orchestration simplifies maintenance and allows future deployment without changing application logic.

Alternatives Considered

Direct manager access.

Advantages

Cleaner architecture.

Lower coupling.

Simpler deployment.

Disadvantages

Slightly larger Pipeline class.

Impact

Entire repository architecture.

Status

Accepted.

---

# Decision 002

Category

Architecture

Decision

Implement separate Managers for Dataset, Pipeline, Registry, Training, Evaluation and Deployment.

Reason

Single responsibility principle.

Advantages

Modular design.

Easy maintenance.

Independent testing.

Status

Accepted.

---

# Decision 003

Category

Research

Decision

Train one universal segmentation model capable of handling all tissues.

Reason

GPU limitations.

Simpler deployment.

Reduced maintenance.

Alternatives

Separate models per tissue.

Status

Accepted.

---

# Decision 004

Category

Research

Decision

Train independent Hb prediction models for each tissue.

Reason

Different tissues exhibit different visual characteristics.

Advantages

Higher specialization.

Flexible routing.

Status

Accepted.

---

# Decision 005

Category

Research

Decision

Use adaptive decision modules rather than a static ensemble.

Reason

Primary scientific contribution.

Advantages

Dynamic inference.

Improved robustness.

Greater novelty.

Status

Accepted.

---

# Decision 006

Category

Software

Decision

All model loading must occur through the Registry.

Reason

Prevent hardcoded checkpoint paths.

Advantages

Version control.

Automatic model discovery.

Cleaner code.

Status

Accepted.

---

# Decision 007

Category

Development

Decision

Separate framework development from model training.

Reason

Avoid coupling implementation with experiments.

Advantages

Better testing.

Cleaner debugging.

Reproducibility.

Status

Accepted.

---

# Decision 008

Category

Experiments

Decision

Always compare the adaptive framework against a static baseline.

Reason

Scientific validity.

Required for publication.

Status

Accepted.

---

# Decision 009

Category

Implementation

Decision

Every training process must support checkpoint recovery and automatic resume.

Reason

Training may require several days.

Advantages

No lost progress.

Robust experiments.

Status

Accepted.

---

# Decision 010

Category

Repository

Decision

All configuration must be externalized into configuration files.

Reason

Avoid hardcoded parameters.

Improve reproducibility.

Status

Accepted.

---

# Decision 011

Decision ID

011

Date

2026-07-06

Category

Repository

Decision

Adopt a `src/` layout with a single installable package `adaptivehb`. The flat
directory list in PROJECT_DESIGN_SPECIFICATION.md Chapter 4 is interpreted as
*logical subpackages* inside `adaptivehb/` (e.g. `adaptivehb/segmentation/`),
not literal top-level directories.

Reason

src-layout prevents accidental imports of the source tree, produces clean installs,
and matches `PROJECT_MANIFEST.yaml` (`src: src/`). A single reconciled tree removes
the conflict between the manifest and PDS Ch.4.

Alternatives Considered

Literal flat top-level layout from PDS Ch.4.

Advantages

Clean installs; unambiguous imports; one canonical structural source.

Disadvantages

Slight divergence from the literal PDS Ch.4 diagram (resolved by interpretation).

Impact

Entire repository layout.

Status

Accepted (provisional — override on request).

---

# Decision 012

Decision ID

012

Date

2026-07-06

Category

Architecture

Decision

Canonical manager roster: PipelineManager, DatasetManager, RegistryManager,
StateManager, CheckpointManager, ExperimentManager, TrainingManager,
SegmentationManager, PredictionManager, AgentManager, EvaluationManager,
DeploymentManager. Fusion and Confidence are **agents inside AgentManager**, not
separate managers. `HbPipeline` is a thin public facade over the internal
PipelineManager.

Reason

The four documents (manifest, index, PDS, pipeline spec) listed differing rosters.
This set unifies them and matches the Agent Specification's three-layer model,
avoiding a fragmented adaptive core.

Alternatives Considered

Separate FusionManager / ConfidenceManager.

Advantages

Single source of truth; no duplicated agent lifecycle code; matches AGENT_SPEC.

Impact

Phase 2 infrastructure and Phase 7 agents.

Status

Accepted (provisional — override on request).

---

# Decision 013

Decision ID

013

Date

2026-07-06

Category

Dataset

Decision

Tissue capture remains per-side (left/right eye, palm, nail; center tongue), but
Hb prediction models operate per tissue **class** (eye, palm, tongue, nail), with
left/right pooled at the dataset layer.

Reason

Reconciles the 7 laterality-specific tissues in PROJECT_MANIFEST.yaml with the 4
tissue classes in DATASET_SPEC.md and the prediction config. Consistent with
Decision 004 (independent Hb model per tissue).

Impact

DatasetManager, filename convention, prediction registry keys.

Status

Accepted (provisional — override on request).

---

# Decision 014

Decision ID

014

Date

2026-07-06

Category

Software

Decision

The Model Registry and Pipeline State use a JSON-on-disk backend (not SQLite) for
the initial implementation.

Reason

Human-readable, git-diffable, dependency-free, and sufficient for single-node
research use. SQLite remains a future option behind the same RegistryManager API.

Alternatives Considered

SQLite.

Advantages

Simplicity; transparency; easy inspection and reproducibility.

Disadvantages

Weaker concurrency guarantees (acceptable for sequential Job execution).

Impact

Phase 2 RegistryManager and StateManager.

Status

Accepted (provisional — override on request).

---

# Decision 015

Decision ID

015

Date

2026-07-06

Category

Documentation

Decision

License the framework under Apache-2.0; standardize tooling on pytest, ruff, black,
and mypy, configured in `pyproject.toml`.

Reason

Apache-2.0 suits open-source research software with a patent grant. The chosen
tools are the mainstream, low-friction quality stack and satisfy Charter §12.

Alternatives Considered

MIT license; deferring license selection.

Impact

Repository packaging and contributor workflow.

Status

Accepted (provisional — override on request).

---

# Decision 016

Decision ID

016

Date

2026-07-06

Category

Development

Decision

Split the Phase 2 "Core Infrastructure" milestone into two implementation
sessions: (A) the persistence/management foundation — core interfaces/types/
utils + RegistryManager, StateManager, CheckpointManager, ExperimentManager;
(B) the orchestration layer — TrainingManager, PipelineManager, and the
HbPipeline facade.

Reason

The orchestration layer coordinates components that arrive later (DatasetManager
in Phase 3; full pipeline modes in Phase 4). Building the foundation first keeps
each milestone cohesive, fully testable on dummy data, and avoids hollow stubs.

Alternatives Considered

Implementing all infrastructure managers in a single session.

Advantages

Higher code quality; genuine unit tests; smaller, reviewable increments.

Impact

Phase 2 sequencing only; no architectural change.

Status

Accepted (provisional — override on request).

---

# Decision 017

Decision ID

017

Date

2026-07-06

Category

Software

Decision

CheckpointManager serializes the checkpoint payload with `pickle` plus a
human-readable JSON metadata sidecar, rather than depending on `torch.save`.

Reason

Keeps the infrastructure layer torch-free and testable without a GPU while
remaining forward-compatible (pickle round-trips torch tensors; the serializer
can be swapped for `torch.save` in a later phase behind the same API).

Alternatives Considered

Requiring torch in the infrastructure layer and using `torch.save`.

Advantages

Fast, dependency-light tests; clean separation of infrastructure from the ML stack.

Disadvantages

Payload format will be revisited when real model states are introduced.

Impact

CheckpointManager implementation.

Status

Accepted (provisional — override on request).

---

# Decision 018

Decision ID

018

Date

2026-07-06

Category

Architecture

Decision

Models are driven by the TrainingManager through a minimal `Trainable` protocol
(`train_epoch`, `validate`, `state_dict`, `load_state_dict`) rather than the
TrainingManager knowing any concrete model type. Pipeline modes that require a
dataset or trained models (training, resume, evaluation, inference, deployment)
are explicitly deferred with informative errors until their phases land.

Reason

Keeps the training loop model-agnostic and torch-free, so segmentation and
prediction models (Phases 5-6) integrate by implementing four methods without
changing the orchestration layer. Deferring unbuilt modes keeps the pipeline
honest and testable now.

Alternatives Considered

Coupling TrainingManager to a concrete BaseModel with torch-specific signatures;
stubbing unbuilt modes as no-ops.

Advantages

Loose coupling; torch-free tests; clear extension contract; no hollow stubs.

Disadvantages

The protocol may gain methods (e.g. predict hooks) as real models arrive.

Impact

TrainingManager, PipelineManager, and all future model implementations.

Status

Accepted (provisional — override on request).

---

# Decision 019

Decision ID

019

Date

2026-07-06

Category

Dataset

Decision

The DatasetManager core (metadata loading, validation, patient-level splitting,
statistics) uses only the Python standard library (`csv`, `statistics`,
`random`). Image decoding, preprocessing, and augmentation — which require
opencv/PIL/albumentations — are deferred to the training data pipeline built with
the segmentation/prediction frameworks (Phases 5-6).

Reason

Keeps the dataset validation/splitting layer fast, importable, and testable
without the heavy ML/vision stack, consistent with the framework-first,
torch-free approach used throughout the infrastructure. Validation and
leakage-free splitting are the parts needed now; pixel-level operations belong
with the model data loaders.

Alternatives Considered

Requiring pandas + opencv in the dataset core from the outset.

Advantages

Lightweight tests; clean separation; no premature coupling to a vision library.

Disadvantages

Corrupt-image detection and pixel statistics arrive in a later phase; a
pandas-backed tabular view may be added behind the same interface.

Impact

`adaptivehb.dataset` package and its tests.

Status

Accepted (provisional — override on request).

---

# Decision 020

Decision ID

020

Date

2026-07-06

Category

Architecture

Decision

The PipelineManager builds models through an injectable trainable factory
(``trainable_factory``), defaulting to a dummy factory. TRAINING/RESUME/
EVALUATION/INFERENCE modes are implemented now against dummy models; only
DEPLOYMENT remains deferred.

Reason

Lets the full data flow (validate → split → train → register → evaluate → infer)
be implemented and tested before real segmentation/prediction networks exist.
Phases 5-6 replace the dummy factory with real model factories without changing
the pipeline or facade.

Alternatives Considered

Waiting until real models exist to implement the modes; hardcoding model
construction inside the pipeline.

Advantages

Full pipeline coverage now; clean seam for real models; no pipeline rewrites in
Phases 5-6.

Disadvantages

Dummy metrics are placeholders until real models and the Evaluation Framework
(Phase 8) arrive.

Impact

PipelineManager, HbPipeline, and the future segmentation/prediction factories.

Status

Accepted (provisional — override on request).

---

# Decision 021

Decision ID

021

Date

2026-07-06

Category

Software

Decision

The segmentation subsystem is torch-optional. The `SegmentationModel` interface
and factory are torch-free; real backends (UNet/DeepLabV3+/SegFormer) live behind
guarded imports and register only when PyTorch is available. When a requested
architecture has no registered builder (e.g. torch absent), the factory falls
back to a torch-free `ReferenceSegmentationModel` with the same interface. Real
training/validation loops (which need image tensors) are provided in the
experiment phase; the framework is verified now with the reference model.

Reason

Keeps the framework installable, importable, and fully testable without the heavy
ML/vision stack (consistent with the torch-free infrastructure), while shipping
real architectures that activate automatically when torch is present. The
segmentation interface implements the existing Trainable contract, so no changes
to the TrainingManager or pipeline are needed for real models.

Alternatives Considered

Making torch a hard dependency; stubbing segmentation until the experiment phase.

Advantages

Testable now; real models ready; graceful degradation; no pipeline rewrites.

Disadvantages

Reference-model metrics are placeholders; real training is exercised only once
torch and the Sample→DataLoader bridge land.

Impact

`adaptivehb.segmentation`, the pipeline's trainable factory, and (by pattern) the
prediction subsystem in Phase 6.

Status

Accepted (provisional — override on request).

---

# Decision 022

Decision ID

022

Date

2026-07-06

Category

Research

Decision

Adaptive agents operate on structured intermediate features (image quality, ROI
metrics, per-tissue predictions and confidences, metadata) rather than raw
images, use deterministic policies in this phase, and communicate only through
the Workflow Controller via a shared context. The Controller is deterministic and
makes no clinical decisions.

Reason

Matches EXPERIMENT_SPEC Ch.12 (agents learn from intermediate predictions, not
raw images), which drastically reduces computational cost and keeps the adaptive
framework torch-free and fully testable now. Deterministic policies establish a
correct, interpretable baseline; the common interface lets learned policies
(decision trees, gradient boosting, lightweight nets, RL) replace them later
without changing the controller or pipeline (Decision 005). Routing outputs via
the controller keeps agents decoupled (AGENT_SPECIFICATION Ch.7).

Alternatives Considered

Agents consuming raw images directly; agents calling one another; an LLM-based
controller (explicitly excluded by the spec).

Advantages

Lightweight; interpretable; testable without the ML stack; clean seam for learned
policies; low coupling.

Disadvantages

Deterministic policies are heuristic until replaced by learned ones during the
experiment phase.

Impact

`adaptivehb.agents` and the pipeline's adaptive workflow.

Status

Accepted (provisional — override on request).

---

# Decision 023

Decision ID

023

Date

2026-07-06

Category

Evaluation

Decision

Evaluation metrics are implemented with the Python standard library only
(numpy/scipy optional), and the EVALUATION mode computes hemoglobin metrics from
per-patient predictions versus ground truth on the test split (exported as
CSV/JSON), rather than surfacing per-model training metrics. Degenerate inputs
(e.g. constant predictions) are handled gracefully.

Reason

Keeps the evaluation layer importable and fully testable without the ML stack
(consistent with the framework-first, dependency-light approach), while giving a
genuine, publication-shaped metric pipeline now. Baseline-vs-adaptive comparison
(EXPERIMENT_SPEC Ch.14) is implemented via ``EvaluationManager.compare`` and will
differentiate once real models yield varying predictions.

Alternatives Considered

Requiring numpy/scipy/sklearn for metrics; deferring real metric computation to
the experiment phase.

Advantages

Lightweight; testable now; real reports; clean baseline-vs-adaptive seam.

Disadvantages

With constant reference predictions, correlation metrics are 0 by construction
until real models are trained; figure generation (matplotlib) remains an
experiment-phase add-on.

Impact

`adaptivehb.evaluation` and the pipeline's EVALUATION mode.

Status

Accepted (provisional — override on request).

---

# Decision 024

Decision ID

024

Date

2026-07-06

Category

Deployment

Decision

Deployment is load-only and torch/web-optional. The `HbInferenceService` loads
registry-confirmed models and serves clinical reports via the adaptive workflow
without any retraining. Deployment targets are pluggable: a dependency-free
desktop target plus FastAPI/Gradio/Streamlit targets whose optional dependencies
are imported lazily and which raise a clear error when the extra is absent
(mirroring the torch-optional model backends, Decision 021).

Reason

Keeps the deployment layer importable and testable without web frameworks or
torch, while providing a real serving path. Load-only serving satisfies the
Charter/PIPELINE_SPEC requirement that deployment never retrains and loads only
registry-approved models.

Alternatives Considered

Hard dependencies on FastAPI/torch; a single fixed transport; retraining at
deploy time (explicitly forbidden).

Advantages

Testable now; multiple transports; no retraining; clean optional-dependency seam.

Disadvantages

Actual web-server launch and PDF report generation require optional extras and
are exercised in the deployment environment, not the framework test suite.

Impact

`adaptivehb.deployment` and the pipeline's DEPLOYMENT mode.

Status

Accepted (provisional — override on request).

---

# Decision 025

Decision ID

025

Date

2026-07-06

Category

Dataset

Decision

The training-data bridge (`adaptivehb.dataloading`) separates dependency-free
batching and label extraction from ML/vision-dependent decoding and tensor
conversion. Batching (`Batch`/`iter_batches`) is pure standard library and fully
tested; image decoding (OpenCV/Pillow), transforms (Albumentations), and the
torch `DataLoader` adapter are guarded optional dependencies that raise clear
errors when absent.

Reason

Completes the deferral from Decision 019 (decoding/preprocessing/augmentation
belong to the training data pipeline) while keeping the bridge importable and
mostly testable without the ML stack, consistent with the torch-optional pattern
used throughout (Decisions 021/024). Real decoding/DataLoaders activate
automatically when the extras are installed.

Alternatives Considered

Hard dependencies on torch/opencv/albumentations; folding batching into the torch
DataLoader (untestable without torch).

Advantages

Batching/label logic tested now; clean seam for real training; graceful
degradation; reuses the standardized Sample schema.

Disadvantages

The torch DataLoader path and the real train loops that consume it are exercised
only where the ML stack is installed.

Impact

`adaptivehb.dataloading`; the forthcoming real train_epoch/validate loops.

Status

Accepted (provisional — override on request).

---

# Decision 026

Decision ID

026

Date

2026-07-06

Category

Implementation

Decision

The real training loops for the torch segmentation and prediction backbones live
in one shared module (`adaptivehb.training_ops`), which separates torch-free,
tested helpers (metric accumulation, optimizer/loss name validation) from guarded
torch builders and epoch loops. The pipeline's TRAINING mode builds and attaches
per-plan dataloaders only to models that accept data (torch models); reference
models train deterministically and are unaffected.

Reason

Keeps training logic in one place (segmentation and prediction share the loop
structure), maximizes the testable surface without torch, and preserves the
torch-optional guarantee: the framework runs and its tests pass without the ML
stack, while real training activates automatically when torch is installed.

Alternatives Considered

Duplicating the loop in each model; requiring torch for the whole package;
attaching dataloaders unconditionally (would break the torch-free path).

Advantages

Single source of training logic; tested helpers; graceful degradation; no changes
to TrainingManager or the pipeline facade.

Disadvantages

The torch epoch loops and builders are executed only where torch is installed;
their integration tests skip in a framework-only environment.

Impact

`adaptivehb.training_ops`, the torch backbones, and the pipeline's TRAINING mode.

Status

Accepted (provisional — override on request).

---

# Decision 027

Decision ID

027

Date

2026-07-06

Category

Evaluation

Decision

Publication figures and tables are generated by a dedicated `adaptivehb.reporting`
subsystem whose plotting (matplotlib) and Excel (openpyxl) dependencies are
optional and guarded. The pipeline's EVALUATION mode emits figures + a metrics
table when the backends are available and skips them gracefully otherwise; CSV
export always works via the standard library.

Reason

Keeps the framework importable and testable without heavyweight plotting/Excel
libraries (consistent with the torch-optional pattern), while producing real
publication assets when the extras are present. Separating reporting from the
metric computation (EvaluationManager) preserves single responsibility.

Alternatives Considered

Hard dependency on matplotlib; embedding plotting inside EvaluationManager;
deferring all figure generation to external notebooks.

Advantages

Real, tested figure/table output; graceful degradation; clean separation from
metric computation; reusable by experiment scripts.

Disadvantages

Figure aesthetics/paper-specific styling will be refined during the experiment
phase against real results.

Impact

`adaptivehb.reporting` and the pipeline's EVALUATION mode.

Status

Accepted (provisional — override on request).

---

# Decision 028

Decision ID

028

Date

2026-07-06

Category

Experiments

Decision

A dedicated `ExperimentRunner` (exposed via `HbPipeline.experiment()`) composes
the managers into one reproducible experiment: create an immutable experiment
directory, train, then evaluate a static baseline against the adaptive
(agent-fused) pipeline on the same test split and compare them, archiving metrics,
the comparison, predictions, tables, and figures. The runner composes existing
primitives (ExperimentManager, EvaluationManager, AgentManager, reporting) rather
than adding a new pipeline mode.

Reason

The baseline-vs-adaptive comparison is the project's core scientific claim
(Decision 008) and must be run under identical conditions with archived,
reproducible outputs (EXPERIMENT_SPEC Ch.13-14, Ch.21). An orchestrator keeps this
end-to-end flow in one place and reuses the tested building blocks, and it runs
without torch (reference models) so the machinery itself is verifiable now.

Alternatives Considered

Adding an EXPERIMENT pipeline mode; running baseline/adaptive comparison ad hoc in
notebooks; duplicating evaluation/reporting logic.

Advantages

Single reproducible entry point; reuses managers; archived outputs; testable now.

Disadvantages

Until registry-backed checkpoint loading lands, predictions come from freshly
built (reference) models, so baseline == adaptive on reference models.

Impact

`adaptivehb.experiment`, `HbPipeline`, and the experiment output directory.

Status

Accepted (provisional — override on request).

---

# Decision 029

Decision ID

029

Date

2026-07-06

Category

Model Registry / Experiments

Decision

Trained weights are loaded into freshly built models through a single helper,
`adaptivehb.model_loading.load_weights_into(model, checkpoints, name)`, exposed on
each domain manager as `load_trained(name, checkpoints, ...)`. The checkpoint is
located by *name* via the CheckpointManager (preferring the `best` tag, then
`latest`); no checkpoint path is ever hardcoded. Evaluation and the experiment's
baseline-vs-adaptive comparison now consume `load_trained` models instead of
freshly built reference models. Loading is best-effort: when no checkpoint exists
the model degrades to an untrained instance (with a warning) rather than raising.

Reason

Evaluate/inference/experiment previously scored freshly built (reference) models,
so metrics did not reflect training and baseline == adaptive (Decision 028's noted
limitation). Registry/checkpoint-backed loading closes that gap so the pipeline
scores the *trained* models, which is a precondition for a genuine experiment
(MODEL_REGISTRY_SPEC Ch.15). A single by-name helper keeps loading uniform across
prediction and segmentation and avoids hardcoded paths.

Alternatives Considered

Hardcoding checkpoint paths per model; loading inside each model's constructor;
having the registry return live model objects instead of checkpoint payloads.

Advantages

One reuse point for all model types; no hardcoded paths; graceful degradation when
untrained; evaluation/experiment now reflect trained weights.

Disadvantages

Reference models still return a constant prediction regardless of loaded state (an
honest limitation of the torch-free reference backend); meaningful baseline vs
adaptive divergence requires the torch backbones + a real dataset.

Impact

`adaptivehb.model_loading`, `PredictionManager.load_trained`,
`SegmentationManager.load_trained`, `managers.pipeline_modes` (evaluation), and
`adaptivehb.experiment` (baseline-vs-adaptive).

Status

Accepted (provisional — override on request).

---

# Decision 030

Decision ID

030

Category

Interface / Reproducibility

Date

2026-07-06

Decision

Expose the framework through a thin, config-driven command-line interface
(`adaptivehb.cli`), registered as the `adaptivehb` console script and runnable as
`python -m adaptivehb`. Each pipeline mode is a subcommand (build/train/resume/
evaluate/predict/deploy/experiment); global options `--config-dir`, `--base-dir`,
`--dataset-root`, plus `--epochs` (training-like modes) and `--name` (experiment)
supply everything from arguments/config with no hardcoded values. The CLI holds no
framework logic: it parses arguments, constructs `HbPipeline.from_config_dir`,
dispatches to the matching public method, and prints the result summary as JSON.

Reason

A real experiment must be launchable reproducibly from one command in the
torch/GPU environment (the current priority) and by external users of the
open-source framework. A subcommand-per-mode CLI over the existing facade gives a
single reproducible entry point without duplicating logic, keeps heavy ML imports
lazy (the CLI imports/tests without torch), and follows the framework's
config-driven, no-hardcoding rules.

Alternatives Considered

Shell scripts per mode (duplicate logic, not testable); a bespoke config runner;
requiring users to write Python against the facade (higher barrier, less
reproducible); Click/Typer dependency (avoided to keep runtime deps minimal —
stdlib argparse suffices).

Advantages

One reproducible entry point; reuses the facade; no new runtime dependency; lazy
ML imports; fully unit-tested; command<->mode mapping is data, so new modes are
one line to expose.

Disadvantages

Argparse subcommand wiring is slightly more verbose than Click; output is a JSON
summary rather than rich human formatting (acceptable for a research CLI).

Impact

`adaptivehb.cli`, `adaptivehb.__main__`, and the `[project.scripts]` entry in
`pyproject.toml`.

Status

Accepted (provisional — override on request).

---

# Decision 031

Decision ID

031

Category

Experiments / Evaluation

Date

2026-07-06

Decision

The baseline-vs-adaptive comparison reports paired statistical significance, not
just a point difference in MAE. A new stdlib-only module
`adaptivehb.evaluation.significance` computes, on the per-sample absolute-error
differences (`|baseline - true| - |adaptive - true|`): a two-sided paired Student
t-test (t distribution tail via the regularized incomplete beta function), a
Wilcoxon signed-rank test (normal approximation with tie/zero handling), a
percentile bootstrap confidence interval for the mean difference (seeded for
reproducibility), and Cohen's d for paired samples. `EvaluationManager.compare`
gains optional paired arrays and attaches a `significance` block when they are
supplied; `ExperimentRunner` passes them so the archived `comparison.json` carries
the full inferential result. Parameters (enabled, bootstrap iterations, confidence
level) come from the `significance` subsection of `evaluation.yaml`.

Reason

A high-impact biomedical/ML journal requires evidence that an improvement is
statistically real, with an effect size and a confidence interval, computed on the
same patients (paired) — a point MAE delta is not publishable on its own. Baseline
and adaptive predict on identical patients, so paired tests are the correct family.
Implementing them with the standard library keeps the evaluation subsystem
torch-free and reproducible, and wiring them through `compare()` (optional,
backward-compatible) means every experiment emits publication-ready statistics with
no change to callers that only want the point comparison.

Alternatives Considered

Depending on scipy (adds a heavy dependency to a deliberately light subsystem);
reporting only the point MAE difference (not publishable); an unpaired/independent
t-test (wrong — the samples are paired); permutation testing only (kept the
bootstrap CI + parametric/non-parametric pair as the more conventional reviewer
expectation).

Advantages

Publication-ready comparison (p-values, CI, effect size); paired tests match the
design; stdlib-only and reproducible; config-driven; backward compatible; graceful
on tiny/degenerate splits.

Disadvantages

Hand-rolled numerics (incomplete beta, normal approximation) require their own
tests rather than leaning on scipy; the Wilcoxon uses a normal approximation rather
than the exact distribution for small n.

Impact

`adaptivehb.evaluation.significance`, `EvaluationManager.compare`,
`EvaluationConfig`, `configs/evaluation.yaml`, and `ExperimentRunner` (archived
`comparison.json`).

Status

Accepted (provisional — override on request).

---

# Decision 032

Decision ID

032

Category

Reproducibility

Date

2026-07-06

Decision

Every archived experiment carries a self-describing reproducibility manifest,
written to ``<experiment>/configuration/provenance.json`` and mirrored in the
experiment summary and the returned ``ExperimentResult``. A new stdlib-only module
``adaptivehb.provenance`` assembles it: the framework version and RNG seed; the
software environment (Python, platform, and the installed versions of the relevant
scientific packages, torch/numpy/… when present); the current git revision read
directly from ``.git`` without invoking the ``git`` executable; and content
fingerprints (SHA-256) of the full configuration and of the dataset roster
(patients, samples, per-tissue/per-split counts, and a hash of the
``(patient_id, tissue, filename, hb)`` tuples). All fields degrade gracefully to
``None``/absent rather than raising.

Reason

Reproducibility is a core project pillar and an explicit requirement of the target
journals (JBI, AIM, CMPB); a point-in-time record of environment, code revision,
configuration, and dataset is needed to reconstruct or audit a published result.
The existing ``ExperimentManager._environment`` captured only Python/platform, and
nothing fingerprinted the config or dataset. Reading git from ``.git`` (no
subprocess) keeps this working in restricted/sandboxed environments, and the
stdlib-only implementation preserves the torch-free guarantee of the core.

Alternatives Considered

Shelling out to ``git rev-parse`` (fails in sandboxes without a git binary and
adds a subprocess dependency); depending on ``pip freeze``/an external tool for the
environment (heavier, less portable); hashing raw image bytes for the dataset
fingerprint (expensive and belongs to the data pipeline — the structural roster
hash detects roster/label changes without reading pixels); storing only a config
path rather than a content hash (would not detect edits).

Advantages

Publication-grade provenance per experiment; detects config/dataset drift across
runs; works without git installed or torch present; additive and backward
compatible (``ExperimentResult`` gains a ``provenance`` field).

Disadvantages

The dataset fingerprint intentionally does not hash image content, so pixel-level
changes that keep the same filenames and labels are not detected; git parsing
covers the common HEAD/loose-ref/packed-refs cases but not every exotic git layout.

Impact

New ``adaptivehb.provenance``; ``ExperimentRunner`` (archives ``provenance.json``,
extends the summary and ``ExperimentResult``). ``ExperimentManager._environment``
is left intact and may later be refactored to delegate to
``provenance.collect_environment`` (noted, not done, to avoid touching a completed
module).

Status

Accepted (provisional — override on request).

---

# Decision 033

Decision ID

033

Category

Reproducibility / Usability

Date

2026-07-07

Decision

Two runnable Jupyter notebooks are the supported entry point for executing
experiments interactively, living in ``notebooks/``. ``smoke_synthetic.ipynb``
runs the whole pipeline torch-free on a generated synthetic dataset (reference
models) to verify a checkout anywhere; ``train_pipeline.ipynb`` (the notebook
named in PROJECT_MANIFEST) runs the real, PyTorch-backed experiment on a dataset.
Both are thin: they only call the public ``HbPipeline`` facade and display the
already-archived outputs (comparison + significance, provenance, figures). They
carry no framework logic, hardcode nothing (paths/dataset/epochs are variables),
and self-locate the repo root so they run with or without ``pip install``. The
notebooks are generated from a script with ``nbformat`` (never hand-edited JSON),
and the torch-free one is executed in CI-style verification with ``nbclient``.

Reason

The user runs experiments in a separate torch/GPU environment and needs a
ready-to-run artifact; PROJECT_MANIFEST already reserves
``notebooks/train_pipeline.ipynb``. A torch-free smoke notebook lets any machine
validate the framework before provisioning a GPU, and executing it headlessly
proves the delivered notebook actually runs (not just that its JSON is valid).
Keeping the notebooks as thin facade callers avoids duplicating logic that already
lives in the framework and keeps them maintainable.

Alternatives Considered

A single notebook (loses the run-anywhere smoke check); a plain Python script
instead of a notebook (the user specifically needs an ``.ipynb``); hand-writing
the notebook JSON (error-prone — chose programmatic generation); embedding
analysis logic in the notebook (would duplicate the framework and rot).

Advantages

Ready-to-run in Colab/local Jupyter; the smoke notebook is executed as part of
verification; thin facade usage means no logic duplication; config-driven and
self-locating; valid-by-construction JSON.

Disadvantages

``train_pipeline.ipynb`` cannot be executed in the dev sandbox (no torch), so it
is verified by JSON validation plus compiling every code cell rather than a full
run; the generator script is not retained in the repo (the notebooks are the
deliverable).

Impact

New ``notebooks/smoke_synthetic.ipynb``, ``notebooks/train_pipeline.ipynb``, and
``notebooks/README.md``. No framework source changed.

Status

Accepted (provisional — override on request).

---

# Decision 034

Decision ID

034

Category

Reporting / Publication

Date

2026-07-07

Decision

Each experiment emits a consolidated, publication-ready report — a Markdown file
(`experiment_report.md`) and a self-contained HTML file (`experiment_report.html`)
in `<experiment>/reports/`. A new module `adaptivehb.reporting.experiment_report`
renders both from the same structured `ExperimentReportData` (metrics, the
baseline-vs-adaptive comparison with paired significance, the provenance manifest,
dataset info, and figure paths) using pure standard-library string building. It
adds a plain-language results sentence (direction and magnitude of the MAE change,
a significance verdict, and a Cohen's-d effect-size interpretation), a metrics
table, a significance table, a reproducibility block, and inline/relative figure
links. `ExperimentRunner` generates it after the figures and records the paths in
the summary and in a new `ExperimentResult.report_paths` field.

Reason

The archive already held the raw evidence as JSON/CSV/PNG, but a reader (and the
authors drafting Papers 1-4) needs a single narrative artifact that ties the
numbers, statistics, provenance, and figures together. Generating it with the
standard library keeps reporting torch-free and dependency-light (no matplotlib,
pandas, or Markdown/HTML library required to produce text) and consistent with the
guarded-optional-dependency approach used elsewhere (Decision 027). Rendering both
Markdown (renders on GitHub / in papers) and HTML (figures embedded for quick
viewing) covers both review and drafting workflows.

Alternatives Considered

Leaving the raw JSON/CSV as the only output (no readable synthesis); depending on
Jinja2/Markdown/pandoc (heavier, and a real dependency for what is simple string
assembly); a PDF report (defer — Markdown+HTML are enough for drafting and both
convert to PDF trivially with external tools).

Advantages

One readable, paper-ready artifact per experiment; plain-language significance and
effect-size interpretation; stdlib-only and torch-free; robust to missing pieces;
additive and backward compatible (`ExperimentResult` gains `report_paths`).

Disadvantages

The HTML is intentionally minimal (no interactive elements); the Markdown->PDF step
is left to external tooling; the report reflects a single run (aggregation across
runs/folds would be a later cross-validation reporting concern).

Impact

New `adaptivehb.reporting.experiment_report`; `ExperimentRunner` (generates the
report, extends the summary and `ExperimentResult`).

Status

Accepted (provisional — override on request).

---

# Decision 035

Decision ID

035

Category

Experiments / Evaluation

Date

2026-07-07

Decision

The framework provides patient-level k-fold cross-validation via a
`CrossValidationRunner` (exposed as `HbPipeline.cross_validate(...)` and the
`adaptivehb crossval` CLI subcommand). A new `k_fold_split` (in
`dataset.splitting`) partitions patients into k balanced, deterministic folds with
no patient appearing in more than one test fold. `DatasetManager` gains an
`apply_split()`/`clear_pinned_split()` mechanism (and `split()` now honors a pinned
split) so an externally-controlled fold assignment survives the internal
`split()` calls made during training. Each fold is run as a fully isolated
`ExperimentRunner` experiment in its own `base_dir` (so checkpoints/registry/
experiments never bleed across folds), and per-fold metrics and baseline-vs-adaptive
comparisons are aggregated (mean/std/min/max) and archived as `cv_summary.json`,
`cv_metrics.csv`, and `cv_report.md`.

Reason

A single held-out split is statistically fragile on the modest datasets typical of
non-invasive hemoglobin studies; high-impact venues expect cross-validated metrics
with dispersion (mean ± std) and per-fold significance. Building the harness by
reusing `ExperimentRunner` per fold maximizes reuse (each fold is a complete,
already-tested experiment with its own report and provenance) and the isolated
per-fold directories guarantee no train/test leakage through shared checkpoints.
Pinning the split (rather than rewriting `run_training`) is the least invasive way
to let CV control the fold assignment while leaving the single-run path untouched.

Alternatives Considered

Rewriting `run_training`/`_baseline_vs_adaptive` to accept an explicit split
(more invasive, touches completed hot paths); sharing one `base_dir` across folds
(risked checkpoint/registry bleed where a fold's "best" could be an earlier fold's
weights); leave-one-out CV (a special case of k=N, still supported by choosing k);
nested CV for hyperparameters (deferred — out of scope for the current milestone).

Advantages

Reviewer-expected cross-validated metrics with dispersion; full reuse of the
experiment path (report + provenance + significance per fold); no leakage via
isolation; least-invasive split pinning; available from both the facade and the CLI.

Disadvantages

Runs training k times (k× cost); the aggregate pools per-fold point metrics rather
than performing a combined mixed-effects analysis; on the torch-free reference
models folds still show baseline == adaptive (real divergence needs torch + data).

Impact

New `adaptivehb.crossval`; `dataset.splitting.k_fold_split`;
`DatasetManager.apply_split`/`clear_pinned_split`/`split()`; `HbPipeline.cross_validate`;
`adaptivehb.cli` (`crossval` subcommand).

Status

Accepted (provisional — override on request).

---

# Decision 036

Decision ID

036

Category

Publication

Date

2026-07-07

Decision

Manuscript drafting begins in parallel with (not after) the real training run,
starting with Paper 3 — the primary thesis paper, "An Adaptive Multi-Agent Decision
Framework for Non-Invasive Hemoglobin Estimation." Drafts live in a top-level
`paper/` directory as Markdown (`paper3_adaptive_framework.md`) with a starter
BibTeX bibliography (`references.bib`), a results-to-file mapping (`paper/README.md`),
and a pandoc-rendered Word copy (`.docx`). Sections grounded in the implemented
framework — abstract, introduction, related work, methods (framework, segmentation,
prediction, the seven agents across three layers, the baseline-vs-adaptive protocol,
statistics, and reproducibility), experimental setup, discussion structure,
limitations, and conclusion — are written in full now. Numeric results are explicit
placeholders (`[[RESULT: …]]`) keyed to the framework's archived outputs and filled
once the models are trained on the study dataset.

Reason

Everything the paper needs except the numbers already exists and is stable; writing
the durable prose now removes it from the critical path so that, once the run
completes, publication is a matter of pasting archived values rather than composing
from scratch. Keeping drafts in-repo as Markdown preserves reproducibility and
version control and lets the automated report (`experiment_report.md`) feed the
Results section directly; the Word copy serves review/editing.

Alternatives Considered

Waiting for results before writing (leaves the whole manuscript on the critical
path); writing in an external word processor (loses version control and the direct
link to archived outputs); starting with Paper 1/2 (their results depend on
segmentation/prediction benchmarks not yet run, whereas Paper 3's methods are fully
realized by the current framework).

Advantages

Removes writing from the critical path; results become copy-paste from archived
files; reproducible, version-controlled drafts; the per-run report populates the
Results section.

Disadvantages

Results/discussion remain templated until the run completes; domain citations
(`[[CITE: …]]`) still need to be gathered.

Impact

New `paper/` directory (`paper3_adaptive_framework.md`, `references.bib`,
`README.md`, `.docx`); `docs/PAPER_PLAN.md` (Paper 3 marked draft-in-progress). No
framework source changed.

Status

Accepted (provisional — override on request).

---

# Decision 037

Decision ID

037

Category

Publication

Date

2026-07-07

Decision

Draft Paper 2 — "Deep Learning Models for Multi-Tissue Non-Invasive Hemoglobin
Estimation" — in parallel with the pending real run, continuing the parallel-writing
strategy of Decision 036. Paper 2 is a standardized benchmark of five image
backbones (EfficientNet, ResNet, DenseNet, ViT, ConvNeXt) for per-tissue Hb
regression across eye/palm/tongue/nail. The manuscript's problem formulation, model
descriptions, training protocol (grounded verbatim in `prediction.yaml`), evaluation
metrics, statistics, and reproducibility are written in full; numeric results are
`[[RESULT: …]]` placeholders keyed to archived evaluation outputs (mapped in
`paper/README.md`). Delivered as Markdown + a pandoc `.docx`, reusing the shared
`references.bib`.

Reason

Paper 2's methods are fully realized by the implemented prediction/evaluation/
reporting subsystems, so — like Paper 3 — its durable prose can be written now and
removed from the critical path; only the numbers await training. Grounding the
training protocol directly in the configuration keeps the paper accurate and
reproducible.

Alternatives Considered

Drafting Paper 1 (segmentation) first — deferred because the evaluation/reporting
machinery is Hb-regression-oriented and Paper 2 maps onto it directly, whereas Paper
1 needs segmentation-specific metrics (IoU/Dice) not yet surfaced; waiting for
results before writing (keeps the manuscript on the critical path).

Advantages

Second manuscript writing-complete before results land; accurate protocol from
config; reuses the paper/ conventions and shared bibliography; results become
copy-paste from archived files.

Disadvantages

Results/discussion remain templated until training; the full backbone×tissue matrix
requires training each backbone on each tissue.

Impact

New `paper/paper2_prediction_models.md` + `.docx`; `paper/README.md` (Paper 2
results map); `docs/PAPER_PLAN.md` (Paper 2 -> draft in progress). No framework
source changed.

Status

Accepted (provisional — override on request).

---

# Decision 038

Decision ID

038

Category

Evaluation / Segmentation

Date

2026-07-07

Decision

Add a standard-library segmentation-metrics module
(`adaptivehb.segmentation.metrics`) providing the region-overlap metrics needed to
evaluate tissue segmentation: IoU/Jaccard, Dice/F1, pixel accuracy, mean per-class
accuracy, and frequency-weighted IoU, computed from a class confusion matrix. It
offers one-shot functions, a `SegmentationMetrics` accumulator for dataset-level
scoring, and a `binarize` helper for probability maps; it accepts nested Python
sequences or numpy arrays, handles an `ignore_index`, and excludes classes absent
from both prediction and truth from the mean (common convention). It is numpy- and
torch-free (numpy used only if the input already is an array).

Reason

The evaluation subsystem scored only Hb regression; segmentation quality had no
metrics, which blocks the segmentation paper (Paper 1) and any real segmentation
evaluation. Implementing the metrics with the standard library keeps the
segmentation domain consistent with the torch-free, numpy-optional design of
`evaluation.metrics`, and the confusion-matrix basis handles binary and multi-class
masks uniformly and supports incremental (batch/epoch) accumulation.

Alternatives Considered

Depending on torchmetrics or scikit-learn for IoU/Dice (adds heavy dependencies to a
deliberately light subsystem and would not run torch-free); computing metrics only
inside a torch inference loop (not unit-testable without torch); folding
segmentation metrics into `evaluation.metrics` (kept them in the segmentation
package where the domain lives).

Advantages

Unblocks segmentation evaluation and Paper 1; stdlib-only and torch-free;
binary/multi-class uniform; incremental accumulator; unit-tested against
hand-computed values.

Disadvantages

No boundary/contour metrics (e.g., boundary F1, Hausdorff) yet; not wired into a
segmentation-evaluation pipeline mode (that needs torch inference + masks), which
remains future work.

Impact

New `adaptivehb.segmentation.metrics`; exported from `adaptivehb.segmentation`.

Status

Accepted (provisional — override on request).

---

# Decision 039

Decision ID

039

Category

Dataset / Pipeline

Date

2026-07-07

Decision

Segmentation may train on a dataset that is separate from the Hb-estimation
dataset. An optional `dataset.segmentation_source` block (root + images_dir/
masks_dir/metadata_file) selects a distinct segmentation dataset; when unset, the
main dataset is reused (backwards compatible). The PipelineManager builds a second
`DatasetManager` (`pm.segmentation_dataset`) for that source; training/validation
split both datasets, segmentation dataloaders draw from the segmentation dataset,
and prediction + evaluation draw from the main (Hb) dataset. The segmentation
source is metadata-optional: a missing `patients.csv` is tolerated (it needs only
images + masks), and patient-level splitting falls back to patient IDs derived from
the image filenames.

Reason

In practice the mask-annotated segmentation set is a different (often smaller,
possibly different-subject) collection than the labelled Hb set; forcing one shared
root would require co-locating them and inventing Hb labels for mask-only images.
Separating the sources matches real datasets, keeps segmentation and prediction
independently trainable (a core project principle), and needs no change to callers
that use a single dataset.

Alternatives Considered

One shared root with masks for a subset (does not fit different-subject
segmentation data; pollutes the Hb split with unlabelled images); a fully separate
segmentation pipeline/config file (heavier; the two share almost all machinery); a
CLI/facade argument instead of config (kept it config-driven and reuses the second
DatasetManager unchanged).

Advantages

Matches real data; segmentation and Hb trained on their own sources; metadata
optional for the mask-only set; backwards compatible; reuses DatasetManager via a
second instance (no duplication).

Disadvantages

Two splits/validations to reason about; the torch dataloader routing for the
segmentation source is validated by logic + reference-model tests here (the real
torch path is exercised on a GPU).

Impact

`DatasetConfig` (segmentation_source), `DatasetManager` (dir overrides +
metadata-optional + split-from-index), `PipelineManager`
(`segmentation_dataset`), `managers.pipeline_modes` (split/attach), and the
notebook control panel (`SEG_DATASET_ROOT`).

Status

Accepted (provisional — override on request).

---

# Future Decisions

New decisions should be appended sequentially.

Decision IDs should never change.

Rejected decisions should remain documented.

Deprecated decisions should never be deleted.

---

# Decision Categories

Architecture

Software

Research

Implementation

Dataset

Training

Evaluation

Deployment

Performance

Optimization

Documentation

---

# Update Rules

Add a new entry only when

Architecture changes

Research methodology changes

Pipeline changes

Repository structure changes

Training strategy changes

Agent design changes

Do not record trivial coding decisions.

---

# End of DECISION_LOG.md