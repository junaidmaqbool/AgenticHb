# IMPLEMENTATION_READINESS_REPORT.md

Version: 1.0
Status: Pre-Implementation Review
Date: 2026-07-06
Author: AI Research Partner (review pass)
Scope: Read-only inspection of the repository and documentation prior to any code being written. No production code has been written. Implementation is on hold pending approval.

---

## 1. Repository Summary

The repository currently contains **documentation only**. There is no source code, no notebooks, no configuration files, no tests, and no Python packaging. Git is **not yet initialized** even though `CURRENT_TASK.yaml` declares `branch: main`.

Present files (19 total):

- Root control files: `START_HERE.md`, `PROJECT_MANIFEST.yaml`, `CURRENT_TASK.yaml`
- `docs/` (16 files): `PROJECT_CHARTER.md`, `PROJECT_DESIGN_SPECIFICATION.md`, `IMPLEMENTATION_ROADMAP.md`, `PIPELINE_SPEC.md`, `DATASET_SPEC.md`, `AGENT_SPECIFICATION.md`, `MODEL_REGISTRY_SPEC.md`, `EXPERIMENT_SPEC.md`, `PROJECT_INDEX.md`, `CLAUDE_DEVELOPMENT_GUIDE.md`, `DEVELOPMENT_RULES.md`, `PROJECT_STATE.md`, `DECISION_LOG.md`, `CHANGELOG.md`, `EXPERIMENT_LOG.md`, `PAPER_PLAN.md`

The documentation is unusually complete and internally consistent in its *philosophy* (modularity, configuration-driven design, registry-mediated model loading, resumability, reproducibility, strict separation of framework development from experimentation). The governing constraints are clear and I will implement to them faithfully.

Declared state: Phase 0 (Documentation) complete; Phase 1 (Repository Construction) is the current milestone; overall completion ~5%; no training permitted until repository, infrastructure, dataset module, and pipeline are complete.

---

## 2. Architecture Summary

The intended system is a layered, manager-mediated pipeline with a single public entry point.

**Entry point.** `HbPipeline` exposes a small public API (`build`, `train`, `resume`, `evaluate`, `test`, `predict`, `deploy`, `shutdown`). A notebook only supplies paths, hardware, experiment name, and mode, then calls `pipeline.run()`. No ML logic lives in the notebook.

**Managers (single-responsibility).** `PipelineManager` orchestrates; specialised managers handle Dataset, Registry, State, Training, Segmentation, Prediction, Agents, Evaluation, Deployment. Managers never call each other directly — all coordination flows through the pipeline. Checkpoint and Experiment management also appear as responsibilities in the specs.

**Adaptive decision framework (the scientific core).** Three layers coordinated by a deterministic Workflow Controller:
- Perception: Quality Assessment, ROI Verification
- Decision: Segmentation Selection, Tissue Selection, Prediction Routing
- Clinical Output: Adaptive Fusion, Confidence Estimation

Agents share a common interface (`initialize/train/predict/evaluate/save/load/reset/shutdown`), communicate only via the pipeline, form an acyclic dependency chain, and are trained not on raw images but on **intermediate prediction records** (predicted Hb, ground truth, quality, ROI, confidence, metadata) produced after segmentation and prediction models are frozen. No LLMs; lightweight models or deterministic policies only.

**Cross-cutting infrastructure.** Model Registry (single source of truth for all trained models; category-separated; versioned; auto-registering; load-by-query never by path), Checkpoint Manager (latest + best + optimizer/scheduler/epoch/seed/history), Pipeline State (`pipeline_state.json` for crash recovery), Experiment Manager (immutable per-experiment archive), and a configuration system of nine YAML files (`project`, `dataset`, `segmentation`, `prediction`, `agents`, `evaluation`, `deployment`, `registry`, `logging`).

**Execution modes.** Build (dummy, no data), Training, Resume, Evaluation, Inference, Deployment. Every mode shares one lifecycle: load config → validate → init logger/registry/state → init managers → load+validate dataset → select mode → execute → save → update registry+state → shutdown. Operations are modelled as dependency-checked **Jobs** with retry/recovery.

**Dataset contract.** Framework is dataset-agnostic. Datasets conform to a documented layout (`metadata/`, `images/<tissue>/`, `masks/<tissue>/`, `splits/`, `statistics/`). Patient-level 80/10/10 split (configurable; CV/external supported). Mandatory metadata fields validated before training; optional fields ignored if absent. A validation report gates training.

---

## 3. Existing Reusable Code

**None.** There is no prior implementation, notebook, utility, dataset loader, evaluation script, or configuration file anywhere in the repository (confirmed via full-tree scan and file-type search). There is nothing to reuse, refactor, or archive at the code level.

The reusable *assets* that do exist are the specifications themselves, which act as the contract for implementation. The only "prior work" decision, therefore, is: **build fresh, to spec.** When we reach model implementations (segmentation backbones, prediction backbones), the project instructions permit cloning mature reference implementations from GitHub rather than reinventing them — that is the appropriate reuse strategy at Phase 5/6, not now.

---

## 4. Existing Problems

These are documentation-level inconsistencies that must be resolved before scaffolding, because they change folder and package layout.

1. **Repository layout conflict (blocking for Phase 1).** `PROJECT_MANIFEST.yaml` declares a `src/` package. `PROJECT_DESIGN_SPECIFICATION.md` Ch.4 instead lists a *flat* top-level layout with ~25 sibling directories (`segmentation/`, `prediction/`, `agents/`, `routing/`, `fusion/`, `confidence/`, `evaluation/`, `deployment/`, `api/`, `utilities/`, `project/`, plus data/output dirs). The roadmap Phase 1 lists yet another set. These cannot all be true. A single canonical tree must be fixed first.

2. **Manager set is inconsistent across four documents.** The manifest, `PROJECT_INDEX.md` §5, `PDS` Ch.7, and `PIPELINE_SPEC.md` Ch.5 each list a different manager roster. Points of divergence: whether `FusionManager` is a separate manager or folded into `AgentManager`; whether `StateManager`, `CheckpointManager`, `TrainingManager`, and `ExperimentManager` are first-class managers (roadmap Phase 2 requires them; the manifest omits them); and whether "Decision Module Manager" and "AgentManager" are the same thing (they are, but naming should be unified).

3. **`HbPipeline` vs `PipelineManager` relationship is implied, not stated.** The specs use both; the intended design is a thin public facade (`HbPipeline`) over an internal orchestrator (`PipelineManager`). This should be written down so the boundary is not blurred during implementation.

4. **Tissue taxonomy mismatch.** The manifest enumerates seven laterality-specific tissues (`left_eye`, `right_eye`, `left_palm`, `right_palm`, `tongue`, `left_nail`, `right_nail`). `DATASET_SPEC.md` and the prediction config group by four tissue *classes* (`eye`, `palm`, `tongue`, `nail`). We need an explicit mapping: capture is per-side, models are per-class (left/right pooled). This affects the dataset loader, the filename convention, and the prediction registry keys.

5. **Git not initialized.** `CURRENT_TASK.yaml` references branch `main`, but the directory is not a git repository. Version control (a stated requirement, Charter §24) does not yet exist.

6. **Registry/state persistence format undecided.** Specs describe registry contents and behaviour richly but do not choose a storage backend (JSON files vs SQLite). This is a Phase 2 decision with reproducibility and concurrency implications.

7. **Reproducibility mechanics underspecified.** Seed capture is required, but deterministic execution also needs `torch`/cuDNN determinism flags, dependency pinning (no lock file strategy stated), and environment capture. Needed before any training claims are reproducible.

None of these are defects in the *vision* — they are the normal seams between documents written separately. They are cheap to fix now and expensive to fix after code exists.

---

## 5. Missing Components

Everything below Phase 0. In dependency order:

**Phase 1 (current) deliverables, all absent:** canonical directory tree; installable Python package; `pyproject.toml`; `requirements.txt` (pinned); `README.md`; `LICENSE`; `.gitignore`; initialized git repository; the nine `configs/*.yaml` files; a typed configuration loader; the logging subsystem; a base exceptions module; and a smoke test proving the package imports and configs load.

**Phase 2 (infrastructure):** common interfaces/abstract base classes (`BaseModel`, `BaseTrainer`, `BaseManager`, `BaseAgent`); `RegistryManager` + backend; `StateManager`; `CheckpointManager`; `ExperimentManager`; `TrainingManager`; `PipelineManager`; `HbPipeline` facade; utilities; unit tests per manager.

**Phase 3:** `DatasetManager` (loaders, validation, statistics, patient-level splitting, preprocessing, augmentation, caching, standardized sample schema) + a synthetic/dummy dataset generator for build-mode testing.

**Phase 4:** pipeline execution across all six modes on dummy models; job queue; dependency resolution; resume/recovery.

**Phases 5–9:** segmentation framework (UNet/SegFormer/DeepLabV3+), prediction framework (EfficientNet/ResNet/DenseNet/ViT/ConvNeXt per tissue), adaptive agents, evaluation framework (metrics/figures/tables/Bland–Altman), deployment (FastAPI/Gradio/Streamlit/Docker).

**Cross-cutting, not yet owned by any phase:** test infrastructure choice (pytest + coverage target), CI, code-quality tooling (ruff/black/mypy per Charter §12), and a synthetic-data fixture strategy that lets every module be tested without the real dataset.

---

## 6. Risks

- **Abstraction-ahead-of-evidence.** The design front-loads ~10 managers plus registry, state, checkpoint, and experiment subsystems before a single model runs. If interfaces are locked against *dummy* data only, they may not survive contact with the real dataset, forcing rework of "completed" modules — which the rules discourage. Mitigation: drive every interface with a realistic synthetic dataset that mirrors the true schema (shapes, dtypes, missing-metadata cases), not trivial stubs.
- **Scope and timeline.** Four papers and nine implementation phases is a multi-year program. Risk of infrastructure gold-plating delaying the first scientific result (segmentation, Paper 1). Mitigation: keep Phases 1–4 deliberately minimal-but-correct; resist building deployment/agent scaffolding early.
- **Agent training depends on frozen upstream models.** Intermediate-prediction-based agent training only works if segmentation/prediction models and their preprocessing are stable and versioned. A change upstream invalidates all intermediate records. Mitigation: version intermediate-prediction artifacts against model registry IDs from day one.
- **Registry as single point of failure.** Everything loads through the registry. Corruption or schema drift stalls the whole pipeline. Mitigation: schema versioning + automatic backup (already in spec) must be implemented in Phase 2, not deferred.
- **Reproducibility debt.** Without pinned dependencies and determinism flags established in Phase 1, later "reproducible experiments" claims will be undermined. Mitigation: pin and capture environment from the first commit.
- **Real dataset unknowns.** No dataset is present. Its true size, class balance, mask quality, and metadata completeness are unknown, and they materially affect model choices and the value of adaptive routing. Mitigation: treat dataset characteristics as an open input; design for the documented contract, validate assumptions when data arrives.

---

## 7. Questions for You

These block or shape Phase 1. My recommended default is given for each so we can move fast if you simply confirm.

1. **Package layout:** Adopt a `src/adaptivehb/` src-layout package (my recommendation — clean installs, matches the manifest, avoids import ambiguity), and treat the flat directory list in PDS Ch.4 as *logical* subpackages inside it (`adaptivehb/segmentation/`, etc.)? Or do you want the literal flat top-level layout from PDS Ch.4?
2. **Import name:** Package/distribution name — `adaptivehb`? (short_name in the manifest is `AdaptiveHb`.)
3. **Manager roster:** Confirm the canonical set as: `PipelineManager`, `DatasetManager`, `RegistryManager`, `StateManager`, `CheckpointManager`, `ExperimentManager`, `TrainingManager`, `SegmentationManager`, `PredictionManager`, `AgentManager`, `EvaluationManager`, `DeploymentManager` — with **fusion and confidence living inside `AgentManager`** (not separate managers). Agree?
4. **Registry/state backend:** JSON-on-disk (human-readable, git-diffable, simplest — my recommendation for a research framework) or SQLite (transactional, scales better)?
5. **Tissue model granularity:** Confirm models are per **class** (eye/palm/tongue/nail) with left/right pooled at the dataset layer, while capture/filenames remain per-side. Agree?
6. **Tooling:** Standardize on `pytest`, `ruff`, `black`, `mypy`, and a pinned `requirements.txt` + `pyproject.toml`? Any constraint on Python patch version or CUDA/torch version I should pin to?
7. **License:** Which license for the (eventually) open-source release — MIT, Apache-2.0, or defer and ship a placeholder?
8. **Where these answers get recorded:** May I log the resolved decisions in `docs/DECISION_LOG.md` (and reconcile the four conflicting manager/layout lists in the specs) as part of Phase 1, since that requires touching existing docs — which the rules otherwise forbid without explicit request?

---

## 8. Recommended First Milestone

**Phase 1 — Repository Construction**, executed in two steps:

- **Step 1 (decisions):** Resolve §7 Q1–Q7 and record them in `DECISION_LOG.md`. This is a five-minute reconciliation that prevents expensive churn later. No code.
- **Step 2 (scaffold):** Build the canonical tree; initialize git; add `pyproject.toml`, pinned `requirements.txt`, `README.md`, `LICENSE`, `.gitignore`; create the nine `configs/*.yaml` with schema-typed defaults; implement the configuration loader, the logging subsystem, and a base exceptions module; add one smoke test asserting the package imports and every config loads and validates. **No ML code** (roadmap Phase 1 is explicit on this).

Definition of done: `pip install -e .` succeeds; `pytest` passes the smoke test; configs load and validate; `PROJECT_STATE.md`, `CHANGELOG.md`, and `CURRENT_TASK.yaml` updated; then stop.

This leaves the repository fully functional and matches the roadmap's "each phase leaves the repo in a working state" rule.

---

## 9. Estimated Development Order

1. **Phase 1** — Repository + config + logging + packaging *(current)*
2. **Phase 2** — Base interfaces → Registry → State → Checkpoint → Experiment → Training → Pipeline/`HbPipeline` (in that dependency order)
3. **Phase 3** — DatasetManager + synthetic dataset generator + validation/statistics/splitting
4. **Phase 4** — Pipeline modes on dummy models; job queue; resume/recovery
5. **Phase 5** — Segmentation framework (UNet, SegFormer, DeepLabV3+) behind one interface
6. **Phase 6** — Prediction framework (per-tissue backbones) behind one interface
7. **Phase 7** — Adaptive agents (deterministic policies first, trainable later) + Workflow Controller
8. **Phase 8** — Evaluation framework (metrics, figures, tables, Bland–Altman, reports)
9. **Phase 9** — Deployment (FastAPI/Gradio/Streamlit/Docker)
10. **Stage B** — Experiments, then Papers 1→4

One milestone per session; stop after each; update living documents each time.

---

## 10. Architectural Review

The architecture is sound and appropriate for a publication-grade, long-lived framework. The manager-mediated, registry-first, configuration-driven design is exactly right for reproducibility and extensibility, and the strict separation of framework construction from experimentation is a genuine strength. The following are recommendations only — I will not implement any of them without your approval, per the governance rules.

- **Single canonical structural source.** After you answer §7, let PDS Ch.4 be the *one* authoritative tree and have the manifest/index/pipeline docs reference it rather than restate it. Restating structure in four places is how the current drift happened.
- **Facade vs orchestrator, stated explicitly.** Document `HbPipeline` (public, thin) over `PipelineManager` (internal orchestrator) so the boundary is unambiguous before code exists.
- **Fusion/confidence as agents, not managers.** Keep them inside `AgentManager` to match the Agent Specification's three-layer model; a separate `FusionManager` would fragment the adaptive core and duplicate lifecycle code.
- **Synthetic-data-first testing.** Make a realistic synthetic dataset a Phase 1/2 deliverable. It is the single highest-leverage decision for keeping "build mode" honest and preventing interface rework.
- **Version intermediate artifacts against registry IDs immediately.** Bake the model-ID → intermediate-prediction linkage into the schema now; it is very hard to retrofit and it underpins the entire agent-training methodology.
- **Establish reproducibility mechanics in Phase 1.** Dependency pinning, determinism flags, and environment capture cost little now and are the foundation of every later scientific claim.
- **Guard against premature breadth.** Deployment targets (FastAPI/Gradio/Streamlit/Docker/HF) and six pipeline modes are correctly *specified*, but I recommend implementing them lazily (interfaces in Phase 4, concrete deployments only in Phase 9) to reach Paper 1's segmentation results sooner.

---

**No production code has been written. Implementation is paused pending your answers to §7 and your approval to begin Phase 1, Step 1.**
