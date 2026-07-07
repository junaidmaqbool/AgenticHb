# IMPLEMENTATION_ROADMAP.md

Version: 1.0

Status: Living Development Document

Last Updated: July 2026

Project Name:

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# 1. Purpose

This document defines the complete implementation strategy for the Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation.

Unlike the Project Charter and the Project Design Specification, this document focuses exclusively on implementation.

It answers one question:

> **What should be built next?**

The roadmap divides development into independent milestones.

Each milestone should leave the repository in a fully functional state.

Each milestone should be independently testable.

Each milestone should support interruption and future continuation.

This document should be updated continuously as development progresses.

---

# 2. Development Philosophy

Development is divided into two completely independent stages.

## Stage A – Framework Development

This stage focuses entirely on software engineering.

No real dataset is required.

No models are trained.

The objective is to build a reusable AI framework.

The completed framework should include

- Repository structure
- Managers
- Configuration system
- Logging
- Registry
- State management
- Checkpoint management
- Pipeline execution
- Dummy testing
- Documentation

At the end of this stage, every module should work using dummy data.

---

## Stage B – Scientific Experiments

Only after the framework has been completed should experiments begin.

This stage includes

- Dataset preparation
- Segmentation training
- Hb prediction training
- Agent training
- Evaluation
- Statistical analysis
- Figure generation
- Deployment

No architectural modifications should occur during this stage unless absolutely necessary.

---

# 3. Development Principles

The following principles govern implementation.

- Never skip implementation phases.
- Never train real models while the framework is still under construction.
- Every module must compile before moving to the next.
- Every phase must leave the repository functional.
- Every phase must include documentation.
- Every phase must include basic testing.
- Every major component must be reusable.
- Never implement temporary solutions.
- Never duplicate code.
- Never redesign completed modules without discussion.

---

# 4. Current Project Status

| Component | Status |
|------------|---------|
| Project Charter | ✅ Completed |
| Project Design Specification | ✅ Completed |
| Repository Architecture | Pending |
| Infrastructure | Pending |
| Dataset Module | Pending |
| Pipeline Framework | Pending |
| Segmentation Framework | Pending |
| Prediction Framework | Pending |
| Adaptive Framework | Pending |
| Evaluation Framework | Pending |
| Deployment Framework | Pending |

---

# 5. Overall Development Timeline

```
Documentation
      │
      ▼
Repository
      │
      ▼
Infrastructure
      │
      ▼
Dataset
      │
      ▼
Pipeline
      │
      ▼
Segmentation
      │
      ▼
Prediction
      │
      ▼
Adaptive Agents
      │
      ▼
Evaluation
      │
      ▼
Deployment
      │
      ▼
Experiments
      │
      ▼
Publication
```

Each phase should be completed before moving to the next.

---

# PHASE 0

# Documentation

## Objective

Create complete project documentation before implementation begins.

## Inputs

None

## Outputs

- PROJECT_CHARTER.md
- PROJECT_DESIGN_SPECIFICATION.md
- IMPLEMENTATION_ROADMAP.md
- DATASET_SPEC.md
- MODEL_REGISTRY_SPEC.md
- PIPELINE_SPEC.md
- AGENT_SPECIFICATION.md
- RESEARCH_PLAN.md

## Deliverables

Complete project blueprint.

## Success Criteria

- Architecture approved.
- Repository design approved.
- Coding standards approved.

Status

✅ Completed

---

# PHASE 1

# Repository Construction

## Objective

Build a professional research repository.

## Tasks

Create

- Root project structure
- Documentation directory
- Configuration directory
- Source code package
- Test directory
- Notebook directory
- Weights directory
- Registry directory
- Results directory
- Logging directory
- Checkpoint directory

Configure

- requirements.txt
- README.md
- LICENSE
- .gitignore
- pyproject.toml
- logging

No machine learning code should be written during this phase.

## Deliverables

Complete repository.

## Success Criteria

- Repository installs successfully.
- Repository structure matches specification.
- Documentation accessible.
- Configuration loads correctly.

---

# PHASE 2

# Core Infrastructure

## Objective

Build reusable infrastructure that will be shared by every future module.

## Managers to Implement

- DatasetManager
- TrainingManager
- RegistryManager
- CheckpointManager
- StateManager
- ExperimentManager
- EvaluationManager
- DeploymentManager
- PipelineManager

## Additional Components

- Configuration Loader
- Logger
- Utilities
- Common Interfaces
- Exception Handling

No deep learning models should be implemented.

## Deliverables

Complete reusable infrastructure.

## Success Criteria

- Managers communicate correctly.
- Configuration loads correctly.
- Logging operational.
- Registry initialized.
- Pipeline state tracking operational.

---

# PHASE 3

# Dataset Module

## Objective

Develop a reusable dataset framework capable of supporting future datasets without source code modification.

## Features

- Image Loader
- Mask Loader
- Metadata Loader
- CSV Parser
- Dataset Validation
- Dataset Statistics
- Visualization
- Dataset Splitting
- Corrupt Image Detection
- Duplicate Detection
- Missing Metadata Detection

## Supported Inputs

- RGB Images
- Segmentation Masks
- CSV Metadata
- Clinical Labels

Optional Metadata

- Age
- Gender
- Height
- Weight
- BMI
- Socioeconomic Status
- Additional Clinical Variables

## Deliverables

Reusable dataset module.

## Success Criteria

- Dataset validation passes.
- Metadata loads correctly.
- Images load correctly.
- Splits generated automatically.
- Statistics generated automatically.

---

# PHASE 4

# Pipeline Framework

## Objective

Develop the central PipelineManager responsible for coordinating every component in the framework.

The PipelineManager should support multiple execution modes.

Initially, all modes should operate using dummy models.

No training should occur during this phase.

## Pipeline Modes

- Build Mode
- Training Mode
- Resume Mode
- Evaluation Mode
- Inference Mode
- Deployment Mode

## Responsibilities

The PipelineManager should

- Load configuration.
- Initialize managers.
- Validate dataset.
- Select execution mode.
- Execute the requested workflow.
- Save pipeline state.
- Recover previous execution when requested.

## Deliverables

Complete execution framework.

## Success Criteria

- All pipeline modes execute successfully.
- Dummy workflow completes.
- Pipeline state updates correctly.
- Resume mechanism functions correctly.

---

# Progress Tracker

| Phase | Status | Completion Date |
|---------|---------|----------------|
| Documentation | ✅ | |
| Repository | ⬜ | |
| Infrastructure | ⬜ | |
| Dataset | ⬜ | |
| Pipeline | ⬜ | |
| Segmentation | ⬜ | |
| Prediction | ⬜ | |
| Agents | ⬜ | |
| Evaluation | ⬜ | |
| Deployment | ⬜ | |

---

# Current Recommended Milestone

The next implementation task is:

**Phase 1 – Repository Construction**

No model development should begin until the repository, infrastructure, dataset module, and pipeline framework are complete.
---

# PHASE 5

# Segmentation Framework

## Objective

Develop the complete segmentation subsystem capable of supporting multiple interchangeable segmentation models.

This phase focuses entirely on implementation.

Model training will occur later during the Experiment Phase.

The framework should initially support

- UNet
- SegFormer
- DeepLabV3+

Additional segmentation models should be easily added without modifying the Pipeline Manager.

---

## Features

Implement

- Segmentation Model Interface
- Segmentation Trainer
- Segmentation Evaluator
- Segmentation Predictor
- Checkpoint Support
- Registry Integration
- Automatic Model Loading

Every segmentation model should expose the same interface.

---

## Deliverables

- Segmentation framework
- Dummy model testing
- Registry integration
- Documentation

---

## Success Criteria

✓ Models load correctly

✓ Models save correctly

✓ Interfaces are identical

✓ Registry integration complete

✓ Dummy inference successful

---

# PHASE 6

# Hemoglobin Prediction Framework

## Objective

Develop the complete hemoglobin estimation framework.

Initially support

- Eye Models
- Palm Models
- Tongue Models
- Nail Models
- Future Multi-Tissue Models

No real training occurs during this phase.

Only architecture implementation.

---

## Features

Implement

- Prediction Model Interface
- Trainer
- Evaluator
- Predictor
- Checkpoint Support
- Registry Integration

Support future replacement of models without changing pipeline logic.

---

## Deliverables

Complete prediction subsystem.

---

## Success Criteria

✓ Models load

✓ Models save

✓ Dummy inference works

✓ Registry updated

---

# PHASE 7

# Adaptive Decision Framework

## Objective

Implement the adaptive decision-making framework.

This phase represents the primary scientific contribution of the project.

The framework should initially support deterministic policies.

Later versions may replace these with trainable policies.

---

## Agents

Quality Assessment Agent

Segmentation Selection Agent

ROI Verification Agent

Tissue Selection Agent

Prediction Routing Agent

Dynamic Fusion Agent

Confidence Agent

Master Controller

---

## Responsibilities

The framework should intelligently decide

- Which segmentation model to use

- Whether segmentation quality is acceptable

- Which tissue(s) should be analysed

- Which Hb models should execute

- How predictions should be fused

- Whether confidence is sufficient

- Whether additional tissue images are required

---

## Deliverables

Complete adaptive framework.

---

## Success Criteria

✓ All agents implemented

✓ Manager integration complete

✓ Dummy workflow successful

✓ Configuration support complete

---

# PHASE 8

# Evaluation Framework

## Objective

Develop a reusable evaluation framework capable of automatically generating publication-quality outputs.

---

## Metrics

Regression

- MAE

- RMSE

- R²

- Pearson Correlation

- Spearman Correlation

Classification

- Accuracy

- Precision

- Recall

- F1

Calibration

- Calibration Error

- Confidence Reliability

Clinical

- Bland–Altman

- Limits of Agreement

Efficiency

- Inference Time

- GPU Memory

- Number of Tissues Used

---

## Automatic Outputs

CSV

Excel

TensorBoard

Plots

Paper Tables

Publication Figures

Prediction Files

Model Comparison Reports

---

## Success Criteria

✓ Automatic evaluation

✓ Automatic figure generation

✓ Automatic report generation

---

# PHASE 9

# Deployment Framework

## Objective

Deploy the completed framework.

Deployment should require only loading previously trained models.

No retraining should occur.

---

## Supported Platforms

Desktop Application

FastAPI

Gradio

Streamlit

Docker

Hugging Face Spaces

Future Mobile Deployment

---

## Pipeline

Load Configuration

↓

Load Registry

↓

Load Models

↓

Accept Images

↓

Predict

↓

Generate Report

↓

Save Results

---

## Deliverables

Complete deployment system.

---

## Success Criteria

✓ Deployment loads automatically

✓ Registry detected

✓ Prediction successful

✓ Report generated

---

# EXPERIMENT PHASE

Only after every implementation phase has been completed should experiments begin.

The experiment workflow should be

Dataset Validation

↓

Train Segmentation Models

↓

Register Models

↓

Train Hb Prediction Models

↓

Register Models

↓

Generate Intermediate Predictions

↓

Train Adaptive Agents

↓

Register Models

↓

Run Independent Testing

↓

Run Adaptive Testing

↓

Generate Publication Figures

↓

Generate Final Reports

---

# Resume Strategy

Every experiment should support interruption.

If execution stops unexpectedly,

the framework should automatically

- Detect previous experiment

- Detect current phase

- Detect latest checkpoint

- Restore optimizer

- Restore scheduler

- Restore epoch

- Restore pipeline state

- Continue automatically

No completed computation should be repeated unnecessarily.

---

# Claude Development Workflow

At the beginning of every conversation Claude should automatically provide

Current Phase

Repository Status

Completed Modules

Pending Modules

Current Recommendation

Estimated Complexity

Potential Risks

---

At the end of every implementation session Claude should produce

Completed Work

Repository Tree

Files Created

Files Modified

Pending Work

Recommended Next Step

Then STOP.

Claude should never continue automatically.

---

# Experiment Workflow

The experiment notebook should be the only user entry point.

The notebook should request only

Dataset Root

Mask Directory

Metadata CSV

Output Directory

Experiment Name

Hardware Configuration

Execution Mode

Everything else should be handled automatically.

The notebook should never contain model implementation.

---

# Progress Tracker

| Phase | Status | Completion |
|---------|--------|------------|
| Documentation | ✅ | |
| Repository | ⬜ | |
| Infrastructure | ⬜ | |
| Dataset | ⬜ | |
| Pipeline | ⬜ | |
| Segmentation | ⬜ | |
| Prediction | ⬜ | |
| Adaptive Framework | ⬜ | |
| Evaluation | ⬜ | |
| Deployment | ⬜ | |
| Experiments | ⬜ | |
| Publication | ⬜ | |

---

# Acceptance Criteria

The implementation roadmap is complete when

✓ Repository is fully functional.

✓ Every module is documented.

✓ Every manager has unit tests.

✓ Dataset validation is automatic.

✓ Training resumes automatically.

✓ Every model registers itself.

✓ Evaluation is fully automated.

✓ Deployment requires no retraining.

✓ Publication figures are generated automatically.

✓ Another researcher can reproduce all experiments using only the provided documentation.

---

# Living Document Policy

This roadmap is a living document.

Any architectural modification should first update

- PROJECT_CHARTER.md

- PROJECT_DESIGN_SPECIFICATION.md

- IMPLEMENTATION_ROADMAP.md

before implementation begins.

This document should always reflect the current implementation state of the repository.

