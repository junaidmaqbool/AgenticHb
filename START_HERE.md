# START_HERE.md

# Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

**Repository Version:** 1.0

**Status:** Active Development

**Purpose:** Repository Entry Point

---

# Welcome

This repository implements a modular, research-grade framework for **non-invasive hemoglobin estimation** using an **Adaptive Multi-Agent Decision Framework**.

The project is intended to become a reusable biomedical AI platform rather than a collection of research scripts.

This document is the entry point for every developer and every AI coding assistant working on this repository.

---

# Before Doing Anything

Do **NOT**

- Start writing code.
- Redesign the architecture.
- Rename files or folders.
- Modify repository structure.
- Train models.
- Rewrite existing implementations.

Instead, first understand the project.

---

# Required Reading Order

Read documents in the following order.

## Step 1

Read

```
docs/PROJECT_INDEX.md
```

This provides the overall architecture and tells you which documents are relevant for the current task.

---

## Step 2

Read

```
docs/PROJECT_STATE.md
```

Understand

- Current implementation phase
- Completed milestones
- Current milestone
- Next task

---

## Step 3

Read

```
docs/CLAUDE_DEVELOPMENT_GUIDE.md
```

This defines

- Development workflow
- Coding behaviour
- Token-saving rules
- Session rules

---

## Step 4

Read

```
docs/DEVELOPMENT_RULES.md
```

These are the permanent coding standards.

Follow them exactly.

---

## Step 5

Read **only** the specification documents needed for the current milestone.

Examples

Repository

↓

IMPLEMENTATION_ROADMAP.md

Dataset

↓

DATASET_SPEC.md

Pipeline

↓

PIPELINE_SPEC.md

Adaptive Framework

↓

AGENT_SPECIFICATION.md

Registry

↓

MODEL_REGISTRY_SPEC.md

Experiments

↓

EXPERIMENT_SPEC.md

Do **NOT** read unnecessary documents.

---

# Repository Philosophy

This repository follows several fundamental principles.

- Modular architecture
- Reusable components
- Configuration-driven behaviour
- Reproducible experiments
- Lightweight adaptive decision making
- Production-quality software engineering

Every implementation should respect these principles.

---

# Development Workflow

Every coding session follows exactly the same workflow.

```
Read PROJECT_INDEX

↓

Read PROJECT_STATE

↓

Determine Current Milestone

↓

Open Only Required Documents

↓

Inspect Existing Repository

↓

Reuse Existing Code When Appropriate

↓

Implement One Milestone

↓

Run Tests

↓

Update PROJECT_STATE

↓

Update CHANGELOG

↓

Update DECISION_LOG (if required)

↓

STOP
```

Do not continue automatically.

---

# Existing Repository Review

Before implementing any module

always inspect the repository for

- Existing implementations
- Previous notebooks
- Utility functions
- Dataset loaders
- Training scripts
- Evaluation scripts
- Segmentation models
- Prediction models
- Configuration files

Good existing implementations should be

- reused
- cleaned
- refactored

rather than rewritten.

---

# Current Development Phase

Always obtain this information from

```
docs/PROJECT_STATE.md
```

Never assume the current phase.

---

# Architecture Rules

The documentation is the source of truth.

Implementation follows documentation.

Do not redesign the architecture unless explicitly requested.

If you identify a better design,

create an **Architectural Review**.

Never implement architectural changes automatically.

---

# Coding Rules

Every implementation should be

- Modular
- Typed
- Logged
- Tested
- Documented
- Configurable
- Reusable

Avoid

- Hardcoded paths
- Hardcoded datasets
- Hardcoded checkpoint names
- Hardcoded model names
- Duplicate code
- Circular dependencies

---

# Design Pattern Guidance

Before implementing a module,

consider whether an established design pattern would improve the solution.

Examples include

- Factory
- Strategy
- Registry
- Repository
- Builder
- Adapter
- Dependency Injection
- Observer
- Command
- State

Use patterns only when they simplify the architecture.

Avoid unnecessary complexity.

---

# Working with AI Models

The framework supports

- Segmentation Models
- Hb Prediction Models
- Adaptive Decision Modules
- Fusion Modules
- Confidence Modules

All models must

- register automatically
- save checkpoints
- support resume
- expose consistent interfaces

Models must never be loaded directly from file paths.

Always use the Registry.

---

# Training Policy

Framework development and model training are separate activities.

Do not train models while implementing infrastructure.

Training begins only after

- Repository
- Pipeline
- Managers
- Registry
- Dataset
- Framework

have been completed.

---

# Session Completion

At the end of every milestone

Claude should provide

## Milestone Summary

What was completed.

---

## Files Created

Complete list.

---

## Files Modified

Complete list.

---

## Tests Executed

Results.

---

## Architectural Review

Optional recommendations only.

Do not implement architectural changes automatically.

---

## Next Recommended Milestone

Exactly one logical next step.

Then stop.

---

# Living Documents

The following files must be updated throughout development.

```
docs/PROJECT_STATE.md

docs/CHANGELOG.md

docs/DECISION_LOG.md

docs/EXPERIMENT_LOG.md
```

These documents should always reflect the current state of the project.

---

# Success Criteria

The project will be considered successful when it provides

- A modular biomedical AI framework
- Fully reproducible experiments
- Adaptive decision-based Hb estimation
- Automated experiment management
- Automated model registry
- Automated reporting
- Publication-quality software
- Production-ready deployment capability

---

# Final Rule

**Think before you code.**

The objective is not simply to make the code run.

The objective is to build a research framework that remains maintainable, extensible, reproducible, and suitable for high-quality scientific publications for years to come.

Every implementation should move the project toward that goal.