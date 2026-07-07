# CLAUDE_DEVELOPMENT_GUIDE.md

Version: 1.0

Status: Permanent Development Rules

Last Updated: July 2026

Project

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# 1. Purpose

This document defines the permanent development rules that Claude must follow throughout the lifetime of the project.

Its objectives are

• Maintain architectural consistency

• Reduce unnecessary token usage

• Prevent code duplication

• Maintain modularity

• Prevent accidental redesign

• Ensure reproducible implementation

Every development session should begin by reading this document.

---

# 2. Primary Mission

Claude is responsible for implementing the project described in the documentation.

Claude is NOT responsible for redesigning the architecture unless explicitly requested.

Claude should implement the existing specifications faithfully.

---

# 3. Documents to Read

Claude should never load every document.

Instead use the following order.

Step 1

Read

PROJECT_INDEX.md

↓

Determine current phase.

↓

Determine current task.

↓

Determine current file.

Only then decide whether additional documentation is required.

---

If implementation concerns

Repository

↓

Read

IMPLEMENTATION_ROADMAP.md

If implementation concerns

Dataset

↓

Read

DATASET_SPEC.md

If implementation concerns

Pipeline

↓

Read

PIPELINE_SPEC.md

If implementation concerns

Agents

↓

Read

AGENT_SPECIFICATION.md

If implementation concerns

Registry

↓

Read

MODEL_REGISTRY_SPEC.md

If implementation concerns

Training

↓

Read

EXPERIMENT_SPEC.md

Never load unnecessary documentation.

---

# 4. Token Saving Policy

Token efficiency is a primary objective.

Claude should

Read only required documentation.

Never restate existing specifications.

Never regenerate documentation.

Never explain previously completed architecture.

Reference existing documents whenever possible.

Do not summarize documents unless requested.

---

# 5. Scope of Every Session

Each development session should implement only one logical milestone.

Examples

Repository

DatasetManager

RegistryManager

PipelineManager

UNet

SegFormer

Prediction Manager

Decision Module

Evaluation Module

One milestone per session.

Do not continue automatically after milestone completion.

---

# 6. Architecture Rules

Claude must not

Redesign the framework

Rename managers

Rename folders

Rename interfaces

Rename public APIs

Change folder structure

Modify documentation

unless explicitly requested.

The documentation is the source of truth.

Implementation follows documentation.

Not the reverse.

---

# 7. Repository Rules

Every new file should belong to the predefined repository structure.

No temporary folders.

No experimental folders.

No duplicate implementations.

No obsolete files.

Every new module should immediately integrate into the existing architecture.

---

# 8. Coding Standards

Python Version

3.11+

Framework

PyTorch

Preferred Libraries

OpenCV

NumPy

Pandas

Albumentations

TorchMetrics

Matplotlib

Scikit-learn

Avoid introducing additional dependencies unless necessary.