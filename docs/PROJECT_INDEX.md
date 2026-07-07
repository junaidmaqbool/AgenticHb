# PROJECT_INDEX.md

Version: 1.0

Status: Master Project Index

Last Updated: July 2026

Project Name

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# 1. Purpose

PROJECT_INDEX.md serves as the central navigation document for the entire project.

It is the first document that should be read before opening any other project documentation.

Its objectives are

• Provide a concise overview of the project

• Track implementation progress

• Summarize architecture

• Track current development phase

• Reduce unnecessary reading of large documents

• Minimize token consumption during AI-assisted development

Detailed information should remain inside the corresponding specification documents.

This document should contain only summaries and references.

---

# 2. Project Summary

Project

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

Primary Goal

Develop a lightweight adaptive AI framework capable of intelligently selecting segmentation models, tissues, prediction models and fusion strategies to improve non-invasive hemoglobin estimation.

Scientific Contribution

Adaptive Decision Framework

Primary Output

Hemoglobin Prediction

Target Platform

Python

PyTorch

JupyterLab

Desktop Deployment

Future Web Deployment

Current Status

Architecture Complete

Implementation Pending

---

# 3. Documentation Index

01_PROJECT_CHARTER.md

Purpose

Project vision, objectives and scope.

Status

Completed

----------------------------------

02_PROJECT_DESIGN_SPECIFICATION.md

Purpose

Complete software architecture.

Status

Completed

----------------------------------

03_IMPLEMENTATION_ROADMAP.md

Purpose

Development phases.

Status

Completed

----------------------------------

04_PIPELINE_SPEC.md

Purpose

Execution workflow.

Status

Completed

----------------------------------

05_DATASET_SPEC.md

Purpose

Dataset architecture.

Status

Completed

----------------------------------

06_AGENT_SPECIFICATION.md

Purpose

Adaptive decision modules.

Status

Completed

----------------------------------

07_MODEL_REGISTRY_SPEC.md

Purpose

Model management.

Status

Completed

----------------------------------

08_EXPERIMENT_SPEC.md

Purpose

Research methodology.

Status

Completed

---

# 4. Repository Overview

Core Components

HbPipeline

↓

Managers

↓

Models

↓

Decision Modules

↓

Registry

↓

Evaluation

↓

Deployment

Every component communicates through the Pipeline.

No module should bypass the Pipeline.

---

# 5. Software Architecture

Framework

↓

Pipeline Manager

↓

Dataset Manager

↓

Training Manager

↓

Segmentation Manager

↓

Prediction Manager

↓

Decision Module Manager

↓

Evaluation Manager

↓

Registry Manager

↓

Deployment Manager

Every manager has a single responsibility.

---

# 6. Adaptive Framework

Perception Layer

Quality Assessment

ROI Verification

↓

Decision Layer

Segmentation Selection

Tissue Selection

Prediction Routing

↓

Clinical Layer

Adaptive Fusion

Confidence Estimation

↓

Workflow Controller

---

# 7. Supported Segmentation Models

UNet

SegFormer

DeepLabV3+

Future Models

Supported through Registry.

---

# 8. Supported Hb Prediction Models

Eye Models

Palm Models

Tongue Models

Nail Models

Multi-Tissue Models

All models loaded through Registry.

---

# 9. Dataset Summary

Supported Tissues

Left Eye

Right Eye

Left Palm

Right Palm

Tongue

Left Nail

Right Nail

Metadata

Patient Information

Hb

Height

Weight

BMI

SES

Additional Variables

Patient-Level Splitting

80

10

10

---

# 10. Current Development Phase

Current Phase

Repository Construction

Current Milestone

Framework Development

Training

Not Started

Deployment

Not Started

---

# 11. Progress Summary

Project Charter

✅

PDS

✅

Roadmap

✅

Pipeline

✅

Dataset

✅

Agents

✅

Registry

✅

Experiment

✅

Coding

⬜

Training

⬜

Evaluation

⬜

Deployment

⬜

Publication

⬜

---

# 12. Current Priority

Highest Priority

Repository Construction

Next Module

DatasetManager

Expected Next Steps

Create Repository

Create Managers

Create Pipeline

No model training yet.

---

# 13. Token Saving Rules

Before reading another document,

determine whether it is actually required.

Use this order

PROJECT_INDEX.md

↓

Specific Specification

↓

Relevant Source Code

↓

Implementation

Never load every document unless architecture changes.

---

# 14. Session Startup

At the beginning of every development session

Claude should

Read PROJECT_INDEX.md

Determine Current Phase

Determine Current Task

Determine Next File

Open only the required specification

Begin implementation

---

# 15. Session Shutdown

At the end of every development session

Claude should update

PROJECT_INDEX.md

Current Phase

Progress

Completed Modules

Remaining Work

Estimated Completion

Next Recommended Task

Then stop.

---

# End of Project Index

This document serves as the primary navigation point for the Adaptive Multi-Agent Framework.

All development sessions should begin here.