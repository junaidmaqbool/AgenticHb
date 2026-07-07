# PROJECT_DESIGN_SPECIFICATION.md

Version: 1.0

Status: Living Technical Specification

Last Updated: July 2026

Project Name:

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Chapter 1

# Introduction

## 1.1 Purpose

This document defines the complete software architecture for the Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation.

Unlike PROJECT_CHARTER.md, which defines the philosophy and objectives of the project, this document specifies exactly how the software should be designed, organized, implemented, trained, evaluated, and deployed.

This document acts as the architectural blueprint for every future implementation.

No source code should contradict this document unless the document itself is updated.

---

# 1.2 Primary Objectives

The framework should

• support multiple segmentation models

• support multiple Hb estimation models

• support adaptive model routing

• support adaptive tissue selection

• support resumable training

• support automatic checkpoint recovery

• support modular deployment

• support future research extensions

The framework should remain computationally lightweight and executable on a single consumer GPU.

---

# 1.3 High-Level Workflow

The entire framework consists of two completely separate stages.

Stage A

Framework Development

↓

Repository Structure

↓

Configuration System

↓

Pipeline

↓

Managers

↓

Agents

↓

Utilities

↓

Dummy Testing

↓

Framework Complete

No actual dataset training occurs.

----------------------------------------------------

Stage B

Experiments

↓

Real Dataset

↓

Train Segmentation

↓

Train Hb Models

↓

Train Agents

↓

Evaluate

↓

Generate Results

↓

Deployment

Software engineering and experimentation should remain completely independent.

---

# Chapter 2

# High-Level Architecture

The framework should follow the architecture shown below.

Input Images

↓

Dataset Manager

↓

Segmentation Manager

↓

Segmentation Selection Agent

↓

ROI Extraction

↓

Quality Assessment Agent

↓

Tissue Selection Agent

↓

Prediction Routing Agent

↓

Hb Prediction Models

↓

Fusion Agent

↓

Confidence Agent

↓

Clinical Report Generator

↓

Output

Every block should exist as an independent module.

Each module should communicate only through documented interfaces.

---

# Chapter 3

# Architectural Principles

The following principles govern the software architecture.

## Principle 1

Single Responsibility

Every module should perform one task only.

Examples

Segmentation Manager

↓

Only segmentation

Prediction Manager

↓

Only prediction

Fusion Manager

↓

Only fusion

No module should perform multiple unrelated tasks.

---

## Principle 2

Loose Coupling

Modules should know as little as possible about each other.

The Prediction Manager should not know how segmentation is implemented.

The Segmentation Manager should not know how prediction is performed.

The Controller coordinates interaction.

---

## Principle 3

Configuration Driven

Nothing should be hardcoded.

Everything should come from configuration files.

Examples

dataset paths

model selection

batch size

epochs

learning rate

augmentation

deployment

checkpoint locations

---

## Principle 4

Replaceability

Every component should be replaceable.

Examples

UNet

↓

replace by

SegFormer

without changing other modules.

Eye ViT

↓

replace by

EfficientNet

without changing the pipeline.

---

## Principle 5

Reproducibility

Every experiment must be reproducible.

The framework should automatically save

configuration

random seed

weights

logs

metrics

execution time

hardware information

---

# Chapter 4

# Repository Architecture

The repository should have the following structure.

project/

│

├── configs/

│

├── project/

│

├── dataset/

│

├── registry/

│

├── weights/

│

├── checkpoints/

│

├── logs/

│

├── tensorboard/

│

├── cache/

│

├── results/

│

├── figures/

│

├── excel/

│

├── notebooks/

│

├── segmentation/

│

├── prediction/

│

├── agents/

│

├── routing/

│

├── fusion/

│

├── confidence/

│

├── deployment/

│

├── api/

│

├── evaluation/

│

├── tests/

│

├── docs/

│

└── utilities/

No folder should serve multiple purposes.

---

# Chapter 5

# Module Overview

The framework consists of seven major modules.

Module 1

Dataset Module

Responsible for

loading

validation

preprocessing

metadata

splitting

augmentation

----------------------------------------------------

Module 2

Segmentation Module

Responsible for

training

prediction

saving

loading

evaluation

of segmentation models.

----------------------------------------------------

Module 3

Prediction Module

Responsible for

Hb estimation

training

validation

testing

saving

loading

----------------------------------------------------

Module 4

Adaptive Agent Module

Responsible for

quality assessment

segmentation selection

tissue selection

prediction routing

dynamic fusion

confidence estimation

----------------------------------------------------

Module 5

Evaluation Module

Responsible for

metrics

plots

statistics

tables

comparison

paper figures

----------------------------------------------------

Module 6

Deployment Module

Responsible for

desktop application

FastAPI

Gradio

Streamlit

Docker

HuggingFace

----------------------------------------------------

Module 7

Infrastructure Module

Responsible for

logging

registry

checkpointing

state management

configuration

experiment tracking

---

# Chapter 6

# Data Flow

The framework follows a strictly sequential data flow.

Patient

↓

Images

↓

Dataset Manager

↓

Segmentation

↓

ROI

↓

Quality Assessment

↓

Segmentation Selection

↓

Tissue Selection

↓

Prediction Routing

↓

Hb Prediction

↓

Dynamic Fusion

↓

Confidence

↓

Clinical Report

↓

Save Results

Each stage should be capable of independent testing.

Each stage should produce intermediate outputs.

Intermediate outputs should be optionally saved.

---

# Chapter 7

# Manager-Based Design

The framework should not expose individual models directly.

Instead it should use managers.

DatasetManager

SegmentationManager

PredictionManager

AgentManager

FusionManager

EvaluationManager

DeploymentManager

TrainingManager

ExperimentManager

PipelineManager

Every manager should expose a clean API.

Example

SegmentationManager

↓

train()

↓

predict()

↓

save()

↓

load()

↓

evaluate()

The implementation details remain hidden.
# Chapter 8

# Configuration System

## 8.1 Philosophy

The framework must be completely configuration-driven.

No dataset-specific information should ever appear inside source code.

The only information supplied by the user should be contained inside configuration files or through the execution notebook.

The configuration system should allow the framework to run on any compatible dataset without modifying Python code.

Every configurable parameter must originate from configuration files.

---

## 8.2 Configuration Files

The framework should contain multiple configuration files rather than one extremely large configuration.

Example

configs/

│

├── project.yaml

├── dataset.yaml

├── segmentation.yaml

├── prediction.yaml

├── agents.yaml

├── evaluation.yaml

├── deployment.yaml

├── registry.yaml

└── logging.yaml

Each configuration file should have a single responsibility.

---

## 8.3 Project Configuration

The project configuration should define

Project Name

Experiment Name

Output Directory

Random Seed

Hardware Selection

Number of Workers

Mixed Precision

Resume Training

Logging Level

TensorBoard

Checkpoint Frequency

---

## 8.4 Dataset Configuration

The dataset configuration should define

Original Image Folder

Mask Folder

Metadata CSV

Patient ID Column

Hemoglobin Column

Age Column

Gender Column

Height Column

Weight Column

BMI Column

Socioeconomic Status Column

Train Split

Validation Split

Test Split

Image Resolution

Image Channels

Normalization

Augmentation

Any optional metadata should simply be ignored if unavailable.

---

## 8.5 Segmentation Configuration

The segmentation configuration should define

Available Models

Default Model

Loss Function

Optimizer

Learning Rate

Batch Size

Epochs

Early Stopping

Scheduler

Checkpoint Directory

Inference Threshold

---

## 8.6 Prediction Configuration

This configuration should define

Available Hb Models

Default Prediction Model

Regression Loss

Optimizer

Scheduler

Learning Rate

Batch Size

Epochs

Input Resolution

Normalization

Feature Fusion

Metadata Fusion

---

## 8.7 Agent Configuration

The adaptive framework should remain configurable.

Example

Quality Agent

Enabled

Segmentation Selection Agent

Enabled

Tissue Selection Agent

Enabled

Prediction Routing Agent

Enabled

Fusion Agent

Enabled

Confidence Agent

Enabled

Future agents should be enabled without modifying source code.

---

# Chapter 9

# Dataset Specification

## 9.1 Dataset Philosophy

The framework should never assume a fixed dataset structure.

Instead it should validate the supplied dataset before training begins.

Validation should include

Folder structure

Image existence

Mask existence

Metadata completeness

Duplicate IDs

Corrupt images

Missing labels

Unsupported formats

A validation report should be generated before training starts.

---

## 9.2 Supported Inputs

The framework should support

RGB Images

Segmentation Masks

CSV Metadata

Patient Information

Clinical Labels

Optional Demographics

Future extensions should require minimal changes.

---

## 9.3 Dataset Splitting

The framework should automatically generate

Training Set

Validation Set

Testing Set

The default split should be

80%

Training

10%

Validation

10%

Testing

However, these values should remain configurable.

The testing dataset should never be used during training.

The testing dataset should only be used once the complete pipeline has been finalized.

---

## 9.4 Metadata Integration

Metadata should be optional.

Possible metadata

Age

Gender

Height

Weight

BMI

Socioeconomic Status

Clinical Information

CBC Parameters

The prediction framework should support image-only experiments as well as image-plus-metadata experiments.

---

# Chapter 10

# Model Registry

## 10.1 Purpose

Every trained model should automatically register itself.

The framework should never load arbitrary checkpoint files.

Instead it should query the registry.

The registry acts as the central catalogue of all trained models.

---

## 10.2 Registered Information

Each registered model should include

Model Name

Task

Architecture

Version

Training Date

Dataset Version

Validation Metrics

Training Configuration

Input Size

Checkpoint Path

Status

Current Best

Every update should create a new version.

Previous versions should never be deleted automatically.

---

## 10.3 Registry Categories

Separate registries should exist for

Segmentation Models

Prediction Models

Agent Models

Fusion Models

Confidence Models

This allows independent development of each subsystem.

---

# Chapter 11

# Checkpoint Manager

## 11.1 Philosophy

Training should never be lost.

Every important training stage should save recoverable checkpoints.

Checkpoints should include

Model Parameters

Optimizer

Scheduler

Epoch

Metrics

Random Seed

Configuration

Training History

Execution Time

Checkpoint creation should be automatic.

---

## 11.2 Recovery

Whenever training begins

the framework should automatically detect existing checkpoints.

If checkpoints exist

the framework should ask

Resume Previous Training?

YES

↓

Continue

NO

↓

Create New Experiment

No manual checkpoint loading should ever be required.

---

# Chapter 12

# Pipeline State Manager

## 12.1 Purpose

The framework should always know exactly where it stopped.

A pipeline state file should continuously record progress.

Examples

Segmentation

Completed

Prediction

Running

Agents

Pending

Testing

Pending

Deployment

Not Started

The user should never manually remember progress.

---

## 12.2 State File

Example

pipeline_state.json

↓

Current Phase

Current Experiment

Completed Modules

Current Module

Epoch

Checkpoint

Elapsed Time

Last Save Time

Current Status

If the computer shuts down unexpectedly

the framework should recover using this file.

---

# Chapter 13

# Experiment Manager

Every experiment should automatically receive

Experiment ID

Configuration Snapshot

Hardware Information

Dataset Version

Training Logs

Results Folder

Model Versions

Generated Figures

Excel Files

CSV Files

Execution Time

Nothing should overwrite previous experiments.

Every experiment should remain permanently reproducible.

---

# Chapter 14

# Execution Philosophy

The entire framework should execute from one notebook.

The notebook should not contain machine learning implementation.

Instead it should simply provide

Dataset Path

Mask Path

Metadata Path

Output Directory

Hardware Selection

Experiment Name

Pipeline Options

The notebook should then call

PipelineManager.run()

Everything else should happen automatically.

No individual training scripts should be manually executed.

The notebook acts as the central controller of the entire framework.

---

# Chapter 15

# Resume Philosophy

The framework should support interruption at any point.

Examples

Computer Shutdown

Power Failure

Manual Stop

GPU Failure

Unexpected Exception

When restarted

the framework should automatically

Load Previous Experiment

Load Pipeline State

Load Latest Checkpoints

Continue Training

Continue Evaluation

Continue Figure Generation

No completed work should ever be repeated unnecessarily.

Training should continue exactly from the last successful checkpoint.

This capability is considered a core requirement of the framework rather than an optional feature.
# Chapter 16

# Adaptive Agent Framework

## 16.1 Philosophy

The Adaptive Agent Framework is the central contribution of this project.

Unlike traditional deep learning pipelines that execute a fixed sequence of operations, the framework should make intelligent computational decisions throughout the inference process.

The framework should remain lightweight.

No Large Language Models should be used.

No external APIs should be required.

Agents should consist of lightweight machine learning models, neural networks, or deterministic controllers.

Future versions may replace deterministic policies with learned policies without changing the remaining architecture.

---

## 16.2 Agent Hierarchy

The framework should initially contain the following agents.

Quality Assessment Agent

↓

Segmentation Selection Agent

↓

ROI Verification Agent

↓

Tissue Selection Agent

↓

Prediction Routing Agent

↓

Dynamic Fusion Agent

↓

Confidence Agent

↓

Master Controller

Every agent should operate independently.

Every agent should expose a documented interface.

Every agent should be replaceable.

---

## 16.3 Agent Communication

Agents should never communicate directly.

Instead they exchange structured outputs through the Pipeline Manager.

Example

Quality Agent

↓

Quality Report

↓

Pipeline Manager

↓

Segmentation Selection Agent

This minimizes coupling.

---

# Chapter 17

# Training Workflow

Training occurs sequentially.

The framework should never attempt to train every component simultaneously.

Step 1

Validate Dataset

↓

Step 2

Train Segmentation Models

↓

Step 3

Evaluate Segmentation Models

↓

Step 4

Register Segmentation Models

↓

Step 5

Train Hb Prediction Models

↓

Step 6

Evaluate Hb Models

↓

Step 7

Register Hb Models

↓

Step 8

Generate Intermediate Predictions

↓

Step 9

Train Adaptive Agents

↓

Step 10

Evaluate Adaptive Framework

↓

Step 11

Final Testing

↓

Step 12

Deployment

Each completed stage should be reusable.

No completed stage should require retraining.

---

# Chapter 18

# Inference Workflow

The inference pipeline should never retrain models.

Instead it should

Load Registry

↓

Load Configuration

↓

Load Best Models

↓

Process Images

↓

Generate Prediction

↓

Generate Report

↓

Save Results

Inference should remain independent from training.

---

# Chapter 19

# Pipeline Modes

The framework should support multiple operating modes.

Mode 1

Build Mode

Purpose

Validate framework

Check repository

Dummy testing

No dataset required.

--------------------------------------------------

Mode 2

Training Mode

Purpose

Train every required model.

Automatically save checkpoints.

Automatically update registry.

--------------------------------------------------

Mode 3

Resume Mode

Purpose

Resume interrupted training.

Automatically recover

checkpoint

optimizer

scheduler

epoch

pipeline state

--------------------------------------------------

Mode 4

Evaluation Mode

Purpose

Generate

metrics

figures

tables

paper results

No retraining.

--------------------------------------------------

Mode 5

Inference Mode

Purpose

Predict Hb for unseen patients.

Generate clinical report.

No retraining.

--------------------------------------------------

Mode 6

Deployment Mode

Purpose

Launch

Desktop

FastAPI

Gradio

Streamlit

Docker

HuggingFace

---

# Chapter 20

# Interfaces

Every module should implement a common interface.

Segmentation

train()

predict()

evaluate()

save()

load()

--------------------------------------------------

Prediction

train()

predict()

evaluate()

save()

load()

--------------------------------------------------

Agent

train()

predict()

save()

load()

--------------------------------------------------

Evaluation

evaluate()

generate_tables()

generate_figures()

export_excel()

export_csv()

--------------------------------------------------

Deployment

load_pipeline()

predict()

launch()

shutdown()

This ensures every component behaves consistently.

---

# Chapter 21

# Future Extensibility

The framework should remain open for future research.

Possible extensions include

Additional tissues

Additional segmentation models

Foundation Models

Vision Language Models

Self-Supervised Learning

Federated Learning

Multimodal Learning

Reinforcement Learning

Edge Deployment

Cloud Deployment

Mobile Deployment

These additions should require minimal modifications to the existing architecture.

---

# Chapter 22

# Project Deliverables

At project completion the repository should contain

Complete Source Code

Configuration Files

Documentation

Training Pipeline

Evaluation Pipeline

Deployment Pipeline

Experiment Tracking

Model Registry

Checkpoint Manager

Pipeline State Manager

Pretrained Models

Example Dataset

Example Notebook

API

Desktop Interface

Documentation

README

Unit Tests

Docker Support

GitHub Repository

HuggingFace Deployment

The repository should be usable by another researcher without requiring direct assistance from the original developer.

---

# Chapter 23

# Acceptance Criteria

The framework shall be considered complete when

✓ Repository builds successfully.

✓ Every module passes unit tests.

✓ Dataset validation succeeds.

✓ Segmentation models train independently.

✓ Hb models train independently.

✓ Adaptive agents train independently.

✓ Pipeline resumes after interruption.

✓ Registry manages every model.

✓ Evaluation generates publication-quality figures.

✓ Deployment loads only saved weights.

✓ New researchers can reproduce experiments.

✓ Complete documentation exists.

---

# End of Project Design Specification

This document should evolve together with the project.

Major architectural modifications should update this document before implementation.

Software implementation should always remain consistent with this specification.