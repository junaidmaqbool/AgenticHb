# PIPELINE_SPEC.md

Version: 1.0

Status: Living Technical Specification

Last Updated: July 2026

Project Name

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Chapter 1

# Purpose

The Pipeline Specification defines the operational behavior of the complete framework.

Unlike the Project Design Specification, which explains the software architecture, this document explains how every component communicates during execution.

The pipeline is responsible for coordinating

• Configuration

• Dataset

• Segmentation

• Prediction

• Adaptive Decision Framework

• Evaluation

• Deployment

Every future execution of the framework must pass through the pipeline.

No module should bypass the pipeline.

---

# Chapter 2

# Pipeline Philosophy

The pipeline is the central controller of the project.

No notebook should directly call individual models.

No notebook should directly train segmentation models.

No notebook should directly train prediction models.

No notebook should directly train adaptive agents.

Instead,

the notebook communicates only with

HbPipeline

↓

HbPipeline communicates with Managers

↓

Managers communicate with Models

↓

Models communicate with Data

This layered architecture minimizes coupling and improves maintainability.

---

# Chapter 3

# Pipeline Overview

The complete framework revolves around one central object.

HbPipeline

The pipeline should expose only a small number of public methods.

Example

pipeline.build()

pipeline.train()

pipeline.resume()

pipeline.evaluate()

pipeline.test()

pipeline.predict()

pipeline.deploy()

Internally,

these methods call different managers.

The user never interacts directly with managers.

---

# Chapter 4

# Pipeline Architecture

The complete execution flow is

User

↓

Experiment Notebook

↓

Configuration Loader

↓

HbPipeline

↓

Pipeline Manager

↓

Managers

↓

Models

↓

Results

The notebook should never contain AI logic.

The notebook acts only as an experiment launcher.

---

# Chapter 5

# Pipeline Managers

The pipeline communicates with the following managers.

DatasetManager

↓

RegistryManager

↓

StateManager

↓

TrainingManager

↓

SegmentationManager

↓

PredictionManager

↓

AgentManager

↓

EvaluationManager

↓

DeploymentManager

Each manager has a single responsibility.

Managers communicate only through the Pipeline Manager.

Managers should never directly call one another.

---

# Chapter 6

# Pipeline Lifecycle

Every execution of the framework follows the same lifecycle.

Initialize

↓

Load Configuration

↓

Validate Configuration

↓

Initialize Managers

↓

Load Registry

↓

Load Pipeline State

↓

Load Dataset

↓

Validate Dataset

↓

Select Execution Mode

↓

Execute

↓

Save Results

↓

Update Registry

↓

Update Pipeline State

↓

Shutdown

This lifecycle should remain identical regardless of execution mode.

---

# Chapter 7

# Execution Modes

The framework should support six execution modes.

Mode 1

Build

Purpose

Validate project

Dummy testing

Verify managers

No dataset required.

---------------------------------------

Mode 2

Training

Purpose

Train models.

Generate checkpoints.

Update registry.

Generate logs.

---------------------------------------

Mode 3

Resume

Purpose

Automatically continue interrupted training.

Load checkpoints.

Restore optimizer.

Restore scheduler.

Restore epoch.

Continue.

---------------------------------------

Mode 4

Evaluation

Purpose

Load trained models.

Generate metrics.

Generate figures.

Generate paper tables.

---------------------------------------

Mode 5

Inference

Purpose

Load trained models.

Predict Hb.

Generate clinical report.

---------------------------------------

Mode 6

Deployment

Purpose

Launch

Desktop

FastAPI

Gradio

Streamlit

Docker

HuggingFace

---

# Chapter 8

# Notebook Interaction

The framework should require only one execution notebook.

The notebook should collect

Dataset Root

Mask Folder

Metadata CSV

Output Folder

Experiment Name

Hardware Selection

Execution Mode

Pipeline Options

Nothing else.

The notebook should never contain

Training loops

Model code

Loss functions

Optimizers

Schedulers

Evaluation logic

Checkpoint handling

Registry handling

Resume logic

All of these belong inside the framework.

The notebook should simply execute

pipeline = HbPipeline(config)

pipeline.run()

---

# Chapter 9

# Pipeline Initialization

The initialization process should always execute in the following order.

1.

Load Configuration

↓

2.

Validate Configuration

↓

3.

Initialize Logger

↓

4.

Initialize Registry

↓

5.

Initialize Pipeline State

↓

6.

Initialize Managers

↓

7.

Load Dataset

↓

8.

Validate Dataset

↓

9.

Select Mode

↓

10.

Begin Execution

If any step fails,

execution should terminate gracefully with an informative error.

---

# Chapter 10

# Manager Initialization Order

Managers should always initialize in the following sequence.

Configuration Manager

↓

Logging Manager

↓

Registry Manager

↓

State Manager

↓

Dataset Manager

↓

Training Manager

↓

Segmentation Manager

↓

Prediction Manager

↓

Agent Manager

↓

Evaluation Manager

↓

Deployment Manager

↓

Pipeline Ready

No manager should initialize before its dependencies are available.

---

# Chapter 11

# Pipeline State Updates

The Pipeline Manager should continuously update

Current Experiment

↓

Current Phase

↓

Current Module

↓

Current Epoch

↓

Elapsed Time

↓

Latest Checkpoint

↓

Status

The pipeline state should be written automatically after every important operation.

If execution stops,

the framework should always know exactly where it ended.

---

# Chapter 12

# Pipeline Outputs

Every execution should generate

Logs

↓

Checkpoints

↓

Registry Updates

↓

Metrics

↓

Figures

↓

Predictions

↓

Configuration Snapshot

↓

Execution Summary

↓

Pipeline State

↓

Experiment Folder

Nothing generated by the pipeline should be lost.

Every execution should remain reproducible.
# Chapter 13

# Training Pipeline

## Purpose

Training Mode is responsible for generating every trainable model used by the framework.

Training should never occur during framework development.

Training begins only after

• Repository is complete

• Managers are implemented

• Pipeline is functional

• Dataset validation succeeds

The Training Pipeline should execute automatically without requiring user intervention after configuration.

---

## Training Workflow

Training Mode should execute the following sequence.

Initialize Project

↓

Load Configuration

↓

Initialize Managers

↓

Load Registry

↓

Load Pipeline State

↓

Validate Dataset

↓

Create Experiment

↓

Prepare Output Directory

↓

Train Segmentation Models

↓

Evaluate Segmentation Models

↓

Register Segmentation Models

↓

Train Hb Prediction Models

↓

Evaluate Hb Prediction Models

↓

Register Hb Models

↓

Generate Intermediate Predictions

↓

Train Adaptive Agents

↓

Evaluate Adaptive Agents

↓

Register Agent Models

↓

Generate Final Summary

↓

Save Experiment

↓

Shutdown

Every completed stage should automatically update the Pipeline State.

---

# Chapter 14

# Resume Pipeline

## Purpose

Resume Mode should recover interrupted execution automatically.

Users should never manually select checkpoints.

The framework should detect

• Experiment

• Phase

• Checkpoint

• Epoch

• Registry

• Configuration

automatically.

---

## Resume Workflow

Load Pipeline State

↓

Locate Latest Checkpoint

↓

Restore Configuration

↓

Restore Random Seed

↓

Restore Optimizer

↓

Restore Scheduler

↓

Restore Epoch

↓

Continue Current Job

↓

Update Registry

↓

Save State

If multiple interrupted experiments exist,

the framework should present available experiments for selection.

---

# Chapter 15

# Evaluation Pipeline

Evaluation Mode should never retrain models.

Instead,

it loads previously trained models and evaluates them.

---

## Evaluation Workflow

Load Registry

↓

Load Best Models

↓

Load Test Dataset

↓

Generate Predictions

↓

Evaluate Individual Models

↓

Evaluate Adaptive Framework

↓

Compare Baselines

↓

Generate Figures

↓

Generate Tables

↓

Export Excel

↓

Export CSV

↓

Save Results

---

## Evaluation Outputs

The Evaluation Manager should automatically generate

Regression Metrics

Classification Metrics

Calibration Metrics

Confusion Matrices

ROC Curves

Precision Recall Curves

Bland–Altman Plots

Scatter Plots

Prediction Distributions

Residual Plots

Comparison Tables

Execution Statistics

Publication Figures

No manual plotting should be required.

---

# Chapter 16

# Inference Pipeline

Inference Mode is designed for unseen patients.

No model training occurs.

---

## Workflow

Load Configuration

↓

Load Registry

↓

Load Best Models

↓

Load Images

↓

Run Segmentation

↓

Verify ROI

↓

Run Quality Assessment

↓

Select Tissue

↓

Run Hb Prediction

↓

Fuse Predictions

↓

Estimate Confidence

↓

Generate Report

↓

Save Results

Inference should complete using only saved models.

---

# Chapter 17

# Deployment Pipeline

Deployment Mode converts the framework into a usable application.

Deployment should require

no retraining

no manual checkpoint selection

no code modification

---

## Deployment Workflow

Load Registry

↓

Load Pipeline

↓

Load Models

↓

Initialize API

↓

Accept Patient Images

↓

Generate Prediction

↓

Generate Report

↓

Return Results

---

# Chapter 18

# Job Queue

Every operation executed by HbPipeline should be represented as a Job.

Examples

ValidateDataset

TrainSegmentation

EvaluateSegmentation

TrainPrediction

EvaluatePrediction

GenerateIntermediatePredictions

TrainAgents

EvaluateAgents

GenerateFigures

Deploy

Each Job should have

Job ID

Status

Start Time

Finish Time

Dependencies

Output

Logs

Retry Count

The Pipeline Manager executes Jobs sequentially.

Future versions may support parallel execution.

---

# Chapter 19

# Dependency Resolution

The Pipeline Manager should never execute Jobs blindly.

Before execution,

dependencies should be verified.

Example

Train Prediction

requires

Completed Segmentation

Completed Dataset Validation

Registered Segmentation Models

If dependencies are missing,

execution should stop with an informative message.

---

# Chapter 20

# Failure Recovery

Failures are expected.

The framework should recover automatically whenever possible.

Possible failures include

Power Failure

GPU Failure

Disk Full

Interrupted Execution

Keyboard Interrupt

Unexpected Exception

Recovery Strategy

Read Pipeline State

↓

Locate Last Successful Job

↓

Restore Checkpoint

↓

Continue Execution

Only failed jobs should repeat.

Completed jobs should never execute again.

---

# Chapter 21

# Logging Pipeline

Every Job should create logs.

Logs should include

Timestamp

Current Phase

Current Job

Current Model

Epoch

Loss

Validation Metrics

Checkpoint Location

Execution Time

Hardware Information

Logs should be human-readable.

Debug logs should remain available when needed.

---

# Chapter 22

# Output Structure

Every experiment should automatically generate

Experiment Folder

│

├── configuration/

├── logs/

├── checkpoints/

├── registry/

├── metrics/

├── figures/

├── predictions/

├── tensorboard/

├── excel/

├── csv/

├── reports/

├── pipeline_state/

└── summary/

The framework should never overwrite previous experiments.

---

# Chapter 23

# Pipeline Interfaces

HbPipeline should expose only a small public API.

Example

initialize()

build()

train()

resume()

evaluate()

test()

predict()

deploy()

shutdown()

Everything else remains internal.

This minimizes complexity for users.

---

# Chapter 24

# Pipeline Completion Criteria

The Pipeline Specification is considered complete when

✓ Every execution mode functions.

✓ Every manager communicates correctly.

✓ Resume Mode restores interrupted execution.

✓ Registry updates automatically.

✓ Checkpoints save automatically.

✓ Experiments are reproducible.

✓ Deployment loads saved models only.

✓ Every stage is independently testable.

---

# End of Pipeline Specification

This document defines the operational behavior of the framework.

All execution within the project should conform to this specification.

Any future modifications to pipeline behavior should first update this document before implementation begins.