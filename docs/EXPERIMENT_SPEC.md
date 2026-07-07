# EXPERIMENT_SPEC.md

Version: 1.0

Status: Living Research Specification

Last Updated: July 2026

Project Name

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Chapter 1

# Purpose

This document defines the complete experimental methodology used throughout the project.

Unlike the previous software specifications, this document focuses on scientific experimentation.

It specifies

• Experimental Design

• Dataset Usage

• Training Protocols

• Validation Protocols

• Testing Procedures

• Baseline Comparisons

• Adaptive Framework Evaluation

• Statistical Analysis

• Reproducibility

Every published experiment should conform to this specification.

---

# Chapter 2

# Experimental Philosophy

The primary objective is not merely to obtain the lowest prediction error.

Instead,

the objective is to demonstrate that adaptive decision making provides measurable advantages over conventional static pipelines.

Every experiment should therefore compare

Static Pipeline

↓

Adaptive Pipeline

The adaptive framework must justify its additional complexity through measurable improvements.

---

# Chapter 3

# Experimental Stages

Every experiment follows the same sequence.

Dataset Validation

↓

Dataset Splitting

↓

Segmentation Training

↓

Segmentation Evaluation

↓

Prediction Training

↓

Prediction Evaluation

↓

Intermediate Prediction Generation

↓

Decision Module Training

↓

Adaptive Evaluation

↓

Statistical Analysis

↓

Publication Figures

↓

Experiment Archiving

No stages should be skipped.

---

# Chapter 4

# Dataset Usage

All experiments should use identical dataset versions.

Every experiment must record

Dataset Version

Split Version

Configuration Version

Framework Version

Random Seed

This ensures complete reproducibility.

No experiment should mix multiple dataset versions.

---

# Chapter 5

# Data Splitting Strategy

The default split is

Training

80%

Validation

10%

Testing

10%

Splitting must always occur

at the patient level.

Images belonging to one patient must never appear in multiple splits.

This rule prevents data leakage.

Alternative evaluation strategies may include

Five-Fold Cross Validation

External Validation

Leave-One-Center-Out

These strategies should be configurable.

---

# Chapter 6

# Training Philosophy

Software development and model training are independent activities.

Only after the framework has been completely implemented should model training begin.

Training should occur exclusively through the experiment notebook.

Training should never occur during software development.

---

# Chapter 7

# Segmentation Training

Segmentation models should be trained independently.

Initially supported models include

UNet

SegFormer

DeepLabV3+

Each segmentation model should

Train

↓

Validate

↓

Evaluate

↓

Save

↓

Register

↓

Generate Reports

No segmentation model should overwrite another.

Every trained version remains available.

---

# Chapter 8

# Segmentation Evaluation

Each segmentation model should be evaluated independently.

Recommended metrics include

Dice Score

Intersection over Union

Precision

Recall

F1 Score

Pixel Accuracy

Boundary Accuracy

Inference Time

GPU Memory

Model Size

Evaluation should generate

Tables

Figures

Example Segmentations

Failure Cases

Comparison Reports

---

# Chapter 9

# Hemoglobin Prediction Training

Prediction models should be trained independently for each tissue.

Supported tissues include

Eye

Palm

Tongue

Nail

Future tissues should integrate without architectural modification.

Each prediction model should

Train

↓

Validate

↓

Evaluate

↓

Register

↓

Archive

Training history should be preserved for every version.

---

# Chapter 10

# Prediction Evaluation

Regression evaluation should include

MAE

RMSE

R²

Pearson Correlation

Spearman Correlation

Mean Bias

Standard Deviation

Clinical evaluation should include

Bland–Altman Analysis

Limits of Agreement

Prediction Error Distribution

Confidence Calibration

Evaluation reports should be generated automatically.

---

# Chapter 11

# Intermediate Prediction Generation

Once all segmentation and prediction models have been trained,

the framework should generate intermediate prediction files.

These files become the training data for the Adaptive Decision Framework.

Intermediate predictions should include

Patient ID

Tissue

Predicted Hb

Ground Truth Hb

Prediction Error

Image Quality

ROI Quality

Model Confidence

Metadata

These intermediate outputs should be permanently stored.

They should never require regeneration unless underlying models change.

---

# Chapter 12

# Adaptive Decision Module Training

Adaptive Decision Modules should never train directly from raw images.

Instead,

they learn from

Intermediate Predictions

↓

Quality Metrics

↓

ROI Metrics

↓

Prediction Confidence

↓

Metadata

↓

Ground Truth

This significantly reduces computational cost.

Each adaptive module should be trained independently.

After training,

every module should be

Validated

↓

Registered

↓

Versioned

↓

Archived

No adaptive module should overwrite previous versions.
# Chapter 13

# Baseline Framework

## Purpose

The adaptive framework must always be compared against a conventional non-adaptive pipeline.

This baseline serves as the primary reference for evaluating the benefits of adaptive decision making.

The baseline pipeline should execute using fixed computational pathways.

No adaptive decisions should be made.

---

## Baseline Workflow

Input Images

↓

Segmentation

↓

ROI Extraction

↓

Fixed Hb Prediction Model

↓

Static Fusion

↓

Final Hb Prediction

Every patient should follow the same computational pathway.

---

## Baseline Evaluation

The baseline should be evaluated using

• MAE

• RMSE

• R²

• Pearson Correlation

• Bland–Altman Analysis

• Inference Time

• Memory Usage

These results become the reference against which the adaptive framework is compared.

---

# Chapter 14

# Adaptive Framework Evaluation

After the baseline has been evaluated,

the Adaptive Decision Framework should be evaluated using the same testing dataset.

The adaptive framework should dynamically determine

• Image Quality

• Segmentation Model

• Tissue Selection

• Prediction Model

• Fusion Strategy

• Confidence

Both pipelines must use

the same dataset

the same test split

the same evaluation metrics.

---

## Adaptive Workflow

Input Images

↓

Quality Assessment

↓

Segmentation Selection

↓

Segmentation

↓

ROI Verification

↓

Tissue Selection

↓

Prediction Routing

↓

Hb Prediction

↓

Adaptive Fusion

↓

Confidence Estimation

↓

Final Hb Prediction

---

## Performance Comparison

The adaptive framework should demonstrate improvements in

Prediction Accuracy

Prediction Robustness

Model Reliability

Generalization

Computational Efficiency

Clinical Confidence

Whenever improvements are not observed,

the reasons should be investigated and documented.

---

# Chapter 15

# Ablation Studies

## Purpose

Ablation studies quantify the contribution of each adaptive module.

Each experiment removes exactly one Decision Module while keeping the remainder of the pipeline unchanged.

---

## Example Experiments

Complete Framework

↓

Without Quality Module

↓

Without Tissue Selection

↓

Without Prediction Routing

↓

Without Fusion Module

↓

Without Confidence Module

↓

Static Pipeline

The performance of every configuration should be reported.

---

## Evaluation

Metrics should include

MAE

RMSE

R²

Inference Time

Number of Selected Tissues

Calibration Error

Confidence Accuracy

Each ablation experiment should explain why performance changed.

---

# Chapter 16

# Hyperparameter Strategy

Hyperparameters should be documented for every experiment.

Examples include

Learning Rate

Batch Size

Epochs

Optimizer

Scheduler

Loss Function

Input Resolution

Weight Decay

Early Stopping

Random Seed

No undocumented hyperparameter changes should occur.

Every experiment should save its configuration automatically.

---

# Chapter 17

# Statistical Analysis

Scientific conclusions should rely on statistical evidence rather than isolated performance values.

Recommended analyses include

Mean

Median

Standard Deviation

95% Confidence Interval

Paired Statistical Tests

Bootstrap Confidence Intervals

Correlation Analysis

Error Distribution Analysis

Whenever multiple runs are performed,

results should be reported as

Mean ± Standard Deviation.

---

# Chapter 18

# Reproducibility

Every experiment should be completely reproducible.

The framework should automatically record

Dataset Version

Configuration Version

Framework Version

Model Versions

Random Seed

Experiment ID

Hardware Information

Training Time

Software Environment

Checkpoint Versions

Anyone using the same dataset and configuration should be able to reproduce the reported results.

---

# Chapter 19

# Automatic Report Generation

After every experiment,

the framework should automatically generate

Experiment Summary

Training Summary

Evaluation Report

Model Comparison

Publication Tables

Publication Figures

Prediction Files

Configuration Snapshot

Execution Logs

These reports should require no manual editing before analysis.

---

# Chapter 20

# Publication Figures

The framework should automatically generate publication-quality figures.

Examples include

Training Curves

Validation Curves

Loss Curves

Scatter Plots

Bland–Altman Plots

Residual Distributions

Prediction Histograms

Model Comparison Charts

Ablation Charts

Tissue Usage Charts

Confidence Distribution

Every figure should be saved in both PNG and PDF formats.

---

# Chapter 21

# Experiment Archive

Every experiment should be archived automatically.

Each archive should contain

Configuration

Logs

Checkpoints

Metrics

Figures

Predictions

Reports

Registry Snapshot

Pipeline State

Execution Summary

No experiment should overwrite a previous experiment.

---

# Chapter 22

# Acceptance Criteria

The Experiment Specification is complete when

✓ Every experiment is reproducible.

✓ Baseline and adaptive pipelines are compared using identical conditions.

✓ Ablation studies quantify the contribution of every Decision Module.

✓ Statistical analyses support reported conclusions.

✓ Publication-ready figures and tables are generated automatically.

✓ Every experiment is archived for future reference.

---

# End of Experiment Specification

This document defines the scientific methodology used to evaluate the Adaptive Multi-Agent Framework.

Every future publication should follow the experimental principles and reporting standards defined in this specification.