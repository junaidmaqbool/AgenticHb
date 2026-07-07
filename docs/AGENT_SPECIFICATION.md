# AGENT_SPECIFICATION.md

Version: 1.0

Status: Living Technical Specification

Last Updated: July 2026

Project Name

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Chapter 1

# Purpose

This document defines the complete Adaptive Decision Framework used by the project.

Unlike conventional deep learning systems that execute a fixed computational pipeline, this framework introduces lightweight decision-making modules capable of dynamically selecting the most appropriate computational pathway for each patient.

These decision modules are referred to as Agents.

Each Agent has a clearly defined responsibility, documented inputs and outputs, standardized interfaces, and independent evaluation criteria.

The Adaptive Agent Framework represents the primary scientific contribution of this project.

---

# Chapter 2

# Agent Philosophy

Agents do not estimate hemoglobin directly.

Instead,

agents make intelligent computational decisions.

Traditional Pipeline

Image

↓

Segmentation

↓

Prediction

↓

Result

Adaptive Pipeline

Image

↓

Agent

↓

Decision

↓

Segmentation

↓

Agent

↓

Decision

↓

Prediction

↓

Agent

↓

Decision

↓

Fusion

↓

Final Result

Every decision should improve either

• prediction accuracy

• computational efficiency

• robustness

• interpretability

• reliability

The framework should remain lightweight and executable on consumer hardware.

---

# Chapter 3

# Adaptive Decision Hierarchy

The adaptive framework is organized into three logical layers.

Level 1

Perception Layer

Purpose

Understand image quality.

Verify segmentation.

Detect unusable inputs.

Modules

Quality Assessment

ROI Verification

-----------------------------------

Level 2

Decision Layer

Purpose

Determine the optimal computational pathway.

Modules

Segmentation Selection

Tissue Selection

Prediction Routing

-----------------------------------

Level 3

Clinical Output Layer

Purpose

Produce the most reliable final prediction.

Modules

Dynamic Fusion

Confidence Estimation

-----------------------------------

Workflow Controller

The Workflow Controller coordinates all agents.

The Workflow Controller is not itself an adaptive agent.

It simply orchestrates execution.

---

# Chapter 4

# General Agent Architecture

Every adaptive module follows the same architecture.

Input

↓

Feature Extraction

↓

Decision Logic

↓

Output

The decision logic may consist of

Rule-Based Logic

Decision Trees

Random Forest

Gradient Boosting

Lightweight Neural Networks

Reinforcement Learning

Future methods may be integrated without modifying the remaining framework.

Large Language Models are explicitly excluded from this project.

---

# Chapter 5

# Common Agent Interface

Every adaptive module should expose the same public interface.

initialize()

train()

predict()

evaluate()

save()

load()

reset()

shutdown()

This common interface allows every agent to be managed through the AgentManager.

---

# Chapter 6

# Agent Lifecycle

Every agent should pass through the following lifecycle.

Initialization

↓

Training

↓

Validation

↓

Registration

↓

Deployment

↓

Inference

↓

Evaluation

↓

Version Update

Every stage should automatically update the Model Registry.

---

# Chapter 7

# Agent Communication

Agents should never communicate directly.

Instead,

communication occurs through the Pipeline Manager.

Example

Quality Agent

↓

Quality Score

↓

Pipeline Manager

↓

Segmentation Selection Agent

↓

Decision

↓

Pipeline Manager

↓

Prediction Routing Agent

↓

Decision

↓

Pipeline Manager

↓

Fusion Agent

This architecture minimizes coupling and improves maintainability.

---

# Chapter 8

# Agent Inputs

Agents may receive

Images

Segmentation Masks

ROI Images

Quality Scores

Prediction Outputs

Metadata

Intermediate Features

Confidence Scores

Patient Metadata

Execution History

Each agent should explicitly declare the inputs it requires.

Unused information should never be passed.

---

# Chapter 9

# Agent Outputs

Agents should never return arbitrary data.

Outputs should follow standardized formats.

Possible outputs include

Quality Score

Segmentation Choice

ROI Accepted

Selected Tissue

Selected Prediction Model

Fusion Weights

Confidence Score

Uncertainty Estimate

Retry Recommendation

Every output should be interpretable.

---

# Chapter 10

# Agent Registry

Every trained adaptive module should automatically register itself.

The registry should contain

Agent Name

Version

Training Dataset

Training Date

Metrics

Configuration

Checkpoint

Status

Performance History

The Pipeline Manager should always load agents through the registry rather than manually loading checkpoint files.

---

# Chapter 11

# Agent Configuration

Every adaptive module should be configurable.

Configuration options include

Enabled

Training Mode

Inference Mode

Model Type

Hyperparameters

Checkpoint Location

Decision Thresholds

Logging

Evaluation

Configuration should originate exclusively from configuration files.

No parameters should be hardcoded.

---

# Chapter 12

# Agent Dependencies

Agents execute sequentially.

Each agent depends only upon outputs generated by previous stages.

Example

Quality Assessment

↓

ROI Verification

↓

Segmentation Selection

↓

Tissue Selection

↓

Prediction Routing

↓

Fusion

↓

Confidence

The dependency graph should remain acyclic.

No circular dependencies should exist.

---

# Chapter 13

# Agent Categories

Adaptive modules are divided into

Perception Agents

Decision Agents

Clinical Output Agents

Future adaptive modules should belong to one of these categories.

This classification simplifies future extensions and documentation.
# Chapter 14

# Quality Assessment Decision Module

## Purpose

The Quality Assessment Decision Module is the first adaptive module executed after image acquisition.

Its responsibility is to determine whether an acquired image is suitable for further processing.

Poor-quality images should not proceed through the pipeline.

Instead, the module should recommend reacquisition whenever image quality falls below predefined thresholds.

---

## Inputs

• Original Image

• Tissue Type

• Optional Acquisition Metadata

---

## Outputs

• Quality Score

• Image Accepted / Rejected

• Estimated Failure Reason

• Recommendation for Reacquisition

---

## Possible Quality Metrics

Blur

Brightness

Contrast

Exposure

Motion Artifacts

Occlusion

Reflection

Noise

ROI Visibility

Color Consistency

---

## Training Targets

Image Quality Labels

Blur Scores

Expert Ratings

Automatically Generated Quality Scores

---

## Success Criteria

High-quality images should proceed.

Low-quality images should trigger reacquisition.

---

# Chapter 15

# ROI Verification Decision Module

## Purpose

After segmentation, this module verifies whether the extracted Region of Interest is suitable for Hb estimation.

The segmentation network may occasionally produce incomplete or inaccurate masks.

This module prevents poor segmentations from propagating through the pipeline.

---

## Inputs

Original Image

Segmentation Mask

Segmented ROI

Segmentation Confidence

---

## Outputs

ROI Accepted

ROI Rejected

ROI Quality Score

Failure Reason

Retry Recommendation

---

## Evaluation

ROI IoU

Dice Score

Boundary Accuracy

Coverage Ratio

---

# Chapter 16

# Segmentation Selection Decision Module

## Purpose

The framework supports multiple segmentation models.

Rather than always executing the same model, this module determines which segmentation model should process the current image.

Initially supported models include

UNet

SegFormer

DeepLabV3+

Future models may be added without modifying the Pipeline Manager.

---

## Inputs

Original Image

Tissue Type

Image Quality Score

Optional Metadata

---

## Outputs

Selected Segmentation Model

Expected Confidence

Execution Priority

---

## Decision Criteria

Image Complexity

Image Quality

Tissue Type

Previous Performance

Expected Inference Time

---

## Training Targets

Best Performing Model

Segmentation Accuracy

Inference Cost

Confidence

---

# Chapter 17

# Tissue Selection Decision Module

## Purpose

Not every tissue contributes equally for every patient.

The Tissue Selection Decision Module determines which available tissues should be analyzed.

The module should maximize prediction quality while minimizing computational cost.

---

## Inputs

Available Tissues

Image Quality Scores

ROI Scores

Patient Metadata (Optional)

---

## Outputs

Selected Tissue List

Priority Order

Ignored Tissues

Selection Confidence

---

## Examples

Eye Only

Palm Only

Tongue Only

Eye + Tongue

Palm + Tongue

Eye + Palm + Tongue

All Available Tissues

---

## Optimization Objectives

Highest Accuracy

Lowest Computation

Maximum Robustness

---

# Chapter 18

# Prediction Routing Decision Module

## Purpose

Different tissues may perform best with different Hb prediction models.

The Prediction Routing Module selects the most appropriate prediction model for each tissue.

---

## Inputs

Selected Tissues

Image Quality

ROI Quality

Available Models

---

## Outputs

Prediction Model Assignment

Expected Confidence

Execution Order

---

## Example

Eye

↓

Vision Transformer

Palm

↓

EfficientNet

Tongue

↓

ConvNeXt

Future models should be selectable without modifying source code.

---

# Chapter 19

# Adaptive Fusion Decision Module

## Purpose

The Adaptive Fusion Module combines predictions from multiple tissues.

Unlike static averaging,

fusion should depend upon

Prediction Confidence

Image Quality

Tissue Reliability

Model Reliability

Patient Characteristics (optional)

---

## Inputs

Prediction Values

Prediction Confidence

Quality Scores

Metadata

---

## Outputs

Final Hb Prediction

Fusion Weights

Prediction Explanation

---

## Future Extensions

Weighted Averaging

Stacking

Mixture of Experts

Attention-Based Fusion

Bayesian Fusion

---

# Chapter 20

# Confidence Decision Module

## Purpose

Every Hb estimate should include a confidence estimate.

Confidence is critical for clinical decision support.

The module should estimate

Prediction Confidence

Prediction Uncertainty

Reliability

Calibration

---

## Outputs

Hb Estimate

Confidence Score

Confidence Interval

Uncertainty Estimate

Clinical Recommendation

---

## Example

Predicted Hb

11.8 g/dL

Confidence

96%

Estimated Error

±0.4 g/dL

Recommendation

Prediction Reliable

---

# Chapter 21

# Workflow Controller

The Workflow Controller coordinates all adaptive decision modules.

Unlike other modules,

it does not make clinical decisions.

Its responsibilities include

Initializing modules

Scheduling execution

Monitoring dependencies

Passing outputs

Handling failures

Updating pipeline state

Updating experiment logs

The Workflow Controller should remain deterministic.

Adaptive behavior belongs only to the Decision Modules.

---

# Chapter 22

# Adaptive Decision Flow

The adaptive framework should execute in the following order.

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

Clinical Report

Every stage should be independently replaceable.

---

# Chapter 23

# Decision Module Evaluation

Each module should be evaluated independently.

Metrics include

Decision Accuracy

Execution Time

Resource Usage

Contribution to Final Performance

Robustness

Generalization

Calibration

Ablation studies should measure the impact of removing each Decision Module.

---

# Chapter 24

# Acceptance Criteria

The Adaptive Decision Framework is complete when

✓ Every Decision Module implements a common interface.

✓ Modules remain independent.

✓ Workflow Controller coordinates execution.

✓ Every decision is logged.

✓ Every module is replaceable.

✓ Every module is configurable.

✓ Every module is independently trainable.

✓ Every module is independently evaluable.

✓ The framework demonstrates measurable improvement over a static pipeline.

---

# End of Agent Specification

This document defines the adaptive decision-making components of the framework.

Future adaptive modules should follow the same interfaces and architectural principles defined in this specification.