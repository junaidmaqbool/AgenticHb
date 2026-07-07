# MODEL_REGISTRY_SPEC.md

Version: 1.0

Status: Living Technical Specification

Last Updated: July 2026

Project Name

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Chapter 1

# Purpose

The Model Registry is the central catalogue of every trainable component used by the framework.

Instead of manually loading checkpoint files, every module should obtain models through the Registry.

The Registry acts as the single source of truth for

• Segmentation Models

• Hb Prediction Models

• Adaptive Decision Modules

• Fusion Models

• Confidence Models

No model should ever be loaded directly from a checkpoint path.

All model loading should occur through the Registry Manager.

---

# Chapter 2

# Registry Philosophy

The registry separates

Model Identity

from

Model Storage.

Users should never need to remember

checkpoint names

folder locations

version numbers

training dates

Instead,

they simply request

"The best Eye Hb model"

or

"The latest SegFormer"

The Registry determines which checkpoint to load.

---

# Chapter 3

# Registry Architecture

The Registry consists of

Registry Manager

↓

Registry Database

↓

Checkpoint Storage

↓

Configuration Snapshots

↓

Performance History

↓

Model Loader

The Registry Manager provides the only interface for model discovery.

---

# Chapter 4

# Registry Responsibilities

The Registry should

Register new models

Store model metadata

Track versions

Track performance

Store configuration

Store training history

Locate checkpoints

Load checkpoints

Retire outdated models

Generate registry reports

The Registry should never perform model training.

---

# Chapter 5

# Registry Categories

Separate registries should exist for

Segmentation Models

Prediction Models

Decision Modules

Fusion Models

Confidence Models

Each category should remain independent.

---

# Chapter 6

# Model Identity

Every registered model should possess a unique identity.

Recommended format

Model Category

↓

Model Name

↓

Version

↓

Unique ID

Example

HB_EYE_VIT_V001

SEG_UNET_V003

FUSION_WEIGHTED_V002

QUALITY_AGENT_V001

These identifiers should never change.

---

# Chapter 7

# Registry Information

Every registered model should contain

Unique ID

Model Name

Task

Category

Architecture

Version

Author

Training Date

Dataset Version

Configuration Version

Framework Version

Checkpoint Location

Input Resolution

Training Status

Current Status

Performance Metrics

Inference Speed

Model Size

Hardware Used

Random Seed

Training Time

Validation Results

Testing Results

Deployment Status

Every field should remain searchable.

---

# Chapter 8

# Model Categories

Supported categories include

Segmentation

↓

Prediction

↓

Decision Module

↓

Fusion

↓

Confidence

↓

Future Categories

The Registry should allow additional categories without changing source code.

---

# Chapter 9

# Model Versioning

Every successful training run creates

a new version.

Versions should never overwrite previous versions.

Example

Eye ViT

↓

Version 1

↓

Version 2

↓

Version 3

↓

Version 4

Older versions remain available.

The Registry decides which version is considered

Best

Latest

Stable

Experimental

Deprecated

---

# Chapter 10

# Registry Status

Every model should possess one status.

Possible values

Training

Validation

Testing

Stable

Experimental

Production

Archived

Deprecated

Failed

Only Stable and Production models should be used for deployment.

---

# Chapter 11

# Registry Storage

The Registry should never store model weights.

Instead,

it stores references.

Example

Registry

↓

Checkpoint Path

↓

Configuration

↓

Metrics

↓

Metadata

↓

Training History

Actual model weights remain inside

weights/

or

checkpoints/

directories.

---

# Chapter 12

# Registry Interfaces

The Registry Manager should expose

register()

update()

remove()

search()

load()

list()

compare()

history()

report()

backup()

restore()

No other component should directly modify the Registry.

All interactions must occur through the Registry Manager.
# Chapter 13

# Model Registration Workflow

## Purpose

Every trainable component should automatically register itself immediately after successful training.

Manual registration should never be required.

The registration workflow should execute automatically after

• Successful Training

• Successful Validation

• Model Saving

• Metric Generation

Only successfully completed models should be registered.

Failed or interrupted training runs should not create production registry entries.

---

## Registration Workflow

Training Completed

↓

Evaluate Model

↓

Generate Metrics

↓

Save Checkpoint

↓

Save Configuration

↓

Generate Model Metadata

↓

Register Model

↓

Update Registry

↓

Generate Registry Report

Every successful registration should receive a unique Registry ID.

---

# Chapter 14

# Model Discovery

The Registry should support intelligent model discovery.

Examples

Find Best Eye Model

Find Latest Palm Model

Find Stable Segmentation Model

Find Experimental Fusion Model

Find Production Confidence Module

Search results should be filterable by

Task

Architecture

Version

Dataset

Performance

Date

Status

Author

Framework Version

---

# Chapter 15

# Automatic Model Selection

The Pipeline should never manually choose checkpoints.

Instead,

the Registry Manager should automatically determine which model satisfies the request.

Example

Request

↓

Eye Hb Prediction

↓

Registry Search

↓

Stable Models

↓

Best Validation MAE

↓

Load Checkpoint

↓

Return Model

The Pipeline should remain completely independent of checkpoint filenames.

---

# Chapter 16

# Registry Reports

The Registry should automatically generate reports.

Examples

Registered Models

Model Performance Summary

Version History

Training Timeline

Deployment Status

Deprecated Models

Experimental Models

Current Production Models

Reports should be exported as

CSV

Excel

JSON

Markdown

---

# Chapter 17

# Registry History

The Registry should preserve complete historical information.

Each model should maintain

Version History

Training History

Performance History

Configuration History

Deployment History

No historical information should be deleted automatically.

Older versions may be archived but should remain accessible.

---

# Chapter 18

# Registry Backup

The Registry should support automatic backup.

Backups should include

Registry Database

Configuration Snapshots

Experiment References

Model Metadata

Training History

Version History

Backups should execute automatically after every successful registration.

---

# Chapter 19

# Registry Recovery

If the Registry becomes corrupted,

the framework should automatically

Locate Latest Backup

↓

Restore Registry

↓

Validate Registry

↓

Resume Normal Operation

Registry recovery should never modify model checkpoints.

---

# Chapter 20

# Registry Validation

Before loading any model,

the Registry should verify

Checkpoint Exists

Configuration Exists

Metrics Available

Compatible Framework Version

Compatible Input Resolution

Compatible Task

If validation fails,

the Registry should reject loading and generate an informative error.

---

# Chapter 21

# Registry Communication

Only the Registry Manager may communicate directly with the Registry.

The following managers interact through the Registry Manager.

Segmentation Manager

Prediction Manager

Agent Manager

Training Manager

Evaluation Manager

Deployment Manager

Pipeline Manager

Direct modification of registry files is prohibited.

---

# Chapter 22

# Registry Search API

The Registry Manager should expose high-level search functions.

Examples

load_best_model()

load_latest_model()

load_production_model()

load_experimental_model()

find_models()

compare_versions()

get_history()

get_metrics()

list_models()

No module should perform manual registry queries.

---

# Chapter 23

# Registry Security

Every registry modification should be logged.

Logs should include

Timestamp

Operation

Model ID

Version

User

Experiment ID

Framework Version

This ensures complete traceability.

---

# Chapter 24

# Registry Integration

The Registry interacts with

Dataset Manager

Training Manager

Checkpoint Manager

Pipeline Manager

Evaluation Manager

Deployment Manager

State Manager

Experiment Manager

The Registry acts as the central information hub connecting all trainable components.

---

# Chapter 25

# Registry Acceptance Criteria

The Model Registry is considered complete when

✓ Every trained model registers automatically.

✓ No manual checkpoint loading is required.

✓ Model versions are tracked automatically.

✓ Historical versions remain available.

✓ Registry backups are generated automatically.

✓ Registry recovery functions correctly.

✓ Registry searches return the correct models.

✓ Deployment loads only registry-approved models.

✓ Every model is completely reproducible.

---

# End of Model Registry Specification

The Model Registry provides the central management system for every trainable component within the Adaptive Multi-Agent Framework.

Every future model, regardless of task or architecture, should integrate with the Registry through the interfaces defined in this document.