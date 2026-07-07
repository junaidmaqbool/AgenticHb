# PROJECT_CHARTER.md

Version: 1.0

Status: Living Document

Last Updated: July 2026

---

# Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

## Project Charter

---

# 1. Purpose of this Document

This document defines the philosophy, engineering standards, research objectives, software architecture principles, collaboration guidelines, and long-term vision for this project.

It serves as the governing document ("constitution") for every future development decision.

Whenever uncertainty exists regarding implementation, software architecture, research methodology, coding style, documentation, experimentation, deployment, or collaboration, this document takes precedence.

This document should evolve together with the project while maintaining backward compatibility wherever possible.

---

# 2. Project Vision

The objective of this project is to develop a publication-quality Artificial Intelligence framework capable of estimating hemoglobin non-invasively using images of exposed human tissues.

Unlike traditional deep learning systems that use fixed pipelines and static ensembles, this project aims to develop an adaptive decision-making framework capable of intelligently selecting the most appropriate computational pathway for every patient.

The framework should ultimately become:

• a reusable research framework

• an open-source software platform

• a deployable clinical prototype

• the basis for multiple peer-reviewed publications

The framework should be modular enough that future researchers can extend individual components without redesigning the complete system.

---

# 3. Long-Term Vision

The final framework should behave similarly to a clinical decision-support system.

Instead of executing one predefined sequence of operations, it should dynamically decide

• which segmentation model to use

• whether image quality is acceptable

• whether another image is required

• which tissue(s) should be analysed

• which prediction models should execute

• whether multiple models should be fused

• how much confidence should be assigned to the prediction

• when the system should stop collecting information

The framework should make intelligent computational decisions while remaining lightweight enough to execute on modest hardware.

The project should prioritize scientific validity over novelty.

Novel methods are valuable only when supported by reproducible experimental evidence.

---

# 4. Research Philosophy

This project is not intended to produce another CNN architecture.

It is not intended to produce another Vision Transformer.

It is not intended to produce another ensemble.

Instead, it seeks to introduce adaptive computational decision making into non-invasive hemoglobin estimation.

Every component added to the framework must answer one question:

"What new capability does this introduce that static pipelines cannot provide?"

If no meaningful answer exists, the component should not be added.

---

# 5. Scientific Objectives

The framework should contribute in several independent directions.

Possible scientific contributions include

• Universal multi-tissue segmentation

• Adaptive segmentation model selection

• Intelligent tissue routing

• Dynamic prediction model selection

• Adaptive fusion

• Confidence estimation

• Explainability

• Clinical decision support

• Lightweight deployment

Each contribution should be independently publishable whenever possible.

---

# 6. Engineering Philosophy

Software quality is equally important as model performance.

Poor software engineering produces irreproducible science.

Therefore every component should satisfy

• modularity

• readability

• reproducibility

• configurability

• maintainability

• extensibility

Every module should have one clear responsibility.

Avoid tightly coupled components.

Prefer composition over monolithic implementations.

---

# 7. Project Scope

The project consists of six major subsystems.

1.

Dataset Management

2.

Segmentation Framework

3.

Hemoglobin Prediction Framework

4.

Adaptive Decision Framework

5.

Evaluation Framework

6.

Deployment Framework

Each subsystem should remain independently usable.

No subsystem should depend unnecessarily on another.

---

# 8. Development Philosophy

This project should never be developed as one large script.

Instead, it should be treated as a software product.

Development occurs in two stages.

Stage A

Software Engineering

↓

Complete framework

↓

No model training

↓

Dummy tests

↓

Integration

↓

Documentation

Stage B

Experiments

↓

Real dataset

↓

Training

↓

Validation

↓

Testing

↓

Paper generation

Software architecture should always exist before experiments begin.

---

# 9. General Rules

The following principles are mandatory.

Never write prototype code.

Never hardcode file paths.

Never hardcode dataset names.

Never hardcode image dimensions.

Never hardcode model names.

Never hardcode checkpoints.

Never duplicate code.

Never bypass configuration files.

Never overwrite trained models.

Never delete checkpoints automatically.

Every important operation should be logged.

Every experiment should be reproducible.

---

# 10. Configuration Philosophy

Every configurable parameter should originate from configuration files.

The source code should contain almost no project-specific constants.

Configuration should include

dataset paths

training parameters

augmentation

hardware

model selection

checkpoint locations

output folders

deployment settings

This allows the same framework to operate on different datasets without changing source code.

---

# 11. Repository Philosophy

The repository represents a reusable research platform.

It should not become a collection of experiments.

Experiments belong inside notebooks.

Reusable functionality belongs inside Python modules.

The repository should always remain deployable.

A new developer should be able to clone the repository and understand its organization without reading the implementation.

---

# 12. Code Quality Standards

Every Python file should satisfy

PEP8

type hints

docstrings

logging

error handling

unit-test compatibility

modularity

avoid global variables

avoid duplicated logic

prefer reusable utilities

prefer object-oriented design where appropriate

Every function should perform one well-defined task.

---

# 13. Documentation Standards

Every module should contain

Purpose

Inputs

Outputs

Dependencies

Example usage

Limitations

Future improvements

Every public function should contain proper documentation.

No undocumented module should remain in the repository.

Documentation is considered part of the implementation.

---

# 14. Collaboration Rules

Claude acts as

Senior AI Research Engineer

Senior Software Architect

Machine Learning Engineer

Research Collaborator

Claude should never behave like a code generator.

Before implementation Claude should always think about architecture.

Claude should explain important design choices.

Claude should identify trade-offs.

Claude should recommend improvements whenever appropriate.

If uncertainty exists Claude should ask rather than assume.

If an implementation conflicts with previous decisions Claude should explain the conflict before writing code.

Backward compatibility should always be preserved whenever possible.
# 15. Model Registry Philosophy

Every trainable component in this framework must register itself after successful training.

No model should ever be loaded directly using a manually specified filename.

Instead, the framework should query the Model Registry.

The registry is the single source of truth for every trained model.

Each registered model should include

• Model Name

• Task

• Tissue

• Version

• Training Date

• Dataset Version

• Performance Metrics

• Training Configuration

• Input Resolution

• Checkpoint Location

• Current Status

Example

Eye_ViT

↓

Task

Hb Prediction

↓

Version

1.0

↓

Validation MAE

0.54

↓

Checkpoint

weights/hb/eye_vit/v1/

This allows future models to replace existing models without modifying any source code.

------------------------------------------------------------

# 16. Checkpoint Philosophy

Training should never be considered temporary.

Every important stage should create recoverable checkpoints.

Examples

Segmentation

Hb Prediction

Quality Agent

Routing Agent

Fusion Agent

Confidence Agent

Training should automatically save

• Latest checkpoint

• Best validation checkpoint

• Optimizer state

• Scheduler state

• Epoch number

• Training history

• Configuration

• Random seed

Power failure should never require restarting training from Epoch 1.

------------------------------------------------------------

# 17. State Manager

The framework must always know its current progress.

A pipeline state file should automatically record

Current Phase

Completed Tasks

Running Tasks

Pending Tasks

Errors

Last Checkpoint

Current Epoch

Current Fold

Experiment ID

When the framework starts, it should automatically detect existing progress and offer to continue from the previous state.

No manual intervention should be required.

------------------------------------------------------------

# 18. Training Philosophy

Software development and model training are separate activities.

The repository should first be completed using dummy data.

Only after the framework is stable should actual model training begin.

Training should occur through one execution notebook.

The notebook should not contain machine learning code.

Instead it should simply configure and launch the framework.

Example

Load Configuration

↓

Create Pipeline

↓

Execute Pipeline

↓

Monitor Progress

↓

Save Results

The notebook acts only as an experiment controller.

------------------------------------------------------------

# 19. Single Entry Point

The complete framework should execute from one primary notebook.

The user should only specify

Dataset Root

Metadata File

Save Directory

Hardware Configuration

Training Options

Everything else should be handled automatically.

The framework itself should discover

Models

Checkpoints

Previous Progress

Configurations

Registry Entries

Existing Results

------------------------------------------------------------

# 20. Dataset Philosophy

The framework should never assume a fixed dataset.

Instead the dataset should follow a documented specification.

The configuration should specify

Original Images

Segmentation Masks

Metadata

Patient Information

Target Labels

Optional Features

The framework should automatically validate the dataset before beginning training.

Any missing or inconsistent files should generate informative reports.

------------------------------------------------------------

# 21. Metadata Support

The framework should support structured patient metadata.

Examples include

Age

Gender

Height

Weight

BMI

Socioeconomic Status

Location

CBC Values

Hemoglobin

Additional metadata should be optional.

The framework should remain functional even when some metadata fields are unavailable.

------------------------------------------------------------

# 22. Experiment Philosophy

Every experiment should have a unique Experiment ID.

Each experiment should automatically save

Configuration

Logs

Metrics

Figures

Confusion Matrices

Regression Plots

Predictions

Checkpoints

TensorBoard Logs

Execution Time

Hardware Information

Nothing should be overwritten.

Every experiment should remain reproducible.

------------------------------------------------------------

# 23. Logging Standards

Every important action should be logged.

Examples

Loading Dataset

Training Started

Checkpoint Saved

Validation Completed

Training Resumed

Testing Started

Deployment Finished

Logs should be human readable.

Debug information should remain available when required.

------------------------------------------------------------

# 24. Version Control Philosophy

The repository should follow Git best practices.

Every major feature should be implemented independently.

Large changes should not affect unrelated modules.

Commit messages should clearly describe

Purpose

Reason

Impact

Future Work

------------------------------------------------------------

# 25. Testing Philosophy

Every module should be testable independently.

Examples

Dataset Loader

Segmentation

Prediction

Quality Agent

Routing Agent

Fusion Agent

Confidence Agent

Testing should occur before integration.

The complete pipeline should also support end-to-end testing.

------------------------------------------------------------

# 26. Evaluation Philosophy

Evaluation should compare

Individual Tissue Models

Static Ensemble

Weighted Ensemble

Dynamic Ensemble

Adaptive Framework

Metrics should include

MAE

RMSE

R²

Pearson Correlation

Spearman Correlation

Bland-Altman Analysis

Prediction within ±0.5 g/dL

Prediction within ±1.0 g/dL

Average Number of Tissues Used

Inference Time

Memory Usage

Model Size

Calibration Error

Confidence Accuracy

------------------------------------------------------------

# 27. Deployment Philosophy

Deployment should require no retraining.

The deployed application should simply

Load Registry

↓

Load Models

↓

Load Configuration

↓

Accept Images

↓

Predict

↓

Generate Report

Deployment targets include

Desktop

FastAPI

Gradio

Streamlit

Docker

HuggingFace Spaces

Future deployment should require minimal additional code.

------------------------------------------------------------

# 28. Publication Philosophy

The repository should support multiple publications.

Possible publication sequence

Paper 1

Universal Multi-Tissue Segmentation

Paper 2

Non-Invasive Hemoglobin Estimation

Paper 3

Adaptive Multi-Agent Decision Framework

Paper 4

Clinical Deployment Framework

Every module should therefore be sufficiently independent to support publication.

------------------------------------------------------------

# 29. Claude Collaboration Rules

Claude is a long-term collaborator.

Claude should

Think before coding.

Explain architectural decisions.

Preserve previous implementations.

Avoid unnecessary redesigns.

Ask questions whenever ambiguity exists.

Produce production-quality code.

Prefer modularity over convenience.

Never rush implementation.

At the beginning of every conversation Claude should

Summarize the current architecture.

Identify completed phases.

Identify pending phases.

Suggest the next logical milestone.

At the end of every coding session Claude should produce

Completed Work

Repository Tree

Files Created

Files Modified

Pending Tasks

Known Risks

Recommended Next Phase

Then stop.

Claude should never automatically continue to the next phase.

------------------------------------------------------------

# 30. Success Criteria

The project will be considered successful if

The framework is modular.

The framework is reproducible.

Training can resume after interruption.

Every model is independently reusable.

The adaptive framework improves over static baselines.

Deployment requires only loading saved models.

The repository can be understood without reading implementation code.

The software supports future research.

The framework becomes suitable for publication in high-impact biomedical AI journals.

------------------------------------------------------------

# 31. Things Never To Do

Never hardcode paths.

Never hardcode checkpoints.

Never hardcode model names.

Never duplicate code.

Never retrain during inference.

Never overwrite trained models.

Never delete experiment history.

Never break backward compatibility without discussion.

Never sacrifice maintainability for short-term convenience.

------------------------------------------------------------

# 32. Final Principle

This repository is not simply a collection of machine learning models.

It is intended to become a reusable biomedical AI platform.

Every design decision should answer one question:

"Will this make the framework easier to extend, easier to reproduce, and more valuable for future research?"

If the answer is no, the design should be reconsidered.

This principle should guide every future implementation throughout the lifetime of the project.