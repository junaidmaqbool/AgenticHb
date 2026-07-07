# Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

**Package:** `adaptivehb` · **Version:** 0.2.0 · **Status:** Active development (Phase 1 — Repository Construction) · **License:** Apache-2.0

A modular, configuration-driven, reproducible research framework for estimating
hemoglobin non-invasively from images of exposed tissues (eye, palm, tongue, nail).
Its scientific core is an **adaptive multi-agent decision framework** that dynamically
selects segmentation models, tissues, prediction models, and fusion strategies per
patient, rather than executing a fixed pipeline.

This repository is intended to become a reusable, open-source biomedical AI platform
supporting multiple peer-reviewed publications. It is developed as a software product,
not a collection of scripts.

## Design principles

- **Manager-mediated:** a single public entry point (`HbPipeline`) coordinates
  single-responsibility managers; modules never call each other directly.
- **Registry-first:** models are loaded by query from a Model Registry, never from
  hardcoded checkpoint paths.
- **Configuration-driven:** every parameter originates from `configs/*.yaml`; no
  dataset names, paths, or image sizes are hardcoded in source.
- **Reproducible & resumable:** seeds, configs, and state are captured; training
  resumes from checkpoints after interruption.
- **Framework first, experiments later:** the framework is built and tested on dummy
  data before any real model is trained.

## Repository layout

```
AgenticHb/
├── configs/            # Nine YAML config files (project, dataset, ... , logging)
├── src/adaptivehb/     # Installable Python package
│   ├── config/         # Typed configuration loading + validation
│   └── logging/        # Logging subsystem
├── tests/              # Unit / smoke tests
├── docs/               # Specifications and living project documents
├── notebooks/          # Experiment launchers (no ML code lives here)
├── checkpoints/  weights/  registry/  logs/  results/
├── figures/  reports/  tensorboard/  cache/  experiments/  outputs/
```

Runtime output directories are tracked (via `.gitkeep`) but their contents are ignored.

## Installation (development)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -e ".[dev]"        # Phase 1 infrastructure + dev tools
pip install -e ".[dev,ml]"     # add the ML stack (needed from Phase 3 onward)
```

The Phase 1 infrastructure (config, logging, exceptions) depends only on `PyYAML`
and is fully importable and testable **without** a GPU or PyTorch.

## Quick check

```python
import adaptivehb
from adaptivehb.config import ConfigLoader
from adaptivehb.logging import setup_logging

config = ConfigLoader("configs").load()
setup_logging(config.logging)
print(adaptivehb.__version__, config.project.name)
```

## Running tests

```bash
pytest
```

## Version control

Initialize git natively on your machine (the development sandbox cannot manage git
metadata on this mount):

```bash
git init
git add .
git commit -m "Phase 1: repository construction"
```

## Documentation

Start with `START_HERE.md`, then `docs/PROJECT_INDEX.md`. The governing documents are
`docs/PROJECT_CHARTER.md`, `docs/PROJECT_DESIGN_SPECIFICATION.md`, and
`docs/IMPLEMENTATION_ROADMAP.md`. Current status lives in `docs/PROJECT_STATE.md`.
