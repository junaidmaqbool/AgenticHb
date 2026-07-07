# Notebooks

Runnable Jupyter notebooks that drive the AdaptiveHb framework end-to-end. Both
locate the repository root automatically and add `src/` to `sys.path`, so they
work whether or not you `pip install` the package — just run them from inside a
clone of this repository.

## `smoke_synthetic.ipynb` — verify your setup (no GPU / no PyTorch)

Generates a small synthetic, spec-conformant dataset and runs a full
baseline-vs-adaptive experiment on the built-in **reference models**, then shows
the comparison (with paired significance), the reproducibility provenance
manifest, and the generated figures. It has no heavy dependencies and runs
anywhere Python 3.11+ is available. Use it first to confirm the checkout is
healthy.

> With the torch-free reference models every tissue returns a constant
> prediction, so `baseline == adaptive` here (improvement 0.0). That is
> expected — this notebook proves the machinery runs and archives correctly.

## `train_pipeline.ipynb` — the real experiment (PyTorch backbones)

The notebook referenced by `PROJECT_MANIFEST.yaml`. Installs the optional ML
stack (`pip install -e ".[ml]"`), lets you point at your dataset (or generate a
synthetic one), trains the real PyTorch backbones, and archives the metrics, the
comparison with paired significance, the provenance manifest, and the publication
figures. A GPU is recommended but not required.

Equivalent one-liner from a terminal:

```bash
adaptivehb experiment --dataset-root <DATASET_ROOT> --base-dir runs --epochs 10 --name real_run
```

## Requirements

- `smoke_synthetic.ipynb`: Python 3.11+, `jupyter` (to open the notebook). The
  framework's core has no third-party runtime requirement beyond PyYAML; figures
  use matplotlib when present.
- `train_pipeline.ipynb`: additionally the `ml` extra (`pip install -e ".[ml]"`),
  which includes PyTorch, torchvision, numpy, and matplotlib.
