"""Evaluation report container with CSV/JSON export (EXPERIMENT_SPEC Ch.19)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptivehb.core.utils import ensure_dir, write_json


@dataclass
class EvaluationReport:
    """Metrics plus per-sample predictions for one evaluated model/pipeline."""

    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report (metrics + sample count)."""
        return {"name": self.name, "metrics": self.metrics, "num_samples": len(self.per_sample)}

    def export_json(self, path: str | Path) -> Path:
        """Write the metrics (and metadata) to a JSON file."""
        return write_json(path, self.to_dict())

    def export_csv(self, path: str | Path) -> Path:
        """Write the per-sample rows to a CSV file (no-op columns if empty)."""
        destination = Path(path)
        ensure_dir(destination.parent)
        fieldnames = list(self.per_sample[0].keys()) if self.per_sample else ["patient_id", "true_hb", "predicted_hb"]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.per_sample)
        return destination


__all__ = ["EvaluationReport"]
