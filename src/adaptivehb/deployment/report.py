"""Clinical report produced by the deployed inference service.

A human-readable, exportable summary of a single patient's non-invasive
hemoglobin estimate and the adaptive framework's decisions (PIPELINE_SPEC Ch.17).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adaptivehb.core.utils import ensure_dir, utcnow_iso, write_json


@dataclass
class ClinicalReport:
    """A patient-level hemoglobin estimation report."""

    patient_id: str
    predicted_hb: float | None = None
    confidence: float | None = None
    interval: float | None = None
    recommendation: str | None = None
    selected_tissues: list[str] = field(default_factory=list)
    prediction_routing: dict[str, str] = field(default_factory=dict)
    fusion_weights: dict[str, float] = field(default_factory=dict)
    generated_at: str = field(default_factory=utcnow_iso)

    @classmethod
    def from_workflow(
        cls, patient_id: str, result: Any, *, include_confidence: bool = True
    ) -> ClinicalReport:
        """Build a report from a :class:`WorkflowResult`."""
        context = result.context
        return cls(
            patient_id=patient_id,
            predicted_hb=result.final_hb,
            confidence=result.confidence if include_confidence else None,
            interval=context.get("interval") if include_confidence else None,
            recommendation=result.recommendation,
            selected_tissues=list(context.get("selected_tissues", [])),
            prediction_routing=dict(context.get("prediction_routing", {})),
            fusion_weights=dict(context.get("fusion_weights", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dictionary."""
        return asdict(self)

    def to_text(self) -> str:
        """Render a human-readable clinical report."""
        hb = f"{self.predicted_hb:.2f} g/dL" if self.predicted_hb is not None else "N/A"
        lines = [
            "Non-Invasive Hemoglobin Estimation Report",
            "=========================================",
            f"Patient ID     : {self.patient_id}",
            f"Estimated Hb   : {hb}",
        ]
        if self.confidence is not None:
            band = f" (+/- {self.interval:.2f} g/dL)" if self.interval is not None else ""
            lines.append(f"Confidence     : {self.confidence:.1%}{band}")
        if self.recommendation:
            lines.append(f"Recommendation : {self.recommendation}")
        if self.selected_tissues:
            lines.append(f"Tissues used   : {', '.join(self.selected_tissues)}")
        lines.append(f"Generated at   : {self.generated_at}")
        return "\n".join(lines)

    def export_json(self, path: str | Path) -> Path:
        """Write the report as JSON."""
        return write_json(path, self.to_dict())

    def export_text(self, path: str | Path) -> Path:
        """Write the human-readable report as a text file."""
        destination = Path(path)
        ensure_dir(destination.parent)
        destination.write_text(self.to_text(), encoding="utf-8")
        return destination


__all__ = ["ClinicalReport"]
