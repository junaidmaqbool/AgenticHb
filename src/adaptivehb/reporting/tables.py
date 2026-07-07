"""Publication table export (CSV always; Excel guarded).

Flattens metric bundles into rows and writes them as CSV (standard library) and,
when ``openpyxl`` is installed, as an Excel workbook. The module imports cleanly
without ``openpyxl`` (Decision 027).
"""

from __future__ import annotations

import csv
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adaptivehb.core.utils import ensure_dir
from adaptivehb.exceptions import ReportingError


def excel_available() -> bool:
    """Whether an Excel writer (openpyxl) is installed."""
    return importlib.util.find_spec("openpyxl") is not None


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Flatten a (possibly nested) metric mapping into ``metric``/``value`` rows.

    Nested mappings are joined with dots; non-scalar leaves are stringified so the
    table always has a flat, exportable shape.
    """
    rows: list[dict[str, Any]] = []
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            rows.extend(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, (int, float)):
            rows.append({"metric": name, "value": float(value)})
        else:
            rows.append({"metric": name, "value": str(value)})
    return rows


def export_table_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write rows to a CSV file (columns inferred from the first row)."""
    destination = Path(path)
    ensure_dir(destination.parent)
    fieldnames = list(rows[0].keys()) if rows else ["metric", "value"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def export_table_excel(rows: list[dict[str, Any]], path: str | Path, sheet: str = "metrics") -> Path:
    """Write rows to an Excel workbook (requires openpyxl).

    Raises:
        ReportingError: If openpyxl is not installed.
    """
    if not excel_available():
        raise ReportingError("Excel export requires 'openpyxl' (install the reporting extra).")
    from openpyxl import Workbook  # pragma: no cover - requires openpyxl

    destination = Path(path)
    ensure_dir(destination.parent)
    fieldnames = list(rows[0].keys()) if rows else ["metric", "value"]
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    worksheet.append(fieldnames)
    for row in rows:
        worksheet.append([row.get(col) for col in fieldnames])
    workbook.save(destination)
    return destination


__all__ = ["flatten_metrics", "export_table_csv", "export_table_excel", "excel_available"]
