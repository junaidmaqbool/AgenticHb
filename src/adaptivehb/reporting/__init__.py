"""Reporting subsystem: publication figures and tables.

Generates evaluation figures (scatter, Bland-Altman, residuals, training curves,
model comparison) via matplotlib and exports metric tables as CSV/Excel. All
plotting/Excel dependencies are optional and guarded (Decision 027).
"""

from adaptivehb.reporting.figures import FigureGenerator, figures_available
from adaptivehb.reporting.manager import ReportingManager
from adaptivehb.reporting.tables import (
    excel_available,
    export_table_csv,
    export_table_excel,
    flatten_metrics,
)

__all__ = [
    "ReportingManager",
    "FigureGenerator",
    "figures_available",
    "flatten_metrics",
    "export_table_csv",
    "export_table_excel",
    "excel_available",
]
