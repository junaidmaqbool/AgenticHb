"""Metadata loading for the dataset subsystem.

The metadata CSV is the central reference for the dataset (DATASET_SPEC Ch.9).
Loading uses only the standard library so the dataset core stays dependency-light
and testable without pandas; a pandas-backed view can be added later behind the
same interface.
"""

from __future__ import annotations

import csv
from pathlib import Path

from adaptivehb.exceptions import DatasetError


class MetadataTable:
    """An in-memory view of the patient metadata CSV."""

    def __init__(self, rows: list[dict[str, str]], columns: list[str], id_column: str) -> None:
        """Initialize from parsed rows.

        Args:
            rows: Metadata rows as dictionaries.
            columns: Column names in file order.
            id_column: Name of the patient-identifier column.
        """
        self._rows = rows
        self._columns = columns
        self._id_column = id_column
        self._by_id: dict[str, dict[str, str]] = {}
        for row in rows:
            pid = (row.get(id_column) or "").strip()
            if pid and pid not in self._by_id:
                self._by_id[pid] = row

    @classmethod
    def load(cls, path: str | Path, id_column: str) -> MetadataTable:
        """Load a metadata CSV from disk.

        Args:
            path: Path to the CSV file.
            id_column: Name of the patient-identifier column.

        Returns:
            A populated :class:`MetadataTable`.

        Raises:
            DatasetError: If the file is missing or empty.
        """
        csv_path = Path(path)
        if not csv_path.is_file():
            raise DatasetError(f"Metadata file not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        if not columns:
            raise DatasetError(f"Metadata file has no header: {csv_path}")
        return cls(rows, columns, id_column)

    @property
    def columns(self) -> list[str]:
        """Column names in file order."""
        return list(self._columns)

    @property
    def rows(self) -> list[dict[str, str]]:
        """All metadata rows."""
        return list(self._rows)

    @property
    def patient_ids(self) -> list[str]:
        """Unique patient identifiers, in first-seen order."""
        return list(self._by_id.keys())

    def has_columns(self, names: list[str] | tuple[str, ...]) -> list[str]:
        """Return the subset of ``names`` that are missing from the table."""
        return [name for name in names if name not in self._columns]

    def get(self, patient_id: str) -> dict[str, str] | None:
        """Return the metadata row for a patient, or ``None`` if absent."""
        return self._by_id.get(patient_id)

    def __len__(self) -> int:
        """Number of rows in the raw table (including any duplicates)."""
        return len(self._rows)


__all__ = ["MetadataTable"]
