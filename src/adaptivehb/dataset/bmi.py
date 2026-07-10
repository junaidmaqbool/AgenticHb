"""Algorithmic BMI derivation for the metadata table.

BMI (Body Mass Index) is frequently absent or blank in clinical spreadsheets even
when height and weight are recorded. This module derives it deterministically from
those two columns so downstream models can rely on the feature being present.

The computation is a pure function over already-parsed metadata rows; column names
and the height unit come from configuration (:class:`~adaptivehb.dataset.config.MetadataSpec`)
so nothing is hardcoded. Existing non-blank BMI values are never overwritten.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Below this value a height is assumed to already be in metres (e.g. 1.75);
# at or above it, in centimetres (e.g. 175). Used only when the unit is unknown.
_METRE_THRESHOLD = 3.0


def _to_float(value: object) -> float | None:
    """Parse a float from a possibly-blank cell, returning ``None`` on failure."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _height_metres(raw_height: float, unit: str) -> float | None:
    """Convert a raw height value to metres given the configured unit."""
    if raw_height <= 0:
        return None
    unit = (unit or "cm").lower()
    if unit == "m":
        return raw_height
    if unit == "cm":
        return raw_height / 100.0
    # Unknown unit: infer from magnitude (heights >= 3 are almost certainly cm).
    return raw_height / 100.0 if raw_height >= _METRE_THRESHOLD else raw_height


def compute_bmi(weight_kg: float, height: float, *, height_unit: str = "cm") -> float | None:
    """Return BMI (kg/m^2) from weight in kg and height in the given unit.

    Returns ``None`` when the inputs are non-positive or otherwise unusable.
    """
    height_m = _height_metres(height, height_unit)
    if height_m is None or height_m <= 0 or weight_kg <= 0:
        return None
    return weight_kg / (height_m * height_m)


def add_bmi(
    rows: Iterable[Mapping[str, str]],
    *,
    height_column: str,
    weight_column: str,
    bmi_column: str,
    height_unit: str = "cm",
    ndigits: int = 1,
) -> int:
    """Fill in a blank/absent BMI column in-place from height and weight.

    Args:
        rows: Mutable metadata rows (dictionaries) to update in place.
        height_column: Name of the height column.
        weight_column: Name of the weight column.
        bmi_column: Name of the BMI column to populate.
        height_unit: Unit of the height column (``cm`` or ``m``).
        ndigits: Rounding precision for the derived BMI.

    Returns:
        The number of rows whose BMI value was newly computed.
    """
    filled = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        existing = (row.get(bmi_column) or "").strip() if bmi_column in row else ""
        if existing:
            continue  # never overwrite a recorded value
        weight = _to_float(row.get(weight_column))
        height = _to_float(row.get(height_column))
        if weight is None or height is None:
            continue
        bmi = compute_bmi(weight, height, height_unit=height_unit)
        if bmi is None:
            continue
        row[bmi_column] = str(round(bmi, ndigits))
        filled += 1
    return filled


__all__ = ["compute_bmi", "add_bmi"]
