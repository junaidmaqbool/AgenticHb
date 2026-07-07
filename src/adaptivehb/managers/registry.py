"""RegistryManager — the single source of truth for trained models.

The registry stores *references* (metadata + checkpoint paths), never weights
(MODEL_REGISTRY_SPEC Ch.11). Models are discovered by query — never by hardcoded
checkpoint path. Each successful training run appends a new, immutable version;
previous versions are never overwritten or deleted automatically.

Backend: a single human-readable JSON document (Decision 014), with automatic
backups on every registration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.types import ModelCategory, ModelRecord, ModelStatus
from adaptivehb.core.utils import ensure_dir, read_json, timestamp_slug, write_json
from adaptivehb.exceptions import RegistryError

_CATEGORY_PREFIX: dict[ModelCategory, str] = {
    ModelCategory.SEGMENTATION: "SEG",
    ModelCategory.PREDICTION: "HB",
    ModelCategory.DECISION_MODULE: "DEC",
    ModelCategory.FUSION: "FUS",
    ModelCategory.CONFIDENCE: "CONF",
}

_DEPLOYABLE = frozenset({ModelStatus.STABLE, ModelStatus.PRODUCTION})


class RegistryManager(BaseManager):
    """Catalogue and discovery service for every trainable component."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the registry manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory; the registry lives under
                ``base_dir / project.paths.registry``.
        """
        super().__init__(config, base_dir)
        self._root = self._base_dir / config.project.paths.registry
        self._db_path = self._root / "registry.json"
        self._backup_dir = self._root / "backups"
        self._db: dict[str, Any] = {"version": 1, "categories": {}}

    # -- lifecycle ---------------------------------------------------------

    def _on_initialize(self) -> None:
        ensure_dir(self._root)
        if self._db_path.is_file():
            self._db = read_json(self._db_path)
            self._log.info("Registry loaded (%d model names).", self._count_names())
        else:
            self._persist()
            self._log.info("Registry created at %s.", self._db_path)

    # -- registration ------------------------------------------------------

    def register(self, record: ModelRecord) -> ModelRecord:
        """Register a new model version.

        The next version number and a stable ``unique_id`` are assigned
        automatically; the record is appended without overwriting prior
        versions, then the registry is persisted and backed up.

        Args:
            record: The model record to register.

        Returns:
            The stored record, with ``version`` and ``unique_id`` populated.

        Raises:
            RegistryError: If the registry has not been initialized.
        """
        self._require_initialized()
        versions = self._versions(record.category, record.name)
        record.version = len(versions) + 1
        prefix = _CATEGORY_PREFIX[record.category]
        record.unique_id = f"{prefix}_{record.name.upper()}_V{record.version:03d}"
        versions.append(record.to_dict())
        self._set_versions(record.category, record.name, versions)
        self._persist()
        self._backup()
        self._log.info("Registered %s (status=%s).", record.unique_id, record.status.value)
        return record

    def update(self, unique_id: str, **changes: Any) -> ModelRecord:
        """Update mutable fields of an existing record (e.g. status, metrics).

        Args:
            unique_id: Identifier of the record to update.
            **changes: Field/value pairs to overwrite. ``status`` accepts a
                :class:`ModelStatus` or its string value.

        Returns:
            The updated record.

        Raises:
            RegistryError: If the record is not found.
        """
        self._require_initialized()
        for category, names in self._db["categories"].items():
            for name, versions in names.items():
                for index, raw in enumerate(versions):
                    if raw.get("unique_id") == unique_id:
                        raw.update(self._normalize_changes(changes))
                        versions[index] = raw
                        self._persist()
                        self._backup()
                        self._log.info("Updated %s.", unique_id)
                        return ModelRecord.from_dict(raw)
        raise RegistryError(f"Model not found in registry: {unique_id}")

    # -- discovery ---------------------------------------------------------

    def find(
        self,
        category: ModelCategory,
        name: str | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelRecord]:
        """Return records matching the given filters.

        Args:
            category: Category to search within.
            name: Optional model name filter.
            status: Optional status filter.

        Returns:
            Matching records (possibly empty).
        """
        self._require_initialized()
        results: list[ModelRecord] = []
        names = self._db["categories"].get(category.value, {})
        for model_name, versions in names.items():
            if name is not None and model_name != name:
                continue
            for raw in versions:
                record = ModelRecord.from_dict(raw)
                if status is not None and record.status != status:
                    continue
                results.append(record)
        return results

    def list_models(self, category: ModelCategory | None = None) -> list[ModelRecord]:
        """Return all records, optionally filtered by category."""
        self._require_initialized()
        categories = (
            [category.value] if category else list(self._db["categories"].keys())
        )
        records: list[ModelRecord] = []
        for cat in categories:
            for versions in self._db["categories"].get(cat, {}).values():
                records.extend(ModelRecord.from_dict(raw) for raw in versions)
        return records

    def history(self, category: ModelCategory, name: str) -> list[ModelRecord]:
        """Return every version of a model, ordered oldest to newest."""
        self._require_initialized()
        return [ModelRecord.from_dict(raw) for raw in self._versions(category, name)]

    def load_latest(self, category: ModelCategory, name: str) -> ModelRecord:
        """Return the most recent version of a model.

        Raises:
            RegistryError: If no version exists.
        """
        versions = self.history(category, name)
        if not versions:
            raise RegistryError(f"No versions registered for {category.value}/{name}.")
        return versions[-1]

    def load_best(
        self,
        category: ModelCategory,
        name: str,
        metric: str,
        *,
        direction: str = "min",
        deployable_only: bool = True,
    ) -> ModelRecord:
        """Return the best version of a model by a validation metric.

        Args:
            category: Model category.
            name: Model name.
            metric: Metric key to compare (must be present in ``metrics``).
            direction: ``"min"`` (lower is better) or ``"max"``.
            deployable_only: Restrict to stable/production models.

        Returns:
            The best-scoring record.

        Raises:
            RegistryError: If no candidate has the metric, or direction is bad.
        """
        if direction not in {"min", "max"}:
            raise RegistryError(f"Invalid direction: {direction!r} (use 'min'/'max').")
        candidates = [
            record
            for record in self.history(category, name)
            if metric in record.metrics
            and (not deployable_only or record.status in _DEPLOYABLE)
        ]
        if not candidates:
            raise RegistryError(
                f"No candidate models with metric {metric!r} for "
                f"{category.value}/{name}."
            )
        chooser = min if direction == "min" else max
        best = chooser(candidates, key=lambda record: record.metrics[metric])
        self._log.info("Best %s/%s by %s: %s.", category.value, name, metric, best.unique_id)
        return best

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Return a summary of registry contents by category and status."""
        self._require_initialized()
        summary: dict[str, Any] = {"total": 0, "by_category": {}, "by_status": {}}
        for record in self.list_models():
            summary["total"] += 1
            summary["by_category"][record.category.value] = (
                summary["by_category"].get(record.category.value, 0) + 1
            )
            summary["by_status"][record.status.value] = (
                summary["by_status"].get(record.status.value, 0) + 1
            )
        return summary

    # -- internals ---------------------------------------------------------

    def _versions(self, category: ModelCategory, name: str) -> list[dict[str, Any]]:
        return self._db["categories"].get(category.value, {}).get(name, [])

    def _set_versions(
        self, category: ModelCategory, name: str, versions: list[dict[str, Any]]
    ) -> None:
        self._db["categories"].setdefault(category.value, {})[name] = versions

    def _count_names(self) -> int:
        return sum(len(names) for names in self._db["categories"].values())

    @staticmethod
    def _normalize_changes(changes: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(changes)
        if isinstance(normalized.get("status"), ModelStatus):
            normalized["status"] = normalized["status"].value
        return normalized

    def _persist(self) -> None:
        write_json(self._db_path, self._db)

    def _backup(self) -> None:
        if not self._config.section("registry")["registry"].get("backup", {}).get(
            "enabled", True
        ):
            return
        ensure_dir(self._backup_dir)
        write_json(self._backup_dir / f"registry_{timestamp_slug()}.json", self._db)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RegistryError("RegistryManager.initialize() must be called first.")


__all__ = ["RegistryManager"]
