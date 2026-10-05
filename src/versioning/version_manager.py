"""
versioning/version_manager.py
==============================
Manages the set of indexed release versions and the "latest" pointer.

Persisted in: paths.version_registry (see config.yaml)
Format:
{
  "versions": {
    "2.3.1": {"collection": "product_docs_2_3_1", "doc_count": 12},
    "3.0.0": {"collection": "product_docs_3_0_0", "doc_count": 15}
  },
  "latest": "3.0.0"
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


class VersionManager:

    def __init__(self):
        self._registry_path = Path(cfg.paths.version_registry)
        self._registry: dict = {"versions": {}, "latest": None}
        self._load()

    def register_version(self, version: str, collection_name: str, doc_count: int = 0):
        self._registry["versions"][version] = {
            "collection": collection_name,
            "doc_count": doc_count,
        }
        log.info(f"Registered version: {version} → {collection_name}")

    def set_latest(self, version: str):
        if version not in self._registry["versions"]:
            raise ValueError(f"Version {version} not registered yet.")
        self._registry["latest"] = version
        log.info(f"Latest version set to: {version}")

    def get_latest(self) -> Optional[str]:
        return self._registry.get("latest")

    def get_collection(self, version: str) -> Optional[str]:
        entry = self._registry["versions"].get(version)
        return entry["collection"] if entry else None

    def get_latest_collection(self) -> Optional[str]:
        latest = self.get_latest()
        if latest:
            return self.get_collection(latest)
        return None

    def all_versions(self) -> list[str]:
        return list(self._registry["versions"].keys())

    def save(self):
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2)

    def _load(self):
        if self._registry_path.exists():
            with open(self._registry_path, "r", encoding="utf-8") as f:
                self._registry = json.load(f)
            log.debug(
                f"Version registry loaded: {list(self._registry['versions'].keys())}, "
                f"latest={self._registry.get('latest')}"
            )
