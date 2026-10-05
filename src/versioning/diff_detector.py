"""
versioning/diff_detector.py
============================
SHA-256 based page-level diff detection.

On every ingestion run, we compare the hash of each page's content
against previously stored hashes. Only CHANGED pages are re-processed,
making incremental updates fast even for 40+ large PDFs.

Hash store: ./cache/page_hashes.json
Format:
{
  "ProductGuide_v2.3.pdf::page_1": "abc123...",
  "ProductGuide_v2.3.pdf::page_2": "def456...",
  ...
}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


class DiffDetector:
    """
    Tracks page-level content hashes to detect changes.

    Usage:
        detector = DiffDetector()
        changed_pages = detector.get_changed_pages(pdf_path, pages)
        # process only changed_pages …
        detector.update_hashes(pdf_path, changed_pages)
        detector.save()
    """

    def __init__(self):
        self._cache_path = Path(cfg.paths.index_cache)
        self._hashes: dict[str, str] = {}
        self._load()

    def _key(self, pdf_path: str | Path, page_number: int) -> str:
        return f"{Path(pdf_path).name}::page_{page_number}"

    def _hash_page(self, text: str, tables_md: list[str]) -> str:
        content = text + "".join(tables_md)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_changed_pages(self, pdf_path, pages) -> list:
        """
        Filter and return only pages whose content has changed since last run.
        On first run (no hash stored), all pages are returned as 'changed'.
        """
        changed = []
        for page in pages:
            tables_md = [t.markdown for t in page.tables]
            new_hash = self._hash_page(page.text, tables_md)
            key = self._key(pdf_path, page.page_number)
            if self._hashes.get(key) != new_hash:
                changed.append(page)
        if changed:
            log.info(f"{Path(pdf_path).name}: {len(changed)}/{len(pages)} pages changed")
        else:
            log.info(f"{Path(pdf_path).name}: no changes detected — skipping")
        return changed

    def update_hashes(self, pdf_path, pages) -> None:
        """Record new hashes for these pages after successful ingestion."""
        for page in pages:
            tables_md = [t.markdown for t in page.tables]
            new_hash = self._hash_page(page.text, tables_md)
            key = self._key(pdf_path, page.page_number)
            self._hashes[key] = new_hash

    def remove_document(self, pdf_path) -> int:
        """Remove all hashes for a deleted document. Returns count removed."""
        prefix = Path(pdf_path).name + "::"
        keys_to_remove = [k for k in self._hashes if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._hashes[k]
        return len(keys_to_remove)

    def save(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._hashes, f, indent=2)

    def _load(self):
        if self._cache_path.exists():
            with open(self._cache_path, "r", encoding="utf-8") as f:
                self._hashes = json.load(f)
            log.debug(f"Loaded {len(self._hashes)} page hashes from cache")
        else:
            self._hashes = {}
