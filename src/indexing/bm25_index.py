"""
indexing/bm25_index.py
=======================
BM25 keyword index for sparse retrieval.

Complements the dense Gemini embedding index — critical for exact keyword matches
(error codes, version strings, product names, CLI commands, etc.)

Persisted as a simple JSON file so we don't re-index on every startup.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from src.config_loader import cfg
from src.ingestion.chunker import Chunk
from src.logger import get_logger

log = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Index:
    """
    Maintains a BM25Okapi index over child chunk texts.

    Stores:
      • corpus tokens (list of token lists)
      • chunk metadata list (parallel to corpus)
      • persists to disk as pickle

    Usage:
        idx = BM25Index()
        idx.add_chunks(children)
        idx.build()
        results = idx.search("SSL certificate path", top_k=20)
    """

    def __init__(self):
        self._corpus: list[list[str]] = []
        self._metadata: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._persist_dir = Path(cfg.bm25.get("storage", cfg.paths.bm25_storage))

    def add_chunks(self, chunks: list[Chunk]):
        """Add child chunks to the index corpus."""
        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            self._corpus.append(tokens)
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "release_version": chunk.release_version,
                "is_latest": chunk.is_latest,
                "section": chunk.section,
                "page_number": chunk.page_number,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "table_caption": chunk.table_caption,
            })

    def build(self):
        """Build the BM25 model from the current corpus."""
        if not self._corpus:
            log.warning("BM25: corpus is empty — nothing to build.")
            return
        log.info(f"Building BM25 index over {len(self._corpus)} chunks…")
        self._bm25 = BM25Okapi(
            self._corpus,
            k1=cfg.bm25.k1,
            b=cfg.bm25.b,
        )
        log.info("BM25 index ready.")

    def search(
        self,
        query: str,
        top_k: int = 20,
        version_filter: Optional[str] = None,
        latest_only: bool = False,
    ) -> list[dict]:
        """
        Return top-k matching chunks as dicts: {score, payload}
        Supports same version/latest filters as the dense index.
        """
        if self._bm25 is None:
            log.warning("BM25 index not built yet.")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        results = []
        for i, score in enumerate(scores):
            meta = self._metadata[i]
            if version_filter and meta["release_version"] != version_filter:
                continue
            if latest_only and not meta["is_latest"]:
                continue
            results.append({"score": float(score), "payload": meta})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self):
        """Persist index to disk."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        with open(self._persist_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({
                "corpus": self._corpus,
                "metadata": self._metadata,
                "bm25": self._bm25,
            }, f)
        log.info(f"BM25 index saved to {self._persist_dir}")

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        pkl_path = self._persist_dir / "bm25_index.pkl"
        if not pkl_path.exists():
            log.debug("No BM25 index found on disk.")
            return False
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._corpus = data["corpus"]
        self._metadata = data["metadata"]
        self._bm25 = data["bm25"]
        log.info(f"BM25 index loaded: {len(self._corpus)} chunks")
        return True
