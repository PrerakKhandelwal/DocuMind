"""
indexing/vector_store.py
=========================
Manages Qdrant collections for dense vector storage and retrieval.

One collection per release version  →  "product_docs_2_3_1"
Plus a special alias "product_docs_latest" pointing to the newest.

Collections store:
  • dense vector (Gemini embeddings, configured dimensions)
  • payload (all Chunk metadata + content)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    ScrollRequest,
)

from src.config_loader import cfg, project_root
from src.ingestion.chunker import Chunk
from src.logger import get_logger

log = get_logger(__name__)


def _make_qdrant_client() -> QdrantClient:
    """
    Embedded mode: no separate Qdrant process (no Docker / qdrant.exe).
    Server mode: connect to Qdrant listening on host:port.
    """
    conn = str(getattr(cfg.qdrant, "connection", "server")).lower()
    if conn == "embedded":
        raw = getattr(cfg.qdrant, "embedded_path", "cache/qdrant_data")
        p = Path(raw)
        if not p.is_absolute():
            p = project_root() / p
        p.mkdir(parents=True, exist_ok=True)
        log.info(f"Qdrant embedded mode (on-disk store): {p}")
        return QdrantClient(path=str(p))
    log.info(f"Qdrant server: {cfg.qdrant.host}:{cfg.qdrant.port}")
    return QdrantClient(
        host=cfg.qdrant.host,
        port=cfg.qdrant.port,
        prefer_grpc=False,
    )


def _vectors_on_disk() -> bool:
    """Embedded Qdrant should not use server-style on-disk vector shards."""
    if str(getattr(cfg.qdrant, "connection", "server")).lower() == "embedded":
        return False
    return bool(getattr(cfg.qdrant, "on_disk_payload", True))


def _version_to_collection_name(version: str) -> str:
    """e.g. "2.3.1" → "product_docs_2_3_1" """
    safe = re.sub(r'[^\w]', '_', version)
    return f"{cfg.qdrant.collection_prefix}_{safe}"


class VectorStore:
    """
    Wraps Qdrant to provide:
      • create_collection
      • upsert_chunks (with embeddings)
      • search (dense, with optional version filter)
      • get_chunks_by_ids (parent retrieval)
    """

    def __init__(self):
        self._client = _make_qdrant_client()

    # ── Collection management ──────────────────────────────────────────────

    def ensure_collection(self, version: str) -> str:
        """Create a collection for this version if it doesn't exist."""
        col_name = _version_to_collection_name(version)
        existing = [c.name for c in self._client.get_collections().collections]
        if col_name not in existing:
            self._client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(
                    size=cfg.qdrant.vector_size,
                    distance=Distance.COSINE,
                    on_disk=_vectors_on_disk(),
                ),
            )
            log.info(f"Created Qdrant collection: {col_name}")
        else:
            log.debug(f"Collection already exists: {col_name}")
        return col_name

    def list_collections(self) -> list[str]:
        prefix = cfg.qdrant.collection_prefix
        return [c.name for c in self._client.get_collections().collections
                if c.name.startswith(prefix)]

    def get_latest_collection(self) -> Optional[str]:
        """Return the collection name that is marked is_latest."""
        for col_name in self.list_collections():
            # Scroll a few points to check is_latest flag
            result = self._client.scroll(
                collection_name=col_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="is_latest", match=MatchValue(value=True))]
                ),
                limit=1,
            )
            if result[0]:  # has at least one point with is_latest=True
                return col_name
        return None

    def delete_collection(self, version: str):
        col_name = _version_to_collection_name(version)
        self._client.delete_collection(col_name)
        log.info(f"Deleted collection: {col_name}")

    # ── Upsert ─────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        collection_name: str,
    ) -> int:
        """
        Upsert child chunks with their embeddings into Qdrant.
        Returns number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = []
        for chunk, vec in zip(chunks, embeddings):
            payload = {
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
            }
            points.append(PointStruct(
                id=_uuid_to_int(chunk.chunk_id),
                vector=vec.tolist(),
                payload=payload,
            ))

        # Batch upsert in groups of 256
        batch_size = 256
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self._client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)

        log.debug(f"Upserted {total} points → {collection_name}")
        return total

    # ── Retrieval ──────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        collection_name: str,
        top_k: int = 20,
        version_filter: Optional[str] = None,
        latest_only: bool = False,
    ) -> list[dict]:
        """
        Dense vector search with optional version / latest filter.

        Returns list of dicts: {score, payload}
        """
        filters = []
        if version_filter:
            filters.append(FieldCondition(
                key="release_version",
                match=MatchValue(value=version_filter)
            ))
        if latest_only:
            filters.append(FieldCondition(
                key="is_latest",
                match=MatchValue(value=True)
            ))

        query_filter = Filter(must=filters) if filters else None

        results = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=cfg.retrieval.min_score_threshold,
        )

        return [{"score": r.score, "payload": r.payload} for r in results]

    def get_parents_by_ids(
        self,
        parent_ids: list[str],
        parent_store: "ParentStore",
    ) -> list[dict]:
        """Fetch full parent chunks by their IDs from the in-memory parent store."""
        return [parent_store.get(pid) for pid in parent_ids if parent_store.get(pid)]

    def search_all_collections(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        version_filter: Optional[str] = None,
        latest_only: bool = False,
    ) -> list[dict]:
        """Search across all version collections, merge results."""
        all_results = []
        for col_name in self.list_collections():
            results = self.search(
                query_vector=query_vector,
                collection_name=col_name,
                top_k=top_k,
                version_filter=version_filter,
                latest_only=latest_only,
            )
            all_results.extend(results)

        # Sort by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]


# ── UUID helper ───────────────────────────────────────────────────────────────

def _uuid_to_int(uid: str) -> int:
    """Convert UUID string to an integer for Qdrant point ID."""
    import uuid
    return uuid.UUID(uid).int % (2**63)  # Qdrant uses uint64


# ── In-memory parent store ────────────────────────────────────────────────────

class ParentStore:
    """
    Stores parent chunks in memory (or optionally on disk as JSON).
    Used to retrieve full context after child chunk retrieval.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._store: dict[str, dict] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load()

    def add(self, chunk: Chunk):
        self._store[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "release_version": chunk.release_version,
            "is_latest": chunk.is_latest,
            "section": chunk.section,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "table_caption": chunk.table_caption,
        }

    def get(self, chunk_id: str) -> Optional[dict]:
        return self._store.get(chunk_id)

    def save(self):
        if self._persist_path:
            import json
            from pathlib import Path
            Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False)
            log.info(f"ParentStore saved: {len(self._store)} entries")

    def _load(self):
        import json
        from pathlib import Path
        p = Path(self._persist_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self._store = json.load(f)
            log.info(f"ParentStore loaded: {len(self._store)} entries")

    def __len__(self):
        return len(self._store)
