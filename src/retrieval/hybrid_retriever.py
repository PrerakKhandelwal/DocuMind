"""
retrieval/hybrid_retriever.py
==============================
Combines dense (Gemini embeddings) and sparse (BM25) retrieval using
Reciprocal Rank Fusion (RRF) to produce a merged, re-ranked list.

Flow:
  1. For each sub-query: run dense search + BM25 search
  2. Merge all results using RRF (standard k=60)
  3. Deduplicate by chunk_id
  4. Fetch parent chunks for top-N results (small-to-big retrieval)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from src.config_loader import cfg
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore, ParentStore
from src.indexing.bm25_index import BM25Index
from src.logger import get_logger

log = get_logger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Executes hybrid retrieval over the Qdrant + BM25 indices.

    For each query (or set of sub-queries):
      - Dense search → Qdrant
      - Sparse search → BM25
      - Fuse with RRF
      - Upgrade child chunks → parent chunks (small-to-big)
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        parent_store: ParentStore,
    ):
        self._embedder = embedder
        self._vs = vector_store
        self._bm25 = bm25_index
        self._parent_store = parent_store

    def retrieve(
        self,
        queries: list[str],
        version_filter: Optional[str] = None,
        latest_only: bool = True,
        search_all_versions: bool = False,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.

        Args:
            queries:             List of sub-queries (including original)
            version_filter:      Specific version to restrict results to
            latest_only:         Only return chunks from the latest release
            search_all_versions: Override — search all versions (no filter)

        Returns:
            List of deduplicated parent-chunk dicts, sorted by RRF score.
        """
        # Determine filter mode
        effective_version = None
        effective_latest = False

        if search_all_versions:
            pass  # no filter
        elif version_filter:
            effective_version = version_filter
        else:
            effective_latest = latest_only

        rrf_accumulator: dict[str, float] = defaultdict(float)
        payload_map: dict[str, dict] = {}

        for query in queries:
            log.debug(f"Sub-query: '{query}'")

            # ── Dense retrieval ────────────────────────────────────
            q_vec = self._embedder.embed_query(query)
            dense_results = self._vs.search_all_collections(
                query_vector=q_vec,
                top_k=cfg.retrieval.dense_top_k,
                version_filter=effective_version,
                latest_only=effective_latest,
            )
            for rank, result in enumerate(dense_results):
                cid = result["payload"]["chunk_id"]
                rrf_accumulator[cid] += _rrf_score(rank, cfg.retrieval.rrf_k)
                payload_map[cid] = result["payload"]

            # ── Sparse (BM25) retrieval ────────────────────────────
            sparse_results = self._bm25.search(
                query=query,
                top_k=cfg.retrieval.sparse_top_k,
                version_filter=effective_version,
                latest_only=effective_latest,
            )
            for rank, result in enumerate(sparse_results):
                cid = result["payload"]["chunk_id"]
                rrf_accumulator[cid] += _rrf_score(rank, cfg.retrieval.rrf_k)
                if cid not in payload_map:
                    payload_map[cid] = result["payload"]

        # Sort by accumulated RRF score
        sorted_ids = sorted(rrf_accumulator, key=lambda c: rrf_accumulator[c], reverse=True)

        # ── Small-to-big: upgrade child → parent ──────────────────
        seen_parent_ids: set[str] = set()
        final_chunks: list[dict] = []

        for cid in sorted_ids:
            payload = payload_map[cid]
            parent_id = payload.get("parent_id")

            if parent_id:
                if parent_id in seen_parent_ids:
                    continue
                parent = self._parent_store.get(parent_id)
                if parent:
                    parent["_rrf_score"] = rrf_accumulator[cid]
                    final_chunks.append(parent)
                    seen_parent_ids.add(parent_id)
                else:
                    # Fallback: use the child chunk itself
                    payload["_rrf_score"] = rrf_accumulator[cid]
                    final_chunks.append(payload)
            else:
                # This IS a parent chunk (e.g. table)
                if cid not in seen_parent_ids:
                    payload["_rrf_score"] = rrf_accumulator[cid]
                    final_chunks.append(payload)
                    seen_parent_ids.add(cid)

        log.info(
            f"Hybrid retrieval: {len(sorted_ids)} child matches "
            f"→ {len(final_chunks)} unique parent chunks"
        )
        return final_chunks
