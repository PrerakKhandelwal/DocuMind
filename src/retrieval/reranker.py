"""Cohere Rerank client for ranking retrieved passages by relevance."""
from __future__ import annotations

import os

import httpx

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


class Reranker:
    """Rerank candidate chunks through Cohere's v2 Rerank API."""

    def __init__(self):
        c = cfg.reranker
        self._api_key = os.getenv("COHERE_API_KEY")
        if not self._api_key:
            raise RuntimeError("COHERE_API_KEY is missing; add it to .env or environment.")
        self._model = c.model
        self._url = c.base_url
        self._http = httpx.Client(timeout=float(c.get("timeout_s", 120)))
        log.info(f"Cohere reranker ready: model={self._model}")

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        if not chunks:
            return []

        top_k = top_k or cfg.retrieval.rerank_top_k
        response = self._http.post(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={
                "model": self._model,
                "query": query,
                "documents": [chunk["content"] for chunk in chunks],
                "top_n": min(top_k, len(chunks)),
            },
        )
        response.raise_for_status()
        ranked = []
        for result in response.json().get("results", []):
            chunk = chunks[result["index"]]
            chunk["_rerank_score"] = float(result["relevance_score"])
            ranked.append(chunk)
        return ranked
