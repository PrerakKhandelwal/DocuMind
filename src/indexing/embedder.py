"""Gemini embedding client for document and retrieval-query vectors."""
from __future__ import annotations

import os
from typing import Union

import httpx
import numpy as np

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


class Embedder:
    """Embed indexed passages and queries using Gemini's retrieval task types."""

    def __init__(self):
        c = cfg.embeddings
        self._model = c.model_name
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is missing; add it to .env or environment.")
        self._dimensions = int(c.dimensions)
        self._url = f"{c.base_url.rstrip('/')}/models/{self._model}:batchEmbedContents"
        self._http = httpx.Client(timeout=float(c.get("timeout_s", 120)))
        log.info(f"Gemini embedding model ready: {self._model} ({self._dimensions} dimensions)")

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)

        vectors = []
        for text in texts:
            body = {"requests": [{
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": self._dimensions,
            }]}
            response = self._http.post(
                self._url,
                headers={"x-goog-api-key": self._api_key},
                json=body,
            )
            response.raise_for_status()
            vectors.append(response.json()["embeddings"][0]["values"])

        result = np.asarray(vectors, dtype=np.float32)
        if result.shape[1] != self._dimensions:
            raise ValueError(
                f"Gemini returned {result.shape[1]} dimensions; expected {self._dimensions}."
            )
        return result

    def embed(self, texts: Union[str, list[str]], batch_size=None,
              show_progress: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query], "RETRIEVAL_QUERY")[0]
