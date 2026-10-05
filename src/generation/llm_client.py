"""Gemini REST client used for grounded answer generation."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0, system_prompt: str | None = None) -> str:
        """Generate text from one prompt."""

    @abstractmethod
    def chat(self, messages: list[dict], max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """Generate text from role-tagged conversation messages."""


class GeminiClient(BaseLLMClient):
    """Gemini generateContent REST client; reads GEMINI_API_KEY from the environment."""

    def __init__(self):
        c = cfg.llm.gemini
        self._model = c.model
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is missing; add it to .env or environment.")
        self._url = f"{c.base_url.rstrip('/')}/models/{self._model}:generateContent"
        self._http = httpx.Client(timeout=float(c.get("timeout_s", 120)))
        self._max_tokens = int(c.max_tokens)
        self._temperature = float(c.temperature)
        log.info(f"Gemini client ready: model={self._model}")

    def _generate(self, contents, max_tokens, temperature, system_prompt=None):
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        response = self._http.post(
            self._url,
            headers={"x-goog-api-key": self._api_key},
            json=payload,
        )
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no generation candidates.")
        return "".join(
            part.get("text", "")
            for part in candidates[0].get("content", {}).get("parts", [])
        ).strip()

    def generate(self, prompt, max_tokens=None, temperature=None, system_prompt=None):
        return self._generate(
            [{"role": "user", "parts": [{"text": prompt}]}],
            max_tokens or self._max_tokens,
            self._temperature if temperature is None else temperature,
            system_prompt,
        )

    def chat(self, messages, max_tokens=1024, temperature=0.0):
        contents, system_parts = [], []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                system_parts.append(message.get("content", ""))
            else:
                contents.append({
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": message.get("content", "")}],
                })
        return self._generate(
            contents,
            max_tokens or self._max_tokens,
            self._temperature if temperature is None else temperature,
            "\n\n".join(system_parts) or None,
        )


def create_llm_client() -> BaseLLMClient:
    """Create the configured Gemini client."""
    if str(cfg.llm.get("backend", "gemini")).lower() != "gemini":
        raise ValueError("This project is configured for Gemini generation only.")
    return GeminiClient()
