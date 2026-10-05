"""
generation/answer_validator.py
================================
Post-generation validation layer — the hallucination firewall.

Checks:
  1. Did the LLM include at least one citation?
  2. Is the answer suspiciously confident despite no context?
  3. Did the LLM paraphrase the "not found" sentinel correctly?

If validation fails, returns the canonical "not found" message.
"""
from __future__ import annotations

import re

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)

# Regex to detect at least one citation in the answer
_CITATION_PATTERN = re.compile(
    r'\[(?:Doc|Source|doc_id)\s*[:=]', re.IGNORECASE
)

# Phrases that can indicate insufficent-evidence responses
_NOT_FOUND_PHRASES = [
    "not available in the documentation",
    "not found in the provided",
    "cannot find this information",
    "no information available",
    "not mentioned in the context",
    "not covered in the documentation",
]


def _insufficient_response() -> str:
    """Canonical insufficient-evidence response."""
    return (
        "Mode: Insufficient\n"
        "Confidence: Low\n"
        "Answer:\n"
        f"- {cfg.generation.not_found_message.strip()}\n"
        "Evidence:\n"
        "- Retrieved context did not contain enough explicit support to provide a reliable answer.\n"
        "Assumptions/Gaps:\n"
        "- Exact details are missing or not clearly stated in the available excerpts."
    )


def _looks_like_insufficient(answer_lower: str) -> bool:
    """Heuristic: detect broad insufficiency/refusal responses."""
    return any(phrase in answer_lower for phrase in _NOT_FOUND_PHRASES)


def validate_answer(answer: str, chunks: list[dict]) -> str:
    """
    Validate the LLM's answer.

    Args:
        answer: Raw LLM output
        chunks: The context chunks that were provided

    Returns:
        The validated answer string (either the original or the not-found message)
    """
    answer = answer.strip()

    # Case 1: No chunks retrieved at all → always "not found"
    if not chunks:
        log.info("Validation: no chunks retrieved → not found")
        return _insufficient_response()

    # Case 2: LLM already said "not found" in some form → normalise it
    answer_lower = answer.lower()
    # If the LLM used a refusal/insufficient response and provided no evidence,
    # normalise to the canonical structured insufficient response.
    if _looks_like_insufficient(answer_lower) and not _CITATION_PATTERN.search(answer):
        log.info("Validation: LLM produced insufficient-evidence response without citations")
        return _insufficient_response()

    # Case 3: No citation found, but chunks were provided → suspicious
    if cfg.generation.require_citations and not _CITATION_PATTERN.search(answer):
        log.warning(
            "Validation: answer contains no citations despite context being provided. "
            "Possible hallucination — returning not-found."
        )
        return _insufficient_response()

    # Case 4: Answer is too short to be meaningful (likely a refusal or error)
    if len(answer.split()) < 5:
        log.warning(f"Validation: suspiciously short answer: '{answer}'")
        return _insufficient_response()

    log.debug("Validation: answer passed all checks.")
    return answer
