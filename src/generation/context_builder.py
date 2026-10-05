"""
generation/context_builder.py
==============================
Builds the final prompt sent to the LLM.

Key design principles:
  • Every chunk is labelled with its source (doc, section, page)
  • The system prompt STRICTLY forbids invention outside provided context
  • Context is trimmed to stay within max_context_chars
  • Tables are rendered in full Markdown for the LLM to read
"""
from __future__ import annotations

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """You are a product documentation assistant.

Your ONLY job is to answer questions using the provided documentation excerpts.

STRICT RULES — you MUST follow these without exception:
1. Use ONLY the provided context. NEVER use outside knowledge.
2. Every factual claim MUST include at least one source citation in this format:
   [Doc: <doc_id> | Version: <version> | Section: <section> | Page: <page>]
3. Never invent any names, parameter names, values, or table rows.
4. If exact wording is not present but nearby evidence supports a careful synthesis,
   provide an inferred answer and clearly mark it as inferred.
5. If evidence is insufficient, explicitly say so and list what is missing.
6. Be concise and factual.

RESPONSE FORMAT (always follow):
Mode: <Exact|Inferred|Insufficient>
Confidence: <High|Medium|Low>
Answer:
- <main answer bullet(s) with citations>
Evidence:
- <short evidence bullet(s) with citations>
Assumptions/Gaps:
- <only when needed; keep short>
"""


def _chunk_header(chunk: dict) -> str:
    """Format a source label for a chunk."""
    return (
        f"[Source: Doc={chunk.get('doc_id','?')} | "
        f"Version={chunk.get('release_version','?')} | "
        f"Section={chunk.get('section','?')} | "
        f"Page={chunk.get('page_number','?')} | "
        f"Type={chunk.get('chunk_type','text')}]"
    )


def build_context_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the LLM.

    The user prompt contains:
      - The documentation context (with source labels)
      - The user's question

    Returns:
        (system_prompt, user_prompt)
    """
    max_chars = cfg.generation.max_context_chars

    # Build context string, respecting the character budget
    context_parts = []
    used_chars = 0

    for i, chunk in enumerate(chunks, 1):
        header = _chunk_header(chunk)
        content = chunk.get("content", "").strip()

        # Add table caption if present
        caption = chunk.get("table_caption", "")
        if caption:
            content = f"Table: {caption}\n{content}"

        part = f"\n--- Excerpt {i} ---\n{header}\n{content}\n"
        if used_chars + len(part) > max_chars:
            log.debug(f"Context budget reached at chunk {i}/{len(chunks)}")
            break

        context_parts.append(part)
        used_chars += len(part)

    if not context_parts:
        # No context at all — will trigger the "not found" path
        context_str = "(No relevant documentation found)"
    else:
        context_str = "".join(context_parts)

    user_prompt = f"""DOCUMENTATION CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER (follow the required response format and cite each factual claim):"""

    return SYSTEM_PROMPT, user_prompt
