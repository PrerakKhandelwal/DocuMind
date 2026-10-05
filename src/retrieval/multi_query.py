"""
retrieval/multi_query.py
=========================
Generates 3-5 sub-questions from a complex user query using Gemini.

This dramatically improves recall for questions whose answer is
distributed across different sections of the documentation.

Example:
  Input:  "How do I set up SSL mutual authentication in cluster mode?"
  Output: [
    "SSL configuration options documentation",
    "mutual TLS mTLS certificate setup",
    "cluster mode network requirements",
    "certificate installation steps",
    "SSL cluster compatibility matrix",
  ]
"""
from __future__ import annotations

import re

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)

_MULTI_QUERY_PROMPT = """\
You are a search query expansion assistant for a product documentation system.

Given the user's question, generate {n} distinct search sub-queries that together
would retrieve all information needed to answer the original question.

Rules:
- Each sub-query must be short (3-8 words), focused, and search-engine style
- Sub-queries should cover DIFFERENT aspects of the original question
- Do NOT repeat the same concept in different words
- Output ONLY a numbered list, nothing else

User question: {query}

Sub-queries:"""


def generate_sub_queries(query: str, llm_client) -> list[str]:
    """
    Use Gemini to expand the query into sub-queries for retrieval.

    Args:
        query:      The clean user query (version stripped)
        llm_client: An instance of LLMClient from generation/llm_client.py

    Returns:
        List of sub-query strings (including the original query)
    """
    if not cfg.multi_query.enabled:
        return [query]

    n = cfg.multi_query.num_sub_questions
    prompt = _MULTI_QUERY_PROMPT.format(n=n, query=query)

    try:
        response = llm_client.generate(prompt, max_tokens=300, temperature=0.3)
        sub_queries = _parse_numbered_list(response)

        if not sub_queries:
            log.warning("Multi-query generation returned empty — using original query only")
            return [query]

        # Always include the original query
        all_queries = [query] + [q for q in sub_queries if q.lower() != query.lower()]
        log.debug(f"Generated {len(all_queries)} sub-queries: {all_queries}")
        return all_queries

    except Exception as e:
        log.warning(f"Multi-query generation failed: {e}. Using original query.")
        return [query]


def _parse_numbered_list(text: str) -> list[str]:
    """Extract lines from a numbered list like '1. ...\n2. ...'"""
    lines = []
    for line in text.strip().split("\n"):
        # Match "1. text" or "- text" or "• text"
        m = re.match(r'^[\d\-•*]+[\.\)]\s*(.+)', line.strip())
        if m:
            item = m.group(1).strip()
            if len(item) > 3:
                lines.append(item)
    return lines
