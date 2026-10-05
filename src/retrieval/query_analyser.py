"""
retrieval/query_analyser.py
============================
Analyses the user's query to extract:
  1. Requested version  (e.g. "in v2.3", "for release 3.0")
  2. Whether to search all versions or latest only
  3. Keywords to prioritise

Uses regex first (fast), falls back to LLM for complex intent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class QueryIntent:
    original_query: str
    clean_query: str           # Query with version mentions stripped
    requested_version: Optional[str]  # None → use latest
    search_all_versions: bool = False  # True if user explicitly asks for history
    keywords: list[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


# Patterns that indicate the user wants a specific version
_VERSION_REQUEST_PATTERNS = [
    (r'\bv(?:ersion)?\s*(\d+\.\d+(?:\.\d+)?)\b', 1),
    (r'\brelease\s+(\d+\.\d+(?:\.\d+)?)\b', 1),
    (r'\bin\s+(\d+\.\d+(?:\.\d+)?)\b', 1),
    (r'\bfor\s+(?:version\s+)?(\d+\.\d+(?:\.\d+)?)\b', 1),
    (r'\b(\d+\.\d+(?:\.\d+)?)\s+(?:docs?|documentation|release|version)\b', 1),
]

# Patterns that indicate the user wants ALL version history
_ALL_VERSIONS_PATTERNS = [
    r'\ball\s+versions?\b',
    r'\bacross\s+(?:all\s+)?versions?\b',
    r'\bversion\s+history\b',
    r'\bchangelog\b',
    r'\bwhat\s+changed\b',
    r'\bcompare\s+versions?\b',
]


def analyse_query(query: str) -> QueryIntent:
    """
    Parse the user's natural language query for version and intent signals.

    Examples:
      "How to configure SSL in v2.3?"
        → requested_version="2.3", clean_query="How to configure SSL?"

      "What's new in the latest release?"
        → requested_version=None, latest_only=True

      "Compare SSL settings across all versions"
        → search_all_versions=True
    """
    clean = query.strip()
    requested_version: Optional[str] = None
    search_all = False

    # Check for all-versions intent
    for pat in _ALL_VERSIONS_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            search_all = True
            break

    # Extract specific version
    if not search_all:
        for pattern, group in _VERSION_REQUEST_PATTERNS:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                requested_version = m.group(group)
                # Remove the version mention from the clean query
                clean = re.sub(pattern, '', clean, flags=re.IGNORECASE).strip()
                clean = re.sub(r'\s{2,}', ' ', clean)
                break

    # Extract potential product/feature keywords (capitalised words, codes)
    keywords = re.findall(r'\b[A-Z][A-Z0-9_\-]{2,}\b', query)

    log.debug(
        f"Query intent: version={requested_version}, all={search_all}, "
        f"keywords={keywords}"
    )

    return QueryIntent(
        original_query=query,
        clean_query=clean,
        requested_version=requested_version,
        search_all_versions=search_all,
        keywords=keywords,
    )
