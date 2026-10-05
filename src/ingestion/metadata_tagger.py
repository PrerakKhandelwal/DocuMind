"""
ingestion/metadata_tagger.py
=============================
Extracts and attaches version + section metadata to each page/chunk.

Version detection order:
  1. Explicit version in filename  (e.g. ProductGuide_v2.3.1.pdf)
  2. "Release" / "Version" text on the first two pages of the PDF
  3. Fallback to config default_version

Section detection:
  Uses the page's heading list to build a section path like:
  "Chapter 3 > Network Configuration > SSL Settings"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config_loader import cfg
from src.logger import get_logger
from src.ingestion.pdf_parser import PageContent

log = get_logger(__name__)


@dataclass
class DocMetadata:
    """Metadata derived from a PDF file and its pages."""
    doc_id: str              # Unique slug derived from filename
    release_version: str     # e.g. "2.3.1"
    is_latest: bool = False  # Set externally by version manager


def _slugify(name: str) -> str:
    """Convert a filename to a clean doc_id slug."""
    name = Path(name).stem  # drop .pdf extension
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s_]+', '-', name.strip()).lower()
    return name


def _extract_version_from_filename(filename: str) -> Optional[str]:
    patterns = cfg.ingestion.version_patterns
    for pattern in patterns:
        m = re.search(pattern, filename)
        if m:
            return m.group(1)
    return None


def _extract_version_from_text(text: str) -> Optional[str]:
    """Scan text (first pages) for common version declaration patterns."""
    patterns = [
        r'[Vv]ersion[:\s]+(\d+\.\d+(?:\.\d+)?)',
        r'[Rr]elease[:\s]+(\d+\.\d+(?:\.\d+)?)',
        r'[Rr]ev(?:ision)?[.:\s]+(\d+\.\d+(?:\.\d+)?)',
        r'[Vv]?(\d+\.\d+\.\d+)',   # bare semver
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def build_doc_metadata(pdf_path: str | Path, pages: list[PageContent]) -> DocMetadata:
    """
    Build DocMetadata for a PDF using filename + first-page content.
    Called once per document before chunking.
    """
    pdf_path = Path(pdf_path)
    doc_id = _slugify(pdf_path.name)

    # Try filename first
    version = _extract_version_from_filename(pdf_path.name)

    # Try first two pages of text
    if not version:
        probe_text = ""
        for p in pages[:2]:
            probe_text += p.text + " ".join(p.headings) + " "
        version = _extract_version_from_text(probe_text)

    if not version:
        version = cfg.ingestion.default_version
        log.debug(f"No version found in {pdf_path.name}, using default: {version}")
    else:
        log.info(f"Version detected: {pdf_path.name} → v{version}")

    return DocMetadata(doc_id=doc_id, release_version=version)


def build_section_path(page: PageContent) -> str:
    """
    Build a human-readable section path from the page's headings.
    e.g. "Chapter 2 > Network Settings > SSL Configuration"
    """
    if not page.headings:
        return f"Page {page.page_number}"
    # Use the first 3 heading levels max, to avoid excessively long paths
    parts = page.headings[:3]
    return " > ".join(parts)
