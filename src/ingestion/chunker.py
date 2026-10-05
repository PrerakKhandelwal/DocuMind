"""
ingestion/chunker.py
=====================
Implements parent-child (small-to-big) chunking strategy.

Parent chunks  (~1024 tokens) — sent to the LLM as context.
Child chunks   (~256 tokens)  — indexed in Qdrant for retrieval.

Each child knows its parent_id so we can fetch the full parent
after retrieval, giving the LLM more context.

Tables are ALWAYS one atomic chunk (never split across chunks).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.config_loader import cfg
from src.logger import get_logger
from src.ingestion.pdf_parser import PageContent, TableData
from src.ingestion.metadata_tagger import DocMetadata, build_section_path

log = get_logger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single retrievable/indexable unit of content."""
    chunk_id: str
    parent_id: Optional[str]      # None for parent chunks themselves
    doc_id: str
    release_version: str
    is_latest: bool
    section: str
    page_number: int
    chunk_type: str               # "text" | "table"
    content: str                  # The actual text / Markdown table
    table_caption: str = ""       # Only for chunk_type == "table"
    # Computed after embedding
    embedding: Optional[list[float]] = field(default=None, repr=False)


# ── Token counting (rough, avoids heavy tokeniser) ────────────────────────────

def _approx_tokens(text: str) -> int:
    """~1 token per 4 characters — fast approximation."""
    return len(text) // 4


# ── Text splitter (sentence-aware) ────────────────────────────────────────────

def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split text into chunks respecting sentence boundaries.
    Overlap is measured in approximate tokens.
    """
    # Split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N tokens worth of sentences
            overlap_buf: list[str] = []
            ob_tokens = 0
            for s in reversed(current):
                st = _approx_tokens(s)
                if ob_tokens + st > overlap_tokens:
                    break
                overlap_buf.insert(0, s)
                ob_tokens += st
            current = overlap_buf
            current_tokens = ob_tokens
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ── Main chunker ──────────────────────────────────────────────────────────────

class ParentChildChunker:
    """
    Produces parent + child chunk pairs from a list of PageContent objects.

    Flow:
      1. For each page, split body text into parent chunks.
      2. Each parent chunk is further split into child chunks.
      3. Each table on the page becomes ONE parent chunk (atomic) with
         child chunks containing: caption + first 3 rows as context.
    """

    def __init__(self):
        self.parent_size = cfg.chunking.parent_chunk_size
        self.child_size = cfg.chunking.child_chunk_size
        self.overlap = cfg.chunking.chunk_overlap

    def chunk_document(
        self,
        pages: list[PageContent],
        doc_meta: DocMetadata,
        is_latest: bool,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Returns:
            parents: list[Chunk]  — stored for context retrieval
            children: list[Chunk] — indexed in Qdrant for search
        """
        parents: list[Chunk] = []
        children: list[Chunk] = []

        for page in pages:
            section = build_section_path(page)

            # ── Body text ──────────────────────────────────────────────
            if page.text:
                parent_texts = _split_text(
                    page.text, self.parent_size, self.overlap
                )
                for parent_text in parent_texts:
                    parent_id = str(uuid.uuid4())
                    parent = Chunk(
                        chunk_id=parent_id,
                        parent_id=None,
                        doc_id=doc_meta.doc_id,
                        release_version=doc_meta.release_version,
                        is_latest=is_latest,
                        section=section,
                        page_number=page.page_number,
                        chunk_type="text",
                        content=parent_text,
                    )
                    parents.append(parent)

                    child_texts = _split_text(
                        parent_text, self.child_size, self.overlap // 2
                    )
                    for child_text in child_texts:
                        child = Chunk(
                            chunk_id=str(uuid.uuid4()),
                            parent_id=parent_id,
                            doc_id=doc_meta.doc_id,
                            release_version=doc_meta.release_version,
                            is_latest=is_latest,
                            section=section,
                            page_number=page.page_number,
                            chunk_type="text",
                            content=child_text,
                        )
                        children.append(child)

            # ── Tables (atomic chunks) ─────────────────────────────────
            for table in page.tables:
                table_content = self._format_table_chunk(table, section)
                parent_id = str(uuid.uuid4())

                parent = Chunk(
                    chunk_id=parent_id,
                    parent_id=None,
                    doc_id=doc_meta.doc_id,
                    release_version=doc_meta.release_version,
                    is_latest=is_latest,
                    section=section,
                    page_number=page.page_number,
                    chunk_type="table",
                    content=table_content,
                    table_caption=table.caption,
                )
                parents.append(parent)

                # Child for a table = caption + first 3 rows (context hint)
                preview = self._table_preview(table)
                child = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    doc_id=doc_meta.doc_id,
                    release_version=doc_meta.release_version,
                    is_latest=is_latest,
                    section=section,
                    page_number=page.page_number,
                    chunk_type="table",
                    content=preview,
                    table_caption=table.caption,
                )
                children.append(child)

        log.info(
            f"[{doc_meta.doc_id} v{doc_meta.release_version}] "
            f"{len(parents)} parent chunks, {len(children)} child chunks"
        )
        return parents, children

    # ── Helpers ────────────────────────────────────────────────────────────

    def _format_table_chunk(self, table: TableData, section: str) -> str:
        """Full table as searchable text: caption + markdown."""
        parts = []
        if table.caption:
            parts.append(f"Table caption: {table.caption}")
        parts.append(f"Section: {section}")
        parts.append(table.markdown)
        return "\n\n".join(parts)

    def _table_preview(self, table: TableData) -> str:
        """Short preview used as the child chunk for indexing."""
        lines = table.markdown.split("\n")
        # header + separator + first 3 data rows
        preview_lines = lines[:5]
        caption = f"Caption: {table.caption}\n" if table.caption else ""
        return caption + "\n".join(preview_lines)
