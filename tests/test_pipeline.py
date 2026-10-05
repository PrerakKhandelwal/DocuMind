"""
tests/test_pipeline.py
========================
Basic smoke tests — verify each component initialises and functions
without a full GPU or Qdrant server (uses mocks where needed).

Run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── Chunker ────────────────────────────────────────────────────────────────

def test_text_splitter():
    from src.ingestion.chunker import _split_text
    # Splitter groups by sentences (. ! ?); punctuation is required for multiple chunks.
    text = ". ".join(["word"] * 500) + "."
    chunks = _split_text(text, max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) > 0


def test_table_to_markdown():
    from src.ingestion.pdf_parser import _table_to_markdown
    table = [
        ["Name", "Value", "Description"],
        ["timeout", "30s", "Connection timeout"],
        ["retries", "3", "Max retry count"],
    ]
    md = _table_to_markdown(table)
    assert "| Name | Value | Description |" in md
    assert "| timeout | 30s | Connection timeout |" in md
    assert "---" in md


# ── Metadata tagger ────────────────────────────────────────────────────────

def test_version_from_filename():
    from src.ingestion.metadata_tagger import _extract_version_from_filename
    assert _extract_version_from_filename("ProductGuide_v2.3.1.pdf") == "2.3.1"
    assert _extract_version_from_filename("manual-3.0.pdf") == "3.0"
    assert _extract_version_from_filename("release_1.5.0_notes.pdf") == "1.5.0"
    assert _extract_version_from_filename("UserGuide.pdf") is None


def test_version_from_text():
    from src.ingestion.metadata_tagger import _extract_version_from_text
    assert _extract_version_from_text("Version: 2.3.1") == "2.3.1"
    assert _extract_version_from_text("Release 3.0.0 documentation") == "3.0.0"
    assert _extract_version_from_text("No version here") is None


# ── Query analyser ─────────────────────────────────────────────────────────

def test_query_analyser_version_extraction():
    from src.retrieval.query_analyser import analyse_query
    result = analyse_query("How do I configure SSL in v2.3?")
    assert result.requested_version == "2.3"
    assert "SSL" in result.clean_query or "configure" in result.clean_query


def test_query_analyser_no_version():
    from src.retrieval.query_analyser import analyse_query
    result = analyse_query("What are the default timeout values?")
    assert result.requested_version is None
    assert result.search_all_versions is False


def test_query_analyser_all_versions():
    from src.retrieval.query_analyser import analyse_query
    result = analyse_query("Compare SSL settings across all versions")
    assert result.search_all_versions is True


# ── BM25 index ─────────────────────────────────────────────────────────────

def test_bm25_basic():
    from src.indexing.bm25_index import BM25Index
    from src.ingestion.chunker import Chunk
    import uuid

    idx = BM25Index()
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            parent_id=str(uuid.uuid4()),
            doc_id="test_doc",
            release_version="1.0.0",
            is_latest=True,
            section="Test Section",
            page_number=1,
            chunk_type="text",
            content="The SSL certificate must be placed in /etc/ssl/certs/",
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            parent_id=str(uuid.uuid4()),
            doc_id="test_doc",
            release_version="1.0.0",
            is_latest=True,
            section="Network Settings",
            page_number=5,
            chunk_type="text",
            content="Configure the timeout value in the network configuration file",
        ),
    ]
    idx.add_chunks(chunks)
    idx.build()

    results = idx.search("SSL certificate path", top_k=5)
    assert len(results) > 0
    # SSL-related chunk should rank first
    assert "ssl" in results[0]["payload"]["content"].lower() or \
           "certificate" in results[0]["payload"]["content"].lower()


# ── Answer validator ────────────────────────────────────────────────────────

def test_validator_passes_cited_answer():
    from src.generation.answer_validator import validate_answer
    chunks = [{"content": "some context", "doc_id": "test", "section": "A", "page_number": 1}]
    answer = "The timeout is 30s [Doc: test | Version: 1.0 | Section: A | Page: 1]"
    result = validate_answer(answer, chunks)
    assert result == answer


def test_validator_blocks_uncited_answer():
    from src.generation.answer_validator import validate_answer
    chunks = [{"content": "some context"}]
    answer = "The timeout is definitely 30 seconds."  # No citation
    result = validate_answer(answer, chunks)
    # Should return the not-found message
    assert "not available" in result.lower() or result != answer


def test_validator_no_chunks():
    from src.generation.answer_validator import validate_answer
    result = validate_answer("Some answer", [])
    assert "not available" in result.lower()


# ── Diff detector ───────────────────────────────────────────────────────────

def test_diff_detector_hash_change():
    from src.versioning.diff_detector import DiffDetector
    from src.ingestion.pdf_parser import PageContent

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override cache path
        detector = DiffDetector()
        detector._cache_path = Path(tmpdir) / "hashes.json"

        page = PageContent(
            doc_path="test.pdf",
            page_number=1,
            text="Original content",
            tables=[],
            headings=[],
        )

        # First run: everything is "changed"
        changed = detector.get_changed_pages("test.pdf", [page])
        assert len(changed) == 1

        detector.update_hashes("test.pdf", [page])

        # Second run: same content → no changes
        changed2 = detector.get_changed_pages("test.pdf", [page])
        assert len(changed2) == 0

        # Modify content → detected as changed
        page.text = "Modified content"
        changed3 = detector.get_changed_pages("test.pdf", [page])
        assert len(changed3) == 1
