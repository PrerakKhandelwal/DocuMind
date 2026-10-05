"""
scripts/ingest.py
==================
Master ingestion pipeline — run this to index your documentation.

What it does:
  1. Scans docs_dir for all PDF files
  2. Detects which pages changed since last run (incremental)
  3. Parses PDF text + tables
  4. Chunks into parent/child pairs
  5. Embeds child chunks with Gemini
  6. Upserts into Qdrant (per-version collection)
  7. Adds to BM25 index
  8. Updates version registry and diff hashes
  9. Saves parent store to disk

Usage:
  python scripts/ingest.py                     # ingest all PDFs
  python scripts/ingest.py --force             # re-ingest everything (ignore diff)
  python scripts/ingest.py --file guide_v2.pdf # single file
  python scripts/ingest.py --set-latest 2.3.1  # manually set latest version
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from tqdm import tqdm

from src.config_loader import cfg
from src.logger import get_logger
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.metadata_tagger import build_doc_metadata
from src.ingestion.chunker import ParentChildChunker
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore, ParentStore
from src.indexing.bm25_index import BM25Index
from src.versioning.diff_detector import DiffDetector
from src.versioning.version_manager import VersionManager

log = get_logger("ingest")

def _collect_pdfs(docs_dir: str, single_file: str | None) -> list[Path]:
    base = Path(docs_dir)
    if not base.exists():
        log.error(f"docs_dir does not exist: {base}")
        sys.exit(1)
    if single_file:
        p = base / single_file
        if not p.exists():
            log.error(f"File not found: {p}")
            sys.exit(1)
        return [p]
    pdfs = sorted(base.rglob("*.pdf"))
    log.info(f"Found {len(pdfs)} PDF(s) in {base}")
    return pdfs


def _determine_latest(version_mgr: VersionManager, versions_ingested: list[str]) -> str:
    """
    Simple heuristic: the lexicographically largest semver is the latest.
    Override manually with --set-latest if needed.
    """
    from packaging.version import Version, InvalidVersion
    valid = []
    for v in version_mgr.all_versions():
        try:
            valid.append((Version(v), v))
        except InvalidVersion:
            valid.append((Version("0.0.0"), v))  # fallback
    if not valid:
        return cfg.ingestion.default_version
    valid.sort(reverse=True)
    return valid[0][1]


@click.command()
@click.option("--force", is_flag=True, help="Re-ingest all pages regardless of changes")
@click.option("--file", "single_file", default=None, help="Ingest a single PDF file")
@click.option("--set-latest", "set_latest", default=None,
              help="Manually set a version as latest (e.g. 3.0.0)")
@click.option("--docs-dir", default=None, help="Override docs_dir from config")
def main(force: bool, single_file: str, set_latest: str, docs_dir: str):
    t0 = time.time()
    log.info("═" * 60)
    log.info("Product Documentation Agent — Ingestion Pipeline")
    log.info("═" * 60)

    docs_path = docs_dir or cfg.paths.docs_dir
    pdfs = _collect_pdfs(docs_path, single_file)

    # ── Initialise components ──────────────────────────────────────────
    parser = PDFParser()
    chunker = ParentChildChunker()
    embedder = Embedder()
    vector_store = VectorStore()
    bm25_index = BM25Index()
    parent_store = ParentStore(cfg.paths.parent_store)
    diff_detector = DiffDetector()
    version_mgr = VersionManager()

    # Load existing BM25 index (we'll rebuild at the end)
    bm25_index.load()

    new_children_all = []
    versions_seen = set()

    # ── Process each PDF ───────────────────────────────────────────────
    for pdf_path in tqdm(pdfs, desc="PDFs", unit="file"):
        log.info(f"\n{'─'*50}")
        log.info(f"Processing: {pdf_path.name}")

        try:
            # 1. Parse all pages
            all_pages = parser.parse(pdf_path)
            if not all_pages:
                log.warning(f"No content extracted from {pdf_path.name}")
                continue

            # 2. Build document metadata (version detection)
            doc_meta = build_doc_metadata(pdf_path, all_pages)

            # 3. Detect changed pages (skip if --force)
            if force:
                changed_pages = all_pages
            else:
                changed_pages = diff_detector.get_changed_pages(pdf_path, all_pages)

            if not changed_pages:
                log.info(f"  No changes — skipping {pdf_path.name}")
                continue

            # 4. Chunk changed pages only
            parents, children = chunker.chunk_document(
                pages=changed_pages,
                doc_meta=doc_meta,
                is_latest=False,  # will be set correctly after all docs are processed
            )

            # 5. Ensure Qdrant collection exists for this version
            col_name = vector_store.ensure_collection(doc_meta.release_version)

            # 6. Embed child chunks
            if children:
                log.info(f"  Embedding {len(children)} child chunks…")
                # Stream embedding + upsert in small batches to keep RAM usage stable
                # on low-memory machines (e.g. 8 GB).
                embed_batch_size = max(1, int(getattr(cfg.embeddings, "batch_size", 1)))
                for i in range(0, len(children), embed_batch_size):
                    child_batch = children[i : i + embed_batch_size]
                    text_batch = [c.content for c in child_batch]
                    embeddings = embedder.embed(text_batch, show_progress=False)
                    vector_store.upsert_chunks(child_batch, embeddings, col_name)
                new_children_all.extend(children)
            

            # 7. Store parent chunks in parent store
            for parent in parents:
                parent_store.add(parent)

            # 8. Register version
            version_mgr.register_version(
                doc_meta.release_version, col_name,
                doc_count=len(pdfs)
            )
            versions_seen.add(doc_meta.release_version)

            # 9. Update diff hashes
            diff_detector.update_hashes(pdf_path, changed_pages)

        except Exception as e:
            log.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
            continue

    # ── Rebuild BM25 index with all new chunks ─────────────────────────
    if new_children_all:
        log.info(f"\nAdding {len(new_children_all)} new chunks to BM25 index…")
        bm25_index.add_chunks(new_children_all)
        bm25_index.build()
        bm25_index.save()

    # ── Set latest version ─────────────────────────────────────────────
    if set_latest:
        version_mgr.set_latest(set_latest)
    elif versions_seen:
        latest = _determine_latest(version_mgr, list(versions_seen))
        version_mgr.set_latest(latest)
        log.info(f"Auto-detected latest version: {latest}")

    # Update is_latest flag in BM25 metadata (in-memory is fine here)
    # The Qdrant payload's is_latest field is informational;
    # the real routing happens via version_mgr.get_latest()

    # ── Save state ─────────────────────────────────────────────────────
    parent_store.save()
    diff_detector.save()
    version_mgr.save()

    elapsed = time.time() - t0
    log.info(f"\n{'═'*60}")
    log.info(f"Ingestion complete in {elapsed:.1f}s")
    log.info(f"  Versions indexed : {version_mgr.all_versions()}")
    log.info(f"  Latest version   : {version_mgr.get_latest()}")
    log.info(f"  Parent chunks    : {len(parent_store)}")
    log.info(f"  BM25 chunks      : {len(new_children_all)} new")
    log.info(f"{'═'*60}")


if __name__ == "__main__":
    main()
