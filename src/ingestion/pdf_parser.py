"""
ingestion/pdf_parser.py
=======================
Extracts structured text and table content from PDF files using:
  • PyMuPDF  (fitz)     — fast text/layout extraction
  • pdfplumber          — accurate table detection & cell merging
  • PaddleOCR (optional)— fallback for scanned/image-only PDFs

Returns a list of PageContent objects, one per PDF page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from src.config_loader import cfg
from src.logger import get_logger

log = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TableData:
    """A table extracted from a PDF page."""
    markdown: str          # Pipe-delimited Markdown representation
    raw_rows: list[list]   # Original cell data for debugging
    bbox: tuple            # (x0, y0, x1, y1) on the page
    caption: str = ""      # Nearest heading/caption text


@dataclass
class PageContent:
    """All content extracted from one PDF page."""
    doc_path: str
    page_number: int       # 1-based
    text: str              # Clean body text (tables excluded)
    tables: list[TableData] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    is_scanned: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _table_to_markdown(table: list[list]) -> str:
    """Convert a 2-D list (rows × cols) to a Markdown pipe table."""
    if not table or not table[0]:
        return ""
    # Normalise cells: strip whitespace, replace None with empty string
    rows = [[str(cell).strip() if cell is not None else "" for cell in row]
            for row in table]
    col_count = max(len(r) for r in rows)
    # Pad short rows
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header = rows[0]
    separator = ["---"] * col_count
    body = rows[1:]

    lines = [_row(header), _row(separator)] + [_row(r) for r in body]
    return "\n".join(lines)


def _is_scanned_page(page: fitz.Page) -> bool:
    """Return True if the page has no selectable text (likely a scan)."""
    text = page.get_text("text").strip()
    return len(text) < 20  # very few characters → probably an image


def _extract_headings(page: fitz.Page) -> list[str]:
    """Heuristic: extract lines whose font size is above average."""
    blocks = page.get_text("dict")["blocks"]
    sizes = []
    lines_with_size = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                sizes.append(span["size"])
                lines_with_size.append((span["size"], span["text"].strip()))

    if not sizes:
        return []
    avg_size = sum(sizes) / len(sizes)
    headings = [txt for size, txt in lines_with_size
                if size > avg_size * 1.15 and txt]
    return list(dict.fromkeys(headings))  # deduplicate preserving order


# ── OCR (optional fallback) ───────────────────────────────────────────────────

def _ocr_page(page: fitz.Page) -> str:
    """Use PaddleOCR to extract text from an image-only page."""
    try:
        from paddleocr import PaddleOCR  # lazy import
        import numpy as np

        ocr = PaddleOCR(use_angle_cls=True, lang=cfg.ingestion.ocr_language,
                        show_log=False)
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        result = ocr.ocr(img, cls=True)
        lines = []
        for line_group in result:
            if line_group:
                for det in line_group:
                    lines.append(det[1][0])
        return "\n".join(lines)
    except ImportError:
        log.warning("PaddleOCR not installed. Scanned page will be empty.")
        return ""


# ── Main parser ───────────────────────────────────────────────────────────────

class PDFParser:
    """
    Parse a single PDF file into a list of PageContent objects.

    Strategy per page:
      1. PyMuPDF extracts all text + detects headings.
      2. pdfplumber detects tables; we extract them and remove their
         bounding-box text from the PyMuPDF text to avoid duplication.
      3. If the page has no text (scanned), optionally run OCR.
    """

    def parse(self, pdf_path: str | Path) -> list[PageContent]:
        pdf_path = Path(pdf_path)
        log.info(f"Parsing: {pdf_path.name}")

        pages: list[PageContent] = []

        fitz_doc = fitz.open(str(pdf_path))
        plumber_doc = pdfplumber.open(str(pdf_path))

        try:
            for page_idx in range(len(fitz_doc)):
                fitz_page = fitz_doc[page_idx]
                plumber_page = plumber_doc.pages[page_idx]
                page_num = page_idx + 1

                scanned = _is_scanned_page(fitz_page)

                # ── 1. Extract tables via pdfplumber ──────────────────────
                table_objects: list[TableData] = []
                table_bboxes: list[tuple] = []

                for tbl in plumber_page.extract_tables():
                    if not tbl:
                        continue
                    md = _table_to_markdown(tbl)
                    if not md:
                        continue
                    # Get bbox of the first detected table region on this page
                    found_tables = plumber_page.find_tables()
                    bbox = found_tables[len(table_bboxes)].bbox \
                        if len(found_tables) > len(table_bboxes) else (0, 0, 0, 0)
                    caption = self._find_caption(fitz_page, bbox)
                    table_objects.append(TableData(
                        markdown=md, raw_rows=tbl,
                        bbox=bbox, caption=caption
                    ))
                    table_bboxes.append(bbox)

                # ── 2. Extract body text (mask out table regions) ─────────
                if scanned and cfg.ingestion.enable_ocr:
                    body_text = _ocr_page(fitz_page)
                else:
                    body_text = self._extract_text_excluding_tables(
                        fitz_page, plumber_page, table_bboxes
                    )

                headings = _extract_headings(fitz_page)

                pages.append(PageContent(
                    doc_path=str(pdf_path),
                    page_number=page_num,
                    text=body_text.strip(),
                    tables=table_objects,
                    headings=headings,
                    is_scanned=scanned,
                ))

        finally:
            fitz_doc.close()
            plumber_doc.close()

        log.info(f"  → {len(pages)} pages, "
                 f"{sum(len(p.tables) for p in pages)} tables extracted")
        return pages

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_text_excluding_tables(
        self,
        fitz_page: fitz.Page,
        plumber_page,
        table_bboxes: list[tuple],
    ) -> str:
        """Extract PyMuPDF text, skipping text blocks that overlap with tables."""
        blocks = fitz_page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,type)
        page_h = fitz_page.rect.height
        # pdfplumber uses top-origin coords; fitz uses top-origin too (same)

        lines = []
        for block in blocks:
            if block[6] != 0:  # skip image blocks
                continue
            bx0, by0, bx1, by1 = block[:4]
            txt = block[4].strip()
            if not txt:
                continue
            # Check overlap with any table bbox
            overlap = False
            for tx0, ty0, tx1, ty1 in table_bboxes:
                # Simple overlap check
                if not (bx1 < tx0 or bx0 > tx1 or by1 < ty0 or by0 > ty1):
                    overlap = True
                    break
            if not overlap:
                lines.append(txt)

        return "\n".join(lines)

    def _find_caption(self, fitz_page: fitz.Page, table_bbox: tuple) -> str:
        """Find the nearest text line just above the table (caption heuristic)."""
        tx0, ty0, tx1, ty1 = table_bbox
        blocks = fitz_page.get_text("blocks")
        candidates = []
        for block in blocks:
            if block[6] != 0:
                continue
            by1 = block[3]
            txt = block[4].strip()
            if txt and by1 <= ty0 and (ty0 - by1) < 60:  # within 60pt above
                candidates.append((ty0 - by1, txt))
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else ""
