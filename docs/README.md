# Documentation PDFs

Place product documentation PDF files in this directory (or subdirectories).

The ingestion script scans recursively:

```bash
python scripts/ingest.py
```

Version tags are inferred from filenames (e.g. `ProductGuide_v2.3.1.pdf`) or document text. Override the latest version manually:

```bash
python scripts/ingest.py --set-latest 3.0.0
```
