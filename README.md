# Hybrid RAG — Product Documentation Agent

A RAG system for answering questions over versioned PDF product documentation.

It combines:

- **Dense retrieval** — Qdrant vector search (embedded, no Docker)
- **Sparse retrieval** — BM25 keyword search
- **Cross-encoder reranking** — Cohere Rerank API
- **Grounded generation** — Gemini with mandatory source citations

PDF parsing, vector storage, and keyword search run locally. Gemini and Cohere are called through their APIs.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Model setup](#model-setup)
5. [Add your documents](#add-your-documents)
6. [Index documents (ingestion)](#index-documents-ingestion)
7. [Ask questions](#ask-questions)
8. [Configuration guide](#configuration-guide)
9. [Project structure](#project-structure)
10. [How the pipeline works](#how-the-pipeline-works)
11. [Troubleshooting](#troubleshooting)
12. [Development & testing](#development--testing)
13. [What not to commit](#what-not-to-commit)

---

## What this project does

1. **Ingest** PDF manuals from the `docs/` folder.
2. **Chunk** them into parent/child pairs (small chunks for search, large chunks for LLM context).
3. **Index** embeddings in Qdrant and keywords in BM25.
4. **Answer** user questions with cited, validated responses.

The agent understands **document versions**. You can ask:

- `How do I configure SSL?` → searches the latest indexed version
- `How do I configure SSL in v2.3?` → searches version 2.3 specifically
- `Compare SSL settings across all versions` → searches every indexed version

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11+** | Tested on 3.13 (Windows) |
| **Gemini API key** | Used for embeddings and answer generation |
| **Cohere API key** | Used for reranking |
| **Internet access** | Required for Gemini and Cohere API requests |
| **Git** | To clone the repository |

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Hybrid_RAG
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This single file includes runtime dependencies, Streamlit, and pytest.

---

## Model setup

The default configuration uses Gemini for embeddings and answer generation, and Cohere for reranking. API keys are loaded from the project `.env` file.

### Gemini and Cohere API keys (required)

Add these variables to `.env` (never commit the file):

```bash
GEMINI_API_KEY=your-gemini-api-key
COHERE_API_KEY=your-cohere-api-key
```

You can start from the committed template with `Copy-Item .env.example .env` in PowerShell or `cp .env.example .env` on macOS/Linux.

| Model | Purpose | Used for |
|-------|---------|----------|
| `gemini-2.5-flash` | Chat / answer generation | `llm.gemini.model` |
| `gemini-embedding-001` | Embeddings (768-dim) | `embeddings.model_name` |
| `rerank-v3.5` | Reranks retrieved chunks | `reranker.model` |

Restart the application after changing API keys or model names.

---

## Add your documents

1. Place PDF files in the `docs/` folder (subfolders are supported).
2. Name files with version hints when possible:

   | Filename | Detected version |
   |----------|------------------|
   | `ProductGuide_v2.3.1.pdf` | `2.3.1` |
   | `manual-3.0.pdf` | `3.0` |
   | `release_1.5.0_notes.pdf` | `1.5.0` |
   | `UserGuide.pdf` | `1.0.0` (default) |

3. PDFs are **not** included in the repository by default — you add your own.

---

## Index documents (ingestion)

Run the ingestion pipeline after adding or updating PDFs:

```bash
python scripts/ingest.py
```

### What ingestion does

1. Scans `docs/` for PDFs
2. Detects changed pages (incremental — only re-processes diffs)
3. Extracts text and tables (PyMuPDF + pdfplumber)
4. Chunks into parent/child pairs
5. Embeds child chunks via Gemini
6. Stores vectors in Qdrant (embedded mode, under `cache/qdrant_data/`)
7. Updates the BM25 keyword index
8. Registers document versions

### Useful ingestion flags

```bash
# Re-index everything from scratch (ignore change detection)
python scripts/ingest.py --force

# Ingest a single file
python scripts/ingest.py --file ProductGuide_v2.3.1.pdf

# Manually set which version is "latest"
python scripts/ingest.py --set-latest 3.0.0

# Use a custom docs folder
python scripts/ingest.py --docs-dir path/to/pdfs
```

### Expected output

On success you should see:

- Versions indexed (e.g. `['2.3.1', '3.0.0']`)
- Latest version set
- Parent chunk and BM25 counts

Generated files (do not edit manually):

```
cache/
├── qdrant_data/          # Vector database
├── bm25/bm25_index.pkl   # Keyword index
├── parent_store.json     # Full parent chunks for LLM context
├── version_registry.json # Version → collection mapping
└── page_hashes.json      # Incremental ingest hashes
```

---

## Ask questions

You need to complete [installation](#installation), [model setup](#model-setup), and [ingestion](#index-documents-ingestion) first.

### Option A — Streamlit web UI (recommended)

```bash
streamlit run streamlit_app.py
```

Opens a browser chat interface with:

- Conversation history
- Source citations per answer
- Optional version lock in the sidebar
- Error messages if something fails

### Option B — Terminal chat

```bash
python scripts/chat.py
```

**Built-in commands:**

| Command | Action |
|---------|--------|
| `/help` | Show help |
| `/versions` | List indexed versions |
| `/latest` | Show latest version |
| `/version 2.3.1` | Lock queries to version 2.3.1 |
| `/unlock` | Remove version lock |
| `/quit` | Exit |

Lock a version from the start:

```bash
python scripts/chat.py --version 2.3.1
```

### Option C — Python API

```python
from src.agent import DocumentationAgent

agent = DocumentationAgent()
result = agent.ask("How do I configure SSL in v2.3?")

print(result.answer)
print(result.sources)
print(f"Latency: {result.latency_ms:.0f} ms")
```

---

## Configuration guide

All settings live in **`config/config.yaml`**. Edit that file directly — that is the normal way to configure the project.

### Optional: `.env` file

Create a file named **`.env`** in the project root (same folder as `README.md`). It is git-ignored and overrides `config/config.yaml`.

Prefix: `AGENT_` — double underscore `__` maps to nested YAML keys:

| `.env` variable | Maps to `config.yaml` |
|-----------------|----------------------|
| `AGENT_EMBEDDINGS__MODEL_NAME` | `embeddings.model_name` |
| `AGENT_LOGGING__LEVEL` | `logging.level` |

Example `.env`:

```bash
GEMINI_API_KEY=your-gemini-api-key
COHERE_API_KEY=your-cohere-api-key
AGENT_LOGGING__LEVEL=INFO
```

You can use **only** `config/config.yaml` and skip `.env` entirely — both work.

### Key settings to know

| Setting | File section | What it controls |
|---------|--------------|------------------|
| `paths.docs_dir` | paths | Where PDFs are read from |
| `embeddings.model_name` | embeddings | Gemini embedding model |
| `qdrant.vector_size` | qdrant | Must match embedding dimensions (768) |
| `qdrant.connection` | qdrant | `embedded` (default) or `server` |
| `llm.gemini.model` | llm | Gemini chat model |
| `reranker.model` | reranker | Cohere rerank model |
| `multi_query.enabled` | multi_query | `false` by default (avoids extra API calls) |
| `generation.confidence_threshold` | generation | Below this rerank score → "not found" |
| `generation.require_citations` | generation | Answers without citations are rejected |

---

## Project structure

```
Hybrid_RAG/
├── config/
│   └── config.yaml          # Main configuration
├── docs/
│   ├── README.md            # PDF placement instructions
│   ├── ARCHITECTURE.md      # Technical deep-dive
│   └── *.pdf                # Your PDF files (add locally; not in git)
├── cache/                   # Generated indexes (not in git)
├── logs/                    # Application logs
├── scripts/
│   ├── ingest.py            # Document indexing pipeline
│   └── chat.py              # Terminal chat interface
├── src/
│   ├── agent.py             # Main orchestrator (DocumentationAgent)
│   ├── config_loader.py     # Config + .env loading
│   ├── ingestion/           # PDF parsing, chunking, metadata
│   ├── indexing/            # Embeddings, Qdrant, BM25
│   ├── retrieval/           # Hybrid search, reranking, query analysis
│   ├── generation/          # LLM client, prompts, validation
│   └── versioning/          # Version registry, diff detection
├── tests/
│   └── test_pipeline.py     # Unit / smoke tests
├── streamlit_app.py         # Web UI
├── requirements.txt         # All Python dependencies (runtime + tests)
└── .gitignore
```

---

## How the pipeline works

### Ingestion

```
PDF → parse → chunk (parent/child) → embed → Qdrant + BM25
```

### Query (each question)

```
Question
  → detect version intent
  → hybrid retrieve (dense + BM25, fused with RRF)
  → rerank top candidates
  → build cited context prompt
  → Gemini generates answer
  → validate citations (hallucination firewall)
  → return answer + sources
```

For a detailed diagram and component breakdown, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Troubleshooting

### `ModuleNotFoundError` (colorlog, qdrant_client, etc.)

```bash
pip install -r requirements.txt
```

Make sure your virtual environment is activated.

### API authentication or connectivity errors

Confirm `GEMINI_API_KEY` and `COHERE_API_KEY` are set in `.env`, the model names in `config/config.yaml` are available to your accounts, and the machine can reach both APIs.

### `BM25 index not found`

You have not ingested any documents yet:

```bash
python scripts/ingest.py
```

### Answers always say "not available in the documentation"

Possible causes:

1. **No PDFs indexed** — run ingestion
2. **Wrong version** — check `/versions` in chat or the Streamlit sidebar
3. **Low retrieval confidence** — lower `generation.confidence_threshold` in config
4. **Embedding model changed** — re-ingest after switching models

### Ingestion is slow

Normal on first run. Subsequent runs use incremental diff detection and only process changed pages. Use `--force` only when you need a full rebuild.

### Vector dimension mismatch

If you switch embedding models, update `qdrant.vector_size` to match:

| Model | Dimensions |
|-------|------------|
| `gemini-embedding-001` | 768 |

Then re-ingest with `--force` so all stored vectors are regenerated:

```bash
python scripts/ingest.py --force
```

### Scanned (image-only) PDFs

Enable OCR in config (requires extra packages):

```yaml
ingestion:
  enable_ocr: true
```

Uncomment `paddlepaddle` and `paddleocr` in `requirements.txt`, then reinstall.

---

## Development & testing

```bash
python -m pytest tests/ -v
```

Tests cover chunking, version detection, query analysis, BM25, answer validation, and diff detection without making Gemini or Cohere API calls.

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
```

Logs are written to `logs/agent.log` when `log_to_file: true`.

---

## What not to commit

The `.gitignore` excludes:

| Path | Reason |
|------|--------|
| `.env` | May contain local overrides |
| `cache/` | Regenerated by ingestion |
| `logs/` | Runtime logs |
| `.venv/` | Virtual environment |
| `docs/*.pdf` | Your proprietary documents (add your own policy) |

Before pushing to GitHub, confirm you are not committing sensitive PDFs, API keys, or large model files.

---

## Quick reference

```bash
# Setup
pip install -r requirements.txt
# Index
python scripts/ingest.py

# Chat
streamlit run streamlit_app.py
python scripts/chat.py

# Test
python -m pytest tests/ -v
```

---

## License

Add a `LICENSE` file before publishing if you plan to make the repository public.
