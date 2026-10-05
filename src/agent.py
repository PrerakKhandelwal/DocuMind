"""
src/agent.py
=============
The Product Documentation Agent.

Orchestrates the full query pipeline:
  query → analyse → sub-queries → hybrid retrieve → rerank → generate → validate

Usage (programmatic):
    agent = DocumentationAgent()
    result = agent.ask("How do I configure SSL in v2.3?")
    print(result.answer)
    print(result.sources)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.config_loader import cfg
from src.logger import get_logger
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore, ParentStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.query_analyser import analyse_query
from src.retrieval.multi_query import generate_sub_queries
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm_client import create_llm_client
from src.generation.context_builder import build_context_prompt
from src.generation.answer_validator import validate_answer
from src.versioning.version_manager import VersionManager

log = get_logger("agent")


@dataclass
class AgentResult:
    """Structured response from the agent."""
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    version_used: Optional[str] = None
    num_chunks_retrieved: int = 0
    latency_ms: float = 0.0
    sub_queries: list[str] = field(default_factory=list)

    def pretty_print(self):
        """Print a formatted response to the terminal."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich import box

        console = Console()

        console.print(Panel(
            f"[bold cyan]{self.question}[/bold cyan]",
            title="[bold]Question[/bold]",
            box=box.ROUNDED,
        ))

        console.print(Panel(
            Markdown(self.answer),
            title="[bold green]Answer[/bold green]",
            box=box.ROUNDED,
        ))

        if self.sources:
            source_lines = []
            seen = set()
            for src in self.sources:
                key = (src.get('doc_id'), src.get('section'), src.get('page_number'))
                if key in seen:
                    continue
                seen.add(key)
                source_lines.append(
                    f"  • [yellow]{src.get('doc_id', '?')}[/yellow] "
                    f"v{src.get('release_version', '?')} | "
                    f"{src.get('section', '?')} | "
                    f"Page {src.get('page_number', '?')}"
                )
            console.print(Panel(
                "\n".join(source_lines),
                title="[bold]Sources[/bold]",
                box=box.ROUNDED,
            ))

        console.print(
            f"[dim]Retrieved {self.num_chunks_retrieved} chunks | "
            f"Version: {self.version_used or 'latest'} | "
            f"{self.latency_ms:.0f}ms[/dim]"
        )


class DocumentationAgent:
    """
    Full RAG agent for product documentation.

    Initialise once, then call .ask() repeatedly.
    All components are loaded once into memory at startup.
    """

    def __init__(self):
        log.info("Initialising Documentation Agent…")
        t0 = time.time()

        self._version_mgr = VersionManager()
        self._embedder = Embedder()
        self._vector_store = VectorStore()

        self._parent_store = ParentStore(cfg.paths.parent_store)
        self._bm25 = BM25Index()

        loaded = self._bm25.load()
        if not loaded:
            log.warning(
                "BM25 index not found. Run 'python scripts/ingest.py' first."
            )

        self._retriever = HybridRetriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            bm25_index=self._bm25,
            parent_store=self._parent_store,
        )
        self._reranker = Reranker()
        self._llm = create_llm_client()

        elapsed = (time.time() - t0) * 1000
        log.info(f"Agent ready in {elapsed:.0f}ms")
        log.info(f"  Known versions : {self._version_mgr.all_versions()}")
        log.info(f"  Latest version : {self._version_mgr.get_latest()}")

    def ask(self, question: str) -> AgentResult:
        """
        Ask a question and return a grounded, cited answer.

        Args:
            question: Natural language question from the user

        Returns:
            AgentResult with answer, sources, latency, etc.
        """
        t0 = time.time()
        log.info(f"\n{'━'*60}")
        log.info(f"Question: {question}")

        # ── 1. Analyse query for version / intent ──────────────────
        intent = analyse_query(question)
        log.info(
            f"Intent: version={intent.requested_version}, "
            f"all={intent.search_all_versions}"
        )

        # ── 2. Determine which version to search ───────────────────
        latest_version = self._version_mgr.get_latest()

        if intent.search_all_versions:
            version_filter = None
            latest_only = False
            version_used = "all"
        elif intent.requested_version:
            # Check the requested version is actually indexed
            if intent.requested_version in self._version_mgr.all_versions():
                version_filter = intent.requested_version
                latest_only = False
                version_used = intent.requested_version
            else:
                log.warning(
                    f"Requested version {intent.requested_version} not indexed. "
                    f"Falling back to latest ({latest_version})."
                )
                version_filter = latest_version
                latest_only = False
                version_used = latest_version
        else:
            # Use explicit version filtering for "latest" instead of relying on
            # is_latest payload flags, which may be stale for previously indexed chunks.
            version_filter = latest_version
            latest_only = False
            version_used = latest_version

        # ── 3. Multi-query expansion ───────────────────────────────
        sub_queries = generate_sub_queries(intent.clean_query, self._llm)
        log.info(f"Sub-queries: {sub_queries}")

        # ── 4. Hybrid retrieval ────────────────────────────────────
        candidate_chunks = self._retriever.retrieve(
            queries=sub_queries,
            version_filter=version_filter,
            latest_only=latest_only,
            search_all_versions=intent.search_all_versions,
        )

        # ── 5. Rerank ──────────────────────────────────────────────
        top_chunks = self._reranker.rerank(
            query=question,
            chunks=candidate_chunks,
            top_k=cfg.retrieval.rerank_top_k,
        )

        # Filter by confidence threshold
        confidence_threshold = cfg.generation.confidence_threshold
        if top_chunks:
            best_score = top_chunks[0].get("_rerank_score", 0)
            if best_score < confidence_threshold:
                log.info(
                    f"Best rerank score {best_score:.4f} < threshold "
                    f"{confidence_threshold} → returning not-found"
                )
                top_chunks = []

        # ── 6. Build prompt & generate ─────────────────────────────
        system_prompt, user_prompt = build_context_prompt(question, top_chunks)

        _max_tok = cfg.llm.gemini.max_tokens

        raw_answer = self._llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=_max_tok,
            temperature=0.0,
        )

        # ── 7. Validate (hallucination firewall) ───────────────────
        final_answer = validate_answer(raw_answer, top_chunks)

        latency = (time.time() - t0) * 1000
        log.info(f"Answer generated in {latency:.0f}ms")

        return AgentResult(
            question=question,
            answer=final_answer,
            sources=top_chunks,
            version_used=version_used,
            num_chunks_retrieved=len(candidate_chunks),
            latency_ms=latency,
            sub_queries=sub_queries,
        )
