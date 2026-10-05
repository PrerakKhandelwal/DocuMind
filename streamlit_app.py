"""
Streamlit chat UI for the Documentation Agent RAG pipeline.

Usage:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent import DocumentationAgent, AgentResult


def _format_sources(sources: list[dict]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple] = set()
    for src in sources:
        key = (
            src.get("doc_id"),
            src.get("section"),
            src.get("page_number"),
        )
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- **{src.get('doc_id', '?')}** "
            f"v{src.get('release_version', '?')} | "
            f"{src.get('section', '?')} | "
            f"Page {src.get('page_number', '?')}"
        )
    return lines


@st.cache_resource(show_spinner="Loading RAG agent (first run may take ~30s)…")
def load_agent() -> DocumentationAgent:
    return DocumentationAgent()


def main() -> None:
    st.set_page_config(
        page_title="Documentation Agent",
        page_icon="📚",
        layout="wide",
    )
    st.title("Product Documentation Agent")
    st.caption("Ask questions about your indexed PDF documentation.")

    with st.sidebar:
        st.header("Settings")
        version_lock = st.text_input(
            "Lock to version (optional)",
            placeholder="e.g. 2.3.1",
        )
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Indexed versions")
        try:
            agent = load_agent()
            versions = agent._version_mgr.all_versions()
            latest = agent._version_mgr.get_latest()
            if versions:
                st.write(", ".join(versions))
                st.write(f"**Latest:** {latest or 'not set'}")
            else:
                st.warning("No versions indexed. Run `python scripts/ingest.py` first.")
        except Exception as e:
            st.error(f"Agent not ready: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    st.markdown("\n".join(msg["sources"]))
            if msg.get("meta"):
                st.caption(msg["meta"])

    prompt = st.chat_input("Ask a question about the documentation…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation…"):
            try:
                agent = load_agent()
                effective = prompt
                if version_lock and version_lock not in prompt:
                    effective = f"{prompt} (for version {version_lock})"

                result: AgentResult = agent.ask(effective)
                st.markdown(result.answer)

                source_lines = _format_sources(result.sources)
                if source_lines:
                    with st.expander("Sources"):
                        st.markdown("\n".join(source_lines))

                meta = (
                    f"Retrieved {result.num_chunks_retrieved} chunks | "
                    f"Version: {result.version_used or 'latest'} | "
                    f"{result.latency_ms:.0f} ms"
                )
                st.caption(meta)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": source_lines,
                    "meta": meta,
                })
            except Exception as e:
                err = f"**Error:** {e}"
                st.error(err)
                with st.expander("Details"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                })


if __name__ == "__main__":
    main()
