"""Documind: upload product documents, build an index, and ask questions."""
from __future__ import annotations

import re
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent import AgentResult, DocumentationAgent
from src.config_loader import cfg
from src.versioning.version_manager import VersionManager


st.set_page_config(page_title="Documind", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { background: #0b0d10; color: #e8edf3; font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#0b0d10; }
[data-testid="stSidebar"] { background: #080a0d; border-right: 1px solid #20242a; }
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }
.brand { display:flex; align-items:center; gap:.65rem; font:800 1.25rem 'Manrope',sans-serif; margin:.2rem 0 2.2rem; }
.brand-mark { display:grid; place-items:center; width:2rem; height:2rem; border-radius:.7rem; color:white; background:linear-gradient(145deg,#4779ff,#7357db); }
.eyebrow { color:#8994a3; font-size:.76rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
.hero { padding:1.4rem 0 .8rem; }
.hero h1 { font-size:2.2rem; margin:.2rem 0 .35rem; }
.hero p { color:#9ba5b2; margin:0; }
.card { background:#11151a; border:1px solid #252b33; border-radius:16px; padding:1.1rem 1.2rem; }
.file-pill { background:#171c23; border:1px solid #2a313a; border-radius:10px; padding:.7rem .85rem; margin:.45rem 0; color:#c9d1db; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#101419; border-color:#252b33; border-radius:16px; }
div[data-testid="stFileUploaderDropzone"] { background:#141920 !important; border:1px dashed #3a4350 !important; border-radius:12px !important; }
div[data-testid="stFileUploaderDropzone"] * { color:#d7dee8 !important; }
div[data-testid="stFileUploaderDropzone"] button { background:#202733 !important; border:1px solid #36404d !important; color:#e8edf3 !important; }
[data-testid="stFileUploader"] section { background:transparent !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:#aab4c1 !important; }
[data-testid="stChatMessage"] { background:#11151a; border:1px solid #222831; border-radius:14px; }
[data-testid="stChatInput"] { background:#11151a !important; border:1px solid #303844 !important; border-radius:14px !important; }
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea:focus { background:transparent !important; color:#e8edf3 !important; caret-color:#8da7ff; }
[data-testid="stChatInput"] textarea::placeholder { color:#7f8a99 !important; }
div[data-baseweb="input"], div[data-baseweb="textarea"] { background:#141920 !important; border-color:#303844 !important; }
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea { color:#e8edf3 !important; }
div.stButton > button { background:#1a2028; color:#e8edf3; border:1px solid #303844; border-radius:10px; transition:all .15s ease; }
div.stButton > button:hover { background:#242d39; border-color:#56657a; color:#fff; }
div.stButton > button[kind="primary"] { background:#5578f6; border-color:#5578f6; color:white; }
div.stButton > button[kind="primary"]:hover { background:#6b89ff; border-color:#6b89ff; }
[data-testid="stAlert"] { background:#141920; border-color:#303844; color:#dce3ec; }
.stCaption { color:#8d98a6; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to your document index…")
def load_agent() -> DocumentationAgent:
    return DocumentationAgent()


def _safe_pdf_name(filename: str) -> str:
    """Return a safe, unique filename for a PDF uploaded into the ignored docs folder."""
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "document"
    return f"{stem[:90]}_{uuid.uuid4().hex[:8]}.pdf"


def _indexed_versions() -> list[str]:
    try:
        return VersionManager().all_versions()
    except Exception:
        return []


def _format_sources(sources: list[dict]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple] = set()
    for source in sources:
        key = (source.get("doc_id"), source.get("section"), source.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- **{source.get('doc_id', '?')}** · v{source.get('release_version', '?')} · "
            f"{source.get('section', '?')} · Page {source.get('page_number', '?')}"
        )
    return lines


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">◈</span>Documind</div>', unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Workspace</div>", unsafe_allow_html=True)
        st.markdown("**▦  Documents & chat**")
        st.divider()
        st.markdown("<div class='eyebrow'>Your library</div>", unsafe_allow_html=True)
        pdfs = sorted(Path(cfg.paths.docs_dir).rglob("*.pdf"))
        versions = _indexed_versions()
        if versions:
            st.success(f"{len(versions)} version{'s' if len(versions) != 1 else ''} indexed")
            st.caption("Versions: " + ", ".join(versions))
        else:
            st.caption("No indexed documents yet")
        if pdfs:
            st.caption(f"{len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''} in your local library")
            for pdf in pdfs[-5:]:
                st.markdown(f"<div class='file-pill'>▤ &nbsp;{pdf.name}</div>", unsafe_allow_html=True)
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Your PDFs and index are stored locally in this project.")


def _render_upload() -> None:
    st.markdown("### Add a document")
    st.caption("Upload a PDF, then build its searchable index.")
    uploaded = st.file_uploader("Choose a PDF document", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        st.markdown(f"<div class='file-pill'>▤ &nbsp;{uploaded.name} &nbsp; · &nbsp; {uploaded.size / 1024:.0f} KB</div>", unsafe_allow_html=True)
        if st.button("Create chunks & embeddings", type="primary", use_container_width=True):
            data = uploaded.getvalue()
            if not data.startswith(b"%PDF"):
                st.error("This file does not look like a valid PDF.")
                return

            docs_dir = Path(cfg.paths.docs_dir)
            upload_dir = docs_dir / "user_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            target = upload_dir / _safe_pdf_name(uploaded.name)
            target.write_bytes(data)

            with st.status("Preparing your document index…", expanded=True) as status:
                command = [sys.executable, str(ROOT / "scripts" / "ingest.py"), "--file", str(target.relative_to(docs_dir))]
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                    if result.returncode == 0:
                        status.update(label="Document indexed", state="complete", expanded=False)
                        st.session_state.flash_message = f"{uploaded.name} is ready. You can ask questions below."
                        if result.stdout.strip():
                            with st.expander("Index details"):
                                st.code(result.stdout[-5000:])
                        load_agent.clear()
                        st.rerun()
                    status.update(label="Indexing failed", state="error", expanded=True)
                    st.error("The document could not be indexed. Check your API keys and PDF, then try again.")
                    with st.expander("Error details", expanded=True):
                        st.code((result.stderr or result.stdout or "No details returned.")[-5000:])
                except OSError as error:
                    status.update(label="Could not start indexing", state="error", expanded=True)
                    st.error(str(error))


def _render_chat() -> None:
    st.markdown("### Ask your documents")
    versions = _indexed_versions()
    if not versions:
        st.info("Index a PDF above to start chatting. Previously indexed documents will appear here automatically.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    st.markdown("\n".join(message["sources"]))
            if message.get("meta"):
                st.caption(message["meta"])

    prompt = st.chat_input("Ask a question about your documents…", disabled=not versions)
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching and preparing an answer…"):
            try:
                result: AgentResult = load_agent().ask(prompt)
                st.markdown(result.answer)
                sources = _format_sources(result.sources)
                if sources:
                    with st.expander("Sources"):
                        st.markdown("\n".join(sources))
                meta = f"{result.num_chunks_retrieved} passages · Version {result.version_used or 'latest'} · {result.latency_ms:.0f} ms"
                st.caption(meta)
                st.session_state.messages.append({"role": "assistant", "content": result.answer, "sources": sources, "meta": meta})
            except Exception as error:
                message = f"I couldn’t answer that yet: {error}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})


def main() -> None:
    _render_sidebar()
    if notice := st.session_state.pop("flash_message", None):
        st.success(notice)
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Your document intelligence workspace</div>
      <h1>Understand your documents.</h1>
      <p>Upload a PDF, create its searchable index, and get grounded answers with page citations.</p>
    </div>
    """, unsafe_allow_html=True)

    upload_col, chat_col = st.columns([0.85, 1.35], gap="large")
    with upload_col:
        with st.container(border=True):
            _render_upload()
    with chat_col:
        with st.container(border=True):
            _render_chat()


if __name__ == "__main__":
    main()
