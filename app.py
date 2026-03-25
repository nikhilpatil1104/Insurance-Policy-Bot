import os
import streamlit as st
import tempfile
from pathlib import Path

try:
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    secret_key = ""

api_key = st.text_input(
    "API Key",
    value=os.getenv("OPENAI_API_KEY", "") or secret_key,
    type="password",
    placeholder="sk-...",
    help="Your key is used only in-session and never stored.",
    label_visibility="collapsed",
) or secret_key

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Q&A",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (show spinner so user isn't staring at a blank screen) ────────
@st.cache_resource(show_spinner=False)
def _load_libs():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter    
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate
    return (
        PyPDFLoader, RecursiveCharacterTextSplitter,
        OpenAIEmbeddings, ChatOpenAI, FAISS, RetrievalQA, PromptTemplate,
    )

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3d;
}

/* ── Header card ─────────────────────────────────────── */
.header-card {
    background: linear-gradient(135deg, #1a2744 0%, #0d1f42 60%, #0a1628 100%);
    border: 1px solid #2e4a7a;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.header-icon { font-size: 48px; }
.header-text h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8f0fe;
    letter-spacing: -0.3px;
}
.header-text p {
    margin: 4px 0 0;
    font-size: 0.95rem;
    color: #8ba3cc;
}

/* ── Status badges ───────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
}
.badge-ready   { background: #0d3320; color: #4ade80; border: 1px solid #16a34a; }
.badge-pending { background: #2a1f0a; color: #fbbf24; border: 1px solid #d97706; }
.badge-error   { background: #2a0d0d; color: #f87171; border: 1px solid #dc2626; }

/* ── Answer box ──────────────────────────────────────── */
.answer-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 22px 26px;
    margin: 18px 0;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.7;
}

/* ── Source chunk cards ──────────────────────────────── */
.chunk-card {
    background: #0d1829;
    border: 1px solid #1e3352;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.chunk-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.chunk-page {
    font-size: 0.75rem;
    color: #4b5563;
    background: #1a2235;
    padding: 2px 10px;
    border-radius: 20px;
}
.chunk-text {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
    font-family: 'Georgia', serif;
}

/* ── Metric tiles ────────────────────────────────────── */
.metric-tile {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Input styling ───────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #111827 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
}

/* ── Sidebar labels ──────────────────────────────────── */
.sidebar-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4b5563;
    margin: 20px 0 8px;
}

/* ── Dividers ────────────────────────────────────────── */
hr { border-color: #1f2937 !important; }

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0d1829 !important;
    border: 1px solid #1e3352 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="header-icon">🛡️</div>
    <div class="header-text">
        <h1>Insurance Policy Q&amp;A</h1>
        <p>Upload your policy PDF · Ask anything · See exactly which clauses answered your question</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown('<div class="sidebar-section-label">Powered by OpenAI</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-label">Chunking</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size (tokens)", 200, 1500, 600, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 80, 20)

    st.markdown('<div class="sidebar-section-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Source chunks to retrieve (k)", 1, 8, 4)

    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "GPT model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    <div style="font-size:0.8rem; color:#4b5563; line-height:1.6">
    <b style="color:#6b7280">How it works</b><br>
    1. PDF → pages → recursive chunks<br>
    2. Chunks → OpenAI embeddings → FAISS index<br>
    3. Question → top-k chunks → GPT answer
    </div>
    """, unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False


# ── Helper: build vectorstore ───────────────────────────────────────────────────
def build_vectorstore(pdf_bytes: bytes, api_key: str, chunk_size: int, chunk_overlap: int):
    (PyPDFLoader, RecursiveCharacterTextSplitter,
     OpenAIEmbeddings, ChatOpenAI, FAISS, RetrievalQA, PromptTemplate) = _load_libs()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        vs = FAISS.from_documents(chunks, embeddings)

        stats = {
            "pages": len(pages),
            "chunks": len(chunks),
            "avg_chunk_len": int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)),
        }
        return vs, stats
    finally:
        os.unlink(tmp_path)


# ── Helper: run QA ──────────────────────────────────────────────────────────────
def run_qa(question: str, vectorstore, api_key: str, model_name: str, top_k: int):
    (PyPDFLoader, RecursiveCharacterTextSplitter,
     OpenAIEmbeddings, ChatOpenAI, FAISS, RetrievalQA, PromptTemplate) = _load_libs()

    prompt_template = """You are an expert insurance policy analyst. Use the provided policy excerpts to answer the question accurately and concisely. If the answer isn't found in the excerpts, say so clearly rather than guessing.

Policy Excerpts:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=api_key)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]


# ── Upload section ──────────────────────────────────────────────────────────────
col_upload, col_status = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Insurance Policy PDF",
        type=["pdf"],
        help="Supports multi-page PDFs up to ~50 MB",
        label_visibility="collapsed",
    )

with col_status:
    if st.session_state.vectorstore:
        d = st.session_state.doc_stats
        st.markdown(f'<span class="badge badge-ready">✓ Index ready</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{d.get("pages","—")}</div><div class="metric-label">Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{d.get("chunks","—")}</div><div class="metric-label">Chunks</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{d.get("avg_chunk_len","—")}</div><div class="metric-label">Avg chars</div></div>', unsafe_allow_html=True)
    elif uploaded_file:
        st.markdown('<span class="badge badge-pending">⏳ Ready to index</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-pending">📄 No file yet</span>', unsafe_allow_html=True)


# ── Build index button ──────────────────────────────────────────────────────────
if uploaded_file:
    if not api_key:
        st.error("❌ OPENAI_API_KEY not set. Please set it in your environment.")

    else:
        btn_label = "🔄 Re-index document" if st.session_state.vectorstore else "⚡ Build Knowledge Index"
        if st.button(btn_label, use_container_width=False):
            with st.spinner("📖 Reading PDF → ✂️ Chunking → 🧠 Embedding → 🗂️ Building FAISS index…"):
                try:
                    pdf_bytes = uploaded_file.read()
                    vs, stats = build_vectorstore(pdf_bytes, api_key, chunk_size, chunk_overlap)
                    st.session_state.vectorstore = vs
                    st.session_state.doc_stats = stats
                    st.session_state.history = []
                    st.success(f"✅ Indexed **{stats['pages']} pages** into **{stats['chunks']} chunks**. Start asking questions below!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")


# ── Q&A section ─────────────────────────────────────────────────────────────────
if st.session_state.vectorstore:
    st.divider()
    st.markdown("### 💬 Ask a Question")

    # Suggested questions
    st.markdown("**Quick questions:**")
    suggestions = [
        "What is the deductible amount?",
        "What events are excluded from coverage?",
        "What is the coverage limit for liability?",
        "How do I file a claim?",
        "What is the policy renewal process?",
    ]
    cols = st.columns(len(suggestions))
    for i, (col, q) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state["prefill_question"] = q

    # Question input
    prefill = st.session_state.pop("prefill_question", "")
    question = st.text_input(
        "Your question",
        value=prefill,
        placeholder="e.g. What are the exclusions for water damage?",
        label_visibility="collapsed",
    )

    ask_col, clear_col = st.columns([5, 1])
    with ask_col:
        ask_clicked = st.button("🔍 Ask", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if ask_clicked and question.strip():
        with st.spinner("🔍 Searching policy… 🤖 Generating answer…"):
            try:
                answer, source_docs = run_qa(
                    question, st.session_state.vectorstore, api_key, model_name, top_k
                )
                st.session_state.history.insert(0, {
                    "question": question,
                    "answer": answer,
                    "sources": source_docs,
                })
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

    # ── History ─────────────────────────────────────────────────────────────────
    for i, entry in enumerate(st.session_state.history):
        st.markdown(f"**Q: {entry['question']}**")

        import re

        def format_math(text):
            # Convert inline LaTeX \( ... \) → $...$
            text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)

            # Convert block LaTeX \[ ... \] → $$...$$
            text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text)
            # Put formulas on new lines
            text = text.replace("$$", "\n$$")
            return text

        formatted_answer = format_math(entry["answer"])

        st.markdown(
            f'<div class="answer-box">{formatted_answer}</div>',
            unsafe_allow_html=True
        )


        with st.expander(f"📄 View {len(entry['sources'])} source chunk(s) used to generate this answer"):
            for j, doc in enumerate(entry["sources"]):
                page_num = doc.metadata.get("page", "?")
                source_file = Path(doc.metadata.get("source", "policy.pdf")).name
                preview = doc.page_content.strip().replace("\n", " ")
                if len(preview) > 600:
                    preview = preview[:600] + "…"

                st.markdown(f"""
<div class="chunk-card">
    <div class="chunk-header">
        <span class="chunk-label">Chunk {j+1} of {len(entry['sources'])}</span>
        <span class="chunk-page">📄 {source_file} · Page {int(page_num)+1 if isinstance(page_num, int) else page_num}</span>
    </div>
    <div class="chunk-text">{preview}</div>
</div>
""", unsafe_allow_html=True)

        if i < len(st.session_state.history) - 1:
            st.divider()

elif not uploaded_file:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #374151;">
        <div style="font-size: 56px; margin-bottom: 16px;">📋</div>
        <div style="font-size: 1.15rem; font-weight: 600; color: #6b7280; margin-bottom: 8px;">
            Upload a PDF to get started
        </div>
        <div style="font-size: 0.9rem; color: #4b5563; max-width: 400px; margin: 0 auto; line-height: 1.7">
            Drop your insurance policy PDF above. The app will chunk and embed it using 
            LangChain + FAISS, then let you ask natural language questions answered by GPT-4o-mini.
        </div>
    </div>
    """, unsafe_allow_html=True)

