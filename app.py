import streamlit as st
import os
import tempfile
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Q&A",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Key (embedded) ─────────────────────────────────────────────────────────
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Lazy imports ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_libs():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    return (
        PyPDFLoader, RecursiveCharacterTextSplitter,
        OpenAIEmbeddings, ChatOpenAI, FAISS, PromptTemplate,
        StrOutputParser, RunnablePassthrough,
    )

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #060b18 0%, #0a1128 50%, #060d1f 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1427 0%, #0a1020 100%);
    border-right: 1px solid rgba(59,130,246,0.15);
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0f2456 0%, #1a1f3c 40%, #0d1829 100%);
    border: 1px solid rgba(99,137,255,0.25);
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #e8f0fe;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
}
.hero-sub {
    font-size: 1rem;
    color: #7b9ccf;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── How it works strip ── */
.how-strip {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 1px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 28px;
}
.how-step {
    background: #080f1e;
    padding: 16px 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.how-num {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white;
    font-size: 0.7rem;
    font-weight: 800;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.how-text strong {
    display: block;
    color: #cbd5e1;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 2px;
}
.how-text span {
    color: #475569;
    font-size: 0.75rem;
    line-height: 1.4;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #3b82f6;
    margin: 24px 0 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(59,130,246,0.2);
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(17,24,39,0.6) !important;
    border: 1.5px dashed rgba(59,130,246,0.3) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.6) !important;
}

/* ── Metric tiles ── */
.metric-tile {
    background: linear-gradient(135deg, #0d1829, #0a1020);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 3px;
}
.metric-desc {
    font-size: 0.68rem;
    color: #334155;
    margin-top: 4px;
    line-height: 1.3;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-ready   { background: rgba(16,185,129,0.1); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-pending { background: rgba(245,158,11,0.1); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* ── Sample PDFs ── */
.pdf-card {
    background: linear-gradient(135deg, #0d1829, #0a1020);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    transition: all 0.2s ease;
    margin-bottom: 8px;
}
.pdf-card:hover {
    border-color: rgba(59,130,246,0.4);
    background: linear-gradient(135deg, #111f3a, #0d1829);
    transform: translateX(3px);
}
.pdf-icon { font-size: 1.4rem; flex-shrink: 0; }
.pdf-info strong {
    display: block;
    color: #93c5fd;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 2px;
}
.pdf-info span { color: #475569; font-size: 0.75rem; }

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, #0d1829, #0a1020);
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 4px solid #3b82f6;
    border-radius: 14px;
    padding: 24px 28px;
    margin: 14px 0;
    color: #e2e8f0;
    font-size: 0.97rem;
    line-height: 1.8;
}

/* ── Guardrail box ── */
.guardrail-box {
    background: linear-gradient(135deg, #1a0d0d, #150808);
    border: 1px solid rgba(239,68,68,0.25);
    border-left: 4px solid #ef4444;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 14px 0;
}

/* ── Chunk cards ── */
.chunk-card {
    background: rgba(13,24,41,0.8);
    border: 1px solid rgba(30,51,82,0.8);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.chunk-label { font-size: 0.75rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.8px; }
.chunk-page { font-size: 0.72rem; color: #475569; background: rgba(26,34,53,0.8); padding: 2px 10px; border-radius: 20px; }
.chunk-text { font-size: 0.87rem; color: #94a3b8; line-height: 1.65; font-family: 'Georgia', serif; }
.chunk-desc { font-size: 0.72rem; color: #334155; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(30,51,82,0.5); font-style: italic; }

/* ── Q input ── */
[data-testid="stTextInput"] input {
    background: rgba(17,24,39,0.8) !important;
    border: 1.5px solid rgba(45,55,72,0.8) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 14px 18px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(59,130,246,0.6) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.3) !important;
}

/* ── Sidebar ── */
.sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #334155 !important;
    margin: 20px 0 8px;
}
hr { border-color: rgba(59,130,246,0.1) !important; }
[data-testid="stExpander"] {
    background: rgba(13,24,41,0.6) !important;
    border: 1px solid rgba(30,51,82,0.6) !important;
    border-radius: 10px !important;
}

/* ── Quick buttons ── */
.stButton > button[kind="secondary"] {
    background: rgba(17,24,39,0.8) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    color: #7b9ccf !important;
    font-size: 0.78rem !important;
    padding: 8px 12px !important;
}
            
/* ── Hide default hamburger, replace with ? icon ── */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border-radius: 50% !important;
    width: 38px !important;
    height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    border: none !important;
}

[data-testid="collapsedControl"] svg {
    display: none !important;
}

[data-testid="collapsedControl"]::after {
    content: '?';
    color: white;
    font-size: 1.1rem;
    font-weight: 800;
    font-family: 'Inter', sans-serif;
    line-height: 1;
}

[data-testid="collapsedControl"] {
    border-radius: 50% !important;
    background: linear-gradient(135deg, #1d4ed8, #2563eb)
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ctext x='7' y='18' font-size='16' font-weight='900' font-family='Arial' fill='white'%3E%3F%3C/text%3E%3C/svg%3E")
        center / 22px no-repeat !important;
    width: 38px !important;
    height: 38px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    border: none !important;
}
[data-testid="collapsedControl"] svg {
    opacity: 0 !important;
    width: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# ── Guardrail config ───────────────────────────────────────────────────────────
GUARDRAIL_SYSTEM = """You are a strict topic classifier for an Insurance Policy Assistant.
Decide if the user question is related to insurance or not.
Insurance topics include: policies, premiums, deductibles, coverage, claims, exclusions,
liability, beneficiaries, endorsements, riders, renewals, cancellations, copays,
out-of-pocket costs, underwriting, or anything in an insurance document.
Reply with ONLY one word — no explanation, no punctuation:
- INSURANCE   (question is about insurance)
- UNRELATED   (question is about anything else)"""

SAMPLE_QUESTIONS = [
    "What does my policy cover for water damage?",
    "What is my deductible amount?",
    "How do I file a claim?",
    "What events are excluded from my coverage?",
    "What is the liability limit in my policy?",
]

SAMPLE_PDFS = [
    {
        "name": "Sample Auto Insurance Policy",
        "desc": "Progressive — personal auto policy template",
        "url": "https://doi.nv.gov/uploadedfiles/doinvgov/_public-documents/Consumers/AU127-1.pdf",
        "icon": "🚗",
    },
    {
        "name": "Sample HOMEOWNERS INSURANCE POLICY",
        "desc": "OID — standard home insurance form",
        "url": "https://www.oid.ok.gov/wp-content/uploads/2019/08/Shelter_HO-4Renters.pdf",
        "icon": "🏠",
    },
    {
        "name": "Sample Health Insurance Policy",
        "desc": "HealthSource UHC — summary of benefits & coverage",
        "url": "https://www.uhc.com/content/dam/uhcdotcom/en/iex-marketplace/sample-medical-policy/Sample-Medical-Policy-MD-IEX-2026.pdf",
        "icon": "🏥",
    },
    {
        "name": "Sample Life Insurance Policy",
        "desc": "HealthSource RI — summary of benefits & coverage",
        "url": "https://healthsourceri.com/wp-content/uploads/2016/11/Principal-Sample-Life-Insurance-Policy.pdf",
        "icon": "💼",
    },
]


def is_insurance_question(question: str, api_key: str) -> bool:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    result = llm.invoke([
        SystemMessage(content=GUARDRAIL_SYSTEM),
        HumanMessage(content=question),
    ])
    return result.content.strip().upper() == "INSURANCE"


def build_vectorstore(pdf_bytes: bytes, api_key: str, chunk_size: int, chunk_overlap: int):
    (PyPDFLoader, RecursiveCharacterTextSplitter,
     OpenAIEmbeddings, ChatOpenAI, FAISS, PromptTemplate,
     StrOutputParser, RunnablePassthrough) = _load_libs()

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
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        vs = FAISS.from_documents(chunks, embeddings)
        stats = {
            "pages": len(pages),
            "chunks": len(chunks),
            "avg_chunk_len": int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)),
        }
        return vs, stats
    finally:
        os.unlink(tmp_path)


def run_qa(question: str, vectorstore, api_key: str, model_name: str, top_k: int):
    (PyPDFLoader, RecursiveCharacterTextSplitter,
     OpenAIEmbeddings, ChatOpenAI, FAISS, PromptTemplate,
     StrOutputParser, RunnablePassthrough) = _load_libs()

    prompt_template = """You are an Insurance Policy Assistant. You ONLY answer questions about the insurance policy document provided.

STRICT DO's:
- Answer ONLY using the policy excerpts below
- Be specific and cite relevant sections when possible
- If a term is ambiguous, ask the user to clarify which aspect they mean
- Use plain English to explain complex insurance terms
- If info is partially covered, say so and suggest what else to ask

STRICT DON'Ts:
- NEVER answer questions unrelated to insurance
- NEVER make up or assume policy details not in the excerpts
- NEVER say "the document doesn't mention it" for non-insurance questions
- NEVER provide legal or financial advice beyond what the policy states
- NEVER use knowledge outside the provided policy excerpts

If the question is about insurance but the answer is not in the excerpts, say:
"I couldn't find a direct answer to that in the uploaded policy. You may want to check [relevant section] or contact your insurer directly."

Policy Excerpts:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=api_key)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    source_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, source_docs


# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [("vectorstore", None), ("doc_stats", {}), ("history", []), ("processing", False)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Sample PDFs — top
    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("GPT model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                               label_visibility="collapsed")
    st.markdown("### 📂 Sample Policies")
    st.caption("Don't have a PDF? Download one of these to try the app.")
    for pdf in SAMPLE_PDFS:
        st.markdown(f"""
        <a href="{pdf['url']}" target="_blank" style="display:flex;align-items:center;gap:10px;
           background:rgba(13,24,41,0.8);border:1px solid rgba(59,130,246,0.15);border-radius:10px;
           padding:10px 14px;text-decoration:none;margin-bottom:6px;transition:all 0.2s;">
            <span style="font-size:1.2rem">{pdf['icon']}</span>
            <span>
                <strong style="color:#93c5fd;font-size:0.8rem;display:block">{pdf['name']}</strong>
                <span style="color:#475569;font-size:0.7rem">{pdf['desc']}</span>
            </span>
        </a>
        """, unsafe_allow_html=True)

    st.divider()

    # Settings — bottom
    st.markdown("### ⚙️ Settings")

    
    st.markdown('<div class="sidebar-section-label">Chunking</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size", 200, 1500, 600, 50,
                           help="Larger chunks = more context per piece, but less precise retrieval")
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 80, 20,
                              help="Overlap prevents answers from being cut off at chunk boundaries")

    st.markdown('<div class="sidebar-section-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 4,
                      help="How many policy sections are fetched to answer each question")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Powered by RAG · GPT-4o-mini · FAISS</div>
    <div class="hero-title">🛡️ Insurance Policy Q&A</div>
    <div class="hero-sub">Upload your policy PDF and ask anything in plain English — get answers backed by exact policy clauses</div>
</div>
""", unsafe_allow_html=True)

# ── How it works (always visible at top) ──────────────────────────────────────
st.markdown("""
<div class="how-strip">
    <div class="how-step">
        <div class="how-num">1</div>
        <div class="how-text">
            <strong>Upload PDF</strong>
            <span>Drop your insurance policy — any insurer, any type</span>
        </div>
    </div>
    <div class="how-step">
        <div class="how-num">2</div>
        <div class="how-text">
            <strong>Auto Chunking</strong>
            <span>Policy is split into overlapping text chunks for precise retrieval</span>
        </div>
    </div>
    <div class="how-step">
        <div class="how-num">3</div>
        <div class="how-text">
            <strong>Semantic Search</strong>
            <span>Your question finds the most relevant policy sections via embeddings</span>
        </div>
    </div>
    <div class="how-step">
        <div class="how-num">4</div>
        <div class="how-text">
            <strong>GPT Answer</strong>
            <span>GPT-4o-mini answers using only those retrieved clauses — no guessing</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    


# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📄 Upload Your Policy</div>', unsafe_allow_html=True)
st.caption("Supports any insurance PDF — auto, home, health, life. Up to ~50 MB.")



col_upload, col_status = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

with col_status:
    if st.session_state.vectorstore:
        d = st.session_state.doc_stats
        st.markdown('<span class="badge badge-ready">✓ Index ready — start asking below</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{d.get('pages','—')}</div>
                <div class="metric-label">Pages</div>
                <div class="metric-desc">Total pages parsed from your PDF</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{d.get('chunks','—')}</div>
                <div class="metric-label">Chunks</div>
                <div class="metric-desc">Text segments indexed for retrieval</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{d.get('avg_chunk_len','—')}</div>
                <div class="metric-label">Avg chars</div>
                <div class="metric-desc">Average characters per chunk</div>
            </div>""", unsafe_allow_html=True)
    elif uploaded_file:
        st.markdown('<span class="badge badge-pending">⏳ PDF ready — click Build Index</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-pending">📄 No file uploaded yet</span>', unsafe_allow_html=True)


# ── Build index ────────────────────────────────────────────────────────────────
if uploaded_file:
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Add it to your Streamlit secrets or environment.")
    else:
        btn_label = "🔄 Re-index Document" if st.session_state.vectorstore else "⚡ Build Knowledge Index"
        st.caption("Clicking this reads your PDF, creates text chunks, generates embeddings, and builds a FAISS vector index.")
        if st.button(btn_label):
            with st.spinner("📖 Reading PDF → ✂️ Chunking → 🧠 Embedding → 🗂️ Building FAISS index…"):
                try:
                    pdf_bytes = uploaded_file.read()
                    vs, stats = build_vectorstore(pdf_bytes, OPENAI_API_KEY, chunk_size, chunk_overlap)
                    st.session_state.vectorstore = vs
                    st.session_state.doc_stats = stats
                    st.session_state.history = []
                    st.success(f"✅ Indexed **{stats['pages']} pages** into **{stats['chunks']} chunks**. Ask your first question below!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Q&A SECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vectorstore:
    st.divider()
    st.markdown('<div class="section-label">💬 Ask a Question</div>', unsafe_allow_html=True)
    st.caption("Ask anything about your uploaded policy — coverage, exclusions, claims, limits, renewals, and more.")

    # Quick question buttons
    st.markdown("**⚡ Quick questions** — click any to instantly search:")
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
                st.session_state["auto_ask"] = True

    st.caption("💡 These are pre-filled common questions. You can also type your own below and press Enter or click Ask.")

    # Input form
    prefill = st.session_state.pop("prefill_question", "")
    auto_ask = st.session_state.pop("auto_ask", False)

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            value=prefill,
            placeholder="e.g. What are the exclusions for water damage?",
            label_visibility="collapsed",
        )
        ask_col, clear_col = st.columns([5, 1])
        with ask_col:
            ask_clicked = st.form_submit_button("🔍 Ask", use_container_width=True)
        with clear_col:
            clear_clicked = st.form_submit_button("🗑️ Clear", use_container_width=True)

    st.caption("Press **Enter** or click **Ask** to search. Click **Clear** to reset the conversation history.")

    if clear_clicked:
        st.session_state.history = []
        st.rerun()

    if (ask_clicked or auto_ask) and question.strip():
        with st.spinner("🔎 Checking question relevance…"):
            try:
                if not is_insurance_question(question, OPENAI_API_KEY):
                    st.session_state.history.insert(0, {
                        "question": question,
                        "answer": "__GUARDRAIL__",
                        "sources": [],
                    })
                    st.rerun()
                else:
                    with st.spinner("📚 Retrieving policy clauses… 🤖 Generating answer…"):
                        answer, source_docs = run_qa(
                            question, st.session_state.vectorstore, OPENAI_API_KEY, model_name, top_k
                        )
                        st.session_state.history.insert(0, {
                            "question": question,
                            "answer": answer,
                            "sources": source_docs,
                        })
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

    # ── History ────────────────────────────────────────────────────────────────
    for i, entry in enumerate(st.session_state.history):
        st.markdown(f"**Q: {entry['question']}**")

        if entry["answer"] == "__GUARDRAIL__":
            suggestions_html = "".join(f"<li>{q}</li>" for q in SAMPLE_QUESTIONS)
            st.markdown(f"""
<div class="guardrail-box">
    <div style="font-size:1.05rem;font-weight:700;color:#fca5a5;margin-bottom:10px;">
        🧑‍💼 I'm your Insurance Policy Assistant
    </div>
    <div style="color:#fecaca;font-size:0.93rem;line-height:1.75;margin-bottom:14px;">
        <em>"{entry['question']}"</em> is outside my area of expertise.<br>
        I'm strictly an insurance specialist — I only answer questions about the policy you uploaded.
        I don't answer general knowledge, coding, science, or any non-insurance questions.
    </div>
    <div style="color:#f87171;font-size:0.82rem;font-weight:600;margin-bottom:8px;">
        Did you mean to ask something like:
    </div>
    <ul style="color:#fca5a5;font-size:0.85rem;line-height:2.1;margin:0;padding-left:20px;">
        {suggestions_html}
    </ul>
</div>
""", unsafe_allow_html=True)

        else:
            st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
            st.caption("↑ Answer generated using only the retrieved chunks from your uploaded policy — no external knowledge used.")

            with st.expander(f"📄 View {len(entry['sources'])} source chunk(s) used to generate this answer"):
                st.caption("These are the exact sections of your policy that were retrieved and passed to GPT to generate the answer above.")
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
    <div class="chunk-desc">This chunk was selected because its semantic meaning was closest to your question in the FAISS vector space.</div>
</div>
""", unsafe_allow_html=True)

        if i < len(st.session_state.history) - 1:
            st.divider()

elif not uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:64px;margin-bottom:18px;filter:drop-shadow(0 0 20px rgba(59,130,246,0.3))">📋</div>
        <div style="font-size:1.2rem;font-weight:700;color:#cbd5e1;margin-bottom:8px;">
            Upload a policy PDF to get started
        </div>
        <div style="font-size:0.9rem;color:#475569;max-width:420px;margin:0 auto;line-height:1.75">
            Drop any insurance PDF above — auto, home, health, or life.<br>
            No PDF? Grab a sample from the sidebar →
        </div>
    </div>
    """, unsafe_allow_html=True)
