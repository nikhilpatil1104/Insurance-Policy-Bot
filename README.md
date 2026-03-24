<div align="center">

# 🛡️ Insurance Policy Q&A

**Upload any insurance policy PDF. Ask questions in plain English. See exactly which clauses answered you.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-00A9E0?style=flat-square)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 What This Does

Most people never fully understand their insurance policies — they're long, dense, and full of legal jargon. This app fixes that.

Drop in any PDF policy (auto, home, health, life) and ask questions like:

- *"What's my deductible for water damage?"*
- *"Are rental cars covered after an accident?"*
- *"What's excluded from liability coverage?"*

The app answers using **Retrieval-Augmented Generation (RAG)** — it finds the relevant policy clauses first, then passes them to GPT-4o-mini to generate a grounded, accurate answer. Every response shows the exact source chunks used, with page numbers.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📄 PDF Ingestion | Multi-page PDFs via LangChain `PyPDFLoader` |
| ✂️ Smart Chunking | `RecursiveCharacterTextSplitter` with configurable size & overlap |
| 🧠 Embeddings | OpenAI `text-embedding-3-small` — fast and cost-efficient |
| 🗂️ Vector Search | FAISS in-memory index with similarity search |
| 💬 RAG Answers | GPT-4o-mini grounded in retrieved policy clauses |
| 🔍 Source Transparency | Every answer shows chunks used, with page numbers |
| 🕘 Conversation History | All Q&A pairs visible in-session |
| ⚙️ Configurable Sidebar | Tune chunk size, overlap, top-k, and model |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Indexing Phase                       │
│                                                             │
│   PDF Upload ──► PyPDFLoader ──► RecursiveCharacterSplitter │
│                                          │                  │
│                              OpenAI Embeddings              │
│                           (text-embedding-3-small)          │
│                                          │                  │
│                                   FAISS VectorStore         │
└─────────────────────────────────────────────────────────────┘
                                           │
┌─────────────────────────────────────────────────────────────┐
│                        Query Phase                          │
│                                                             │
│   User Question ──► Similarity Search (top-k chunks)        │
│                                 │                           │
│                    GPT-4o-mini (RetrievalQA chain)          │
│                                 │                           │
│             Answer + Source Documents (with page #s)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o-mini` and `text-embedding-3-small`

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/insurance-policy-qa.git
cd insurance-policy-qa

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Usage

1. **Paste your OpenAI API key** in the sidebar *(session-only, never stored)*
2. **Upload a PDF** insurance policy using the file uploader
3. **Tune settings** if needed — chunk size, overlap, retrieval k, and model
4. **Click ⚡ Build Knowledge Index** — the app chunks, embeds, and indexes the PDF
5. **Ask a question** by typing or clicking a quick-question button
6. **Expand "View source chunks"** to see exactly which clauses were used to generate the answer

---

## ⚙️ Configuration

All settings are adjustable from the sidebar — no code changes needed.

| Setting | Default | Range | Description |
|---|---|---|---|
| Chunk size | `600` | 200 – 1500 | Max characters per chunk |
| Chunk overlap | `80` | 0 – 300 | Overlap between adjacent chunks |
| Top-k chunks | `4` | 1 – 8 | Chunks retrieved per question |
| Model | `gpt-4o-mini` | gpt-4o, gpt-3.5-turbo | Generation model |

---

## ☁️ Deployment

### Streamlit Community Cloud *(recommended — free)*

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → connect your repo
3. Add your secret under **Settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Click **Deploy** — you get a live public URL in ~2 minutes

---

## 📦 Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[LangChain](https://langchain.com)** — document loading, chunking, RAG chain
- **[FAISS](https://faiss.ai)** — vector similarity search
- **[OpenAI](https://openai.com)** — embeddings (`text-embedding-3-small`) + generation (`gpt-4o-mini`)
- **[PyPDF](https://pypdf.readthedocs.io)** — PDF parsing

---

## 🗺️ Roadmap

- [ ] Persist FAISS index across sessions (ChromaDB or Pinecone)
- [ ] Support multiple PDF uploads and cross-document Q&A
- [ ] Add a comparison mode (e.g., compare two policy PDFs)
- [ ] Export Q&A session as a PDF report
- [ ] Add hybrid search (BM25 + semantic)

---



---

<div align="center">
Built with LangChain · FAISS · OpenAI · Streamlit
</div>
