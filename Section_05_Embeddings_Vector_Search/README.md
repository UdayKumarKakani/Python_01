# Section 05 — Embeddings, Vector Search & Semantic Systems

A 6-day, fresher-friendly walkthrough of **embeddings** and **semantic search**. Every day fits in roughly **1 hour 15 minutes** of teaching.

Section 4 covered LLMs (talking to models). This section covers **how machines find meaning** — the foundation for RAG, semantic search, and recommendation systems.

We use **Sentence Transformers** (free, open-source) as the default provider and touch **OpenAI text-embedding-3** once for comparison. **ChromaDB** is our main vector database.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | Embeddings 101 + providers side-by-side | `Day_1_Embeddings_101/` |
| 2 | Similarity & naive search (cosine, linear scan) | `Day_2_Similarity_Naive_Search/` |
| 3 | FAISS & vector indexes (ANN in plain English) | `Day_3_FAISS_Vector_Indexes/` |
| 4 | Vector databases — Chroma, Pinecone, pgvector | `Day_4_Vector_Databases/` |
| 5 | Chunking, metadata filters, hybrid search | `Day_5_Chunking_Metadata_Hybrid/` |
| 6 | Capstone — Semantic Search Engine over PDFs | `Day_6_Capstone_Semantic_Search/` |

## How each day is organized

Each day folder contains:
- `concepts.ipynb` — 75-minute teaching notebook, plain English, works in Colab
- `main.py` — runnable Python script (`python main.py`)
- `assignments.ipynb` — 2–3 exercises + optional stretch

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Create a `.env` file in the section root (optional — most days work fully offline):

```env
# Optional — only used on Day 1 for the OpenAI embedding comparison
OPENAI_API_KEY=sk-...

# Optional — only used on Day 4 for the Pinecone demo
PINECONE_API_KEY=...
```

## Stack

- **sentence-transformers** — free open-source embedding models (default)
- **openai** — used once on Day 1 for `text-embedding-3-small` comparison
- **numpy** — vector math for the naive-search day
- **faiss-cpu** — local vector index (Day 3)
- **chromadb** — main vector database (Days 4–6)
- **pinecone** — managed vector DB intro (Day 4)
- **rank-bm25** — keyword search half of hybrid search (Day 5)
- **pypdf** — PDF ingestion for the capstone
- **fastapi + uvicorn** — capstone search API

## Prerequisites

- Completed Sections 1–4 (Python, REST APIs, and LLM basics)
- Python 3.10+
- No GPU needed — everything runs on CPU

## What you'll build

By the end of Day 6 you'll have a **Semantic Search API** that:
- Ingests real PDFs, chunks them, and embeds each chunk
- Stores vectors in ChromaDB with metadata (source file, page number)
- Exposes `/search` via FastAPI with a `top_k` parameter and metadata filters
- Returns the most relevant passages along with their source citations
