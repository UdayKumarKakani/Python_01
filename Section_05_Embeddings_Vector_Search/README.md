# Section 05 — Embeddings, Vector Search & Semantic Systems

A 4-day, fresher-friendly walkthrough of **embeddings** and **semantic search**. Every day fits in roughly **1 hour 15 minutes** of teaching. Day 1 is split into two ~75-min sessions (1.1 + 1.2).

Section 4 covered LLMs (talking to models). This section covers **how machines find meaning** — the foundation for RAG, semantic search, and recommendation systems.

We use **Sentence Transformers** (free, open-source) as the default provider and touch **OpenAI text-embedding-3** once for comparison. **ChromaDB** is our main vector database.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1.1 | Embeddings 101 + providers side-by-side | `Day_1.1_Embeddings_101/` |
| 1.2 | Similarity & naive search (cosine, linear scan) | `Day_1.2_Similarity_Naive_Search/` |
| 2 | Vector databases — Chroma, Pinecone, pgvector | `Day_2_Vector_Databases/` |
| 3 | Chunking, metadata filters, hybrid search | `Day_3_Chunking_Metadata_Hybrid/` |
| 4 | Capstone — Semantic Search Engine over PDFs | `Day_4_Capstone_Semantic_Search/` |

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
# Optional — only used on Day 1.1 for the OpenAI embedding comparison
OPENAI_API_KEY=sk-...

# Optional — only used on Day 2 for the Pinecone demo
PINECONE_API_KEY=...
```

## Stack

- **sentence-transformers** — free open-source embedding models (default)
- **openai** — used once on Day 1.1 for `text-embedding-3-small` comparison
- **numpy** — vector math for the naive-search day
- **chromadb** — main vector database (Days 2–4)
- **pinecone** — managed vector DB intro (Day 2)
- **rank-bm25** — keyword search half of hybrid search (Day 3)
- **pypdf** — PDF ingestion for the capstone
- **fastapi + uvicorn** — capstone search API

## Prerequisites

- Completed Sections 1–4 (Python, REST APIs, and LLM basics)
- Python 3.10+
- No GPU needed — everything runs on CPU

## What you'll build

By the end of Day 4 you'll have a **Semantic Search API** that:
- Ingests real PDFs, chunks them, and embeds each chunk
- Stores vectors in ChromaDB with metadata (source file, page number)
- Exposes `/search` via FastAPI with a `top_k` parameter and metadata filters
- Returns the most relevant passages along with their source citations
