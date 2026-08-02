# Section 06 — RAG Engineering

A 6-day, fresher-friendly walkthrough of **Retrieval-Augmented Generation** (RAG) — the single most in-demand AI-engineering skill of 2026. Every day fits in roughly **1 hour 15 minutes** of teaching.

RAG = **give the LLM the right documents, then ask it your question.** It's how ChatGPT plugins, enterprise search, "chat with your PDF", and every AI-powered support bot work.

This section builds directly on Section 5 — we already have chunking, embeddings, and ChromaDB. Now we hook them up to an LLM.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | RAG in one hour — build the full loop | `Day_1_RAG_In_One_Hour/` |
| 2 | Document loaders — DOCX, web, markdown + a `LoaderRegistry` | `Day_2_Document_Loaders/` |
| 3 | Reranking + query expansion (HyDE-lite) | `Day_3_Reranking_Query_Expansion/` |
| 4 | Prompt design & context window management | `Day_4_Prompt_And_Context/` |
| 5 | Hallucinations, prompt injection & evaluation | `Day_5_Hallucinations_Guardrails_Eval/` |
| 6 | Capstone — Enterprise RAG Chatbot with streaming | `Day_6_Capstone_RAG_Chatbot/` |

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

Create a `.env` file in the section root:

```env
# Required — the LLM that generates answers (free tier on sign-up)
TOGETHER_API_KEY=...

# Optional — used only if you want to compare providers on Day 1
OPENAI_API_KEY=sk-...
```

Get a Together AI key: https://together.ai

## Stack

- **sentence-transformers** — embeddings (default: `all-MiniLM-L6-v2`)
- **chromadb** — vector database (persistent, on disk)
- **together** — Together AI Python client (LLM for generation)
- **openai** — optional, for provider comparison
- **python-docx** — DOCX loader
- **trafilatura** — clean text extraction from web pages
- **pypdf** — PDF loader (from Section 5)
- **rank-bm25** — keyword side of hybrid search (from Section 5)
- **fastapi + uvicorn** — capstone chatbot API
- **python-jose + passlib** — JWT reused from Section 2
- **tiktoken** — token counting for context-window budgeting

## Prerequisites

- Completed Sections 1–5 (Python, REST APIs, LLMs, embeddings + vector search)
- Python 3.10+
- A Together AI API key (free tier is plenty for the whole section)

## What you'll build

By the end of Day 6 you'll have an **Enterprise RAG Chatbot API** that:
- Ingests PDFs, DOCX, and web pages into a persistent Chroma knowledge base
- Retrieves top-k chunks, reranks them, and stuffs them into a prompt
- Streams the answer with inline citations to source documents
- Rejects prompt-injection attempts and says "I don't know" when the context is weak
- Tracks tokens + cost per user via JWT auth (reused from Section 2)
