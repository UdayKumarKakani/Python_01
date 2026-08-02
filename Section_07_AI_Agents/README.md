# Section 07 — AI Agents & Agentic Workflows

A 5-day, fresher-friendly walkthrough of **AI agents** — LLMs that don't just answer, they **act**. Every day fits in roughly **1 hour 15 minutes** of teaching.

An **agent** is an LLM in a loop that can decide to use **tools** — web search, HTTP calls, calculators, code — and observe the results before deciding what to do next. Every autonomous AI system in production today (Claude Code, Cursor's agent mode, OpenAI's `o1` reasoning, most enterprise AI assistants) is built on the ideas in this section.

Since **function calling** and **tool schemas** were already covered in Section 4 Day 5.1, Day 1 here is a condensed recap focused on **the loop**. Day 2 then goes deep on **LangGraph** — the framework used in most 2026 AI-eng job listings.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | What makes an agent? (Loop + tools recap) | `Day_1_Agents_And_Tools/` |
| 2 | LangGraph — stateful, multi-step agent workflows | `Day_2_LangGraph_Stateful_Agents/` |
| 3 | Agent memory — short-term + long-term (vector store) | `Day_3_Agent_Memory/` |
| 4 | Reliability, human-in-the-loop, cost governance | `Day_4_Reliability_HITL_Cost/` |
| 5 | Capstone — Autonomous Research Assistant | `Day_5_Capstone_Research_Assistant/` |

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
# Required — Together AI hosts openai/gpt-oss-20b (our default LLM)
TOGETHER_API_KEY=...

# Optional — used in Day 2 for the LangChain-wrapped Together model
# (langchain-together)

# Optional — Tavily is the friendliest web-search API for agents (free tier)
TAVILY_API_KEY=tvly-...
```

Get keys: https://together.ai · https://tavily.com

## Stack

- **together** — Together AI Python client (default LLM)
- **langgraph** — stateful agent workflows (Day 2+)
- **langchain-core** / **langchain-together** — used with `create_react_agent`
- **tavily-python** — LLM-friendly web search API
- **sentence-transformers + chromadb** — long-term memory (reused from Section 5)
- **fastapi + uvicorn** — capstone trigger endpoint
- **httpx** — HTTP fetch tool

## Prerequisites

- Completed Sections 1–6 (Python, REST APIs, LLMs, embeddings, RAG)
- Python 3.10+
- A Together AI API key (free tier is enough)
- Optional: Tavily key for real web search (or use the mock tool provided)

## What you'll build

By the end of Day 5 you'll have an **Autonomous Research Assistant** that:
- Accepts a research question via a FastAPI endpoint
- Plans steps (search web → fetch pages → summarize → save)
- Uses tools to actually fetch information from the internet
- Remembers past research in a long-term vector store
- Stops if it exceeds a max-iterations / max-cost budget
- Asks for human approval before spending real money
