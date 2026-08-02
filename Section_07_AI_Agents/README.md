# Section 07 — AI Agents & Agentic Workflows

A 6-day, fresher-friendly walkthrough of **AI agents** — LLMs that don't just answer, they **act**. Every day fits in roughly **1 hour 15 minutes** of teaching.

An **agent** is an LLM in a loop that can decide to use **tools** — web search, HTTP calls, calculators, code — and observe the results before deciding what to do next. Every autonomous AI system in production today (Claude Code, Cursor's agent mode, OpenAI's `o1` reasoning, most enterprise AI assistants) is built on the ideas in this section.

We build the first agent **from scratch** so you understand the loop. Then we adopt **LangGraph** to make it production-worthy.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | What is an agent? Build a ReAct loop from scratch | `Day_1_ReAct_From_Scratch/` |
| 2 | Tools — function calling, web search, safe execution | `Day_2_Tools_And_Function_Calling/` |
| 3 | LangGraph — stateful, multi-step agent workflows | `Day_3_LangGraph_Stateful_Agents/` |
| 4 | Agent memory — short-term + long-term (vector store) | `Day_4_Agent_Memory/` |
| 5 | Reliability, human-in-the-loop, cost governance | `Day_5_Reliability_HITL_Cost/` |
| 6 | Capstone — Autonomous Research Assistant | `Day_6_Capstone_Research_Assistant/` |

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
# Required — Together AI hosts the Llama models we use
TOGETHER_API_KEY=...

# Optional — OpenAI shown once for the strict function-calling API on Day 2
OPENAI_API_KEY=sk-...

# Optional — Tavily is the friendliest web-search API for agents (free tier)
TAVILY_API_KEY=tvly-...
```

Get keys: https://together.ai · https://tavily.com · https://platform.openai.com

## Stack

- **together** — Together AI Python client (default LLM)
- **openai** — used once on Day 2 for strict function-calling API
- **langgraph** — stateful agent workflows (Day 3+)
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

By the end of Day 6 you'll have an **Autonomous Research Assistant** that:
- Accepts a research question via a FastAPI endpoint
- Plans steps (search web → fetch pages → summarize → save)
- Uses tools to actually fetch information from the internet
- Remembers past research in a long-term vector store
- Stops if it exceeds a max-iterations / max-cost budget
- Asks for human approval before spending real money
