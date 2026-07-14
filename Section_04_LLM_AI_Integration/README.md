# Section 04 — LLM Engineering & AI Integration

A 7-day, non-technical-friendly walkthrough of building with Large Language Models. Every day fits in roughly **one hour** of teaching.

From Day 3 onward we use **Together AI** as the default provider — it hosts open-source models like LLaMA 3, Mistral, and Qwen via one API, and it's cheaper than the closed-source options. OpenAI and Claude appear in Days 2 and 5 for API mastery and tool calling.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | What is an LLM? — tokens, attention, temperature, hallucinations | `Day_1_LLM_Basics/` |
| 2 | Using AI APIs — OpenAI + Claude (+ Hugging Face concept) | `Day_2_AI_APIs/` |
| 3 | Hugging Face hands-on + Together AI | `Day_3_HuggingFace_and_Together/` |
| 4 | Prompt engineering — role/task/format, few-shot, CoT, ReAct | `Day_4_Prompt_Engineering/` |
| 5 | Structured outputs & tool calling (function schemas) | `Day_5_Structured_and_Tools/` |
| 6 | Cost control, streaming & async LLM calls | `Day_6_Cost_Streaming_Async/` |
| 7 | Capstone — Multi-Model AI Assistant (Together + OpenAI + Claude) | `Day_7_Capstone_AI_Assistant/` |

## How each day is organized

Each day folder contains:
- `concepts.ipynb` — 1-hour teaching notebook, plain English, works in Colab
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
# Required from Day 3 onward
TOGETHER_API_KEY=...

# Used on Day 2 and Day 5 (and unlocks extra routing in the capstone)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Where to get keys**
- Together AI: https://together.ai (free credits on sign-up)
- OpenAI: https://platform.openai.com
- Anthropic (Claude): https://console.anthropic.com

## Stack

- **together** — Together AI Python client (default provider from Day 3)
- **openai** — OpenAI SDK (Day 2, tool calling on Day 5, capstone fallback)
- **anthropic** — Claude SDK (Day 2, code routing in capstone)
- **transformers**, **sentence-transformers** — Hugging Face pipelines (Day 3)
- **tiktoken** — accurate token counting for cost estimation
- **fastapi + uvicorn** — capstone backend
- **pydantic v2** — request/response schemas + structured outputs
- **sqlalchemy** — SQLite log of every call for the usage dashboard
- **python-jose + passlib** — JWT auth in the capstone
- **python-dotenv** — reads `.env`

## Prerequisites

- Completed Section 1 (Python) and Section 2 (Backend / REST APIs)
- Python 3.10+
- A Together AI API key (free tier is enough for Days 3–7)

## Topic coverage vs the AI Engineering Bootcamp doc

| Original Section 4 topic | Where it's covered |
|---|---|
| LLM internals — transformers, attention, tokenization | Day 1 (simplified) |
| OpenAI & Claude API mastery | Day 2 |
| Hugging Face — loading & using open source models | Day 3 |
| Local LLMs (originally Ollama) | Replaced by Together AI hosted open-source models on Day 3 |
| Advanced prompt engineering — system prompts, few-shot, CoT, ReAct | Day 4 |
| Structured outputs — JSON mode, function schemas | Day 5 |
| Token optimization & cost control | Day 6 |
| Streaming responses & async LLM calls | Day 6 |
| Multi-model routing | Day 7 capstone |
| Project: Production AI Assistant with multi-model support | Day 7 capstone |

## What you'll build

By the end of Day 7 you'll have an **AI Assistant API** that:
- Accepts chat requests with JWT auth
- Automatically routes each request to the best provider (Together AI for cheap chat, Claude for code and long documents, OpenAI as a fallback)
- Streams tokens as they're generated
- Returns typed JSON on the `/extract` endpoint
- Tracks token usage & cost per user in SQLite and caps daily spend
