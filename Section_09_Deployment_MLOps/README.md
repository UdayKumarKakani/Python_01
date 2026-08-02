# Section 09 — Deployment, MLOps & LLMOps

A 6-day, fresher-friendly walkthrough of **shipping AI systems to production**. Every day fits in roughly **1 hour 15 minutes** of teaching.

Everything you've built so far runs on your laptop. This section teaches you to put it on the internet with **Docker + CI/CD + a cheap cloud host + observability**.

We deliberately favor **cheap, fast, fresher-friendly** platforms (Render, Railway, Fly.io) over full AWS/Azure/GCP deep-dives. You'll learn transferable skills without spending money on cloud bills.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | Docker for AI apps | `Day_1_Docker_For_AI/` |
| 2 | CI/CD with GitHub Actions | `Day_2_CI_CD_GitHub_Actions/` |
| 3 | Cloud deployment (Render / Railway / Fly / HF Spaces) | `Day_3_Cloud_Deployment/` |
| 4 | LLM observability with Langfuse | `Day_4_Observability_Langfuse/` |
| 5 | Cost tracking + A/B testing | `Day_5_Cost_AB_Testing/` |
| 6 | Capstone — deploy your Section 6 RAG chatbot end-to-end | `Day_6_Capstone_Deploy_RAG/` |

## How each day is organized

Each day folder contains:
- `concepts.ipynb` — 75-minute teaching notebook, plain English
- `main.py` (or `Dockerfile` / `.github/workflows`) — runnable artifact
- `assignments.ipynb` — 2–3 exercises

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need:
- **Docker Desktop** (Days 1, 2, 6) — https://docker.com/products/docker-desktop
- **A GitHub account** (Days 2, 3, 6)
- **A free account** on one of Render / Railway / Fly.io (Days 3, 6)
- **A free Langfuse account** (Days 4, 5) — https://langfuse.com

```env
TOGETHER_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Stack

- **Docker** — containerize apps
- **GitHub Actions** — CI/CD
- **Render / Railway / Fly.io** — cheap cloud PaaS
- **Hugging Face Spaces** — free demo hosting (Gradio/Streamlit)
- **Langfuse** — LLM tracing + observability (open-source, free tier)
- **FastAPI** — from Sections 2–7
- **httpx** — load-test helper
- **rich** — pretty tables for cost reports

## Prerequisites

Sections 1–7 complete. Ideally Section 8 too (though this section doesn't require GPU).

## What you'll build

By Day 6 you'll take the **Section 6 RAG Chatbot** and:
- Containerize it with Docker
- Set up a GitHub Actions pipeline that tests + builds + pushes on every merge
- Deploy to Render (or Railway / Fly) on a free tier
- Wire in Langfuse for end-to-end tracing of every RAG call
- Add a `/metrics` endpoint with token + cost totals
- Load-test it to know its p50/p95 latency
- Set an SLO and a Grafana-style dashboard URL you can share
