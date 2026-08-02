# Section 11 — Final Capstone + Corporate Demo

**A compact 2-day, guided build section.** Each day fits in roughly **1 hour 15 minutes** of class time.

Class time is only 75 min/day — **real capstone work happens on your own time over 1–2 weeks** using the plan you set on Day 1. This section gives you the plan, the reference code, and the demo template.

## Pick your track (Day 1)

| Track | What it is | Example |
|---|---|---|
| **A — AI SaaS Product** | Consumer- or team-facing AI web product | "AI writing coach", "PR review copilot" |
| **B — Enterprise Knowledge AI** | Internal knowledge assistant | "HR handbook Q&A", "Contract search" |
| **C — AI Automation Platform** | Multi-agent workflow that automates a real business process | "Support ticket triage bot" |

## Capstone requirements (must-have)

- **Production backend** with auth (JWT) + a database
- **LLM integration** (Together AI default; OpenAI/Anthropic optional)
- **RAG pipeline** with a vector database
- **Agentic workflow** — at least one autonomous / multi-step agent
- **Full cloud deployment** with CI/CD
- **LLM monitoring & observability** (Langfuse)
- **Architecture doc** + cost analysis
- **Recorded 3-minute demo video**

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | **Track selection, architecture doc, backend + auth kick-off** | `Day_1_Plan_And_Backend/` |
| 2 | **Build (RAG + Agent) → deploy → monitor → demo & cost report** | `Day_2_Build_Deploy_Demo/` |

## How each day is organized

- `concepts.ipynb` — 75-min class walkthrough
- `main.py` — starter code (cost estimator on Day 1, capstone helpers on Day 2)
- `assignments.ipynb` — the actual build checklist (this is where the week of work lives)
- `ARCHITECTURE_TEMPLATE.md` (Day 1) — fill-in-the-blanks doc

## Setup

Same combined stack as Sections 5–9. See `requirements.txt`.

## Prerequisites

**Sections 1–10 completed.**

## What you'll deliver

A single portfolio page (Notion / GitHub Pages / your site) with:

- Live demo URL
- Public GitHub repo (main branch green in CI)
- 3-minute recorded demo video
- Architecture diagram
- Cost analysis: cost per request, per active user, monthly projection
- Langfuse dashboard link (or screenshots)
- "What I'd do next" — 5 bullets

**That page is your job-hunting artifact for the rest of the year.**
