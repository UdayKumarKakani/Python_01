# Section 08 — Fine-Tuning & Model Customization

A 6-day, fresher-friendly walkthrough of **fine-tuning** open-source LLMs. Every day fits in roughly **1 hour 15 minutes** of teaching.

Fine-tuning = **teaching an existing LLM new skills, style, or knowledge with a small dataset.** It's the third leg of the stool alongside prompt engineering (Section 4) and RAG (Section 6).

We use **LoRA / QLoRA** — the modern, cheap way to fine-tune. All hands-on work runs on a free Google Colab GPU. No local GPU needed.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | When to fine-tune vs prompt vs RAG + dataset preparation | `Day_1_When_To_Fine_Tune/` |
| 2 | LoRA / QLoRA basics + your first fine-tune | `Day_2_LoRA_QLoRA_Basics/` |
| 3 | Instruction-tuning format + fine-tune on your own data | `Day_3_Instruction_Tuning/` |
| 4 | Experiment tracking (W&B) + evaluation | `Day_4_Tracking_And_Eval/` |
| 5 | Merging, exporting, and serving fine-tuned models | `Day_5_Merging_Exporting_Serving/` |
| 6 | Capstone — domain-specific fine-tuned model | `Day_6_Capstone_Domain_Model/` |

## How each day is organized

Each day folder contains:
- `concepts.ipynb` — 75-minute teaching notebook (open in Colab for GPU days)
- `main.py` — runnable script (some require a GPU environment)
- `assignments.ipynb` — 2–3 exercises + optional stretch

## Setup

**For GPU work (Days 2, 3, 6): use Google Colab (free T4 GPU).** Upload the notebook and run it there.

**For local CPU work (Days 1, 4, 5):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables (`.env`):
```env
HF_TOKEN=hf_...          # Hugging Face token (free) — needed to download gated models
WANDB_API_KEY=...        # Weights & Biases token (free) — Day 4
TOGETHER_API_KEY=...     # Used in Day 5 for hosted serving comparison
```

## Stack

- **transformers**, **peft**, **trl** — Hugging Face fine-tuning stack
- **unsloth** — modern 2×-faster fine-tuning wrapper (2026 standard)
- **bitsandbytes** — 4-bit quantization for QLoRA
- **datasets** — Hugging Face datasets library
- **wandb** — experiment tracking
- **ollama** — serve fine-tuned models locally (Day 5)
- **together** — hosted inference comparison (Day 5)

## Prerequisites

- Sections 1–7 completed
- A free Google account (for Colab GPU access)
- A Hugging Face account + token (free)
- A W&B account (free) — for Day 4

## What you'll build

By the end of Day 6 you'll have a **domain-specific fine-tuned LLaMA-3.2-3B model** that:
- Was trained on a small (~500 example) dataset in your chosen domain (legal, medical, finance, code, or customer support)
- Beats the base model on your held-out eval set
- Was tracked in W&B with train/eval curves
- Is exported to GGUF format for local serving with Ollama
- Costs <$1 to reproduce end-to-end (Colab free tier is enough)
