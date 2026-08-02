"""
Day 4 - Prompt design & context management
-------------------------------------------
Requires TOGETHER_API_KEY in .env.

Shows:
1. A reusable RAG prompt template with system + numbered context
2. Deduplication and token-based context capping
3. Inline-citation-friendly one-shot example

Run:
    python main.py
"""

import os

import tiktoken
from dotenv import load_dotenv
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"
llm = Together()
enc = tiktoken.encoding_for_model("gpt-4o-mini")


def n_tokens(text: str) -> int:
    return len(enc.encode(text))


def dedupe(chunks: list[str], min_diff: int = 30) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        key = c[:min_diff].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def cap_tokens(chunks: list[str], max_tokens: int = 500) -> list[str]:
    total, out = 0, []
    for c in chunks:
        t = n_tokens(c)
        if total + t > max_tokens:
            break
        out.append(c)
        total += t
    return out


def build_prompt(question: str, chunks: list[str]) -> list[dict]:
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    system = (
        "You are a helpful assistant. Answer using ONLY the numbered context. "
        "If the answer is not there, say 'I don't know.' "
        "Cite sources with bracket numbers.\n\n"
        "Format your answer like this:\n"
        "The Pro plan costs $29/month [2]."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Context:\n{numbered}\n\nQuestion: {question}"},
    ]


def full_rag(question: str, raw_chunks: list[str]) -> str:
    chunks = cap_tokens(dedupe(raw_chunks), max_tokens=500)
    resp = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=build_prompt(question, chunks),
        temperature=0.2,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    retrieved = [
        "The free tier includes 10 GB of storage.",
        "The free tier includes 10 GB storage and 100 API calls/day.",  # near-dupe
        "The Pro plan costs $29/month and includes 500 GB.",
        "Enterprise customers receive 24/7 phone support.",
    ]
    print("--- Deduped + capped context ---")
    for c in cap_tokens(dedupe(retrieved), max_tokens=200):
        print(" -", c)
    print()

    print("--- Full RAG answer ---")
    print(full_rag("What's the price difference between free and Pro?", retrieved))
