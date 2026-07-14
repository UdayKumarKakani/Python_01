"""
Day 6 - Cost, Streaming & Async
-------------------------------
Three demos via Together AI:

1. Cost estimation with tiktoken.
2. Streaming with first-token latency measurement.
3. asyncio.gather across 4 parallel calls (wall-clock time comparison).

Setup:
    Add TOGETHER_API_KEY to your .env file.

Run:
    python main.py
"""

import asyncio
import os
import sys
import time

import tiktoken
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("TOGETHER_API_KEY"):
    sys.exit("Missing TOGETHER_API_KEY. Add it to .env and re-run.")

from openai import AsyncOpenAI
from together import Together

client = Together()
aio = AsyncOpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# 1. Cost estimator
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, max_output_tokens: int = 300,
                  price_in_per_M: float = 0.18,
                  price_out_per_M: float = 0.18) -> tuple[int, float]:
    in_tokens = len(enc.encode(prompt))
    cost = (in_tokens / 1_000_000) * price_in_per_M + \
           (max_output_tokens / 1_000_000) * price_out_per_M
    return in_tokens, cost


def demo_cost() -> None:
    print("--- 1. Cost estimator ---")
    for p in [
        "Summarize this tweet.",
        "Classify: " + "lorem ipsum " * 30,
        "Extract invoice: " + "line item " * 100,
    ]:
        tokens, cost = estimate_cost(p)
        print(f"  {tokens:5d} tokens  ->  ${cost:.6f}/req  ->  ${cost*10_000:.2f} for 10k reqs")
    print()


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------
def demo_streaming() -> None:
    print("--- 2. Streaming with TTFT ---")
    t0 = time.time()
    first = None

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "List 3 unusual uses for a paperclip."}],
        stream=True,
        max_tokens=300,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first is None:
                first = time.time() - t0
            print(delta, end="", flush=True)

    print(f"\n  TTFT  : {first:.2f}s")
    print(f"  Total : {time.time()-t0:.2f}s\n")


# ---------------------------------------------------------------------------
# 3. Async parallel
# ---------------------------------------------------------------------------
async def _ask(prompt: str) -> str:
    r = await aio.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
    )
    return r.choices[0].message.content.strip()


async def _run_parallel() -> None:
    prompts = [
        "One 1-line joke about databases.",
        "One 1-line joke about Python.",
        "One 1-line joke about AI.",
        "One 1-line joke about coffee.",
    ]
    t0 = time.time()
    results = await asyncio.gather(*(_ask(p) for p in prompts))
    print(f"  parallel: {time.time()-t0:.2f}s for {len(prompts)} calls")
    for r in results:
        print("  -", r)


def demo_async() -> None:
    print("--- 3. Async parallel ---")
    asyncio.run(_run_parallel())
    print()


if __name__ == "__main__":
    demo_cost()
    demo_streaming()
    demo_async()
