"""
Day 6 - Load-test helper for your deployed RAG API
---------------------------------------------------
Usage:
    python main.py https://your-app.onrender.com/ask
"""

import asyncio
import sys
import time

import httpx


async def hit(client: httpx.AsyncClient, url: str, question: str):
    t0 = time.time()
    try:
        r = await client.post(url, json={"question": question}, timeout=30)
        return time.time() - t0, r.status_code
    except Exception:
        return time.time() - t0, 599


async def load_test(url: str, n: int = 20, concurrency: int = 5):
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def one():
            async with sem:
                return await hit(client, url, "What is 2+2?")

        results = await asyncio.gather(*[one() for _ in range(n)])

    lats = sorted(r[0] for r in results)
    print(f"n={n}  concurrency={concurrency}")
    print(f"  p50   = {lats[n // 2]:.2f}s")
    print(f"  p95   = {lats[int(n * 0.95) - 1]:.2f}s")
    print(f"  max   = {lats[-1]:.2f}s")
    print(f"  errors= {sum(1 for r in results if r[1] != 200)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <url>")
        sys.exit(1)
    asyncio.run(load_test(sys.argv[1]))
