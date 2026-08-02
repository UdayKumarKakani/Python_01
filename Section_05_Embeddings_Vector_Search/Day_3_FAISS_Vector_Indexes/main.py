"""
Day 3 - FAISS Vector Index
---------------------------
No API keys needed.

Shows:
1. Building an HNSW index over 8 documents
2. Saving and loading the index
3. Naive vs FAISS speed benchmark on 50k vectors

Run:
    python main.py
"""

import os
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def demo_build_and_search() -> None:
    print("--- 1. Build FAISS HNSW index & search ---\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = [
        "Python is a popular programming language.",
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning models are trained on data.",
        "Croissants are a famous French pastry.",
        "FastAPI is a modern Python web framework.",
        "Neural networks are inspired by the human brain.",
        "The Louvre museum houses the Mona Lisa.",
        "Django is another Python web framework.",
    ]
    vectors = model.encode(docs).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexHNSWFlat(vectors.shape[1], 32)
    index.add(vectors)

    q = model.encode(["web development in Python"]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, 3)
    for s, i in zip(scores[0], ids[0]):
        print(f"  {s:.3f}  {docs[i]}")
    print()
    return index


def demo_save_load(index) -> None:
    print("--- 2. Save & reload the index ---\n")
    faiss.write_index(index, "notes.faiss")
    size = os.path.getsize("notes.faiss")
    loaded = faiss.read_index("notes.faiss")
    print(f"  Saved to notes.faiss ({size} bytes). Reloaded {loaded.ntotal} vectors.\n")


def demo_speedup() -> None:
    print("--- 3. Naive vs FAISS on 50,000 vectors ---\n")
    N, dim = 50_000, 384
    docs = np.random.randn(N, dim).astype("float32")
    faiss.normalize_L2(docs)
    query = np.random.randn(1, dim).astype("float32")
    faiss.normalize_L2(query)

    start = time.time()
    _ = np.argsort(-(docs @ query[0]))[:10]
    naive_ms = (time.time() - start) * 1000

    idx = faiss.IndexHNSWFlat(dim, 32)
    idx.add(docs)
    start = time.time()
    idx.search(query, 10)
    faiss_ms = (time.time() - start) * 1000

    print(f"  Naive: {naive_ms:7.2f} ms")
    print(f"  FAISS: {faiss_ms:7.2f} ms   (~{naive_ms/faiss_ms:.0f}x faster)\n")


if __name__ == "__main__":
    idx = demo_build_and_search()
    demo_save_load(idx)
    demo_speedup()
