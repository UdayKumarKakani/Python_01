"""
Day 2 - Similarity & Naive Search
----------------------------------
No API keys needed.

Shows:
1. Cosine similarity between sentence pairs
2. A mini semantic search engine over 8 documents
3. Timing naive search on 50k fake vectors

Run:
    python main.py
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer, util


def demo_similarity() -> None:
    print("--- 1. Cosine similarity between sentences ---\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    a = model.encode("A dog is running in the park")
    b = model.encode("A puppy plays on the grass")
    c = model.encode("The stock market crashed today")
    print(f"  dog vs puppy    : {util.cos_sim(a, b).item():.3f}")
    print(f"  dog vs stocks   : {util.cos_sim(a, c).item():.3f}")
    print(f"  puppy vs stocks : {util.cos_sim(b, c).item():.3f}\n")


def demo_mini_search() -> None:
    print("--- 2. Mini semantic search engine ---\n")
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
    doc_vectors = model.encode(docs)

    for query in ["web development in Python", "places to visit in France"]:
        print(f"  Query: {query!r}")
        q = model.encode(query)
        scores = util.cos_sim(q, doc_vectors)[0].numpy()
        for i in np.argsort(-scores)[:3]:
            print(f"    {scores[i]:.3f}  {docs[i]}")
        print()


def demo_naive_speed() -> None:
    print("--- 3. Naive search over 50,000 fake vectors ---\n")
    N, dim = 50_000, 384
    fake_docs = np.random.randn(N, dim).astype("float32")
    fake_query = np.random.randn(dim).astype("float32")

    start = time.time()
    scores = fake_docs @ fake_query
    top10 = np.argsort(-scores)[:10]
    elapsed = (time.time() - start) * 1000
    print(f"  Searched {N:,} docs in {elapsed:.1f} ms  (top idx: {top10[0]})\n")


if __name__ == "__main__":
    demo_similarity()
    demo_mini_search()
    demo_naive_speed()
