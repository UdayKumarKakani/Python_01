"""
Day 5 - Chunking, Metadata, Hybrid Search
------------------------------------------
No API keys needed.

Shows:
1. Recursive chunking of a paragraph
2. Chunks in Chroma with source + page metadata + filtered query
3. Hybrid search combining BM25 + embeddings

Run:
    python main.py
"""

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para
    if current:
        chunks.append(current)
    with_overlap = []
    for i, c in enumerate(chunks):
        with_overlap.append(c if i == 0 else chunks[i - 1][-overlap:] + c)
    return with_overlap


def demo_chunking() -> None:
    print("--- 1. Recursive chunking ---\n")
    sample = (
        "Python is a high-level programming language.\n\n"
        "FastAPI is a modern web framework built on Python.\n\n"
        "Machine learning is a field of AI focused on models that learn from data."
    )
    for i, c in enumerate(recursive_chunk(sample, chunk_size=80, overlap=20)):
        print(f"  chunk {i}: {c}")
    print()


def demo_metadata_filter() -> None:
    print("--- 2. Chunks in Chroma with metadata filter ---\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    col = client.create_collection("kb")

    chunks = [
        ("Python is a popular language.",            "python_intro.md", 1),
        ("It has readable, elegant syntax.",         "python_intro.md", 1),
        ("FastAPI is fast and modern.",              "fastapi_guide.md", 3),
        ("It uses Pydantic for request validation.", "fastapi_guide.md", 3),
    ]
    col.add(
        documents=[c[0] for c in chunks],
        embeddings=model.encode([c[0] for c in chunks]).tolist(),
        metadatas=[{"source": c[1], "page": c[2]} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    q = model.encode(["request validation"]).tolist()
    r = col.query(query_embeddings=q, n_results=2, where={"source": "fastapi_guide.md"})
    for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
        print(f"  {meta['source']} p{meta['page']}: {doc}")
    print()


def demo_hybrid() -> None:
    print("--- 3. Hybrid search: BM25 + embeddings ---\n")
    docs = [
        "Python is a popular programming language.",
        "Error code ERR-4041 means the file was not found.",
        "FastAPI is a modern Python web framework.",
        "The Eiffel Tower is in Paris.",
        "Machine learning trains models on data.",
    ]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_vecs = model.encode(docs)

    for query in ["ERR-4041", "web framework in Python", "landmarks in France"]:
        print(f"  Query: {query!r}")
        kw = bm25.get_scores(query.lower().split())
        kw = kw / (kw.max() or 1)
        q_vec = model.encode(query)
        sem = util.cos_sim(q_vec, doc_vecs).numpy()[0]
        combined = 0.5 * kw + 0.5 * sem
        for i in np.argsort(-combined)[:3]:
            print(f"    {combined[i]:.3f}  {docs[i]}")
        print()


if __name__ == "__main__":
    demo_chunking()
    demo_metadata_filter()
    demo_hybrid()
