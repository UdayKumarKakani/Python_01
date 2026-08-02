"""
Day 4 - Vector Databases (Chroma)
----------------------------------
No API keys needed for the Chroma demo.

Shows:
1. Creating a Chroma collection and adding docs with metadata
2. Semantic search over the collection
3. Metadata-filtered search

Run:
    python main.py
"""

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> None:
    print("--- Chroma: create, add, search, filter ---\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    collection = client.create_collection(name="my_docs")

    docs = [
        "Python is a popular programming language.",
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning models are trained on data.",
        "Croissants are a famous French pastry.",
        "FastAPI is a modern Python web framework.",
        "The Louvre museum houses the Mona Lisa.",
    ]
    metadatas = [
        {"topic": "programming", "year": 2024},
        {"topic": "travel",      "year": 2023},
        {"topic": "programming", "year": 2024},
        {"topic": "food",        "year": 2022},
        {"topic": "programming", "year": 2025},
        {"topic": "travel",      "year": 2023},
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]
    vectors = model.encode(docs).tolist()

    collection.add(documents=docs, embeddings=vectors, metadatas=metadatas, ids=ids)
    print(f"  Added {collection.count()} docs\n")

    q_vec = model.encode(["Python web framework"]).tolist()

    print("  Unfiltered search for 'Python web framework':")
    r = collection.query(query_embeddings=q_vec, n_results=3)
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"    dist={dist:.3f}  ({meta['topic']}, {meta['year']})  {doc}")
    print()

    print("  Same query, filtered to year=2024:")
    r = collection.query(query_embeddings=q_vec, n_results=3, where={"year": 2024})
    for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
        print(f"    ({meta['year']})  {doc}")


if __name__ == "__main__":
    main()
