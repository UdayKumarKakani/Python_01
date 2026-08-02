"""
Day 3 - Reranking + Query Expansion
------------------------------------
Requires TOGETHER_API_KEY in .env for the expansion demo.
Reranking works fully offline.

Shows:
1. Cross-encoder reranking with bge-reranker-base
2. Two-stage retrieve -> rerank pipeline
3. HyDE-lite query expansion using an LLM

Run:
    python main.py
"""

import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from together import Together

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base")
llm = Together() if os.getenv("TOGETHER_API_KEY") else None


DOCS = [
    "AcmeCloud's free tier includes 10 GB of storage.",
    "The Pro plan costs $29/month and includes 500 GB.",
    "AcmeCloud servers are located in AWS us-east-1 and eu-west-1.",
    "To reset your password, click 'Forgot Password' on the login page.",
    "Enterprise customers receive 24/7 phone support.",
    "AcmeCloud was founded in 2019 by Priya Rao and Marcus Chen.",
    "Passwords must be at least 12 characters and include a symbol.",
    "You can enable two-factor authentication in the security settings.",
]
kb = chromadb.Client().create_collection("acme_v2")
kb.add(
    documents=DOCS,
    embeddings=embedder.encode(DOCS).tolist(),
    ids=[f"d{i}" for i in range(len(DOCS))],
)


def retrieve_and_rerank(question: str, k_candidates: int = 6, k_final: int = 3):
    q_vec = embedder.encode([question]).tolist()
    candidates = kb.query(query_embeddings=q_vec, n_results=k_candidates)["documents"][0]
    scores = reranker.predict([(question, c) for c in candidates])
    return [d for d, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])[:k_final]]


def expand_query(question: str) -> str:
    if llm is None:
        return question
    resp = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{
            "role": "user",
            "content": (
                "Rewrite the user's question as a short, keyword-rich search query. "
                "Add synonyms and likely terms that would appear in a documentation page. "
                "Return only the rewritten query.\n\n"
                f"User question: {question}\n\nRewritten query:"
            ),
        }],
        temperature=0.0,
        max_tokens=60,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print("--- Reranking demo ---\n")
    for q in ["how do I change my password?", "where is my data hosted?"]:
        print(f"Q: {q}")
        for c in retrieve_and_rerank(q):
            print(f"  - {c}")
        print()

    if llm:
        print("--- Query expansion demo ---\n")
        for q in ["is it safe?", "is it fast enough?"]:
            print(f"Original : {q}")
            print(f"Expanded : {expand_query(q)}\n")
