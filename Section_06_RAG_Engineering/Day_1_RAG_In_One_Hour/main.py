"""
Day 1 - RAG in one hour
------------------------
Requires TOGETHER_API_KEY in .env.

Shows:
1. Building a tiny knowledge base in Chroma
2. Retrieve -> Augment -> Generate loop
3. Returning answer + sources

Run:
    python main.py
"""

import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"


DOCS = [
    "AcmeCloud's free tier includes 10 GB of storage and 100 API calls per day.",
    "The Pro plan costs $29/month and includes 500 GB of storage and unlimited API calls.",
    "AcmeCloud servers are located in AWS us-east-1 and eu-west-1 regions.",
    "To reset your password, click 'Forgot Password' on the login page or email support@acmecloud.io.",
    "Enterprise customers receive 24/7 phone support and a dedicated account manager.",
    "AcmeCloud was founded in 2019 by Priya Rao and Marcus Chen in Austin, Texas.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
kb = chromadb.Client().create_collection("acme_kb")
kb.add(
    documents=DOCS,
    embeddings=model.encode(DOCS).tolist(),
    ids=[f"doc_{i}" for i in range(len(DOCS))],
)
llm = Together()


def rag(question: str, top_k: int = 3) -> dict:
    q_vec = model.encode([question]).tolist()
    chunks = kb.query(query_embeddings=q_vec, n_results=top_k)["documents"][0]
    context = "\n\n".join(f"- {c}" for c in chunks)
    prompt = f"""You are a helpful assistant for AcmeCloud.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    resp = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {"answer": resp.choices[0].message.content, "sources": chunks}


if __name__ == "__main__":
    for q in [
        "How much does the Pro plan cost?",
        "Who founded AcmeCloud?",
        "What is the airspeed velocity of an unladen swallow?",
    ]:
        r = rag(q)
        print(f"Q: {q}")
        print(f"A: {r['answer']}\n")
