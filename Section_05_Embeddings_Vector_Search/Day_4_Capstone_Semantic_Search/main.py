"""
Day 4 - Capstone: Semantic Search API
--------------------------------------
Runs entirely on your laptop. No API keys needed.

Start the server:
    uvicorn main:app --reload

Then open http://localhost:8000/docs to try endpoints.

Endpoints:
    POST /ingest    body: {"pdf_path": "..."}
    GET  /search    query: q, top_k, source (optional filter)
    GET  /stats
"""

from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


app = FastAPI(title="Semantic Search API")
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./search_db")
collection = chroma.get_or_create_collection(name="documents")


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
    return chunks


class IngestRequest(BaseModel):
    pdf_path: str


@app.post("/ingest")
def ingest(req: IngestRequest):
    path = Path(req.pdf_path)
    if not path.exists():
        raise HTTPException(404, f"File not found: {req.pdf_path}")

    reader = PdfReader(str(path))
    all_chunks, all_meta, all_ids = [], [], []
    counter = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for chunk in recursive_chunk(text, 500, 50):
            all_chunks.append(chunk)
            all_meta.append({"source": path.name, "page": page_num})
            all_ids.append(f"{path.stem}_{counter}")
            counter += 1

    if not all_chunks:
        return {"chunks_added": 0}

    vectors = model.encode(all_chunks).tolist()
    collection.add(
        documents=all_chunks,
        embeddings=vectors,
        metadatas=all_meta,
        ids=all_ids,
    )
    return {"chunks_added": len(all_chunks), "source": path.name}


@app.get("/search")
def search(q: str, top_k: int = 5, source: str | None = None):
    q_vec = model.encode([q]).tolist()
    where = {"source": source} if source else None
    r = collection.query(query_embeddings=q_vec, n_results=top_k, where=where)
    return {
        "query": q,
        "results": [
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "distance": round(dist, 3),
            }
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ],
    }


@app.get("/stats")
def stats():
    return {"total_chunks": collection.count()}
