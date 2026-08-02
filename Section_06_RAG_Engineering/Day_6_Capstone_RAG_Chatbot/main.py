"""
Day 6 - Capstone: Enterprise RAG Chatbot
-----------------------------------------
Requires TOGETHER_API_KEY in .env.

Start:
    uvicorn main:app --reload
Then open http://localhost:8000/docs

Endpoints:
    POST /login    { "username": "...", "password": "..." }  -> { "token": "..." }
    POST /ingest   { "source": "path or URL" }               (JWT required)
    POST /ask      { "question": "...", "top_k": 5 }         (JWT required, streams)
    GET  /usage                                              (JWT required)

NOTE: /login is a STUB. Replace with your real user store from Section 2.
"""

import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import chromadb
import tiktoken
import trafilatura
from docx import Document
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer
from together import Together

load_dotenv()

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base")
llm = Together()
enc = tiktoken.encoding_for_model("gpt-4o-mini")

chroma = chromadb.PersistentClient(path="./rag_db")
kb = chroma.get_or_create_collection("enterprise_kb")

conn = sqlite3.connect("./usage.db", check_same_thread=False)
conn.execute(
    "CREATE TABLE IF NOT EXISTS usage ("
    "user TEXT, ts TEXT, input_tokens INT, output_tokens INT)"
)
conn.commit()

app = FastAPI(title="Enterprise RAG Chatbot")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)


# ------------------------------------------------------------
# Loaders (Day 2)
# ------------------------------------------------------------
def load_pdf(path: str) -> str:
    return "\n\n".join(
        (p.extract_text() or "").strip()
        for p in PdfReader(path).pages
        if (p.extract_text() or "").strip()
    )


def load_docx(path: str) -> str:
    return "\n\n".join(p.text for p in Document(path).paragraphs if p.text.strip())


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_url(url: str) -> str:
    html = trafilatura.fetch_url(url)
    return trafilatura.extract(html) or "" if html else ""


LOADERS: dict[str, Callable[[str], str]] = {
    ".pdf": load_pdf, ".docx": load_docx, ".md": load_text, ".txt": load_text,
}


def load(source: str) -> str:
    if source.startswith("http"):
        return load_url(source)
    ext = Path(source).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported format: {ext}")
    return LOADERS[ext](source)


# ------------------------------------------------------------
# Chunking + prompt utilities
# ------------------------------------------------------------
def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


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


INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above)",
    r"you are now [A-Z]",
    r"reveal (the |your )?(system|initial) prompt",
]


def looks_like_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def build_messages(question: str, chunks: list[str]) -> list[dict]:
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    system = (
        "You are a helpful assistant. Answer using ONLY the numbered context. "
        "If the answer is not there, say 'I don't know.' "
        "Cite sources with bracket numbers.\n\n"
        "Format your answer like this:\n"
        "The Pro plan costs $29/month [2]."
    )
    user = (
        f"Context:\n{numbered}\n\n"
        f"<user_question>\n{question}\n</user_question>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ------------------------------------------------------------
# Auth (Section 2 pattern - simplified)
# ------------------------------------------------------------
class LoginBody(BaseModel):
    username: str
    password: str


def make_token(username: str) -> str:
    payload = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=8)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def current_user(token: Optional[str] = Depends(oauth2_scheme)) -> str:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return data["sub"]
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.post("/login")
def login(body: LoginBody):
    # STUB: replace with real user check
    if not body.username or not body.password:
        raise HTTPException(400, "username & password required")
    return {"token": make_token(body.username)}


class IngestBody(BaseModel):
    source: str


@app.post("/ingest")
def ingest(body: IngestBody, user: str = Depends(current_user)):
    text = clean(load(body.source))
    chunks = recursive_chunk(text)
    if not chunks:
        return {"chunks_added": 0}

    ids = [f"{user}::{body.source}::{i}" for i in range(len(chunks))]
    kb.add(
        documents=chunks,
        embeddings=embedder.encode(chunks).tolist(),
        metadatas=[{"user": user, "source": body.source} for _ in chunks],
        ids=ids,
    )
    return {"chunks_added": len(chunks), "source": body.source}


class AskBody(BaseModel):
    question: str
    top_k: int = 5


DISTANCE_THRESHOLD = 1.0
CANDIDATE_POOL = 20


def log_usage(user: str, in_tok: int, out_tok: int) -> None:
    conn.execute(
        "INSERT INTO usage VALUES (?, ?, ?, ?)",
        (user, datetime.utcnow().isoformat(), in_tok, out_tok),
    )
    conn.commit()


def stream_answer(user: str, question: str, chunks: list[str]):
    messages = build_messages(question, chunks)
    input_tokens = sum(len(enc.encode(m["content"])) for m in messages)
    output_text = ""

    stream = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content or ""
        output_text += delta
        if delta:
            yield delta

    log_usage(user, input_tokens, len(enc.encode(output_text)))


@app.post("/ask")
def ask(body: AskBody, user: str = Depends(current_user)):
    if looks_like_injection(body.question):
        return {"answer": "I can't help with that."}

    q_vec = embedder.encode([body.question]).tolist()
    r = kb.query(query_embeddings=q_vec, n_results=CANDIDATE_POOL)

    docs = r["documents"][0] if r["documents"] else []
    dists = r["distances"][0] if r["distances"] else []
    if not docs or dists[0] > DISTANCE_THRESHOLD:
        return {"answer": "I don't know - I couldn't find a confident match."}

    scores = reranker.predict([(body.question, d) for d in docs])
    top = [d for d, _ in sorted(zip(docs, scores), key=lambda x: -x[1])[: body.top_k]]

    return StreamingResponse(
        stream_answer(user, body.question, top),
        media_type="text/plain",
    )


@app.get("/usage")
def usage(user: str = Depends(current_user)):
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) "
        "FROM usage WHERE user = ?",
        (user,),
    ).fetchone()
    n, in_tok, out_tok = row
    # openai/gpt-oss-20b pricing on Together (2026): $0.05/M in, $0.20/M out
    cost = (in_tok / 1_000_000) * 0.05 + (out_tok / 1_000_000) * 0.20
    return {
        "user": user,
        "questions": n,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "est_cost_usd": round(cost, 6),
    }
