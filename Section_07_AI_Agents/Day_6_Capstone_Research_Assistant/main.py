"""
Day 6 - Capstone: Autonomous Research Assistant
------------------------------------------------
Requires TOGETHER_API_KEY. Optional: TAVILY_API_KEY (mock is used otherwise).

Start:
    uvicorn main:app --reload
Open http://localhost:8000/docs

Endpoints:
    POST /research                 { "question": "..." } -> { "job_id": "..." }
    GET  /research/{job_id}        -> { "status": ..., "report": ... }
    GET  /research/memory?q=...    semantic search of past reports
"""

import operator
import os
import time
import uuid
from typing import Annotated, TypedDict

import chromadb
import httpx
import tiktoken
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
llm = Together()
MODEL = "openai/gpt-oss-20b"
enc = tiktoken.encoding_for_model("gpt-4o-mini")
embed = SentenceTransformer("all-MiniLM-L6-v2")
memory = chromadb.PersistentClient(path="./research_db").get_or_create_collection("reports")
app = FastAPI(title="Autonomous Research Assistant")
JOBS: dict[str, dict] = {}


# ------------------------------------------------------------
# Budget (Day 5)
# ------------------------------------------------------------
class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_steps: int = 6, max_tokens: int = 20_000):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.steps = 0
        self.tokens = 0

    def charge_step(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded("max_steps")

    def charge_tokens(self, text: str) -> None:
        self.tokens += len(enc.encode(text))
        if self.tokens > self.max_tokens:
            raise BudgetExceeded("max_tokens")


# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
def web_search(query: str) -> list[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return [{"title": f"[MOCK] {query}", "url": "https://example.com",
                 "content": "(set TAVILY_API_KEY for real results)"}]
    from tavily import TavilyClient
    tv = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    r = tv.search(query, max_results=3)
    return [{"title": x["title"], "url": x["url"], "content": x["content"][:500]}
            for x in r["results"]]


def fetch_url(url: str) -> str:
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        return r.text[:2000]
    except Exception as e:
        return f"error: {e}"


# ------------------------------------------------------------
# State + nodes
# ------------------------------------------------------------
class ResearchState(TypedDict):
    question: str
    queries: list[str]
    findings: list[dict]                       # [{title, url, content}]
    report: str
    sources: list[str]
    steps: Annotated[list[str], operator.add]
    budget: Budget


def _chat(prompt: str, state: ResearchState, temperature: float = 0.2) -> str:
    state["budget"].charge_step()
    r = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = r.choices[0].message.content.strip()
    state["budget"].charge_tokens(prompt + text)
    return text


def plan_node(state: ResearchState) -> dict:
    prompt = (
        f"You are a research planner. The user asks: {state['question']!r}\n"
        "List 2-3 short web search queries that would help answer it. "
        "One per line, no numbering, no extra text."
    )
    raw = _chat(prompt, state)
    queries = [q.strip("- ") for q in raw.splitlines() if q.strip()][:3]
    return {"queries": queries, "steps": ["planned"]}


def search_node(state: ResearchState) -> dict:
    findings: list[dict] = []
    for q in state["queries"]:
        state["budget"].charge_step()
        findings.extend(web_search(q))
    return {"findings": findings, "steps": [f"searched({len(findings)})"]}


def synthesize_node(state: ResearchState) -> dict:
    context = "\n\n".join(
        f"[{i+1}] {f['title']}\n{f['content']}" for i, f in enumerate(state["findings"])
    )
    prompt = (
        f"You are a research writer. Answer the question using ONLY the numbered "
        f"sources below. Cite with [1], [2], etc. Keep it under 250 words.\n\n"
        f"Question: {state['question']}\n\nSources:\n{context}\n\nReport:"
    )
    report = _chat(prompt, state)
    sources = [f["url"] for f in state["findings"]]
    return {"report": report, "sources": sources, "steps": ["synthesized"]}


def save_node(state: ResearchState) -> dict:
    memory.add(
        documents=[state["report"]],
        embeddings=embed.encode([state["report"]]).tolist(),
        metadatas=[{"question": state["question"], "ts": time.time()}],
        ids=[f"report_{time.time_ns()}"],
    )
    return {"steps": ["saved"]}


# ------------------------------------------------------------
# Graph
# ------------------------------------------------------------
builder = StateGraph(ResearchState)
builder.add_node("plan",       plan_node)
builder.add_node("search",     search_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("save",       save_node)
builder.add_edge(START,        "plan")
builder.add_edge("plan",       "search")
builder.add_edge("search",     "synthesize")
builder.add_edge("synthesize", "save")
builder.add_edge("save",       END)
graph = builder.compile()


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
class ResearchBody(BaseModel):
    question: str


def run_job(job_id: str, question: str) -> None:
    initial: ResearchState = {
        "question": question, "queries": [], "findings": [],
        "report": "", "sources": [], "steps": [], "budget": Budget(),
    }
    try:
        final = graph.invoke(initial)
        JOBS[job_id] = {
            "status": "done",
            "report": final["report"],
            "sources": final["sources"],
            "steps": final["steps"],
        }
    except BudgetExceeded as e:
        JOBS[job_id] = {"status": "stopped_budget", "reason": str(e)}
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}


@app.post("/research")
def start_research(body: ResearchBody, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running"}
    bg.add_task(run_job, job_id, body.question)
    return {"job_id": job_id, "status": "running"}


@app.get("/research/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.get("/research/memory")
def memory_search(q: str, top_k: int = 3):
    r = memory.query(query_embeddings=embed.encode([q]).tolist(), n_results=top_k)
    return [
        {"question": m["question"], "report": d}
        for d, m in zip(r["documents"][0], r["metadatas"][0])
    ]
