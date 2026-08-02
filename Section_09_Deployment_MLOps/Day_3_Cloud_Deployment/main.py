"""Minimal deployable FastAPI app with health check + CORS."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deploy demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/healthz")
def health():
    return {"ok": True}
