"""Tiny FastAPI app for the Docker demo."""
from fastapi import FastAPI

app = FastAPI(title="Docker demo")


@app.get("/")
def root():
    return {"status": "ok", "message": "hello from inside a container"}


@app.get("/healthz")
def health():
    return {"ok": True}
