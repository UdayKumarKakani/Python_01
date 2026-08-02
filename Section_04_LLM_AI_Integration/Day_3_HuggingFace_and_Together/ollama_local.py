"""
Section 4 - Day 3 bonus: talking to a locally-running Ollama server
--------------------------------------------------------------------
Ollama is the easiest way to run an open-source LLM on your own laptop
(no GPU required, though GPU helps). It exposes an HTTP API on port 11434
that looks a lot like OpenAI's.

Setup (one-time):
    1. Install Ollama:  https://ollama.com  (macOS / Linux / Windows)
    2. Pull a small model:
           ollama pull llama3.2:3b
       (~2 GB. Alternative small ones: qwen2.5:3b, phi3:mini, mistral:7b)
    3. Confirm it works:
           ollama run llama3.2:3b "say hi"

Then just:
    python ollama_local.py

This file demonstrates the two most common calls:
    - one-shot chat completion
    - streaming chat (token-by-token)

Uses `httpx` directly so you can see the exact request shape.
Requires only:  pip install httpx
"""

import json
import sys

import httpx


OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.1:8b"       # change if you pulled a different model


# ------------------------------------------------------------
# 1. One-shot chat
# ------------------------------------------------------------
def chat(messages: list[dict], model: str = MODEL,
         temperature: float = 0.3) -> str:
    """Send a chat request, wait for the full response."""
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        timeout=60,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ------------------------------------------------------------
# 2. Streaming chat (prints tokens as they arrive)
# ------------------------------------------------------------
def stream(messages: list[dict], model: str = MODEL,
           temperature: float = 0.3) -> str:
    """Stream tokens - print each as it arrives, return the joined text."""
    out_parts: list[str] = []
    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/chat",
        timeout=60,
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        },
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                out_parts.append(piece)
                print(piece, end="", flush=True)
            if data.get("done"):
                break
    print()
    return "".join(out_parts)


# ------------------------------------------------------------
# 3. Helper: check if the server is up
# ------------------------------------------------------------
def is_up() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------
if __name__ == "__main__":
    if not is_up():
        print(
            "Ollama server not reachable at", OLLAMA_URL,
            "\n  Install:  https://ollama.com",
            f"\n  Pull:     ollama pull {MODEL}",
            "\n  Then leave `ollama serve` running (macOS: it auto-starts).",
            sep=" ",
        )
        sys.exit(1)

    system = {"role": "system", "content": "You are a concise Python tutor."}
    user1  = {"role": "user",   "content": "In 3 lines, explain what a Python decorator is."}

    print(f"--- 1. one-shot chat with {MODEL} ---\n")
    print(chat([system, user1]))

    print(f"\n\n--- 2. streaming chat with {MODEL} ---\n")
    stream([system, {"role": "user", "content": "In 3 lines, explain what a Python decorator is."}])
