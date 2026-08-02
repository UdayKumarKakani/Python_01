"""
Day 5 - Serving helper (Ollama HTTP call)
-----------------------------------------
Requires a locally running Ollama with a model named 'triage' (see class notebook).
"""

import httpx


OLLAMA = "http://localhost:11434/api/chat"


def chat(model: str, prompt: str, system: str = "") -> str:
    r = httpx.post(OLLAMA, timeout=60, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


if __name__ == "__main__":
    print(chat("triage", "My card was charged twice.",
               system="Classify into billing / technical / feature-request / other."))
