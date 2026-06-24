"""
Day 1 - Backend & REST Principles: Demo Script
----------------------------------------------
A small script that exercises the standard HTTP verbs against the public
jsonplaceholder API. This is NOT a FastAPI server — it's a *client* that
shows how to call a REST API from Python.

Run:
    python main.py
"""

import requests

BASE = "https://jsonplaceholder.typicode.com"


def call(method: str, path: str, body: dict | None = None) -> None:
    """Make an HTTP call and print a one-line summary plus the response body."""
    url = f"{BASE}{path}"
    response = requests.request(method, url, json=body)
    print(f"{method} {path} -> {response.status_code}")
    try:
        print("  body:", response.json())
    except Exception:
        # 404 / empty bodies don't always parse as JSON
        print("  body:", response.text[:120])


if __name__ == "__main__":
    # --- GET: read a single resource ---
    call("GET", "/posts/1")

    # --- POST: create a new resource ---
    call("POST", "/posts", {"title": "hello", "body": "first post", "userId": 1})

    # --- PUT: replace an existing resource ---
    call("PUT", "/posts/1", {"id": 1, "title": "updated", "body": "new", "userId": 1})

    # --- DELETE: remove a resource ---
    call("DELETE", "/posts/1")

    # --- GET a missing resource → 404 ---
    call("GET", "/posts/9999")
