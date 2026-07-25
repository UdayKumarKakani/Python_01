"""
Day 2 - FastAPI Basics
----------------------
A tiny FastAPI app demonstrating root, path params, query params, and POST.

Run:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs    # Swagger UI
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Day 2 - FastAPI Basics")


@app.get("/")
def root():
    return {"hello": "Udaykumar"}


@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None):
    # `item_id` comes from the path, `q` is an optional query parameter
    return {"item_id": item_id, "q": q}


@app.post("/items")
def create_item(item: dict):
    # We accept any JSON object here — Day 3 will replace this with a Pydantic model
    return {"received": item}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
