"""
Day 4 - Middleware & Dependency Injection
-----------------------------------------
A small FastAPI app showing:
  - a reusable pagination dependency
  - a fake current-user dependency reading the X-User header
  - a custom timing middleware that adds X-Process-Time
  - CORS middleware allowing http://localhost:3000

Run:
    uvicorn main:app --reload

Try:
    curl -i http://127.0.0.1:8000/items?skip=2&limit=5
    curl -i -H "X-User: alice" http://127.0.0.1:8000/me
"""

import time

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Day 4 - Middleware & DI")


# --- CORS middleware (built-in) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Custom timing middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


# --- Dependencies ---
def common_pagination(skip: int = 0, limit: int = 10) -> dict:
    """Reusable pagination query params."""
    return {"skip": skip, "limit": limit}


def current_user(x_user: str = Header(default="")) -> str:
    """Fake auth: identify the user by the X-User header."""
    if not x_user:
        raise HTTPException(status_code=401, detail="missing X-User header")
    return x_user


# --- Routes ---
@app.get("/")
def root():
    return {"message": "Day 4 - Middleware & DI"}


@app.get("/items")
def list_items(page: dict = Depends(common_pagination)):
    items = [{"id": i, "name": f"item-{i}"} for i in range(page["skip"], page["skip"] + page["limit"])]
    return {"page": page, "items": items}


@app.get("/me")
def me(user: str = Depends(current_user)):
    return {"user": user}


if __name__ == "__main__":
    uvicorn.run(app)
