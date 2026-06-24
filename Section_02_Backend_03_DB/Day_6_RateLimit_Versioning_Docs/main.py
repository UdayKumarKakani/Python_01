"""
Day 6 - Rate Limiting, Versioning & OpenAPI Docs
------------------------------------------------
A FastAPI app combining:
  - slowapi rate limiting on /ping (3/minute)
  - v1 and v2 routers (URL-prefix versioning)
  - customized OpenAPI metadata (title, description, version)
  - a deprecated endpoint with a summary pointing to its replacement

Run:
    uvicorn main:app --reload

Open in a browser:
    http://127.0.0.1:8000/docs       Swagger UI
    http://127.0.0.1:8000/redoc      ReDoc
    http://127.0.0.1:8000/openapi.json
"""

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Day 6 Demo",
    description="Rate limiting (slowapi), URL-prefix versioning, and customized OpenAPI docs.",
    version="1.0.0",
    contact={"name": "Bootcamp", "email": "bootcamp@example.com"},
)

# Hook the limiter into the app so the decorator works
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Versioned routers ---
v1 = APIRouter(prefix="/v1", tags=["v1"])
v2 = APIRouter(prefix="/v2", tags=["v2"])


@v1.get("/items", summary="List items (v1)", description="Returns minimal item info.")
def items_v1():
    return [{"id": 1, "name": "pen"}, {"id": 2, "name": "notebook"}]


@v2.get(
    "/items",
    summary="List items (v2, enriched)",
    description="Returns items with price and stock fields.",
)
def items_v2():
    return [
        {"id": 1, "name": "pen", "price": 2.5, "stock": 100},
        {"id": 2, "name": "notebook", "price": 4.0, "stock": 50},
    ]


app.include_router(v1)
app.include_router(v2)


# --- Rate-limited endpoint ---
@app.get("/ping", tags=["util"], summary="Health check (rate limited)")
@limiter.limit("3/minute")
def ping(request: Request):
    return {"pong": True}


# --- Deprecated endpoint ---
@app.get(
    "/old",
    deprecated=True,
    tags=["legacy"],
    summary="Use /v2/items instead",
    description="This endpoint will be removed in the next major release.",
)
def old():
    return {"warning": "deprecated, use /v2/items"}


@app.get("/", tags=["util"])
def root():
    return {"message": "Day 6 - Rate limiting, versioning, docs"}


if __name__ == "__main__":
    uvicorn.run(app)
