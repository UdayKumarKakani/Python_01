"""
Notes API - Capstone

Run:
    uvicorn main:app --reload
Open http://127.0.0.1:8000/docs for Swagger.
"""
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from database import Base, engine
from routers import notes, tags, users

# Create tables (in production, use Alembic migrations instead).
Base.metadata.create_all(bind=engine)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Notes API",
    description="JWT-authenticated note-taking API with tags, built with FastAPI + SQLAlchemy 2.0.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach an X-Process-Time header (milliseconds) to every response."""
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.2f}ms"
    return response


# Mount routers under the /v1 prefix.
app.include_router(users.router, prefix="/v1")
app.include_router(notes.router, prefix="/v1")
app.include_router(tags.router, prefix="/v1")


@app.get("/", tags=["health"])
def root():
    """Simple health/landing endpoint."""
    return {"status": "ok", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
