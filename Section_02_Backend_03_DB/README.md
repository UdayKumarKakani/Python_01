# Section 02 — Backend Engineering + Section 03 — Databases

A 9-day program covering REST API design, FastAPI, Pydantic v2, authentication, rate limiting, OpenAPI docs, and relational databases with SQLAlchemy — ending in a production-style **Notes API** capstone.

## Day-by-day

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | Backend Architecture & REST Principles | `Day_1_Backend_REST_Principles/` |
| 2 | FastAPI Basics — Routing | `Day_2_FastAPI_Basics/` |
| 3 | Pydantic v2 — Request Validation | `Day_3_Pydantic_Validation/` |
| 4 | Middleware & Dependency Injection | `Day_4_Middleware_DI/` |
| 5 | Authentication — API Keys & JWT | `Day_5_Auth_JWT/` |
| 6 | Rate Limiting, Versioning & OpenAPI Docs | `Day_6_RateLimit_Versioning_Docs/` |
| 7 | SQL & SQLAlchemy Basics | `Day_7_SQL_SQLAlchemy/` |
| 8 | Relationships, Migrations & FastAPI Integration | `Day_8_Relationships_Migrations/` |
| 9 | Git Workflow + Capstone — Notes API | `Day_9_Capstone_Notes_API/` |

## How each day is organized

Each day folder contains:
- `concepts.ipynb` — main teaching notebook (Colab-friendly)
- `main.py` — runnable Python equivalent (run with `uvicorn main:app --reload` when applicable)
- `assignments.ipynb` — tasks for the day (4 tasks + bonus)

## Setup

```bash
# Local
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Run any day's FastAPI app
cd Day_2_FastAPI_Basics
uvicorn main:app --reload
# Open http://127.0.0.1:8000/docs
```

### Google Colab

Each `concepts.ipynb` works in Colab. Install deps in the first cell:
```python
!pip install fastapi uvicorn pydantic python-jose[cryptography] passlib[bcrypt] slowapi sqlalchemy
```

For FastAPI in Colab, use `nest_asyncio` + `uvicorn.run(app)` in a thread, or test endpoints with `TestClient` from `fastapi.testclient`. The capstone day shows both approaches.

## Stack

- **FastAPI 0.115** + **Uvicorn**
- **Pydantic v2** for validation
- **python-jose** + **passlib[bcrypt]** for JWT auth
- **slowapi** for rate limiting
- **SQLAlchemy 2.0** ORM + **Alembic** migrations
- **SQLite** as the database (zero-config)

## Prerequisites

- Completed `Python_basics/` (especially OOP, exception handling, type hints, JSON/CSV)
- Python 3.10+
