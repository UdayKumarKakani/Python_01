# Notes API — Capstone Project

A production-style, JWT-authenticated note-taking API that brings together everything covered in Days 1–8: REST design, FastAPI, Pydantic v2, middleware, dependency injection, JWT auth, rate limiting, OpenAPI customization, and SQLAlchemy 2.0 ORM with relationships.

## Features

- JWT-based authentication (OAuth2 password flow)
- CRUD for user-owned notes
- Tags with a many-to-many relationship to notes (find-or-create by name)
- Rate limiting on auth endpoints via `slowapi` (5/minute)
- Versioned routes mounted under `/v1`
- Process-time middleware emitting an `X-Process-Time` header
- Auto-generated OpenAPI / Swagger UI at `/docs`
- SQLAlchemy 2.0 typed ORM (`Mapped`, `mapped_column`, `select`) on SQLite

## Architecture

```
[Client] -> [FastAPI middleware] -> [Routers] -> [auth.py / db dep] -> [SQLAlchemy models] -> [SQLite app.db]
                                                  |
                                                  +-> [Pydantic schemas in/out]
```

## Endpoints

| Method | Path                       | Auth | Description                                  |
|--------|----------------------------|------|----------------------------------------------|
| GET    | `/`                        | No   | Health / landing                             |
| POST   | `/v1/users/register`       | No   | Create a new user (rate limited 5/min)       |
| POST   | `/v1/users/login`          | No   | Exchange creds for JWT (rate limited 5/min)  |
| GET    | `/v1/users/me`             | Yes  | Current user's profile                       |
| POST   | `/v1/notes/`               | Yes  | Create a note (optional `tag_names`)         |
| GET    | `/v1/notes/`               | Yes  | List my notes (`skip`, `limit`)              |
| GET    | `/v1/notes/{id}`           | Yes  | Get one of my notes                          |
| PATCH  | `/v1/notes/{id}`           | Yes  | Partial update; replaces tags if provided    |
| DELETE | `/v1/notes/{id}`           | Yes  | Delete one of my notes                       |
| GET    | `/v1/tags/`                | Yes  | List all tags                                |
| GET    | `/v1/tags/{id}/notes`      | Yes  | My notes carrying this tag                   |

## Setup

```bash
uv venv notesapi --python 3.10 && .\my_env\Scripts\Activate.ps1 && uv pip install -r requirements.txt && uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/docs>.

## Project structure

```
Day_9_Capstone_Notes_API/
  README.md
  requirements.txt
  .gitignore
  main.py              # FastAPI app, middleware, router wiring
  database.py          # Engine, Base, get_db dependency
  models.py            # SQLAlchemy 2.0 typed models
  schemas.py           # Pydantic v2 request/response schemas
  auth.py              # Password hashing + JWT + get_current_user
  routers/
    __init__.py
    users.py           # /v1/users  (register, login, me)
    notes.py           # /v1/notes  (CRUD)
    tags.py            # /v1/tags   (list, notes-by-tag)
  walkthrough.ipynb    # Runnable end-to-end walkthrough
```

## Git workflow

- Branch per change: `feature/notes-crud`, `fix/jwt-expiry`, `docs/readme-endpoints`
- Conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- PR checklist:
  - [ ] Tests / manual smoke pass
  - [ ] `python -m py_compile` clean
  - [ ] Docs / README updated if endpoints changed
  - [ ] No secrets, no `.db` files committed

## Extension ideas

- Full-text search across note content
- Soft delete + restore (`deleted_at` column)
- Attachments / file uploads (S3 or local)
- Roles + admin endpoints
- Refresh tokens and token revocation list
- Alembic migrations instead of `create_all`
- Per-user tag namespaces
- WebSocket live updates
