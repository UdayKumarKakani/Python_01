# Day 7 Capstone — AI Assistant (Together AI + OpenAI + Claude)

A small but production-shaped FastAPI service that ties together everything from Section 4:

- **Three providers** you can switch between: **Together AI** (default, cheap open source), OpenAI, and Claude.
- **Automatic routing** — cheapest model that fits the task, with fallback if the primary fails.
- **Streaming** responses so replies feel instant.
- **Structured extraction** endpoint using JSON mode.
- **Token & cost tracking** to SQLite per user.
- **JWT auth** so only logged-in users can call it.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/auth/token` | Log in with a demo user, get a JWT |
| POST | `/chat` | Send a message, get a reply. Router picks the model. |
| POST | `/chat/stream` | Same but streams tokens as they arrive |
| POST | `/extract` | Extract structured JSON (name/age/email) from any text |
| GET | `/usage/me` | See your token usage & cost for the last 24 hours |
| GET | `/health` | Liveness check |

## Routing rules (see `router.py`)

1. Client sent `force_model="provider/model"` → use that.
2. Client sent `privacy=True` → Together AI (open source, cheap, easy to self-host later).
3. Prompt has code keywords (`code`, `python`, `sql`, `refactor`) → **Claude 3.5 Sonnet**.
4. Prompt longer than 800 chars → **Claude 3.5 Sonnet** (long-context strength).
5. Otherwise → **Together AI LLaMA 3.1 8B Turbo** (cheapest default).

Fallback order: if the primary provider fails, we try the next-cheapest in the same tier before giving up.

## Files

```
main.py           - FastAPI app
auth.py           - JWT auth (same pattern as Section 2)
router.py         - route(prompt) + fallback cascade
providers.py      - Async wrappers around Together AI, OpenAI, Claude
costs.py          - Pricing table + tiktoken-based estimation
database.py       - SQLite via SQLAlchemy — UsageLog model
schemas.py        - Pydantic request/response models
walkthrough.ipynb - Guided demo of every endpoint via httpx
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env — TOGETHER_API_KEY is REQUIRED, the others are optional but
# unlock more routing rules
TOGETHER_API_KEY=...
OPENAI_API_KEY=sk-...           # optional
ANTHROPIC_API_KEY=sk-ant-...    # optional
JWT_SECRET=change-me-in-prod
BUDGET_USD_PER_DAY=1.00

uvicorn main:app --reload
# Open http://127.0.0.1:8000/docs
```

## Quick smoke test

```bash
# 1. Get a token (demo user)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"wonderland"}' | jq -r .access_token)

# 2. Chat — routes to Together AI (cheap default)
curl -s -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"prompt":"Say hi in 5 words."}'

# 3. Code prompt — routes to Claude (if key set)
curl -s -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"prompt":"Refactor this Python code: def double(xs): out=[]; ..."}'

# 4. Structured extraction
curl -s -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"text":"Rohan Mehta, 34, rohan@example.com"}'

# 5. Usage report
curl -s http://localhost:8000/usage/me -H "Authorization: Bearer $TOKEN"
```
