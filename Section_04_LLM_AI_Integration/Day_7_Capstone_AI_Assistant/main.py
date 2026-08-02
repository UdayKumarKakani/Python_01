"""
Day 7 Capstone — AI Assistant (Together AI + OpenAI + Claude)
-------------------------------------------------------------
FastAPI service that stitches together:
- JWT auth
- Multi-provider routing (default = Together AI open-source LLaMA)
- Blocking + streaming chat endpoints
- Structured extraction with JSON mode
- Token & cost tracking to SQLite
- Per-user daily budget cap

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs
"""

import json
import os
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sqlalchemy import func, select

import providers
from auth import authenticate, current_user, issue_token
from costs import cost_usd, count_tokens
from database import SessionLocal, UsageLog, init_db
from router import FALLBACK, route
from schemas import (
    ChatRequest,
    ChatResponse,
    ExtractRequest,
    LoginRequest,
    PersonExtract,
    TokenResponse,
    UsageSummary,
)

load_dotenv()
init_db()

BUDGET_USD_PER_DAY = float(os.getenv("BUDGET_USD_PER_DAY", "1.00"))

app = FastAPI(title="Section 4 Capstone — Multi-Model AI Assistant", version="1.1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log_usage(username: str, provider: str, model: str,
               in_tokens: int, out_tokens: int, cost: float, latency_ms: int) -> None:
    with SessionLocal() as s:
        s.add(UsageLog(
            username=username, provider=provider, model=model,
            input_tokens=in_tokens, output_tokens=out_tokens,
            cost_usd=cost, latency_ms=latency_ms,
        ))
        s.commit()


def _enforce_budget(username: str) -> None:
    since = datetime.utcnow() - timedelta(hours=24)
    with SessionLocal() as s:
        spent = s.execute(
            select(func.coalesce(func.sum(UsageLog.cost_usd), 0.0))
            .where(UsageLog.username == username, UsageLog.ts >= since)
        ).scalar_one()
    if spent >= BUDGET_USD_PER_DAY:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Daily budget of ${BUDGET_USD_PER_DAY:.2f} reached (spent ${spent:.4f}).",
        )


async def _call_with_fallback(prompt: str, decision) -> tuple[str, int, int, str, str]:
    """Try primary → each fallback. Return (text, in, out, provider, model)."""
    attempts = [(decision.provider, decision.model)] + FALLBACK.get(decision.provider, [])
    last_err = None
    for provider, model in attempts:
        try:
            text, in_tokens, out_tokens = await providers.call(provider, model, prompt)
            return text, in_tokens, out_tokens, provider, model
        except Exception as e:
            last_err = e
            continue
    raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"All providers failed: {last_err}")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/auth/token", response_model=TokenResponse)
def login(req: LoginRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad credentials")
    return TokenResponse(access_token=issue_token(req.username))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, username: str = Depends(current_user)):
    _enforce_budget(username)
    decision = route(req.prompt, privacy=req.privacy, force_model=req.force_model)

    t0 = time.time()
    text, in_tokens, out_tokens, provider, model = await _call_with_fallback(req.prompt, decision)
    latency_ms = int((time.time() - t0) * 1000)

    cost = cost_usd(provider, model, in_tokens, out_tokens)
    _log_usage(username, provider, model, in_tokens, out_tokens, cost, latency_ms)

    return ChatResponse(
        text=text,
        provider=provider,
        model=model,
        reason=decision.reason,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, username: str = Depends(current_user)):
    _enforce_budget(username)
    decision = route(req.prompt, privacy=req.privacy, force_model=req.force_model)

    async def gen():
        t0 = time.time()
        buf: list[str] = []
        try:
            async for delta in providers.stream(decision.provider, decision.model, req.prompt):
                buf.append(delta)
                yield delta
        except Exception as e:
            yield f"\n\n[stream error: {e}]"

        full_out = "".join(buf)
        in_tokens = count_tokens(req.prompt)
        out_tokens = count_tokens(full_out)
        cost = cost_usd(decision.provider, decision.model, in_tokens, out_tokens)
        latency_ms = int((time.time() - t0) * 1000)
        _log_usage(username, decision.provider, decision.model,
                   in_tokens, out_tokens, cost, latency_ms)

    headers = {
        "X-Route-Provider": decision.provider,
        "X-Route-Model": decision.model,
        "X-Route-Reason": decision.reason,
    }
    return StreamingResponse(gen(), media_type="text/plain", headers=headers)


# ---------------------------------------------------------------------------
# Structured extraction — Together AI JSON mode (cheap default)
# ---------------------------------------------------------------------------
@app.post("/extract", response_model=PersonExtract)
async def extract(req: ExtractRequest, username: str = Depends(current_user)):
    _enforce_budget(username)

    tg = AsyncOpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    model = "openai/gpt-oss-20b"

    prompt = (
        "Extract as JSON with keys name (string), age (int), email (string). "
        "Reply with ONLY the JSON object.\n\nText: " + req.text
    )

    t0 = time.time()
    r = await tg.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    latency_ms = int((time.time() - t0) * 1000)

    try:
        data = json.loads(r.choices[0].message.content)
        person = PersonExtract(**data)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"Model returned unparseable output: {e}")

    usage = r.usage
    in_t = getattr(usage, "prompt_tokens", None) or count_tokens(prompt)
    out_t = getattr(usage, "completion_tokens", None) or count_tokens(r.choices[0].message.content)
    _log_usage(username, "together", model, in_t, out_t,
               cost_usd("together", model, in_t, out_t), latency_ms)

    return person


# ---------------------------------------------------------------------------
# Usage report
# ---------------------------------------------------------------------------
@app.get("/usage/me", response_model=UsageSummary)
def my_usage(username: str = Depends(current_user)):
    since = datetime.utcnow() - timedelta(hours=24)
    with SessionLocal() as s:
        row = s.execute(
            select(
                func.count(UsageLog.id),
                func.coalesce(func.sum(UsageLog.input_tokens), 0),
                func.coalesce(func.sum(UsageLog.output_tokens), 0),
                func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
            ).where(UsageLog.username == username, UsageLog.ts >= since)
        ).one()
    calls, in_tokens, out_tokens, total_cost = row
    return UsageSummary(
        username=username,
        calls=calls,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_cost_usd=total_cost,
    )


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}
