"""
Section 11 - Day 1 helpers
---------------------------
1. Cost model for your capstone architecture doc
2. Minimal FastAPI backend skeleton with JWT auth

Run the cost model:
    python main.py

Or start the API skeleton in a real app:
    uvicorn main:app --reload
"""

import os
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel


# ------------------------------------------------------------
# Part 1 - cost model helper (run standalone)
# ------------------------------------------------------------
def cost_model(dau: int, q_per_user: int, tokens_per_q: int,
               price_in_per_m: float, price_out_per_m: float,
               ratio_out_to_in: float = 0.4) -> dict:
    in_tok = dau * q_per_user * tokens_per_q
    out_tok = int(in_tok * ratio_out_to_in)
    daily = (in_tok / 1e6) * price_in_per_m + (out_tok / 1e6) * price_out_per_m
    return {
        "dau": dau, "queries_per_user": q_per_user,
        "in_tokens_day": in_tok, "out_tokens_day": out_tok,
        "cost_day_usd": round(daily, 4),
        "cost_month_usd": round(daily * 30, 2),
        "cost_per_active_user_month": round(daily * 30 / max(dau, 1), 3),
    }


# ------------------------------------------------------------
# Part 2 - FastAPI skeleton
# ------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-change-me")
JWT_ALG = "HS256"

app = FastAPI(title="Capstone skeleton")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)
USERS: dict[str, dict] = {}       # swap for SQLAlchemy in the real capstone


class RegisterBody(BaseModel):
    email: str
    password: str


def make_token(sub: str) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(hours=8)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def current_user(token: str | None = Depends(oauth2)) -> str:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing token")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])["sub"]
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


@app.get("/")
def root():
    return {"name": "capstone", "version": "0.1.0"}


@app.get("/healthz")
def health():
    return {"ok": True}


@app.post("/auth/register")
def register(body: RegisterBody):
    if body.email in USERS:
        raise HTTPException(400, "email already registered")
    USERS[body.email] = {"email": body.email, "hashed": pwd.hash(body.password)}
    return {"ok": True}


@app.post("/auth/login")
def login(body: RegisterBody):
    u = USERS.get(body.email)
    if not u or not pwd.verify(body.password, u["hashed"]):
        raise HTTPException(401, "bad credentials")
    return {"token": make_token(body.email)}


@app.get("/me")
def me(email: str = Depends(current_user)):
    return {"email": email}


if __name__ == "__main__":
    # Example: 50 DAU, 20 q/day, ~700 tokens, openai/gpt-oss-20b on Together
    # Pricing (2026): $0.05/M input, $0.20/M output
    print(cost_model(50, 20, 700, 0.05, 0.20))
