"""Pydantic request/response schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


# --- Chat ---
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    privacy: bool = False
    force_model: Optional[str] = None  # e.g. "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


# --- Extraction ---
class PersonExtract(BaseModel):
    name: str
    age: int
    email: str


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


# --- Usage ---
class UsageSummary(BaseModel):
    username: str
    calls: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    window_hours: int = 24
