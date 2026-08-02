"""Pydantic request/response schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Auth ---
class LoginRequest(BaseModel):
    """Kept for backwards compatibility; /auth/token now uses OAuth2PasswordRequestForm."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


# --- Chat ---
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    privacy: bool = False
    # Format: "<provider>/<model_id>" where provider is one of
    # together / openai / anthropic.
    #   "together/openai/gpt-oss-20b"       - Together-hosted OSS default
    #   "together/openai/gpt-oss-120b"      - Together-hosted big sibling
    #   "openai/gpt-4o-mini"                - OpenAI directly
    #   "anthropic/claude-3-5-sonnet-latest"
    # Note: "openai/gpt-oss-20b" is treated as OpenAI-direct. Prepend
    # "together/" to route through Together.
    force_model: Optional[str] = None


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
