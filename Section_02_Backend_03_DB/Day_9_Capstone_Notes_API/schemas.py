"""Pydantic v2 schemas for request/response validation."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

try:
    from pydantic import EmailStr
except ImportError:  # pragma: no cover
    # Fallback if `email-validator` is missing. Install with: pip install pydantic[email]
    EmailStr = str  # type: ignore


# ---------- Users ----------

class UserCreate(BaseModel):
    """Payload for registering a new user."""
    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Public representation of a user."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime


# ---------- Auth ----------

class Token(BaseModel):
    """OAuth2 bearer token response."""
    access_token: str
    token_type: str = "bearer"


# ---------- Tags ----------

class TagOut(BaseModel):
    """Tag as returned in responses."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


# ---------- Notes ----------

class NoteCreate(BaseModel):
    """Payload for creating a note. Tags are referenced by name (find-or-create)."""
    title: str
    content: str
    tag_names: List[str] = []


class NoteUpdate(BaseModel):
    """Partial update for a note. Missing fields stay unchanged."""
    title: Optional[str] = None
    content: Optional[str] = None
    tag_names: Optional[List[str]] = None


class NoteOut(BaseModel):
    """Note as returned in responses, with embedded tags."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    tags: List[TagOut]
    created_at: datetime
    updated_at: datetime
