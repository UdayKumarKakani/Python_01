"""Read-only endpoints for tags."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import Note, Tag, User
from schemas import NoteOut, TagOut

router = APIRouter(prefix="/tags", tags=["tags"])


@router.get("/", response_model=List[TagOut])
def list_tags(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Tag]:
    """List all tags in the system (tags are global, not per-user)."""
    return list(db.execute(select(Tag).order_by(Tag.name)).scalars().all())


@router.get("/{tag_id}/notes", response_model=List[NoteOut])
def notes_for_tag(
    tag_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Note]:
    """Return notes owned by the current user that carry the given tag."""
    tag = db.get(Tag, tag_id)
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    return [n for n in tag.notes if n.owner_id == current_user.id]
