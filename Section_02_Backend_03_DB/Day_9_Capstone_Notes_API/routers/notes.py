"""CRUD endpoints for notes, scoped to the authenticated user."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import Note, Tag, User
from schemas import NoteCreate, NoteOut, NoteUpdate

router = APIRouter(prefix="/notes", tags=["notes"])


def _get_or_create_tags(db: Session, names: List[str]) -> List[Tag]:
    """Resolve tag names to Tag rows, creating any that don't yet exist."""
    tags: List[Tag] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        tag = db.execute(select(Tag).where(Tag.name == name)).scalar_one_or_none()
        if tag is None:
            tag = Tag(name=name)
            db.add(tag)
            db.flush()  # assign PK before linking
        tags.append(tag)
    return tags


@router.post("/", response_model=NoteOut, status_code=status.HTTP_201_CREATED)
def create_note(
    payload: NoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Note:
    """Create a new note owned by the current user."""
    note = Note(title=payload.title, content=payload.content, owner_id=current_user.id)
    if payload.tag_names:
        note.tags = _get_or_create_tags(db, payload.tag_names)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@router.get("/", response_model=List[NoteOut])
def list_notes(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Note]:
    """List notes belonging to the current user, paginated."""
    stmt = (
        select(Note)
        .where(Note.owner_id == current_user.id)
        .order_by(Note.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


@router.get("/{note_id}", response_model=NoteOut)
def get_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Note:
    """Fetch one note by id, 404 if it doesn't belong to the current user."""
    note = db.execute(
        select(Note).where(Note.id == note_id, Note.owner_id == current_user.id)
    ).scalar_one_or_none()
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.patch("/{note_id}", response_model=NoteOut)
def update_note(
    note_id: int,
    payload: NoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Note:
    """Partial update for a note. When `tag_names` is provided, tags are replaced."""
    note = db.execute(
        select(Note).where(Note.id == note_id, Note.owner_id == current_user.id)
    ).scalar_one_or_none()
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    if payload.title is not None:
        note.title = payload.title
    if payload.content is not None:
        note.content = payload.content
    if payload.tag_names is not None:
        note.tags = _get_or_create_tags(db, payload.tag_names)

    db.commit()
    db.refresh(note)
    return note


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a note owned by the current user."""
    note = db.execute(
        select(Note).where(Note.id == note_id, Note.owner_id == current_user.id)
    ).scalar_one_or_none()
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(note)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
