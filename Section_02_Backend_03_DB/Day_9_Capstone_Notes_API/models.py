"""SQLAlchemy 2.0 typed ORM models for Users, Notes, and Tags."""
from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Table, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


# Association table for the many-to-many Note <-> Tag relationship.
note_tags = Table(
    "note_tags",
    Base.metadata,
    Column("note_id", Integer, ForeignKey("notes.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    """Application user with hashed password."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    notes: Mapped[List["Note"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class Note(Base):
    """A note owned by a user, optionally tagged with multiple tags."""
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False, default="")
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    owner: Mapped["User"] = relationship(back_populates="notes")
    tags: Mapped[List["Tag"]] = relationship(
        secondary=note_tags,
        back_populates="notes",
    )


class Tag(Base):
    """A tag that can be attached to many notes."""
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)

    notes: Mapped[List["Note"]] = relationship(
        secondary=note_tags,
        back_populates="tags",
    )
