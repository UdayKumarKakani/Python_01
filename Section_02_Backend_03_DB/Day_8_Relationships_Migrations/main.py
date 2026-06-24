"""
Day 8 - Relationships & FastAPI Integration
-------------------------------------------
A small FastAPI app showing a one-to-many relationship (User -> Posts)
backed by SQLite via SQLAlchemy 2.0, with cascade delete and a clean
Pydantic <-> SQLAlchemy split.

Run:
    uvicorn main:app --reload

Try:
    curl -X POST http://127.0.0.1:8000/users \
         -H "Content-Type: application/json" \
         -d '{"name":"Alice","email":"a@x.com"}'

    curl -X POST http://127.0.0.1:8000/users/1/posts \
         -H "Content-Type: application/json" \
         -d '{"title":"Hello","content":"first post"}'

    curl http://127.0.0.1:8000/users/1/posts
    curl -X DELETE http://127.0.0.1:8000/users/1
"""

from __future__ import annotations

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import ForeignKey, create_engine, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


# --- DB setup --------------------------------------------------------------
engine = create_engine(
    "sqlite:///./day8_app.db",
    connect_args={"check_same_thread": False},
)


class Base(DeclarativeBase):
    pass


# --- SQLAlchemy models -----------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id:    Mapped[int] = mapped_column(primary_key=True)
    name:  Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)

    posts: Mapped[list["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",
    )


class Post(Base):
    __tablename__ = "posts"

    id:        Mapped[int] = mapped_column(primary_key=True)
    title:     Mapped[str]
    content:   Mapped[str]
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    author: Mapped["User"] = relationship(back_populates="posts")


# --- Pydantic schemas (kept separate from SQLAlchemy models) ---------------
class UserCreate(BaseModel):
    name: str
    email: str


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    model_config = ConfigDict(from_attributes=True)


class PostCreate(BaseModel):
    title: str
    content: str


class PostOut(BaseModel):
    id: int
    title: str
    content: str
    author_id: int
    model_config = ConfigDict(from_attributes=True)


# --- Session factory + dep -------------------------------------------------
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- App + startup ---------------------------------------------------------
app = FastAPI(title="Day 8 - Relationships & Integration")
Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"message": "Day 8 - Relationships & Integration"}


# --- User endpoints --------------------------------------------------------
@app.post("/users", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = User(name=payload.name, email=payload.email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.execute(select(User)).scalars().all()


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    return user


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    db.delete(user)  # cascade="all, delete-orphan" removes related posts
    db.commit()
    return None


# --- Post endpoints (nested under user) ------------------------------------
@app.post("/users/{user_id}/posts", response_model=PostOut, status_code=201)
def create_post(user_id: int, payload: PostCreate, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    post = Post(title=payload.title, content=payload.content, author=user)
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


@app.get("/users/{user_id}/posts", response_model=list[PostOut])
def list_user_posts(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    return user.posts


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
