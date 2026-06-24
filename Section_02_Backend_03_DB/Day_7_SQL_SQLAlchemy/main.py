"""
Day 7 - SQL & SQLAlchemy Basics
-------------------------------
Standalone script (no FastAPI). Demonstrates SQLAlchemy 2.0 typed CRUD
against a local SQLite file (./day7_app.db).

Run:
    python main.py

Re-running is safe — the script wipes the `users` table at the start so the
demo output stays deterministic.
"""

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


# --- Engine + session factory ---
engine = create_engine("sqlite:///./day7_app.db", echo=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id:    Mapped[int] = mapped_column(primary_key=True)
    name:  Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    age:   Mapped[int]

    def __repr__(self) -> str:
        return f"User(id={self.id}, name={self.name!r}, email={self.email!r}, age={self.age})"


SessionLocal = sessionmaker(bind=engine)


def demo() -> None:
    Base.metadata.create_all(engine)

    with SessionLocal() as session:
        # Clean slate so the script is idempotent
        session.execute(delete(User))
        session.commit()

        # --- CREATE ---
        session.add_all([
            User(name="Alice", email="alice@example.com", age=30),
            User(name="Bob",   email="bob@example.com",   age=17),
            User(name="Carol", email="carol@example.com", age=42),
        ])
        session.commit()

        # --- READ all ---
        print("\n[all users]")
        for u in session.execute(select(User)).scalars():
            print(" ", u)

        # --- READ with WHERE ---
        print("\n[adults only (age >= 18)]")
        stmt = select(User).where(User.age >= 18).order_by(User.age.desc())
        for u in session.execute(stmt).scalars():
            print(" ", u)

        # --- UPDATE ---
        alice = session.execute(select(User).where(User.name == "Alice")).scalar_one()
        alice.age = 31
        session.commit()
        print("\n[after update] Alice ->", session.get(User, alice.id))

        # --- DELETE ---
        bob = session.execute(select(User).where(User.name == "Bob")).scalar_one()
        session.delete(bob)
        session.commit()

        # --- Final state ---
        print("\n[final state]")
        for u in session.execute(select(User)).scalars():
            print(" ", u)


if __name__ == "__main__":
    demo()
