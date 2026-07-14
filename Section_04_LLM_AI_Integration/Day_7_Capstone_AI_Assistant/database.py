"""SQLAlchemy engine + UsageLog model.

One row per completed LLM call. Used for cost dashboards and the
per-user daily budget cap.
"""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

engine = create_engine("sqlite:///./assistant.db", echo=False)


class Base(DeclarativeBase):
    pass


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(index=True)
    provider: Mapped[str]                # together | openai | anthropic
    model: Mapped[str]
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cost_usd: Mapped[float]
    latency_ms: Mapped[int]
    ts: Mapped[datetime] = mapped_column(default=datetime.utcnow, index=True)


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
