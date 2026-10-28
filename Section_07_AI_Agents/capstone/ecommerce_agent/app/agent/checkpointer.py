"""Pluggable conversation-memory backend, selected by `CHECKPOINTER`.

Each backend is an async context manager so connections are opened at startup
and closed cleanly on shutdown (see the FastAPI lifespan in app/main.py).

Adding a backend (e.g. MongoDB): write an `@asynccontextmanager` that yields a
`BaseCheckpointSaver` and register it in `_BACKENDS`.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

CheckpointerFactory = Callable[[Settings], AbstractAsyncContextManager[BaseCheckpointSaver]]


class CheckpointerConfigurationError(RuntimeError):
    pass


@asynccontextmanager
async def _memory(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    yield MemorySaver()


@asynccontextmanager
async def _sqlite(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        raise CheckpointerConfigurationError(
            "CHECKPOINTER=sqlite requires `pip install langgraph-checkpoint-sqlite`"
        ) from exc

    db_path = Path(settings.checkpoint_sqlite_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        await saver.setup()  # create tables now rather than on the first request
        yield saver


_BACKENDS: dict[str, CheckpointerFactory] = {
    "memory": _memory,
    "sqlite": _sqlite,
}


@asynccontextmanager
async def open_checkpointer(settings: Settings | None = None) -> AsyncIterator[BaseCheckpointSaver]:
    settings = settings or get_settings()
    factory = _BACKENDS.get(settings.checkpointer)
    if factory is None:
        raise CheckpointerConfigurationError(
            f"Unknown CHECKPOINTER '{settings.checkpointer}'. Available: {sorted(_BACKENDS)}"
        )

    async with factory(settings) as saver:
        logger.info("Checkpointer ready backend=%s", settings.checkpointer)
        yield saver
