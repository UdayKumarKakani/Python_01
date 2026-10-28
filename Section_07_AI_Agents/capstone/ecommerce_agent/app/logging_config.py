"""Console logging with per-request context.

Every log line carries the request ID and session ID of the chat turn that produced it,
so one conversation can be followed through the router, tools and escalation even when
requests interleave:

    14:02:11.402 INFO    [req=7c1e9a2b sid=demo-1] app.agent.nodes: [router] LLM call #1 ...

The IDs live in context variables, which asyncio copies into tasks and `asyncio.to_thread`
calls, so tools running in parallel or in worker threads are tagged too.
"""

import contextvars
import logging
import logging.config
import sys

from app.config import get_settings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")

# Chatty third-party loggers: keep warnings and errors only.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "aiosqlite", "pinecone", "sentence_transformers", "filelock")


class _RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Send app logs to stdout. Safe to call more than once (e.g. under `uvicorn --reload`)."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # keep loggers created at import time
            "filters": {"request_context": {"()": _RequestContextFilter}},
            "formatters": {
                "console": {
                    "format": "%(asctime)s.%(msecs)03d %(levelname)-7s [req=%(request_id)s sid=%(session_id)s] "
                    "%(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "console",
                    "filters": ["request_context"],
                }
            },
            "root": {"level": level.upper(), "handlers": ["console"]},
            "loggers": {name: {"level": "WARNING"} for name in _QUIET_LOGGERS},
        }
    )


def preview(text: object) -> str:
    """Single-line, length-capped rendering of message or tool content for logs.

    Returns only the length when LOG_MESSAGE_CONTENT=false, so logs can't leak customer data.
    """
    settings = get_settings()
    flat = " ".join(str(text).split())
    if not settings.log_message_content:
        return f"<{len(flat)} chars>"
    limit = settings.log_preview_chars
    return repr(flat) if len(flat) <= limit else f"{flat[:limit]!r}… (+{len(flat) - limit} chars)"
