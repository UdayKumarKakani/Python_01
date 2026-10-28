"""SQLite connection handling and schema for the store (products, orders).

By default the store shares the checkpointer's SQLite file (`data/checkpoints.db`);
its tables don't collide with LangGraph's `checkpoints` / `writes` tables.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku            TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    category       TEXT NOT NULL CHECK (category IN ('Books', 'Electronics')),
    subcategory    TEXT,
    brand          TEXT,              -- manufacturer (electronics) or publisher (books)
    author         TEXT,              -- books only
    description    TEXT NOT NULL,
    price_cents    INTEGER NOT NULL CHECK (price_cents >= 0),
    currency       TEXT NOT NULL DEFAULT 'USD',
    stock_qty      INTEGER NOT NULL DEFAULT 0 CHECK (stock_qty >= 0),
    rating         REAL CHECK (rating BETWEEN 0 AND 5),
    review_count   INTEGER NOT NULL DEFAULT 0,
    attributes     TEXT NOT NULL DEFAULT '{}',  -- JSON object of specs
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);

CREATE TABLE IF NOT EXISTS orders (
    order_id           TEXT PRIMARY KEY,
    customer_name      TEXT NOT NULL,
    customer_email     TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (
                           status IN ('Processing', 'Shipped', 'Out for delivery', 'Delivered', 'Cancelled')
                       ),
    placed_at          TEXT NOT NULL,
    estimated_arrival  TEXT,
    carrier            TEXT,
    tracking_number    TEXT,
    shipping_method    TEXT NOT NULL,
    subtotal_cents     INTEGER NOT NULL,
    shipping_cents     INTEGER NOT NULL DEFAULT 0,
    total_cents        INTEGER NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id          TEXT NOT NULL REFERENCES orders (order_id) ON DELETE CASCADE,
    sku               TEXT NOT NULL REFERENCES products (sku),
    quantity          INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_cents  INTEGER NOT NULL,
    PRIMARY KEY (order_id, sku)
);
"""


@asynccontextmanager
async def connect(settings: Settings | None = None) -> AsyncIterator[aiosqlite.Connection]:
    settings = settings or get_settings()
    async with aiosqlite.connect(settings.resolved_store_path, timeout=settings.store_query_timeout_seconds) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        yield conn


async def init_db(settings: Settings | None = None) -> int:
    """Create tables if needed. Returns the number of products currently stored."""
    settings = settings or get_settings()
    Path(settings.resolved_store_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    async with connect(settings) as conn:
        # WAL lets the checkpointer write while tools read the same file.
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.executescript(SCHEMA)
        await conn.commit()
        async with conn.execute("SELECT COUNT(*) FROM products") as cursor:
            (count,) = await cursor.fetchone()
    logger.info("Store DB ready path=%s products=%d", settings.resolved_store_path, count)
    return count
