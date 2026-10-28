"""Read queries for products and orders. Every call is time-boxed and raises StoreError on failure."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from app.config import get_settings
from app.db.database import connect
from app.exceptions import StoreError

logger = logging.getLogger(__name__)

T = TypeVar("T")
Category = Literal["Books", "Electronics"]

_STOPWORDS = {
    "a", "an", "and", "any", "about", "are", "do", "does", "for", "have", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "show", "tell", "the", "to", "what", "with", "you", "your",
}  # fmt: skip
_MAX_TERMS = 8


class Product(BaseModel):
    sku: str
    name: str
    category: Category
    subcategory: str | None
    brand: str | None
    author: str | None
    description: str
    price_cents: int
    currency: str
    stock_qty: int
    rating: float | None
    review_count: int
    attributes: dict[str, Any]


class OrderItem(BaseModel):
    sku: str
    name: str
    quantity: int
    unit_price_cents: int


class Order(BaseModel):
    order_id: str
    customer_email: str
    status: str
    placed_at: str
    estimated_arrival: str | None
    carrier: str | None
    tracking_number: str | None
    shipping_method: str
    subtotal_cents: int
    shipping_cents: int
    total_cents: int
    items: list[OrderItem]


async def _timed(coro: Awaitable[T], what: str) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=get_settings().store_query_timeout_seconds)
    except TimeoutError as exc:
        raise StoreError(f"{what} timed out") from exc
    except StoreError:
        raise
    except Exception as exc:
        raise StoreError(f"{what} failed: {exc}") from exc


def _row_to_product(row: Any) -> Product:
    data = dict(row)
    data["attributes"] = json.loads(data["attributes"] or "{}")
    data.pop("updated_at", None)
    return Product(**data)


def _search_terms(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-+.']*", query.lower())
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS][:_MAX_TERMS]


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def _search_products(
    query: str, category: Category | None, max_price_cents: int | None, limit: int
) -> list[Product]:
    terms = _search_terms(query)
    # Relevance: SKU exact match >> name >> author/brand >> description/specs.
    score_sql = ["(CASE WHEN upper(sku) = ? THEN 100 ELSE 0 END)"]
    params: list[Any] = [query.strip().upper()]
    for term in terms:
        pattern = _like(term)
        score_sql.append(
            "(CASE WHEN lower(name) LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END)"
            " + (CASE WHEN lower(coalesce(author, '') || ' ' || coalesce(brand, '')) LIKE ? ESCAPE '\\' THEN 2 ELSE 0 END)"
            " + (CASE WHEN lower(description || ' ' || coalesce(subcategory, '') || ' ' || attributes)"
            " LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"
        )
        params += [pattern, pattern, pattern]

    filters, filter_params = ["score > 0"], []
    if category:
        filters.append("category = ?")
        filter_params.append(category)
    if max_price_cents is not None:
        filters.append("price_cents <= ?")
        filter_params.append(max_price_cents)

    sql = (
        f"SELECT * FROM (SELECT *, {' + '.join(score_sql)} AS score FROM products) "
        f"WHERE {' AND '.join(filters)} ORDER BY score DESC, rating DESC LIMIT ?"
    )
    async with connect() as conn, conn.execute(sql, [*params, *filter_params, limit]) as cursor:
        rows = await cursor.fetchall()
    return [_row_to_product({k: row[k] for k in row.keys() if k != "score"}) for row in rows]


async def search_products(
    query: str, category: Category | None = None, max_price_cents: int | None = None, limit: int = 3
) -> list[Product]:
    return await _timed(_search_products(query, category, max_price_cents, limit), "Product search")


async def _get_order(order_id: str) -> Order | None:
    async with connect() as conn:
        async with conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)) as cursor:
            order_row = await cursor.fetchone()
        if order_row is None:
            return None
        async with conn.execute(
            """
            SELECT oi.sku, p.name, oi.quantity, oi.unit_price_cents
            FROM order_items oi JOIN products p ON p.sku = oi.sku
            WHERE oi.order_id = ? ORDER BY p.name
            """,
            (order_id,),
        ) as cursor:
            item_rows = await cursor.fetchall()

    data = dict(order_row)
    for unused in ("customer_name", "updated_at"):
        data.pop(unused, None)
    return Order(**data, items=[OrderItem(**dict(r)) for r in item_rows])


async def get_order(order_id: str) -> Order | None:
    return await _timed(_get_order(order_id), "Order lookup")
