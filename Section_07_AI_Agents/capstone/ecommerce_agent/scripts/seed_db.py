"""Create the store tables and load Acme's demo products and orders. Safe to re-run.

Run from the project root:  python -m scripts.seed_db
"""

import asyncio
import json
from datetime import UTC, datetime

from app.config import get_settings
from app.db import connect, init_db
from app.db.seed_data import PRODUCTS, build_orders

_PRODUCT_COLUMNS = (
    "sku", "name", "category", "subcategory", "brand", "author", "description",
    "price_cents", "stock_qty", "rating", "review_count", "attributes", "updated_at",
)  # fmt: skip

_ORDER_COLUMNS = (
    "order_id", "customer_name", "customer_email", "status", "placed_at", "estimated_arrival",
    "carrier", "tracking_number", "shipping_method", "subtotal_cents", "shipping_cents",
    "total_cents", "updated_at",
)  # fmt: skip


def _upsert_sql(table: str, columns: tuple[str, ...], key: str) -> str:
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != key)
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT({key}) DO UPDATE SET {updates}"


async def main() -> None:
    settings = get_settings()
    await init_db(settings)
    now = datetime.now(UTC).isoformat()

    product_rows = [
        tuple(
            json.dumps(p["attributes"]) if c == "attributes" else now if c == "updated_at" else p.get(c)
            for c in _PRODUCT_COLUMNS
        )
        for p in PRODUCTS
    ]

    orders = build_orders()
    known_skus = {p["sku"] for p in PRODUCTS}
    order_rows, item_rows = [], []
    for order in orders:
        subtotal = sum(qty * unit for _, qty, unit in order["items"])
        values = {**order, "subtotal_cents": subtotal, "total_cents": subtotal + order["shipping_cents"]}
        order_rows.append(tuple(values[c] for c in _ORDER_COLUMNS))
        for sku, qty, unit in order["items"]:
            if sku not in known_skus:
                raise ValueError(f"Order {order['order_id']} references unknown SKU {sku}")
            item_rows.append((order["order_id"], sku, qty, unit))

    async with connect(settings) as conn:
        await conn.executemany(_upsert_sql("products", _PRODUCT_COLUMNS, "sku"), product_rows)
        await conn.executemany(_upsert_sql("orders", _ORDER_COLUMNS, "order_id"), order_rows)
        await conn.executemany("DELETE FROM order_items WHERE order_id = ?", [(o["order_id"],) for o in orders])
        await conn.executemany(
            "INSERT INTO order_items (order_id, sku, quantity, unit_price_cents) VALUES (?, ?, ?, ?)", item_rows
        )
        await conn.commit()

    print(f"Seeded {len(product_rows)} products and {len(order_rows)} orders into {settings.resolved_store_path}")


if __name__ == "__main__":
    asyncio.run(main())
