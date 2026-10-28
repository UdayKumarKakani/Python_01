"""Order status tool backed by the SQLite store."""

import json
import logging
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.db.repository import get_order
from app.exceptions import StoreError

logger = logging.getLogger(__name__)

ORDER_NOT_FOUND_PREFIX = "ORDER_NOT_FOUND"
STORE_UNAVAILABLE_PREFIX = "STORE_UNAVAILABLE"


class TrackOrderInput(BaseModel):
    order_id: str = Field(
        ...,
        min_length=3,
        max_length=40,
        description="The customer's order ID, e.g. 'ORD-10021'.",
    )


def normalize_order_id(raw: str) -> str:
    """Accept 'ord 10021', 'ord_10021', '#10021', '10021' -> 'ORD-10021'."""
    cleaned = re.sub(r"[\s_]+", "-", raw.strip().upper()).strip("#")
    if cleaned.isdigit():
        cleaned = f"ORD-{cleaned}"
    return re.sub(r"^ORD-?(\d+)$", r"ORD-\1", cleaned)


def is_successful_lookup(tool_output: str) -> bool:
    return not tool_output.startswith((ORDER_NOT_FOUND_PREFIX, STORE_UNAVAILABLE_PREFIX))


@tool("track_order", args_schema=TrackOrderInput)
async def track_order(order_id: str) -> str:
    """Look up an Acme order by its order ID: status, items, totals, carrier,
    tracking number and estimated arrival date."""
    normalized = normalize_order_id(order_id)
    try:
        order = await get_order(normalized)
    except StoreError as exc:
        logger.error("Order lookup failed order_id=%s: %s", normalized, exc)
        return (
            f"{STORE_UNAVAILABLE_PREFIX}: The order database could not be reached. "
            "Apologize and offer to connect the customer with a team member."
        )

    if order is None:
        logger.info("[track_order] %s not found", normalized)
        return (
            f"{ORDER_NOT_FOUND_PREFIX}: No order exists with ID '{normalized}'. "
            "Ask the customer to double-check the ID from their confirmation email."
        )

    logger.info("[track_order] %s found: status=%s, %d item(s)", normalized, order.status, len(order.items))
    # Customer PII (name, email) stays out of the model's context.
    return json.dumps(
        {
            "order_id": order.order_id,
            "status": order.status,
            "placed_at": order.placed_at,
            "estimated_arrival": order.estimated_arrival,
            "shipping_method": order.shipping_method,
            "carrier": order.carrier,
            "tracking_number": order.tracking_number,
            "items": [
                {"name": i.name, "sku": i.sku, "quantity": i.quantity, "unit_price": f"${i.unit_price_cents / 100:,.2f}"}
                for i in order.items
            ],
            "subtotal": f"${order.subtotal_cents / 100:,.2f}",
            "shipping": f"${order.shipping_cents / 100:,.2f}",
            "total": f"${order.total_cents / 100:,.2f}",
        }
    )
