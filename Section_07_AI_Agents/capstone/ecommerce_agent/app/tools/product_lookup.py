"""Product catalog tool backed by the SQLite store."""

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.db.repository import Category, Product, search_products
from app.exceptions import StoreError
from app.logging_config import preview

logger = logging.getLogger(__name__)


class ProductLookupInput(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Product name, SKU, author, brand or keywords, e.g. 'Paperlight e-reader' or 'python book'.",
    )
    category: Category | None = Field(None, description="Restrict to 'Books' or 'Electronics' when known.")
    max_price: float | None = Field(None, gt=0, description="Maximum price in USD, if the customer gave a budget.")
    limit: int = Field(3, ge=1, le=5, description="How many products to return.")


def _money(cents: int, currency: str = "USD") -> str:
    return f"${cents / 100:,.2f}" if currency == "USD" else f"{cents / 100:,.2f} {currency}"


def _value(value: Any) -> str:
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def format_product(product: Product) -> str:
    availability = f"In stock ({product.stock_qty} available)" if product.stock_qty > 0 else "Out of stock"
    lines = [
        f"{product.name} (SKU {product.sku})",
        f"- category: {product.category}" + (f" > {product.subcategory}" if product.subcategory else ""),
    ]
    if product.author:
        lines.append(f"- author: {product.author}")
    if product.brand:
        lines.append(f"- {'publisher' if product.category == 'Books' else 'brand'}: {product.brand}")
    lines += [
        f"- price: {_money(product.price_cents, product.currency)}",
        f"- availability: {availability}",
        f"- description: {product.description}",
    ]
    if product.rating is not None:
        lines.append(f"- rating: {product.rating}/5 ({product.review_count:,} reviews)")
    lines += [f"- {key.replace('_', ' ')}: {_value(val)}" for key, val in product.attributes.items()]
    return "\n".join(lines)


@tool("product_lookup", args_schema=ProductLookupInput)
async def product_lookup(
    query: str, category: Category | None = None, max_price: float | None = None, limit: int = 3
) -> str:
    """Look up Acme products (books and electronics) for exact descriptions, prices, stock,
    ratings and specifications such as dimensions, weight, battery life, page count or format.
    Use this for ANY question about a specific product or for product recommendations."""
    max_price_cents = round(max_price * 100) if max_price is not None else None
    try:
        products = await search_products(query, category, max_price_cents, limit)
    except StoreError as exc:
        logger.error("Product lookup failed query=%r: %s", query, exc)
        return (
            "STORE_UNAVAILABLE: The product database could not be reached. "
            "Do not answer from memory. Apologize and offer to connect the customer with a team member."
        )

    logger.info(
        "[product_lookup] query=%s category=%s max_price=%s -> %s",
        preview(query),
        category or "any",
        max_price if max_price is not None else "none",
        ", ".join(f"{p.sku} ({p.name})" for p in products) or "no match",
    )
    if not products:
        return (
            "PRODUCT_NOT_FOUND: No Acme products matched this search. "
            "Ask the customer for the exact product name or SKU, or suggest a broader search."
        )
    return "\n\n".join(format_product(p) for p in products)
