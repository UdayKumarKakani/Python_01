"""Token counting + cost estimation."""

import tiktoken

# $/million tokens (input, output). Rough early-2026 numbers.
PRICING = {
    ("together", "openai/gpt-oss-20b"):  (0.18, 0.18),
    ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"): (0.88, 0.88),
    ("together", "mistralai/Mistral-7B-Instruct-v0.3"):           (0.20, 0.20),
    ("openai", "gpt-4o"):                (2.50, 10.00),
    ("openai", "gpt-4o-mini"):           (0.15,  0.60),
    ("anthropic", "claude-3-5-sonnet-latest"): (3.00, 15.00),
    ("anthropic", "claude-3-5-haiku-latest"):  (0.80,  4.00),
}


def _encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(_encoder(model).encode(text))


def cost_usd(provider: str, model: str, in_tokens: int, out_tokens: int) -> float:
    price = PRICING.get((provider, model), (0.0, 0.0))
    return (in_tokens * price[0] + out_tokens * price[1]) / 1_000_000
