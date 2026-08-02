"""Multi-model routing.

`route(prompt, privacy, force_model)` picks (provider, model) + a
human-readable reason (logged for audit).

Together AI is the default cheap tier. Claude gets code + long-context.
OpenAI is available as an alternative or fallback.
"""

import re
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, status

CODE_KEYWORDS = re.compile(r"\b(code|python|sql|refactor|debug|algorithm|regex)\b", re.I)

# Together AI - both are serverless in 2026
TOGETHER_DEFAULT = "openai/gpt-oss-20b"     # small + cheap
TOGETHER_BIG     = "openai/gpt-oss-120b"    # bigger sibling

VALID_PROVIDERS = {"together", "openai", "anthropic"}

# If a Together-hosted model id is passed without the "together/" prefix,
# we auto-prefix. These are the vendor prefixes Together hosts.
_TOGETHER_MODEL_PREFIXES = ("openai/", "meta-llama/", "mistralai/", "Qwen/", "google/")


@dataclass
class RouteDecision:
    provider: str
    model: str
    reason: str


def _split(force_model: str) -> RouteDecision:
    """Parse a `force_model` string.

    Accepted shapes:
      - "together/openai/gpt-oss-20b"        (canonical: provider + slash + model_id)
      - "openai/gpt-4o-mini"                 (canonical for OpenAI)
      - "anthropic/claude-3-5-sonnet-latest" (canonical for Anthropic)
      - "openai/gpt-oss-20b"                 (Together-hosted; auto-prefixed with "together/")

    Raises HTTPException(400) with an example on bad input so Swagger / clients
    see a helpful message instead of a 500.
    """
    example = (
        'force_model must look like "<provider>/<model_id>". '
        'Examples: "together/openai/gpt-oss-20b", "openai/gpt-4o-mini", '
        '"anthropic/claude-3-5-sonnet-latest".'
    )

    if not force_model or "/" not in force_model:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, example)

    provider, model = force_model.split("/", 1)

    # Auto-prefix a Together-hosted model that came in without "together/"
    if provider not in VALID_PROVIDERS and force_model.startswith(_TOGETHER_MODEL_PREFIXES):
        provider, model = "together", force_model

    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f'Unknown provider "{provider}". Must be one of: '
            f'{sorted(VALID_PROVIDERS)}. {example}',
        )

    return RouteDecision(provider, model, f"forced by client: {provider}/{model}")


def route(prompt: str, privacy: bool = False, force_model: Optional[str] = None) -> RouteDecision:
    if force_model:
        return _split(force_model)
    if privacy:
        return RouteDecision("together", TOGETHER_DEFAULT,
                             "privacy=true: open-source model on Together AI")
    if CODE_KEYWORDS.search(prompt):
        return RouteDecision("anthropic", "claude-3-5-sonnet-latest",
                             "code/reasoning keywords")
    if len(prompt) > 800:
        return RouteDecision("anthropic", "claude-3-5-sonnet-latest",
                             "long prompt -> long-context model")
    return RouteDecision("together", TOGETHER_DEFAULT,
                         "short + simple: cheapest default")


# Fallback order per provider — main.py cascades through these on failure.
FALLBACK = {
    "together":  [("openai", "gpt-4o-mini")],
    "openai":    [("together", TOGETHER_DEFAULT), ("anthropic", "claude-3-5-haiku-latest")],
    "anthropic": [("together", TOGETHER_BIG), ("openai", "gpt-4o-mini")],
}
