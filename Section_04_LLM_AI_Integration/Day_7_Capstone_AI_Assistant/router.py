"""Multi-model routing.

`route(prompt, privacy, force_model)` picks (provider, model) + a
human-readable reason (logged for audit).

Together AI is the default cheap tier. Claude gets code + long-context.
OpenAI is available as an alternative or fallback.
"""

import re
from dataclasses import dataclass
from typing import Optional

CODE_KEYWORDS = re.compile(r"\b(code|python|sql|refactor|debug|algorithm|regex)\b", re.I)

TOGETHER_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
TOGETHER_BIG     = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


@dataclass
class RouteDecision:
    provider: str
    model: str
    reason: str


def _split(force_model: str) -> RouteDecision:
    """`force_model` comes as 'provider/model_id'. Model may contain '/'."""
    if "/" not in force_model:
        raise ValueError("force_model must be 'provider/model'")
    provider, model = force_model.split("/", 1)
    return RouteDecision(provider, model, f"forced by client: {force_model}")


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
