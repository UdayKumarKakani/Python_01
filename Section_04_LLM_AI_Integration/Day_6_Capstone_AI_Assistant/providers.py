"""Thin async wrappers over Together AI, OpenAI, and Anthropic.

Together AI is exposed via the OpenAI SDK's `AsyncOpenAI` with the
Together base URL — its API is OpenAI-compatible so we get the same
`.chat.completions.create(...)` shape.
"""

import os
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from costs import count_tokens

_together: Optional[AsyncOpenAI] = None
_openai:   Optional[AsyncOpenAI] = None
_anthropic: Optional[AsyncAnthropic] = None


def _tg() -> AsyncOpenAI:
    global _together
    if _together is None:
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            raise RuntimeError("TOGETHER_API_KEY is not set")
        _together = AsyncOpenAI(api_key=key, base_url="https://api.together.xyz/v1")
    return _together


def _oa() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI()  # picks up OPENAI_API_KEY
    return _openai


def _an() -> AsyncAnthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = AsyncAnthropic()  # picks up ANTHROPIC_API_KEY
    return _anthropic


# ---------- Together AI ----------
async def together_chat(model: str, prompt: str) -> tuple[str, int, int]:
    r = await _tg().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    text = r.choices[0].message.content or ""
    usage = r.usage
    in_t = getattr(usage, "prompt_tokens", None) or count_tokens(prompt)
    out_t = getattr(usage, "completion_tokens", None) or count_tokens(text)
    return text, in_t, out_t


async def together_stream(model: str, prompt: str) -> AsyncIterator[str]:
    stream = await _tg().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------- OpenAI ----------
async def openai_chat(model: str, prompt: str) -> tuple[str, int, int]:
    r = await _oa().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = r.choices[0].message.content or ""
    return text, r.usage.prompt_tokens, r.usage.completion_tokens


async def openai_stream(model: str, prompt: str) -> AsyncIterator[str]:
    stream = await _oa().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------- Anthropic ----------
async def anthropic_chat(model: str, prompt: str) -> tuple[str, int, int]:
    r = await _an().messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = r.content[0].text if r.content else ""
    return text, r.usage.input_tokens, r.usage.output_tokens


async def anthropic_stream(model: str, prompt: str) -> AsyncIterator[str]:
    async with _an().messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ---------- Dispatch ----------
async def call(provider: str, model: str, prompt: str) -> tuple[str, int, int]:
    if provider == "together":
        return await together_chat(model, prompt)
    if provider == "openai":
        return await openai_chat(model, prompt)
    if provider == "anthropic":
        return await anthropic_chat(model, prompt)
    raise ValueError(f"unknown provider: {provider}")


async def stream(provider: str, model: str, prompt: str) -> AsyncIterator[str]:
    if provider == "together":
        async for d in together_stream(model, prompt):
            yield d
    elif provider == "openai":
        async for d in openai_stream(model, prompt):
            yield d
    elif provider == "anthropic":
        async for d in anthropic_stream(model, prompt):
            yield d
    else:
        raise ValueError(f"unknown provider: {provider}")
