"""
Day 5 - Reliability, HITL, cost governance
-------------------------------------------
No API keys required for the demos below.

Shows:
1. Retry helper with exponential backoff
2. Budget class enforcing max_steps / max_tokens / max_usd
3. Human-in-the-loop wrapper for dangerous tools

Run:
    python main.py
"""

import functools
import random
import time

import tiktoken


enc = tiktoken.encoding_for_model("gpt-4o-mini")
PRICE_PER_MTOK = 0.10   # blended $/1M for openai/gpt-oss-20b (~$0.05 in + $0.20 out, ~70/30 mix)


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_steps: int = 8, max_tokens: int = 20_000, max_usd: float = 0.10):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.steps = 0
        self.tokens = 0

    def charge_step(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded("max_steps")

    def charge_tokens(self, text: str) -> None:
        self.tokens += len(enc.encode(text))
        if self.tokens > self.max_tokens:
            raise BudgetExceeded("max_tokens")
        if (self.tokens / 1_000_000) * PRICE_PER_MTOK > self.max_usd:
            raise BudgetExceeded("max_usd")


def with_retries(fn, max_attempts: int = 3, base_delay: float = 0.5):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_err = None
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                time.sleep(base_delay * (2 ** attempt))
        return f"tool failed after {max_attempts} attempts: {last_err}"
    return wrapper


DANGEROUS = {"send_email", "delete_row", "charge_card"}


def run_tool_with_approval(name: str, args: dict, tools: dict) -> str:
    if name in DANGEROUS:
        prompt = f"HITL: agent wants to call {name}({args}). Approve? [y/N] "
        try:
            ok = input(prompt).strip().lower() == "y"
        except EOFError:
            ok = False
        if not ok:
            return "denied by human"
    return tools[name](**args)


@with_retries
def flaky(x: str) -> str:
    if random.random() < 0.7:
        raise RuntimeError("network glitch")
    return f"processed: {x}"


if __name__ == "__main__":
    print("--- 1. Retries ---")
    for _ in range(3):
        print(" ", flaky("hello"))

    print("\n--- 2. Budget ---")
    b = Budget(max_steps=3, max_tokens=50, max_usd=1.0)
    try:
        for i in range(10):
            b.charge_step()
            b.charge_tokens("some prompt text that adds up")
            print(f"  step {i} ok  tokens={b.tokens}")
    except BudgetExceeded as e:
        print(f"  HALT: {e}")

    print("\n--- 3. HITL (skipped in main - see class notebook) ---")
