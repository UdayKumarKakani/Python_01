"""
Day 5 - Structured Outputs & Tool Calling
-----------------------------------------
Two demos:

1. Extract a JSON dict {name, age, email} from a paragraph using
   Together AI JSON mode.
2. Full tool-calling round-trip with OpenAI: model asks for
   get_weather(city='Paris'), we run it, model writes final answer.

Setup:
    Add TOGETHER_API_KEY (required) and OPENAI_API_KEY (for demo 2) to .env.

Run:
    python main.py
"""

import json
import os

from dotenv import load_dotenv
from together import Together

load_dotenv()

if not os.getenv("TOGETHER_API_KEY"):
    raise SystemExit("Missing TOGETHER_API_KEY. Add it to .env and re-run.")

tg = Together()
TG_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


# ---------------------------------------------------------------------------
# 1. Structured JSON via Together AI JSON mode
# ---------------------------------------------------------------------------
def demo_structured() -> None:
    print("--- 1. Structured JSON (Together AI) ---")
    text = "Rohan Mehta is a 34-year-old software engineer. Reach him at rohan@example.com."
    prompt = (
        "Extract as JSON with keys name, age, email. "
        "Reply with ONLY the JSON object.\n\nText: " + text
    )
    r = tg.chat.completions.create(
        model=TG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    data = json.loads(r.choices[0].message.content)
    print(f"  {data}\n")


# ---------------------------------------------------------------------------
# 2. Tool calling round-trip on OpenAI
# ---------------------------------------------------------------------------
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Fake weather function. In real life this would call an API."""
    fake = {
        "Paris":     {"temp": 21, "condition": "sunny"},
        "London":    {"temp": 15, "condition": "cloudy"},
        "Bangalore": {"temp": 27, "condition": "humid"},
    }
    data = fake.get(city, {"temp": 20, "condition": "unknown"})
    return {"city": city, "unit": unit, **data}


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def demo_tool_call() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("--- 2. Tool calling (OpenAI) — skipped (no OPENAI_API_KEY) ---\n")
        return
    from openai import OpenAI
    oa = OpenAI()
    model = "gpt-4o-mini"

    print("--- 2. Tool calling round-trip (OpenAI) ---")
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user",   "content": "What's the weather like in Paris right now?"},
    ]

    r1 = oa.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    msg = r1.choices[0].message
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    print(f"  Model asked for : {call.function.name}({args})")

    result = get_weather(**args)
    print(f"  We returned     : {result}")

    messages.append(msg)
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    })
    r2 = oa.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    print(f"  Final answer    : {r2.choices[0].message.content}\n")


if __name__ == "__main__":
    demo_structured()
    demo_tool_call()
