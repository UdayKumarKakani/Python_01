"""
Day 1 - Fine-tuning dataset prep (offline)
-------------------------------------------
No API keys, no GPU needed.

Shows:
1. Building a small chat-format dataset
2. Saving as JSONL
3. Train/eval split

Run:
    python main.py
"""

import json
import random


SYS = ("You are a support triage bot. Classify each ticket into exactly one of: "
       "billing, technical, feature-request, other. Respond with only the label.")


TICKETS = [
    ("My card was charged twice for last month.",              "billing"),
    ("The dashboard is not loading in Chrome.",                "technical"),
    ("It would be great if you could add dark mode.",          "feature-request"),
    ("Why is my monthly bill higher this month?",              "billing"),
    ("The API returns 500 errors when I POST large files.",    "technical"),
    ("Can you tell me what time your office is open?",         "other"),
]


def to_chat(user_text: str, label: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYS},
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": label},
    ]}


def save_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    data = [to_chat(u, l) for u, l in TICKETS]
    save_jsonl("triage.jsonl", data)

    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    save_jsonl("triage_train.jsonl", data[:split])
    save_jsonl("triage_eval.jsonl",  data[split:])
    print(f"Saved: triage_train.jsonl ({split}), triage_eval.jsonl ({len(data)-split})")
