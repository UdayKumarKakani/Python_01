"""
Day 4 - Evaluation helpers (offline for the LLM-as-judge part; GPU for accuracy)
"""

import json
import os

from dotenv import load_dotenv
from together import Together

load_dotenv()


def llm_judge(question: str, expected: str, got: str) -> dict:
    llm = Together()
    prompt = (
        "You are a strict evaluator. Given a question, an expected answer, "
        "and a model answer, return ONLY a JSON object of the form "
        '{"score": 0-5, "reason": "..."}. \n\n'
        f"Question: {question}\nExpected: {expected}\nModel: {got}"
    )
    r = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        return json.loads(r.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return {"score": -1, "reason": "unparseable"}


def exact_match(expected: str, got: str) -> bool:
    return expected.strip().lower() == got.strip().lower()


if __name__ == "__main__":
    if not os.getenv("TOGETHER_API_KEY"):
        print("Set TOGETHER_API_KEY in .env to run the judge demo.")
    else:
        result = llm_judge(
            "Classify: 'my card was charged twice'",
            "billing", "billing issue",
        )
        print(result)
