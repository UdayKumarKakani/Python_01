"""
Day 4 - Langfuse tracing demo
------------------------------
Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, TOGETHER_API_KEY.
"""

import os

from dotenv import load_dotenv
from langfuse import Langfuse
from together import Together

load_dotenv()
assert os.getenv("LANGFUSE_SECRET_KEY"), "Set Langfuse keys in .env"
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"

lf = Langfuse()
llm = Together()


@lf.observe(name="retrieve")
def retrieve(question: str) -> list[str]:
    # placeholder - swap in real Chroma call
    return [f"pretend chunk about '{question}'"]


@lf.observe(as_type="generation", name="generate")
def generate(question: str, chunks: list[str]) -> str:
    context = "\n".join(chunks)
    resp = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": f"Context: {context}\nQ: {question}"}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


@lf.observe(name="rag")
def rag(question: str) -> str:
    return generate(question, retrieve(question))


if __name__ == "__main__":
    print(rag("Who founded AcmeCloud?"))
    print("Open https://cloud.langfuse.com to see the trace.")
