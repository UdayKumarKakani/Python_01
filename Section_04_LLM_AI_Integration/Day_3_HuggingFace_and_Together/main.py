"""
Day 3 - Hugging Face & Together AI
----------------------------------
Runs three tiny demos:
1. Sentiment analysis with the `transformers` pipeline (local, no key).
2. Text embeddings with sentence-transformers (local, no key).
3. A single Together AI chat call.

The first two download model weights on first run (~250 MB total).

Setup:
    Optional: add TOGETHER_API_KEY to .env to enable demo 3.

Run:
    python main.py
"""

import os

from dotenv import load_dotenv

load_dotenv()


def demo_sentiment() -> None:
    print("--- 1. Hugging Face sentiment analysis (local) ---")
    from transformers import pipeline
    clf = pipeline("sentiment-analysis")

    for text in [
        "I love learning about AI!",
        "This coffee is terrible.",
        "Meh, it's fine.",
    ]:
        r = clf(text)[0]
        print(f"  {r['label']:8s}  {r['score']:.2%}  ::  {text}")
    print()


def demo_embeddings() -> None:
    print("--- 2. Sentence embeddings (local) ---")
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [
        "I love cats.",
        "Dogs are wonderful.",
        "My favorite programming language is Python.",
    ]
    vecs = model.encode(sentences)

    def cos(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    print(f"  cats vs dogs   : {cos(vecs[0], vecs[1]):.3f}")
    print(f"  cats vs python : {cos(vecs[0], vecs[2]):.3f}")
    print()


def demo_together() -> None:
    print("--- 3. Together AI call ---")
    if not os.getenv("TOGETHER_API_KEY"):
        print("  (TOGETHER_API_KEY not set — skipping)\n")
        return

    from together import Together
    client = Together()
    r = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Say hello in 3 languages."}],
    )
    print(" ", r.choices[0].message.content, "\n")


if __name__ == "__main__":
    demo_sentiment()
    demo_embeddings()
    demo_together()
