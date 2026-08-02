"""
Day 1 - Embeddings 101 (offline demo)
--------------------------------------
No API keys needed. Downloads ~90 MB of models on first run.

Shows:
1. Turning sentences into embedding vectors
2. Comparing two open-source models on the same sentence

Run:
    python main.py
"""

from sentence_transformers import SentenceTransformer


def demo_first_embedding() -> None:
    print("--- 1. Your first embedding ---\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [
        "The cat sat on the mat",
        "A kitten rested on the rug",
        "Bitcoin hit an all-time high",
    ]
    vectors = model.encode(sentences)
    print(f"  Shape: {vectors.shape}   (rows=sentences, cols=dims)")
    for s, v in zip(sentences, vectors):
        print(f"  {s[:45]:45s} -> first 4 nums: {v[:4].round(3).tolist()}")
    print()


def demo_model_comparison() -> None:
    print("--- 2. Same sentence, two models ---\n")
    sentence = "The Eiffel Tower is in Paris."
    for name in ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
        m = SentenceTransformer(name)
        v = m.encode(sentence)
        print(f"  {name:35s} -> {len(v)} dims, first 3: {v[:3].round(3).tolist()}")
    print()


if __name__ == "__main__":
    demo_first_embedding()
    demo_model_comparison()
