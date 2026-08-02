"""
Section 10 - Day 1 helpers
---------------------------
Two utilities you should reach for while practicing system design:

1. Back-of-envelope cost estimator
2. In-memory semantic cache demo (uses sentence-transformers)

Run:
    python main.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Back-of-envelope
# ------------------------------------------------------------
def estimate(dau: int, queries_per_user: int, tokens_per_query: int,
             price_per_million: float, chunks: int = 0, dim: int = 384) -> None:
    qpd = dau * queries_per_user
    qps_avg = qpd / 86_400
    qps_peak = qps_avg * 8
    tpd = qpd * tokens_per_query
    cost_day = (tpd / 1_000_000) * price_per_million
    print(f"DAU               : {dau:>10,}")
    print(f"Queries/day       : {qpd:>10,}")
    print(f"QPS avg / peak    : {qps_avg:>8.2f} / {qps_peak:.2f}")
    print(f"Tokens/day        : {tpd:>10,}")
    print(f"Cost/day / month  : ${cost_day:>8.2f} / ${cost_day*30:.2f}")
    if chunks:
        gb = chunks * dim * 4 / 1e9
        print(f"Vector storage    : {chunks:>10,} chunks x {dim} dim = {gb:.2f} GB")


# ------------------------------------------------------------
# 2. Semantic cache
# ------------------------------------------------------------
class SemanticCache:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.vecs: list[np.ndarray] = []
        self.answers: list[str] = []

    def _embed(self, q: str) -> np.ndarray:
        v = self.model.encode(q).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def get(self, q: str) -> str | None:
        if not self.vecs:
            return None
        sims = np.stack(self.vecs) @ self._embed(q)
        i = int(sims.argmax())
        return self.answers[i] if sims[i] >= self.threshold else None

    def set(self, q: str, ans: str) -> None:
        self.vecs.append(self._embed(q))
        self.answers.append(ans)


if __name__ == "__main__":
    print("--- Estimator: 10k DAU RAG chatbot, openai/gpt-oss-20b ---")
    # gpt-oss-20b on Together: ~$0.05/M in + $0.20/M out; blended ~$0.10/M for a ~70/30 in-out mix
    estimate(10_000, 5, 700, 0.10, chunks=500_000)

    print("\n--- Semantic cache demo ---")
    c = SemanticCache(threshold=0.7)
    c.set("What is the refund policy?", "Refunds within 30 days, minus fees.")
    print("miss:", c.get("How do dogs bark?"))
    print("hit :", c.get("How do I get a refund?"))
