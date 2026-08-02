"""
Day 3 - Agent memory (short-term + long-term)
----------------------------------------------
Requires TOGETHER_API_KEY.

Shows:
1. Sliding-window short-term memory
2. Chroma-backed long-term memory with per-user isolation
3. A chatbot that recalls facts from earlier sessions

Run:
    python main.py
"""

import os
import time

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"

llm = Together()
embed = SentenceTransformer("all-MiniLM-L6-v2")
long_mem = chromadb.Client().create_collection("long_term")


def remember(user_id: str, text: str) -> None:
    long_mem.add(
        documents=[text],
        embeddings=embed.encode([text]).tolist(),
        metadatas=[{"user_id": user_id, "ts": time.time()}],
        ids=[f"{user_id}_{time.time_ns()}"],
    )


def recall(user_id: str, query: str, top_k: int = 3) -> list[str]:
    r = long_mem.query(
        query_embeddings=embed.encode([query]).tolist(),
        n_results=top_k,
        where={"user_id": user_id},
    )
    return r["documents"][0] if r["documents"] else []


class ChatMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]

    def as_prompt(self, system: str) -> list[dict]:
        return [{"role": "system", "content": system}] + self.messages


class Chatbot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short = ChatMemory()

    def talk(self, message: str) -> str:
        memories = recall(self.user_id, message, top_k=3)
        mem_block = "\n".join(f"- {m}" for m in memories) or "(none)"
        system = (
            "You are a helpful assistant. Things you remember about the user:\n"
            f"{mem_block}\n\nUse them if relevant."
        )
        self.short.add("user", message)
        r = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.short.as_prompt(system),
            temperature=0.3,
        )
        answer = r.choices[0].message.content.strip()
        self.short.add("assistant", answer)
        return answer


if __name__ == "__main__":
    for fact in [
        "Uday is a Python developer based in Bangalore.",
        "Uday's favorite framework is FastAPI.",
        "Uday is building an AI course.",
    ]:
        remember("uday", fact)

    bot = Chatbot("uday")
    for msg in [
        "Should I try building a frontend for my course?",
        "What do you know about my job?",
    ]:
        print(f"\nYou: {msg}\nBot: {bot.talk(msg)}")
