"""
Section 10 - Day 2 helpers
---------------------------
1. Random interview question prompter (set a 3-min timer, answer aloud)
2. Resume checklist

Run:
    python main.py                # both demos
    python main.py --q            # just a random question
    python main.py --checklist    # just the resume checklist
"""

import random
import sys


QUESTIONS = [
    "Explain the difference between RAG and fine-tuning.",
    "What does an embedding vector represent?",
    "Why use a cross-encoder reranker after bi-encoder retrieval?",
    "Explain LoRA at a high level.",
    "Why do LLMs hallucinate and how do you mitigate?",
    "Design a chatbot over 100k internal docs.",
    "Your chatbot's answer quality just dropped. How do you debug?",
    "How do you defend against prompt injection?",
    "Your LLM bill is $10k/mo. How do you cut it?",
    "Explain streaming from LLM to browser.",
    "When would you pick OpenAI over open-source Llama?",
    "Design an autonomous agent for a business workflow.",
    "Tell me about an AI project you're proud of.",
    "Tell me about a wrong technical decision you made.",
    "Why AI engineering vs backend or data science?",
]


CHECKLIST = [
    "One page (US letter or A4)",
    "Every project bullet has at least one number",
    "Tech stack matches real 2026 JDs you plan to apply to",
    "GitHub URL clickable and points to a profile with 6 pinned repos",
    "LinkedIn URL clickable, profile has your headline",
    "Portfolio URL (or top pinned repo) has a live demo link",
    "No paragraphs - only bullet points",
    "No spelling errors (spellchecker run)",
    "Fits printed at 100% - no tiny 8pt fonts",
    "PDF export (not .docx) - preserves formatting",
]


def random_question() -> None:
    print(f"\nYou have 3 minutes. Start.\n\nQ: {random.choice(QUESTIONS)}\n")


def resume_checklist() -> None:
    print("\nResume checklist:")
    for i, c in enumerate(CHECKLIST, 1):
        print(f"  [ ] {i:2}. {c}")


if __name__ == "__main__":
    flag = sys.argv[1] if len(sys.argv) > 1 else ""
    if flag == "--q":
        random_question()
    elif flag == "--checklist":
        resume_checklist()
    else:
        random_question()
        resume_checklist()
