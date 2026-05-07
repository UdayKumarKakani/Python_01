# ─────────────────────────────────────────────────────────────────────────────
# 2_simple_llm_workflow.py
#
# Simple LLM Q&A Workflow using LangGraph + Groq
#
# What this script does:
#   - Takes a question as input via the state
#   - Sends it to the Groq LLM (llama-3.3-70b-versatile)
#   - Returns and prints the LLM's answer
#
# This is the simplest LangGraph + LLM pattern:
#   START → llm_qa node → END
#
# Prerequisites:
#   - Add your Groq API key to a .env file:  GROQ_API_KEY=your_key_here
#   - Get a free key at: https://console.groq.com
#
# Run:  python 2_simple_llm_workflow.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

# Load GROQ_API_KEY from .env file
load_dotenv()


# ── 1. Initialise the LLM ─────────────────────────────────────────────────────
# llama-3.3-70b-versatile is a powerful free model on Groq.
# You can swap to "llama3-8b-8192" for a faster, lighter response.

model = ChatGroq(model='llama-3.3-70b-versatile')


# ── 2. Define the State ───────────────────────────────────────────────────────

class LLMState(TypedDict):
    question: str   # input from the user
    answer: str     # filled in by the llm_qa node


# ── 3. Define the Node ────────────────────────────────────────────────────────

def llm_qa(state: LLMState) -> LLMState:
    """Single node: takes the question, sends it to Groq, stores the answer."""
    question = state['question']

    prompt = f'Answer the following question: {question}'
    answer = model.invoke(prompt).content

    state['answer'] = answer
    return state


# ── 4. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(LLMState)

graph.add_node('llm_qa', llm_qa)

graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

workflow = graph.compile()


# ── 5. Run the Workflow ───────────────────────────────────────────────────────

if __name__ == '__main__':
    initial_state = {
        'question': 'How far is the Moon from the Earth?'
    }

    final_state = workflow.invoke(initial_state)

    print('─' * 40)
    print('SIMPLE LLM Q&A RESULT')
    print('─' * 40)
    print(f"Question : {final_state['question']}")
    print(f"Answer   : {final_state['answer']}")
    print('─' * 40)
