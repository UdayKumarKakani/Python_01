# ─────────────────────────────────────────────────────────────────────────────
# 9_basic_chatbot.py
#
# Basic Conversational Chatbot using LangGraph + Groq
#
# What this script does:
#   - Maintains a running conversation history in state
#   - Each user message is added to the messages list
#   - The LLM receives the FULL conversation history every turn
#     (so it remembers what was said earlier)
#   - Replies are appended back to the messages list via add_messages
#   - Runs as an interactive loop in the terminal — type 'quit' to exit
#
# Key concepts shown:
#   ✔ Annotated[list[BaseMessage], add_messages]
#       → add_messages is a reducer that APPENDS new messages instead of
#         overwriting — essential for multi-turn memory
#   ✔ Stateful graph — state is carried across multiple invocations
#   ✔ HumanMessage / AIMessage round-trip
#
# Graph flow (single node):
#   START → chat_node → END
#   (called once per user message; state persists across calls)
#
# Prerequisites:
#   - Add GROQ_API_KEY to a .env file
#   - Get a free key at: https://console.groq.com
#
# Run:  python 9_basic_chatbot.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()


# ── 1. Initialise the LLM ─────────────────────────────────────────────────────
# llama-3.3-70b-versatile is conversational, fast, and free on Groq.

llm = ChatGroq(model='llama-3.3-70b-versatile')


# ── 2. Define the State ───────────────────────────────────────────────────────
# add_messages is a special LangGraph reducer:
#   - When you return {'messages': [new_msg]}, it APPENDS to the list
#   - Without it, each return would overwrite the whole messages list
#   - This is what gives the chatbot its memory across turns

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 3. Define the Node ────────────────────────────────────────────────────────

def chat_node(state: ChatState):
    """Single node: send the full message history to the LLM, get a reply."""
    messages = state['messages']
    response = llm.invoke(messages)
    # Returning a list causes add_messages to APPEND the response
    return {'messages': [response]}


# ── 4. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START,       'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile()


# ── 5. Single-turn helper (for quick testing) ─────────────────────────────────

def ask(question: str, history: list = None) -> str:
    """Send one question and return the reply text."""
    if history is None:
        history = []
    history.append(HumanMessage(content=question))
    result   = chatbot.invoke({'messages': history})
    reply    = result['messages'][-1].content
    history.append(result['messages'][-1])   # keep conversation going
    return reply, history


# ── 6. Interactive Loop ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print('─' * 50)
    print('BASIC CHATBOT  (type "quit" to exit)')
    print('─' * 50)

    conversation_history = []

    while True:
        user_input = input('\nYou: ').strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            print('Goodbye!')
            break
        if not user_input:
            continue

        reply, conversation_history = ask(user_input, conversation_history)
        print(f'\nBot: {reply}')
