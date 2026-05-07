# ─────────────────────────────────────────────────────────────────────────────
# 3_prompt_chaining.py
#
# Prompt Chaining Workflow — Blog Writer + Rater using LangGraph + Groq
#
# What this script does:
#   - Takes a blog TITLE as input
#   - Node 1 (create_outline) : asks the LLM to generate a detailed outline
#   - Node 2 (create_blog)    : asks the LLM to write the full blog using
#                               the outline from Node 1
#   - Node 3 (rate_content)   : asks the LLM to compare the blog against
#                               the outline and rate it
#   - Prints outline, blog content, and the rating
#
# This demonstrates PROMPT CHAINING — the output of one LLM call feeds
# directly into the next as part of the shared state.
#
# Graph flow:  START → create_outline → create_blog → rate_content → END
#
# Prerequisites:
#   - Add GROQ_API_KEY to a .env file
#   - Get a free key at: https://console.groq.com
#
# Run:  python 3_prompt_chaining.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()


# ── 1. Initialise the LLM ─────────────────────────────────────────────────────
# llama-3.3-70b-versatile handles long-form writing tasks well.

model = ChatGroq(model='llama-3.3-70b-versatile')


# ── 2. Define the State ───────────────────────────────────────────────────────

class BlogState(TypedDict):
    title:   str    # input: blog topic
    outline: str    # created by create_outline node
    content: str    # created by create_blog node
    rating:  str    # created by rate_content node


# ── 3. Define the Nodes ───────────────────────────────────────────────────────

def create_outline(state: BlogState) -> BlogState:
    """Node 1: Generate a structured outline for the blog topic."""
    title = state['title']
    prompt = f'Generate a detailed outline for a blog on the topic: {title}'
    outline = model.invoke(prompt).content

    state['outline'] = outline
    return state


def create_blog(state: BlogState) -> BlogState:
    """Node 2: Write the full blog using the title and outline from Node 1."""
    title   = state['title']
    outline = state['outline']

    prompt = (
        f'Write a detailed blog on the title: {title} '
        f'using the following outline:\n{outline}'
    )
    content = model.invoke(prompt).content

    state['content'] = content
    return state


def rate_content(state: BlogState) -> BlogState:
    """Node 3: Compare the blog content against the outline and rate it."""
    outline = state['outline']
    content = state['content']

    prompt = (
        f'Compare the generated content with the outline and rate the following '
        f'content as per the outline.\n\nOutline:\n{outline}\n\nContent:\n{content}'
    )
    rating = model.invoke(prompt).content

    state['rating'] = rating
    return state


# ── 4. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(BlogState)

graph.add_node('create_outline', create_outline)
graph.add_node('create_blog',    create_blog)
graph.add_node('rate_content',   rate_content)

graph.add_edge(START,            'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog',    'rate_content')
graph.add_edge('rate_content',   END)

workflow = graph.compile()


# ── 5. Run the Workflow ───────────────────────────────────────────────────────

if __name__ == '__main__':
    initial_state = {'title': 'Rise of AI in India'}

    final_state = workflow.invoke(initial_state)

    print('─' * 60)
    print('PROMPT CHAINING — BLOG WRITER RESULT')
    print('─' * 60)

    print('\n📋 OUTLINE:\n')
    print(final_state['outline'])

    print('\n✍️  BLOG CONTENT:\n')
    print(final_state['content'])

    print('\n⭐ RATING:\n')
    print(final_state['rating'])
    print('─' * 60)
