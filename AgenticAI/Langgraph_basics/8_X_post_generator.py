# ─────────────────────────────────────────────────────────────────────────────
# 8_X_post_generator.py
#
# X / Twitter Post Generator with Iterative Improvement Loop
# using LangGraph + Groq
#
# What this script does:
#   - Takes a topic as input
#   - Node 1 (generate)  : Writes a funny/viral tweet on the topic
#   - Node 2 (evaluate)  : Critiques the tweet — returns "approved" or
#                          "needs_improvement" + detailed feedback
#   - Conditional router (route_evaluation):
#       • approved  OR  iteration >= max_iteration  →  END
#       • needs_improvement                         →  optimize
#   - Node 3 (optimize)  : Rewrites the tweet based on the feedback
#   - Loop continues until approved or max iterations reached
#
# Key concepts shown:
#   ✔ LOOP / CYCLE in LangGraph (optimize → evaluate → optimize ...)
#   ✔ Annotated list accumulator for tweet_history and feedback_history
#   ✔ Structured output (TweetEvaluation schema)
#   ✔ Multi-LLM setup (generator / evaluator / optimizer)
#   ✔ Iteration counter to prevent infinite loops
#
# Graph flow:
#   START → generate → evaluate ──── approved ──────────→ END
#                          ↑                                |
#                          └─── needs_improvement ← optimize
#
# Prerequisites:
#   - Add GROQ_API_KEY to a .env file
#   - Get a free key at: https://console.groq.com
#
# Run:  python 8_X_post_generator.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator

load_dotenv()


# ── 1. Initialise Models ──────────────────────────────────────────────────────
# Three separate LLM instances — one per role.
# All use llama-3.3-70b-versatile; mixtral is used for structured eval output.

generator_llm = ChatGroq(model='llama-3.3-70b-versatile')
evaluator_llm = ChatGroq(model='mixtral-8x7b-32768')    # structured output
optimizer_llm = ChatGroq(model='llama-3.3-70b-versatile')


# ── 2. Pydantic Schema for Structured Evaluation Output ───────────────────────

class TweetEvaluation(BaseModel):
    evaluation: Literal['approved', 'needs_improvement'] = Field(
        ..., description='Final evaluation result.'
    )
    feedback: str = Field(
        ..., description='Feedback for the tweet.'
    )


structured_evaluator = evaluator_llm.with_structured_output(TweetEvaluation)


# ── 3. Define the State ───────────────────────────────────────────────────────
# tweet_history and feedback_history use operator.add so each iteration
# APPENDS rather than overwrites — giving a full audit trail.

class TweetState(TypedDict):
    topic:           str
    tweet:           str
    evaluation:      Literal['approved', 'needs_improvement']
    feedback:        str
    iteration:       int
    max_iteration:   int
    tweet_history:   Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


# ── 4. Define the Nodes ───────────────────────────────────────────────────────

def generate_tweet(state: TweetState):
    """Node 1: Generate a funny, viral tweet on the given topic."""
    messages = [
        SystemMessage(content='You are a funny and clever Twitter/X influencer.'),
        HumanMessage(content=(
            f'Write a short, original, and hilarious tweet on the topic: "{state["topic"]}".\n\n'
            f'Rules:\n'
            f'- Do NOT use question-answer format.\n'
            f'- Max 280 characters.\n'
            f'- Use observational humor, irony, sarcasm, or cultural references.\n'
            f'- Think in meme logic, punchlines, or relatable takes.\n'
            f'- Use simple, everyday English.'
        ))
    ]
    response = generator_llm.invoke(messages).content
    return {'tweet': response, 'tweet_history': [response]}


def evaluate_tweet(state: TweetState):
    """Node 2: Critically evaluate the tweet for humor, virality, and format."""
    messages = [
        SystemMessage(content=(
            'You are a ruthless Twitter critic. You evaluate tweets based on '
            'humor, originality, virality, and tweet format.'
        )),
        HumanMessage(content=(
            f'Evaluate the following tweet:\n\nTweet: "{state["tweet"]}"\n\n'
            f'Criteria:\n'
            f'1. Originality – Is this fresh?\n'
            f'2. Humor – Did it make you laugh?\n'
            f'3. Punchiness – Short, sharp, scroll-stopping?\n'
            f'4. Virality Potential – Would people share it?\n'
            f'5. Format – Well-formed tweet? Under 280 characters?\n\n'
            f'Auto-reject if:\n'
            f'- Written in question-answer format\n'
            f'- Exceeds 280 characters\n'
            f'- Reads like a traditional setup-punchline joke\n\n'
            f'Respond ONLY in structured format:\n'
            f'- evaluation: "approved" or "needs_improvement"\n'
            f'- feedback: One paragraph explaining strengths and weaknesses'
        ))
    ]
    response = structured_evaluator.invoke(messages)
    return {
        'evaluation':      response.evaluation,
        'feedback':        response.feedback,
        'feedback_history': [response.feedback]
    }


def optimize_tweet(state: TweetState):
    """Node 3: Rewrite the tweet based on evaluator feedback."""
    messages = [
        SystemMessage(content='You punch up tweets for virality and humor based on feedback.'),
        HumanMessage(content=(
            f'Improve the tweet based on this feedback:\n"{state["feedback"]}"\n\n'
            f'Topic: "{state["topic"]}"\n'
            f'Original Tweet:\n{state["tweet"]}\n\n'
            f'Re-write as a short, viral-worthy tweet. '
            f'Avoid Q&A style and stay under 280 characters.'
        ))
    ]
    response  = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1
    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}


# ── 5. Conditional Router ─────────────────────────────────────────────────────

def route_evaluation(state: TweetState) -> Literal['approved', 'needs_improvement']:
    """Stop if approved OR if we've hit the max iteration limit."""
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    return 'needs_improvement'


# ── 6. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(TweetState)

graph.add_node('generate',  generate_tweet)
graph.add_node('evaluate',  evaluate_tweet)
graph.add_node('optimize',  optimize_tweet)

graph.add_edge(START,      'generate')
graph.add_edge('generate', 'evaluate')

graph.add_conditional_edges(
    'evaluate',
    route_evaluation,
    {'approved': END, 'needs_improvement': 'optimize'}
)

graph.add_edge('optimize', 'evaluate')     # ← this is the LOOP edge

workflow = graph.compile()


# ── 7. Run the Workflow ───────────────────────────────────────────────────────

if __name__ == '__main__':
    initial_state = {
        'topic':         'Monday morning meetings that could have been an email',
        'iteration':     1,
        'max_iteration': 4,
    }

    result = workflow.invoke(initial_state)

    print('─' * 60)
    print('X POST GENERATOR — ITERATIVE IMPROVEMENT RESULT')
    print('─' * 60)
    print(f"Topic      : {initial_state['topic']}")
    print(f"Iterations : {result['iteration']}")
    print(f"Final eval : {result['evaluation']}")

    print('\n📜 TWEET HISTORY (each iteration):')
    for i, tweet in enumerate(result['tweet_history'], 1):
        print(f'\n  [{i}] {tweet}')

    print('\n💬 FEEDBACK HISTORY:')
    for i, fb in enumerate(result['feedback_history'], 1):
        print(f'\n  [{i}] {fb}')

    print('\n✅ FINAL TWEET:')
    print(f"  {result['tweet']}")
    print('─' * 60)
