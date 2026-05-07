# ─────────────────────────────────────────────────────────────────────────────
# 7_review_reply_workflow.py
#
# Customer Review Reply Workflow — Sentiment + Conditional Routing
# using LangGraph + Groq
#
# What this script does:
#   - Takes a customer review as input
#   - Node 1 (find_sentiment) : classifies the review as "positive" or
#     "negative" using structured output (SentimentSchema)
#   - Conditional router (check_sentiment):
#       • positive → positive_response   (warm thank-you message)
#       • negative → run_diagnosis       → negative_response
#   - run_diagnosis uses DiagnosisSchema to extract:
#       issue_type (UX / Performance / Bug / Support / Other)
#       tone       (angry / frustrated / disappointed / calm)
#       urgency    (low / medium / high)
#   - negative_response crafts an empathetic resolution message
#
# This demonstrates:
#   ✔ Structured output with Pydantic schemas
#   ✔ Conditional routing based on LLM classification
#   ✔ Multi-step negative path (diagnose → respond)
#
# Graph flow:
#   START → find_sentiment
#               ↓ (conditional)
#     positive_response          run_diagnosis → negative_response
#           ↓                                         ↓
#          END ←────────────────────────────────────END
#
# Prerequisites:
#   - Add GROQ_API_KEY to a .env file
#   - Get a free key at: https://console.groq.com
#
# Run:  python 7_review_reply_workflow.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ── 1. Initialise Models ──────────────────────────────────────────────────────
# mixtral-8x7b-32768 is reliable for structured output / JSON parsing.
# llama-3.3-70b-versatile is used for free-text response generation.

structured_llm_base = ChatGroq(model='mixtral-8x7b-32768')
response_llm        = ChatGroq(model='llama-3.3-70b-versatile')


# ── 2. Pydantic Schemas for Structured Output ─────────────────────────────────

class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Sentiment of the review'
    )


class DiagnosisSchema(BaseModel):
    issue_type: Literal['UX', 'Performance', 'Bug', 'Support', 'Other'] = Field(
        description='The category of issue mentioned in the review'
    )
    tone: Literal['angry', 'frustrated', 'disappointed', 'calm'] = Field(
        description='The emotional tone expressed by the user'
    )
    urgency: Literal['low', 'medium', 'high'] = Field(
        description='How urgent or critical the issue appears to be'
    )


sentiment_model  = structured_llm_base.with_structured_output(SentimentSchema)
diagnosis_model  = structured_llm_base.with_structured_output(DiagnosisSchema)


# ── 3. Define the State ───────────────────────────────────────────────────────

class ReviewState(TypedDict):
    review:    str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response:  str


# ── 4. Define the Nodes ───────────────────────────────────────────────────────

def find_sentiment(state: ReviewState):
    """Node 1: Classify the review as positive or negative."""
    prompt = f'For the following review, find out the sentiment:\n\n{state["review"]}'
    sentiment = sentiment_model.invoke(prompt).sentiment
    return {'sentiment': sentiment}


def check_sentiment(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
    """Conditional router — decides which branch to take."""
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'


def positive_response(state: ReviewState):
    """Branch A: Write a warm thank-you reply for positive reviews."""
    prompt = (
        f'Write a warm thank-you message in response to this review:\n\n'
        f'"{state["review"]}"\n\n'
        f'Also kindly ask the user to leave feedback on our website.'
    )
    response = response_llm.invoke(prompt).content
    return {'response': response}


def run_diagnosis(state: ReviewState):
    """Branch B – Step 1: Diagnose the negative review (issue, tone, urgency)."""
    prompt = (
        f'Diagnose this negative review and return issue_type, tone, and urgency.\n\n'
        f'{state["review"]}'
    )
    result = diagnosis_model.invoke(prompt)
    return {'diagnosis': result.model_dump()}


def negative_response(state: ReviewState):
    """Branch B – Step 2: Write an empathetic resolution message."""
    d = state['diagnosis']
    prompt = (
        f'You are a support assistant.\n'
        f'The user had a "{d["issue_type"]}" issue, sounded "{d["tone"]}", '
        f'and the urgency is "{d["urgency"]}".\n'
        f'Write an empathetic, helpful resolution message.'
    )
    response = response_llm.invoke(prompt).content
    return {'response': response}


# ── 5. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(ReviewState)

graph.add_node('find_sentiment',   find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis',    run_diagnosis)
graph.add_node('negative_response', negative_response)

graph.add_edge(START, 'find_sentiment')

graph.add_conditional_edges('find_sentiment', check_sentiment)

graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis',     'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()


# ── 6. Run the Workflow ───────────────────────────────────────────────────────

REVIEW_NEGATIVE = (
    "I've been trying to log in for over an hour now, and the app keeps "
    "freezing on the authentication screen. I even tried reinstalling it, "
    "but no luck. This kind of bug is unacceptable, especially when it "
    "affects basic functionality."
)

REVIEW_POSITIVE = (
    "Absolutely love this app! The interface is clean, everything works "
    "smoothly, and the support team was super helpful. 10/10 would recommend!"
)

if __name__ == '__main__':
    for label, review in [('NEGATIVE', REVIEW_NEGATIVE), ('POSITIVE', REVIEW_POSITIVE)]:
        print('─' * 60)
        print(f'REVIEW REPLY WORKFLOW — {label} REVIEW')
        print('─' * 60)
        print(f'Review    : {review[:80]}...')

        final_state = workflow.invoke({'review': review})

        print(f'Sentiment : {final_state["sentiment"]}')
        if final_state.get('diagnosis'):
            print(f'Diagnosis : {final_state["diagnosis"]}')
        print(f'\nResponse:\n{final_state["response"]}')
        print()
