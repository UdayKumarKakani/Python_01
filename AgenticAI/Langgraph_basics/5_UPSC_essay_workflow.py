# ─────────────────────────────────────────────────────────────────────────────
# 5_UPSC_essay_workflow.py
#
# UPSC Essay Evaluator — Parallel Evaluation + Structured Output
# using LangGraph + Groq
#
# What this script does:
#   - Takes a UPSC-style essay as input
#   - Runs THREE evaluation nodes IN PARALLEL (fan-out):
#       • evaluate_language  → language quality feedback + score
#       • evaluate_analysis  → depth of analysis feedback + score
#       • evaluate_thought   → clarity of thought feedback + score
#   - individual_scores list is auto-accumulated using Annotated + operator.add
#   - final_evaluation node (fan-in):
#       • summarises all three feedbacks
#       • computes the average score
#   - Prints overall feedback and average score
#
# Key concepts shown:
#   ✔ Parallel fan-out nodes
#   ✔ Annotated list accumulator (operator.add) for merging scores
#   ✔ Structured output with Pydantic (EvaluationSchema)
#   ✔ with_structured_output() for guaranteed JSON-shaped responses
#
# Prerequisites:
#   - Add GROQ_API_KEY to a .env file
#   - Get a free key at: https://console.groq.com
#
# Run:  python 5_UPSC_essay_workflow.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()


# ── 1. Initialise Models ──────────────────────────────────────────────────────
# Use mixtral-8x7b-32768 for structured output — it handles JSON reliably.
# llama-3.3-70b-versatile is used for the final free-text summary.

structured_llm_base = ChatGroq(model='qwen/qwen3-32b')
summary_llm         = ChatGroq(model='llama-3.3-70b-versatile')


# ── 2. Pydantic Schema for Structured Output ─────────────────────────────────

class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the essay')
    score:    int = Field(description='Score out of 10', ge=0, le=10)


# Wrap the model with structured output — responses are parsed into
# EvaluationSchema objects automatically
structured_model = structured_llm_base.with_structured_output(EvaluationSchema)


# ── 3. Define the State ───────────────────────────────────────────────────────
# individual_scores uses Annotated[list[int], operator.add] so that
# each parallel node APPENDS its score rather than overwriting.

class UPSCState(TypedDict):
    essay:             str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback:  str
    overall_feedback:  str
    individual_scores: Annotated[list[int], operator.add]   # auto-accumulates
    avg_score:         float


# ── 4. Define the Nodes ───────────────────────────────────────────────────────

def evaluate_language(state: UPSCState):
    """Parallel Node 1: Evaluate language quality."""
    prompt = (
        f'Evaluate the language quality of the following essay '
        f'and provide feedback and a score out of 10.\n\n{state["essay"]}'
    )
    output = structured_model.invoke(prompt)
    return {
        'language_feedback': output.feedback,
        'individual_scores': [output.score]     # appended to the list
    }


def evaluate_analysis(state: UPSCState):
    """Parallel Node 2: Evaluate depth of analysis."""
    prompt = (
        f'Evaluate the depth of analysis of the following essay '
        f'and provide feedback and a score out of 10.\n\n{state["essay"]}'
    )
    output = structured_model.invoke(prompt)
    return {
        'analysis_feedback': output.feedback,
        'individual_scores': [output.score]
    }


def evaluate_thought(state: UPSCState):
    """Parallel Node 3: Evaluate clarity of thought."""
    prompt = (
        f'Evaluate the clarity of thought of the following essay '
        f'and provide feedback and a score out of 10.\n\n{state["essay"]}'
    )
    output = structured_model.invoke(prompt)
    return {
        'clarity_feedback': output.feedback,
        'individual_scores': [output.score]
    }


def final_evaluation(state: UPSCState):
    """Fan-in Node: Summarise all feedbacks and compute average score."""
    prompt = (
        f'Based on the following feedbacks, create a concise summarised feedback.\n\n'
        f'Language feedback      : {state["language_feedback"]}\n'
        f'Depth of analysis      : {state["analysis_feedback"]}\n'
        f'Clarity of thought     : {state["clarity_feedback"]}'
    )
    overall_feedback = summary_llm.invoke(prompt).content
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])

    return {
        'overall_feedback': overall_feedback,
        'avg_score':        round(avg_score, 2)
    }


# ── 5. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(UPSCState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought',  evaluate_thought)
graph.add_node('final_evaluation',  final_evaluation)

# Fan-out: all three evaluations start simultaneously
graph.add_edge(START,               'evaluate_language')
graph.add_edge(START,               'evaluate_analysis')
graph.add_edge(START,               'evaluate_thought')

# Fan-in: all three feed into final_evaluation
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought',  'final_evaluation')

graph.add_edge('final_evaluation',  END)

workflow = graph.compile()


# ── 6. Sample Essays ──────────────────────────────────────────────────────────

ESSAY_GOOD = """\
India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI),
India stands at a critical juncture — one where it can either emerge as a global
leader in AI innovation or risk falling behind in the technology race. The age of
AI brings with it immense promise as well as unprecedented challenges, and how
India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers,
a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM
graduates annually and a burgeoning base of AI researchers, India possesses the
intellectual capital required to build cutting-edge AI systems.

However, the path to AI-led growth is riddled with challenges. Chief among them is
the digital divide. Without effective skilling and re-skilling programs, AI could
exacerbate existing socio-economic inequalities. Data privacy and ethics also remain
a pressing concern as India is still shaping its data protection laws.

In conclusion, India in the age of AI is a story in the making — one of opportunity,
responsibility, and transformation. The decisions we make today will determine India's
future as an inclusive, equitable, and innovation-driven society.
"""

ESSAY_POOR = """\
India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI).
India also want become big in this AI thing. If work hard, India can go top. But if
no careful, India go back.

India have many good. We have smart student, many engineer, and good IT peoples.
Big company like TCS already use AI.

But problem come also. Many villager no have phone or internet. Many people lose job.

So, in short, AI time in India have many hope and many danger. We must go right road.
"""


# ── 7. Run the Workflow ───────────────────────────────────────────────────────

if __name__ == '__main__':
    # Change to ESSAY_GOOD to test a high-quality essay
    initial_state = {'essay': ESSAY_GOOD}

    final_state = workflow.invoke(initial_state)

    print('─' * 60)
    print('UPSC ESSAY EVALUATION RESULT')
    print('─' * 60)
    print(f"\n📝 LANGUAGE FEEDBACK:\n{final_state['language_feedback']}")
    print(f"\n🔍 ANALYSIS FEEDBACK:\n{final_state['analysis_feedback']}")
    print(f"\n💡 CLARITY FEEDBACK:\n{final_state['clarity_feedback']}")
    print(f"\n📊 INDIVIDUAL SCORES : {final_state['individual_scores']}")
    print(f"⭐ AVERAGE SCORE     : {final_state['avg_score']} / 10")
    print(f"\n✅ OVERALL FEEDBACK:\n{final_state['overall_feedback']}")
    print('─' * 60)
