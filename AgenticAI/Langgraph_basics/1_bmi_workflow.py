# ─────────────────────────────────────────────────────────────────────────────
# 1_bmi_workflow.py
#
# BMI Calculator Workflow using LangGraph
#
# What this script does:
#   - Takes weight (kg) and height (m) as input
#   - Node 1: Calculates the BMI value
#   - Node 2: Labels the BMI into a category (Underweight / Normal /
#             Overweight / Obese)
#   - Prints the final state with BMI value and category
#
# No LLM is used here — this is a pure logic workflow to demonstrate
# how LangGraph State and sequential nodes work.
#
# Run:  python 1_bmi_workflow.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ── 1. Define the State ───────────────────────────────────────────────────────
# TypedDict defines what data travels through the graph.
# Every node reads from and writes back to this shared state.

class BMIState(TypedDict):
    weight_kg: float    # input: user's weight in kilograms
    height_m: float     # input: user's height in metres
    bmi: float          # calculated by calculate_bmi node
    category: str       # labelled by label_bmi node


# ── 2. Define the Nodes ───────────────────────────────────────────────────────
# Each node is a plain Python function that receives the current state
# and returns a (partial) updated state.

def calculate_bmi(state: BMIState) -> BMIState:
    """Node 1: Calculate BMI from weight and height."""
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight / (height ** 2)
    state['bmi'] = round(bmi, 2)

    return state


def label_bmi(state: BMIState) -> BMIState:
    """Node 2: Assign a WHO category label based on BMI value."""
    bmi = state['bmi']

    if bmi < 18.5:
        state['category'] = 'Underweight'
    elif 18.5 <= bmi < 25:
        state['category'] = 'Normal'
    elif 25 <= bmi < 30:
        state['category'] = 'Overweight'
    else:
        state['category'] = 'Obese'

    return state


# ── 3. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(BMIState)

# Register nodes
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# Connect nodes with edges:  START → calculate_bmi → label_bmi → END
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# Compile the graph into a runnable workflow
workflow = graph.compile()


# ── 4. Run the Workflow ───────────────────────────────────────────────────────

if __name__ == '__main__':
    initial_state = {
        'weight_kg': 80,
        'height_m': 1.73
    }

    final_state = workflow.invoke(initial_state)

    print('─' * 40)
    print('BMI WORKFLOW RESULT')
    print('─' * 40)
    print(f"Weight   : {final_state['weight_kg']} kg")
    print(f"Height   : {final_state['height_m']} m")
    print(f"BMI      : {final_state['bmi']}")
    print(f"Category : {final_state['category']}")
    print('─' * 40)
