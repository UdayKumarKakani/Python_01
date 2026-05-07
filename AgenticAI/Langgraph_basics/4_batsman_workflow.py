# ─────────────────────────────────────────────────────────────────────────────
# 4_batsman_workflow.py
#
# Cricket Batsman Stats Workflow using LangGraph (parallel nodes)
#
# What this script does:
#   - Takes a batsman's raw scorecard (runs, balls, fours, sixes) as input
#   - Runs THREE stat-calculation nodes IN PARALLEL:
#       • calculate_sr              → Strike Rate
#       • calculate_bpb             → Balls per Boundary
#       • calculate_boundary_percent → Boundary % of total runs
#   - After all three finish, the summary node collects and prints the stats
#
# This demonstrates PARALLEL / FAN-OUT execution in LangGraph:
#   START → [calculate_sr | calculate_bpb | calculate_boundary_percent]
#                                   ↓ (all merge into summary)
#                                 summary → END
#
# No LLM is used — pure calculation workflow.
#
# Run:  python 4_batsman_workflow.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ── 1. Define the State ───────────────────────────────────────────────────────

class BatsmanState(TypedDict):
    # Inputs
    runs:   int
    balls:  int
    fours:  int
    sixes:  int

    # Outputs (filled by parallel nodes)
    sr:                 float   # Strike Rate
    bpb:                float   # Balls per Boundary
    boundary_percent:   float   # % of runs scored via boundaries

    summary:            str     # Final text summary


# ── 2. Define the Nodes ───────────────────────────────────────────────────────

def calculate_sr(state: BatsmanState):
    """Strike Rate = (Runs / Balls) × 100"""
    sr = (state['runs'] / state['balls']) * 100
    return {'sr': round(sr, 2)}


def calculate_bpb(state: BatsmanState):
    """Balls per Boundary = Balls / (Fours + Sixes)"""
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb': round(bpb, 2)}


def calculate_boundary_percent(state: BatsmanState):
    """Boundary % = ((4×fours + 6×sixes) / Runs) × 100"""
    boundary_runs    = (state['fours'] * 4) + (state['sixes'] * 6)
    boundary_percent = (boundary_runs / state['runs']) * 100
    return {'boundary_percent': round(boundary_percent, 2)}


def summary(state: BatsmanState):
    """Collect all computed stats into a readable summary string."""
    text = (
        f"Strike Rate           : {state['sr']}\n"
        f"Balls per Boundary    : {state['bpb']}\n"
        f"Boundary Percentage   : {state['boundary_percent']}%"
    )
    return {'summary': text}


# ── 3. Build the Graph ────────────────────────────────────────────────────────
# All three calculation nodes fan out from START and converge at 'summary'.

graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr',               calculate_sr)
graph.add_node('calculate_bpb',              calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary',                    summary)

# Fan-out: START fires all three nodes in parallel
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

# Fan-in: all three merge into summary
graph.add_edge('calculate_sr',               'summary')
graph.add_edge('calculate_bpb',              'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()


# ── 4. Run the Workflow ───────────────────────────────────────────────────────

if __name__ == '__main__':
    initial_state = {
        'runs':  65,
        'balls': 45,
        'fours': 6,
        'sixes': 3
    }

    final_state = workflow.invoke(initial_state)

    print('─' * 40)
    print('BATSMAN STATS WORKFLOW RESULT')
    print('─' * 40)
    print(f"Runs   : {final_state['runs']}")
    print(f"Balls  : {final_state['balls']}")
    print(f"Fours  : {final_state['fours']}")
    print(f"Sixes  : {final_state['sixes']}")
    print()
    print(final_state['summary'])
    print('─' * 40)
