# ─────────────────────────────────────────────────────────────────────────────
# 6_quadratic_equation_workflow.py
#
# Quadratic Equation Solver — Conditional Routing in LangGraph
#
# What this script does:
#   - Takes coefficients a, b, c of the equation ax² + bx + c = 0
#   - Node 1 (show_equation)        : formats a display string for the equation
#   - Node 2 (calculate_discriminant): computes discriminant = b² - 4ac
#   - Conditional router (check_condition):
#       • discriminant > 0  →  real_roots       (two distinct real roots)
#       • discriminant = 0  →  repeated_roots   (one repeated root)
#       • discriminant < 0  →  no_real_roots    (no real roots / complex)
#   - The chosen branch node calculates and stores the result
#
# This demonstrates CONDITIONAL EDGES in LangGraph — the graph takes
# different paths based on the current state value.
#
# No LLM is used — pure maths workflow.
#
# Graph flow:
#   START → show_equation → calculate_discriminant
#                                  ↓ (conditional)
#                    ┌─────────────┼─────────────┐
#                real_roots  repeated_roots  no_real_roots
#                    └─────────────┼─────────────┘
#                                 END
#
# Run:  python 6_quadratic_equation_workflow.py
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


# ── 1. Define the State ───────────────────────────────────────────────────────

class QuadState(TypedDict):
    a:             int     # coefficient of x²
    b:             int     # coefficient of x
    c:             int     # constant term
    equation:      str     # formatted equation string
    discriminant:  float   # b² - 4ac
    result:        str     # final answer text


# ── 2. Define the Nodes ───────────────────────────────────────────────────────

def show_equation(state: QuadState):
    """Format the equation as a readable string."""
    a, b, c = state['a'], state['b'], state['c']

    # Build sign-aware equation string
    b_part = f'+ {b}x' if b >= 0 else f'- {abs(b)}x'
    c_part = f'+ {c}'  if c >= 0 else f'- {abs(c)}'

    equation = f'{a}x² {b_part} {c_part} = 0'
    return {'equation': equation}


def calculate_discriminant(state: QuadState):
    """Compute discriminant = b² - 4ac."""
    discriminant = state['b'] ** 2 - (4 * state['a'] * state['c'])
    return {'discriminant': discriminant}


def real_roots(state: QuadState):
    """Two distinct real roots (discriminant > 0)."""
    a, b, d = state['a'], state['b'], state['discriminant']
    root1 = (-b + d ** 0.5) / (2 * a)
    root2 = (-b - d ** 0.5) / (2 * a)
    result = f'Two real roots: x₁ = {round(root1, 4)},  x₂ = {round(root2, 4)}'
    return {'result': result}


def repeated_roots(state: QuadState):
    """One repeated root (discriminant = 0)."""
    root = (-state['b']) / (2 * state['a'])
    result = f'One repeated root: x = {round(root, 4)}'
    return {'result': result}


def no_real_roots(state: QuadState):
    """No real roots — complex roots (discriminant < 0)."""
    result = 'No real roots (discriminant < 0 → complex roots)'
    return {'result': result}


# ── 3. Conditional Router ─────────────────────────────────────────────────────
# The return value is the NAME of the next node to go to.

def check_condition(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'


# ── 4. Build the Graph ────────────────────────────────────────────────────────

graph = StateGraph(QuadState)

graph.add_node('show_equation',          show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots',             real_roots)
graph.add_node('repeated_roots',         repeated_roots)
graph.add_node('no_real_roots',          no_real_roots)

graph.add_edge(START,                 'show_equation')
graph.add_edge('show_equation',       'calculate_discriminant')

# add_conditional_edges: after calculate_discriminant, call check_condition
# to decide which branch node to visit next
graph.add_conditional_edges('calculate_discriminant', check_condition)

graph.add_edge('real_roots',      END)
graph.add_edge('repeated_roots',  END)
graph.add_edge('no_real_roots',   END)

workflow = graph.compile()


# ── 5. Run the Workflow ───────────────────────────────────────────────────────

def solve(a, b, c):
    """Helper to solve and print results for given coefficients."""
    final_state = workflow.invoke({'a': a, 'b': b, 'c': c})
    print(f"  Equation      : {final_state['equation']}")
    print(f"  Discriminant  : {final_state['discriminant']}")
    print(f"  Result        : {final_state['result']}")


if __name__ == '__main__':
    test_cases = [
        (1,  -5,  6),    # discriminant > 0  → two real roots
        (2,   4,  2),    # discriminant = 0  → repeated root
        (1,   2,  5),    # discriminant < 0  → no real roots
    ]

    print('─' * 55)
    print('QUADRATIC EQUATION SOLVER — CONDITIONAL ROUTING')
    print('─' * 55)

    for a, b, c in test_cases:
        print(f'\nInput: a={a}, b={b}, c={c}')
        solve(a, b, c)

    print('─' * 55)
