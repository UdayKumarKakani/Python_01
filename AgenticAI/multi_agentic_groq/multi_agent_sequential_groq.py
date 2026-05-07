"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       SEQUENTIAL MULTI-AGENT SYSTEM USING LANGGRAPH + GROQ API             ║
║                                                                              ║
║  Model  : llama-3.3-70b-versatile  (via Groq — ultra fast inference)        ║
║                                                                              ║
║  Agents (in sequential order):                                               ║
║   1. 🔍 Research Agent   — searches the web using DuckDuckGo                 ║
║   2. 📊 Analysis Agent   — extracts key findings and insights                ║
║   3. ✍️  Writer Agent     — writes a polished markdown report                 ║
║   4. ✅ Reviewer Agent   — scores quality and gives improvement feedback     ║
║                                                                              ║
║  Pipeline Flow:                                                              ║
║  User Query → Research → Analysis → Writer → Reviewer → Final Output        ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW THIS SYSTEM WORKS — ARCHITECTURE EXPLANATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 OVERVIEW
 ─────────
 This is a Sequential Multi-Agent system built on LangGraph's StateGraph.

 Instead of one LLM doing everything, work is divided across 4 specialized
 AI agents. Each agent has a single responsibility and passes its output
 to the next agent through a shared state object.

 WHY LANGGRAPH?
 ───────────────
 LangGraph lets us define agents as "nodes" in a directed graph, and
 connect them with "edges" (transitions). The graph engine handles:
   • Routing state from one node to the next
   • Termination (via the END sentinel)
   • Stateful persistence across nodes
   • Error isolation per node

 SHARED STATE (AgentState)
 ──────────────────────────
 All agents share a single Python TypedDict called AgentState.
 Think of it as a shared clipboard that gets passed down the chain.
 Each agent reads from it and writes its result back to it.

   AgentState = {
     "query"         → Original user question (never changed)
     "raw_research"  → Written by Research Agent
     "analysis"      → Written by Analysis Agent
     "final_report"  → Written by Writer Agent
     "review"        → Written by Reviewer Agent
     "quality_score" → Integer score (1–10) from Reviewer Agent
     "agent_logs"    → Execution trail, appended by every agent
   }

 THE GRAPH STRUCTURE
 ────────────────────
   [START]
      │
      ▼
  research_agent   (Node 1) — DuckDuckGo search + LLM consolidation
      │
      ▼
  analysis_agent   (Node 2) — Extracts themes, findings, gaps
      │
      ▼
   writer_agent    (Node 3) — Writes final markdown report
      │
      ▼
  reviewer_agent   (Node 4) — Scores quality, gives feedback
      │
      ▼
    [END]

 All edges are unconditional — every node always runs in order.
 No branching or loops. Pure linear pipeline.

 WHY GROQ?
 ──────────
 Groq provides ultra-fast LLM inference via its LPU hardware.
 We use llama-3.3-70b-versatile — a powerful 70B model that runs
 faster than most hosted APIs, making this multi-agent chain feel snappy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import requests
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Shared state that flows through every agent node.
    LangGraph passes this dict from node to node automatically.
    Each agent reads what it needs and writes its output back.
    """
    query: str                      # Original user query — set once, never changed
    raw_research: Optional[str]     # Written by Research Agent (Agent 1)
    analysis: Optional[str]         # Written by Analysis Agent (Agent 2)
    final_report: Optional[str]     # Written by Writer Agent   (Agent 3)
    review: Optional[str]           # Written by Reviewer Agent (Agent 4)
    quality_score: Optional[int]    # Score 1–10 from Reviewer Agent
    agent_logs: list[str]           # Execution log — each agent appends here


# ══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY — GROQ
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Returns a ChatGroq instance using llama-3.3-70b-versatile.
    GROQ_API_KEY must be set as an environment variable.
    Get your free key at: https://console.groq.com
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — RESEARCH AGENT
# ══════════════════════════════════════════════════════════════════════════════

def research_agent(state: AgentState) -> AgentState:
    """
    RESEARCH AGENT
    ═══════════════
    Role    : Information Gatherer
    Input   : state["query"]
    Output  : state["raw_research"]

    What it does:
      1. Runs a DuckDuckGo search for the user's query
      2. Collects top 6 web results (title, snippet, URL)
      3. Passes raw results to the Groq LLM (llama-3.3-70b)
      4. LLM organises the web data into structured research notes

    Why a dedicated research agent?
      Real answers need real data. This agent is like a junior analyst
      who goes out, collects sources, and returns with raw notes.
      It doesn't judge or write — it just gathers.
    """
    print("\n🔍 [Agent 1] Research Agent running...")
    query = state["query"]

    # ── Step 1: Web search via DuckDuckGo ──────────────────────────────────
    search_results = []
    try:
        results = DDGS().text(query, max_results=6)
        for r in results:
            search_results.append(
                f"SOURCE: {r['href']}\nTITLE: {r['title']}\nSNIPPET: {r['body']}"
            )
        raw_web_data = "\n\n---\n\n".join(search_results)
    except Exception as e:
        raw_web_data = f"Search error: {str(e)}. Using LLM knowledge only."

    print(f"   ✓ Collected {len(search_results)} search results from DuckDuckGo")

    # ── Step 2: Groq LLM consolidates the raw search data ──────────────────
    llm = get_llm(temperature=0.1)
    prompt = f"""You are a Research Agent. Your job is to consolidate raw web search results 
into structured, detailed research notes.

USER QUERY: {query}

RAW SEARCH DATA:
{raw_web_data}

Instructions:
- Extract ALL relevant facts, statistics, data points, and key information
- Organise into clearly labelled sections (e.g. Background, Key Facts, Recent Developments)
- Preserve specific numbers, dates, names, and details — do NOT over-summarise
- Note any conflicting or uncertain information
- Write as thorough research notes (NOT a final report — that comes later)
- Aim for comprehensiveness over brevity

Output your research notes below:"""

    response = llm.invoke([
        SystemMessage(content="You are a meticulous research analyst. Always be thorough and detailed."),
        HumanMessage(content=prompt)
    ])
    raw_research = response.content
    print(f"   ✓ Research notes ready ({len(raw_research)} chars)")

    return {
        **state,
        "raw_research": raw_research,
        "agent_logs": state["agent_logs"] + [
            f"✅ Research Agent: Searched web ({len(search_results)} sources), produced {len(raw_research)} chars of research notes."
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

def analysis_agent(state: AgentState) -> AgentState:
    """
    ANALYSIS AGENT
    ═══════════════
    Role    : Critical Thinker & Pattern Extractor
    Input   : state["raw_research"]
    Output  : state["analysis"]

    What it does:
      Reads the raw research notes from Agent 1 and performs structured analysis:
        • Key Findings      — the most important facts
        • Main Themes       — recurring patterns or categories
        • Critical Insights — non-obvious conclusions
        • Data Gaps         — missing or unclear information
        • Contradictions    — conflicting data points
        • Relevance Score   — how well research answers the original query

    Why a dedicated analysis agent?
      Raw data is noise. Analysis turns noise into signal. This agent
      acts like a senior analyst who reads field reports and extracts
      exactly what matters for decision-making — separate from writing.
    """
    print("\n📊 [Agent 2] Analysis Agent running...")
    query = state["query"]
    raw_research = state["raw_research"]

    llm = get_llm(temperature=0.2)
    prompt = f"""You are an Analysis Agent. Your job is to critically analyse research notes
and produce a structured, insightful analysis.

ORIGINAL QUERY: {query}

RESEARCH NOTES (from Research Agent):
{raw_research}

Produce a structured analysis with these exact sections:

## 1. KEY FINDINGS
(Top 5–7 most important, specific facts directly relevant to the query)

## 2. MAIN THEMES
(2–4 recurring patterns, categories, or threads in the research)

## 3. CRITICAL INSIGHTS
(Non-obvious conclusions — what does this data actually mean?)

## 4. DATA GAPS
(What is missing, unclear, or not well-supported by the research?)

## 5. CONTRADICTIONS
(Any conflicting information found across sources)

## 6. RELEVANCE ASSESSMENT
(How well does the gathered research answer the original query? Score: X/10, with reason)

Be specific, analytical, and critical. The Writer Agent will use this analysis to write the final report."""

    response = llm.invoke([
        SystemMessage(content="You are a sharp, critical analyst. Identify patterns and insights that others miss."),
        HumanMessage(content=prompt)
    ])
    analysis = response.content
    print(f"   ✓ Analysis complete ({len(analysis)} chars)")

    return {
        **state,
        "analysis": analysis,
        "agent_logs": state["agent_logs"] + [
            f"✅ Analysis Agent: Produced structured analysis — key findings, themes, insights ({len(analysis)} chars)."
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — WRITER AGENT
# ══════════════════════════════════════════════════════════════════════════════

def writer_agent(state: AgentState) -> AgentState:
    """
    WRITER AGENT
    ═════════════
    Role    : Report Generator
    Input   : state["query"] + state["raw_research"] + state["analysis"]
    Output  : state["final_report"]

    What it does:
      Takes the original query, the raw research notes, and the structured
      analysis from the previous two agents, then composes a polished,
      professional markdown report that directly and completely answers
      the user's original question.

    Why a dedicated writer agent?
      Analysis alone isn't readable. The writer transforms structured
      insights into a document that humans can actually use — like a
      senior journalist converting an analyst's briefing into a published article.
      Separating writing from analysis keeps each step higher quality.
    """
    print("\n✍️  [Agent 3] Writer Agent running...")
    query = state["query"]
    analysis = state["analysis"]
    raw_research = state["raw_research"]

    llm = get_llm(temperature=0.5)
    prompt = f"""You are a Writer Agent. Your job is to write a clear, professional, 
comprehensive report in markdown format.

ORIGINAL USER QUERY: {query}

ANALYSIS (from Analysis Agent):
{analysis}

SUPPORTING RESEARCH (from Research Agent — use for specific facts/figures):
{raw_research[:3000]}

Write a comprehensive, well-structured markdown report with EXACTLY these sections:

# [Create an engaging, descriptive title]

## Executive Summary
(2–3 sentences that directly answer the original query. Most important section — be direct.)

## Key Findings
(5–7 bullet points of the most important facts, each with a brief explanation)

## Detailed Analysis
(3–4 paragraphs discussing the topic in depth, organised by the main themes from analysis.
 Use headers for each theme. Include specific data, examples, and context.)

## What This Means
(Practical implications and actionable takeaways — what should the reader do or understand?)

## Limitations & Caveats
(Honest acknowledgement of gaps, uncertainties, or caveats in the research)

---
*Report generated by Sequential Multi-Agent System | Powered by LangGraph + Groq (Llama 3.3 70B)*

Make it professional, informative, and genuinely useful. Use markdown formatting throughout."""

    response = llm.invoke([
        SystemMessage(content="You are an expert research writer. Write clearly, precisely, and engagingly."),
        HumanMessage(content=prompt)
    ])
    final_report = response.content
    print(f"   ✓ Report written ({len(final_report)} chars)")

    return {
        **state,
        "final_report": final_report,
        "agent_logs": state["agent_logs"] + [
            f"✅ Writer Agent: Produced final markdown report ({len(final_report)} chars)."
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — REVIEWER AGENT
# ══════════════════════════════════════════════════════════════════════════════

def reviewer_agent(state: AgentState) -> AgentState:
    """
    REVIEWER AGENT
    ═══════════════
    Role    : Quality Controller & Editor
    Input   : state["query"] + state["final_report"]
    Output  : state["review"] + state["quality_score"]

    What it does:
      Acts as a critical editor. Reads the final report and evaluates it
      against the original query on multiple dimensions:
        • Does it directly answer the query?
        • Is the information accurate and well-supported?
        • Is it well-structured and easy to read?
        • Are there any gaps, errors, or weaknesses?
      Gives a score 1–10 and specific improvement suggestions.

    Why a dedicated reviewer agent?
      Quality assurance. Every professional pipeline needs a review step.
      This ensures the final output is actually good — not just generated.
      In a production system, you could use this score to trigger a
      revision loop (re-run the writer if score < 6, for example).
    """
    print("\n✅ [Agent 4] Reviewer Agent running...")
    query = state["query"]
    final_report = state["final_report"]

    llm = get_llm(temperature=0.1)
    prompt = f"""You are a Reviewer Agent. Your job is to critically evaluate a report 
for quality, accuracy, and usefulness.

ORIGINAL QUERY: {query}

REPORT TO REVIEW:
{final_report}

Evaluate the report and provide your review in EXACTLY this format:

QUALITY_SCORE: [integer from 1 to 10]

## Strengths
- [specific strength 1]
- [specific strength 2]
- [specific strength 3]

## Weaknesses
- [specific weakness 1]
- [specific weakness 2]

## Improvement Suggestions
- [concrete suggestion 1]
- [concrete suggestion 2]
- [concrete suggestion 3]

## Verdict
[One clear, honest sentence summarising the report's overall quality]

## Query Answered?
[Yes / Partially / No] — [brief explanation]

Scoring Guide:
  9–10 : Exceptional — directly and completely answers the query, comprehensive, well-written
  7–8  : Good — answers the query well with minor gaps or style issues
  5–6  : Average — partially answers the query, some structural or content issues  
  3–4  : Poor — misses key aspects of the query, poorly organised
  1–2  : Very poor — fails to answer the query or is incoherent

Be honest and rigorous. A high score must be earned."""

    response = llm.invoke([
        SystemMessage(content="You are a rigorous, honest quality reviewer. Do not inflate scores."),
        HumanMessage(content=prompt)
    ])
    review_text = response.content

    # Extract the numeric score from the response
    quality_score = 7  # safe default
    for line in review_text.split("\n"):
        if line.startswith("QUALITY_SCORE:"):
            try:
                quality_score = int(line.split(":")[1].strip())
                quality_score = max(1, min(10, quality_score))  # clamp 1–10
            except Exception:
                pass

    print(f"   ✓ Review done — Quality Score: {quality_score}/10")

    return {
        **state,
        "review": review_text,
        "quality_score": quality_score,
        "agent_logs": state["agent_logs"] + [
            f"✅ Reviewer Agent: Quality Score = {quality_score}/10. Review complete."
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# BUILD LANGGRAPH SEQUENTIAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline():
    """
    Constructs and compiles the LangGraph StateGraph.

    Nodes  : research → analysis → writer → reviewer
    Edges  : All unconditional (linear chain, no branching)
    Entry  : research_agent
    Exit   : END (after reviewer_agent)

    The compiled graph can be .invoke()'d with an initial AgentState dict.
    LangGraph handles state passing, node scheduling, and termination.
    """
    graph = StateGraph(AgentState)

    # Register each agent as a named node
    graph.add_node("research",  research_agent)
    graph.add_node("analysis",  analysis_agent)
    graph.add_node("writer",    writer_agent)
    graph.add_node("reviewer",  reviewer_agent)

    # Define the sequential execution order
    graph.set_entry_point("research")
    graph.add_edge("research",  "analysis")
    graph.add_edge("analysis",  "writer")
    graph.add_edge("writer",    "reviewer")
    graph.add_edge("reviewer",  END)

    return graph.compile()


# Compile once at module load — reuse for all queries
PIPELINE = build_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str) -> dict:
    """
    Run the full 4-agent sequential pipeline on any research query.

    Args:
        query : Any research question or topic

    Returns a dict with all agent outputs:
        raw_research   → Research Agent output
        analysis       → Analysis Agent output
        final_report   → Writer Agent output (main deliverable)
        review         → Reviewer Agent feedback
        quality_score  → int (1–10) from Reviewer Agent
        agent_logs     → list of execution log strings
    """
    print(f"\n{'═'*65}")
    print(f"  🚀 SEQUENTIAL MULTI-AGENT PIPELINE (Groq + LangGraph)")
    print(f"  Query: {query}")
    print(f"{'═'*65}")

    initial_state: AgentState = {
        "query":         query,
        "raw_research":  None,
        "analysis":      None,
        "final_report":  None,
        "review":        None,
        "quality_score": None,
        "agent_logs":    [],
    }

    result = PIPELINE.invoke(initial_state)

    print(f"\n{'═'*65}")
    print(f"  ✅ PIPELINE COMPLETE  |  Quality Score: {result['quality_score']}/10")
    print(f"{'═'*65}\n")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def run_app():
    import streamlit as st

    st.set_page_config(
        page_title="Sequential Multi-Agent | Groq",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Bricolage+Grotesque:wght@400;600;800&display=swap');

    :root {
        --bg:       #07090f;
        --surface:  #0e1117;
        --surface2: #141824;
        --border:   #1a2030;
        --c1: #f97316;   /* orange  — Research */
        --c2: #8b5cf6;   /* purple  — Analysis */
        --c3: #06b6d4;   /* cyan    — Writer   */
        --c4: #22c55e;   /* green   — Reviewer */
        --text:  #e2e8f0;
        --muted: #4b5675;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Bricolage Grotesque', sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    .hero {
        text-align: center;
        padding: 1.8rem 0 0.8rem;
    }
    .hero h1 {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #f97316, #8b5cf6, #06b6d4, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 0.4rem;
    }

    .flow-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin: 1.2rem 0;
        flex-wrap: wrap;
    }
    .flow-node {
        padding: 0.5rem 1.1rem;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid;
    }
    .flow-arrow { color: var(--muted); font-size: 1.2rem; padding: 0 0.3rem; }

    .agent-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .agent-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .agent-desc {
        font-size: 0.78rem;
        color: var(--muted);
        line-height: 1.45;
    }

    .log-item {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--c4);
        padding: 0.28rem 0;
        border-bottom: 1px solid var(--border);
    }

    .score-box {
        text-align: center;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    .score-num {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
    }
    .score-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #f97316, #8b5cf6) !important;
        color: #07090f !important;
        font-family: 'Bricolage Grotesque', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 0.65rem !important;
        letter-spacing: -0.2px !important;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'Bricolage Grotesque', sans-serif !important;
    }
    .stSelectbox > div > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0 0.8rem;'>
            <div style='font-size:2rem;'>⚡</div>
            <div style='font-family:Bricolage Grotesque,sans-serif; font-weight:800; font-size:1rem; color:#f97316;'>Groq Multi-Agent</div>
        </div>
        """, unsafe_allow_html=True)

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com"
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#4b5675;
                    text-transform:uppercase; letter-spacing:0.5px; margin:0.8rem 0 0.4rem;'>
            Model
        </div>
        <div style='background:#141824; border:1px solid #1a2030; border-radius:8px;
                    padding:0.6rem 0.8rem; font-family:IBM Plex Mono,monospace;
                    font-size:0.78rem; color:#f97316;'>
            llama-3.3-70b-versatile
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#4b5675;
                    text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.6rem;'>
            Agent Pipeline
        </div>

        <div class='agent-card' style='border-left:3px solid #f97316;'>
            <div class='agent-title' style='color:#f97316;'>🔍 Research Agent</div>
            <div class='agent-desc'>DuckDuckGo web search → LLM consolidation into structured research notes</div>
        </div>
        <div class='agent-card' style='border-left:3px solid #8b5cf6;'>
            <div class='agent-title' style='color:#8b5cf6;'>📊 Analysis Agent</div>
            <div class='agent-desc'>Extracts key findings, themes, insights, gaps & contradictions</div>
        </div>
        <div class='agent-card' style='border-left:3px solid #06b6d4;'>
            <div class='agent-title' style='color:#06b6d4;'>✍️ Writer Agent</div>
            <div class='agent-desc'>Transforms analysis into a polished, structured markdown report</div>
        </div>
        <div class='agent-card' style='border-left:3px solid #22c55e;'>
            <div class='agent-title' style='color:#22c55e;'>✅ Reviewer Agent</div>
            <div class='agent-desc'>Reviews quality, scores 1–10, and provides improvement feedback</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#4b5675; margin-bottom:0.4rem;'>
            Get Groq API key free at:
        </div>
        <div style='font-family:IBM Plex Mono,monospace; font-size:0.75rem; color:#f97316;'>
            console.groq.com
        </div>
        """, unsafe_allow_html=True)

    # ── Main ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero'>
        <h1>Sequential Multi-Agent System</h1>
        <p>⚡ LangGraph + Groq (Llama 3.3 70B) · Research → Analysis → Writing → Review</p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline flow diagram
    st.markdown("""
    <div class='flow-bar'>
        <div class='flow-node' style='border-color:#f97316; color:#f97316; background:#f9731611;'>🔍 Research</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-node' style='border-color:#8b5cf6; color:#8b5cf6; background:#8b5cf611;'>📊 Analysis</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-node' style='border-color:#06b6d4; color:#06b6d4; background:#06b6d411;'>✍️ Writer</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-node' style='border-color:#22c55e; color:#22c55e; background:#22c55e11;'>✅ Reviewer</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-node' style='border-color:#4b5675; color:#4b5675;'>📋 Report</div>
    </div>
    """, unsafe_allow_html=True)

    # Example query selector
    examples = [
        "What is the current state of AI in healthcare in 2024?",
        "How does quantum computing threaten current encryption methods?",
        "What are the latest breakthroughs in renewable energy storage?",
        "How is climate change affecting global food security?",
        "What is the economic impact of generative AI on the job market?",
    ]
    col_ex, col_space = st.columns([3, 1])
    with col_ex:
        selected = st.selectbox(
            "📌 Example queries",
            ["— type your own query below, or pick an example —"] + examples,
            label_visibility="collapsed"
        )

    query = st.text_area(
        "Research Query",
        value=selected if "pick an example" not in selected else "",
        placeholder="Enter any research question — the agent pipeline will research, analyse, write, and review it automatically...",
        height=90,
        label_visibility="collapsed"
    )

    run = st.button("⚡ Run Multi-Agent Pipeline", use_container_width=True)

    # ── Handle run ────────────────────────────────────────────────────────────
    if run:
        if not os.environ.get("GROQ_API_KEY"):
            st.warning("⚠️ Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
        elif not query.strip():
            st.warning("⚠️ Please enter a research query.")
        else:
            with st.status("⚡ Sequential Multi-Agent Pipeline Running...", expanded=True) as status:
                st.write("🔍 **Agent 1 — Research:** Searching the web with DuckDuckGo...")
                try:
                    result = run_pipeline(query.strip())

                    st.write("📊 **Agent 2 — Analysis:** Extracting findings and insights...")
                    st.write("✍️  **Agent 3 — Writer:** Composing the report...")
                    st.write("✅ **Agent 4 — Reviewer:** Evaluating quality...")
                    status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

                    # Score display
                    score = result.get("quality_score", 0)
                    score_color = "#22c55e" if score >= 7 else "#f97316" if score >= 5 else "#ef4444"
                    score_bg = f"{score_color}15"

                    # Execution log
                    st.markdown("### 📋 Execution Log")
                    log_html = "".join([
                        f"<div class='log-item'>{log}</div>"
                        for log in result["agent_logs"]
                    ])
                    st.markdown(
                        f"<div style='background:#0e1117; border:1px solid #1a2030; border-radius:10px; padding:0.8rem 1rem;'>{log_html}</div>",
                        unsafe_allow_html=True
                    )

                    # Quality score
                    st.markdown(f"""
                    <div class='score-box' style='background:{score_bg}; border:1px solid {score_color}44;'>
                        <div class='score-label'>Quality Score</div>
                        <div class='score-num' style='color:{score_color};'>{score}<span style='font-size:1.5rem; opacity:0.5;'>/10</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Results tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📄 Final Report",
                        "🔍 Research Notes",
                        "📊 Analysis",
                        "✅ Review"
                    ])

                    with tab1:
                        st.markdown(result.get("final_report", ""))

                    with tab2:
                        st.text_area(
                            "Raw Research Notes",
                            value=result.get("raw_research", ""),
                            height=450,
                            label_visibility="collapsed"
                        )

                    with tab3:
                        st.text_area(
                            "Analysis",
                            value=result.get("analysis", ""),
                            height=450,
                            label_visibility="collapsed"
                        )

                    with tab4:
                        st.text_area(
                            "Review",
                            value=result.get("review", ""),
                            height=350,
                            label_visibility="collapsed"
                        )

                except Exception as e:
                    status.update(label="❌ Pipeline Failed", state="error")
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure your GROQ_API_KEY is valid and all packages are installed.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_app()
