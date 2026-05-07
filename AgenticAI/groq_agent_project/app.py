"""
app.py — Streamlit UI using GROQ
"""

import json
import os
import streamlit as st
from agent import run_agent


st.set_page_config(page_title="Groq AI Agent")

st.title("🚀 Groq AI Research Agent")

groq_key = st.sidebar.text_input(
    "Enter GROQ API Key",
    type="password"
)

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "tools_log" not in st.session_state:
    st.session_state.tools_log = []


user_input = st.text_input("Ask Anything")

if st.button("Send"):

    if not groq_key:
        st.warning("Please enter Groq API Key")

    else:

        response, updated_history, tools_used, tool_steps = run_agent(
            user_input,
            st.session_state.chat_history
        )

        st.session_state.chat_history = updated_history
        st.session_state.tools_log.append(tool_steps)


for msg in st.session_state.chat_history:

    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")

    else:
        st.markdown(f"**Assistant:** {msg['content']}")

if st.session_state.tools_log:
    st.markdown("---")
    st.subheader("Tool usage")

    for run_index, tool_steps in enumerate(st.session_state.tools_log, start=1):
        if not tool_steps:
            st.markdown(f"**Run {run_index}:** No tools used.")
            continue

        with st.expander(f"Run {run_index}: {len(tool_steps)} tool step(s)"):
            for step_index, step in enumerate(tool_steps, start=1):
                st.markdown(f"**{step_index}. {step['name']}**")
                args_text = json.dumps(step.get('args', {}), indent=2)
                st.code(args_text, language='json')
                st.markdown(f"**Output:**\n{step.get('output', 'No output returned')}\n")
