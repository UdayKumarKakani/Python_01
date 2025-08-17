# utils/chat_ui.py

import streamlit as st
from utils.retriever import retrieve_context
from utils.generator import generate_answer


def render_bot_ui(bot_id: str, title: str, index_name: str):
    """
    Renders a chatbot UI with consistent behavior:
    - Input form with auto-clear
    - Chat history
    - Status messaging

    Parameters:
    - bot_id: short identifier (e.g. 'ds', 'legal')
    - title: UI title shown on top
    - index_name: Pinecone index to use for context retrieval
    """
    st.header(title)

    # Auto-clear input handling
    if st.session_state.get(f"{bot_id}_clear_input", False):
        st.session_state.pop(f"{bot_id}_query", None)
        st.session_state[f"{bot_id}_clear_input"] = False

    # Text input with form
    with st.form(f"{bot_id}_form"):
        query = st.text_input("Ask a question:", key=f"{bot_id}_query")
        submitted = st.form_submit_button("Submit")

    if submitted and query:
        with st.spinner("Thinking..."):
            context = retrieve_context(index_name, query)
            if context.strip() == "":
                answer = "I don’t have enough information in my knowledge base to answer that."
            else:
                answer = generate_answer(context, query)

            # Append to chat history
            history_key = f"{bot_id}_chat_history"
            if history_key not in st.session_state:
                st.session_state[history_key] = []
            st.session_state[history_key].append((query, answer))

            # Trigger input clear + rerun
            st.session_state[f"{bot_id}_clear_input"] = True
            st.rerun()

    # Display chat history
    history_key = f"{bot_id}_chat_history"
    if history_key in st.session_state:
        for q, a in reversed(st.session_state[history_key]):
            with st.expander(f"🧠 You: {q}", expanded=False):
                st.markdown(f"**🤖 Bot:** {a}")
