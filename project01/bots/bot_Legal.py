from utils.chat_UI import render_bot_ui

def run():
    render_bot_ui(
        bot_id="legal",
        title="🧠 Legal Assistant",
        index_name="langchain-embeddings"
    )