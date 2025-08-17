from utils.chat_UI import render_bot_ui

def run():
    render_bot_ui(
        bot_id="general",
        title="🧠 Datascience Assistant",
        index_name="langchain-embeddings"
    )