import streamlit as st
from config import load_env

# Import bots
from bots import bot_datascience,bot_Legal

# Load environment
load_env()

st.set_page_config(page_title="🧠 Multi-Chatbot Hub", page_icon="🤖", layout="wide")
st.sidebar.title("🤖 Choose Your Assistant")

# Sidebar selection
bot_choice = st.sidebar.radio("Select Bot", ["DataScience Assistant", "Legal Assistant", "Tech Support"])

# Render the selected bot dynamically
if bot_choice == "DataScience Assistant":
    bot_datascience.run()
elif bot_choice == "Legal Assistant":
    bot_Legal.run()
elif bot_choice == "Tech Support":
    bot_Legal.run()
