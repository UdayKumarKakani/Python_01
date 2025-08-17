import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")
