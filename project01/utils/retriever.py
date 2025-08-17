from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
import os

def retrieve_context(index_name: str, query: str, top_k: int = 3) -> str:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    embedder = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        api_key=os.environ["TOGETHER_API_KEY"]
    )
    query_vector = embedder.embed_documents([query])[0]

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    chunks = [m["metadata"].get("text", "") for m in results.matches]
    return "\n\n".join(chunks)
