import os
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_core.prompts import PromptTemplate
import pinecone
from pinecone import Pinecone as PC

# Load environment
load_dotenv()
os.environ["TOGETHER_API_KEY"] = "tgp_v1_Gdl66OKThh1KsJjEym9JEgDMqFWqd6bXtlZhviYqf34"
os.environ["PINECONE_API_KEY"] = "pcsk_3875g1_PSfiVC6hgEBa7mPwUMFf6dbhmZa68JiueGaf5eSYDwKoyt8JABHRYsirkcLfRnm"

# Constants
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")


# === RETRIEVER ===
def retrieve_context(index_name: str, query_text: str):
    pc = PC(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    embedding = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        api_key=os.getenv("TOGETHER_API_KEY")
    )

    query_embedding = embedding.embed_documents([query_text])[0]

    search_response = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Concatenate metadata/documents as raw context
    context = "\n\n".join(
        doc["metadata"].get("text", "No content found.")
        for doc in search_response["matches"]
    )
    return context


# === GENERATOR ===
def generate_answer(context: str, question: str) -> str:
    llm = ChatTogether(
        together_api_key=os.environ["TOGETHER_API_KEY"],
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.3,
        max_tokens=512
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the provided context to answer the question as accurately as possible.
If the context is empty or not relevant, respond:
I’m sorry, I don’t have enough information in my knowledge base to answer that.

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = prompt | llm

    result = chain.invoke({"context": context, "question": question})
    return result.content
