import os
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import pinecone
from pinecone import Pinecone, ServerlessSpec


# === Load Environment Variables ===
def load_env():
    # Set API keys
    os.environ["TOGETHER_API_KEY"] = "tgp_v1_Gdl66OKThh1KsJjEym9JEgDMqFWqd6bXtlZhviYqf34"
    os.environ["PINECONE_API_KEY"] = "pcsk_3875g1_PSfiVC6hgEBa7mPwUMFf6dbhmZa68JiueGaf5eSYDwKoyt8JABHRYsirkcLfRnm"
    os.environ["PINECONE_ENVIRONMENT"] = "gcp-starter"  # Replace with your Pinecone environment


# === Initialize LangChain VectorStore Retriever ===
def contextRetriever(index_name: str, query_text: str):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    index= pc.Index(index_name)

    embedding = TogetherEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    api_key=os.getenv("TOGETHER_API_KEY")
    )

    query_embedding = embedding.embed_documents([query_text])[0]

    # Query Pinecone for top-k similar vectors
    top_k = 3
    search_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return search_response

# === Generate Answers using LLM ===
def generateAnswers(context:str, question :str):

    llm = ChatTogether(
        together_api_key=os.environ["TOGETHER_API_KEY"],
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.3,
        max_tokens=512
    )

    
        # If the context is empty, missing, or does not contain information relevant to the question, respond with:
        # "I’m sorry, I don’t have enough information in my knowledge base to answer that."
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the provided context to answer the question as accurately as possible.
        If the context is empty, missing, or does not contain information relevant to the question, respond with:
        I’m sorry, I don’t have enough information in my knowledge base to answer that.


        Context: {context}
        Question: {question}

        Answer:"""
        )
    chain = prompt|llm

    data = chain.invoke({
        "context": context,
        "question": question
    })
   
    return data


# === Main Chat Loop ===
def main():
    load_env()

    index_name = "langchain-embeddings"
    print("🤖 Chatbot is ready. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        context = contextRetriever(index_name, query)
        results = generateAnswers(context,query)
        
        print(f"\nBot: {results.content}")


if __name__ == "__main__":
    main()
