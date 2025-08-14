import os
import streamlit as st
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_core.prompts import PromptTemplate
import pinecone
from pinecone import Pinecone

# === Load Environment Variables ===
def load_env():
    os.environ["TOGETHER_API_KEY"] = "tgp_v1_Gdl66OKThh1KsJjEym9JEgDMqFWqd6bXtlZhviYqf34"
    os.environ["PINECONE_API_KEY"] = "pcsk_3875g1_PSfiVC6hgEBa7mPwUMFf6dbhmZa68JiueGaf5eSYDwKoyt8JABHRYsirkcLfRnm"
    os.environ["PINECONE_ENVIRONMENT"] = "gcp-starter"  # Replace with your Pinecone environment



# === Initialize Retriever ===
def contextRetriever(index_name: str, query_text: str):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    embedding = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        api_key=os.environ["TOGETHER_API_KEY"]
    )

    query_embedding = embedding.embed_documents([query_text])[0]

    search_response = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    return search_response


# === Generate Answer ===
def generateAnswers(context: str, question: str):
    llm = ChatTogether(
        together_api_key=os.environ["TOGETHER_API_KEY"],
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.3,
        max_tokens=512
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the provided context to answer the question as accurately as possible.

Context: {context}
Question: {question}

Answer:"""
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# === Streamlit UI ===
def main():
    load_env()
    index_name = "langchain-embeddings"

    st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
    st.title("🔍 Semantic Search Chatbot")
    st.markdown("Ask a question based on the indexed documents.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Your question:", placeholder="Type your question and press Enter...")

    if query:
        with st.spinner("Searching context and generating response..."):
            pinecone_results = contextRetriever(index_name, query)

            # Extract readable context from metadata
            context_chunks = [match['metadata'].get('text', '') for match in pinecone_results.matches]
            context_text = "\n\n".join(context_chunks).strip()

            if not context_text:
                answer = "I’m sorry, I don’t have enough information in my knowledge base to answer that."
            else:
                answer = generateAnswers(context_text, query)

            # Store in session history
            st.session_state.chat_history.append((query, answer, context_text))

    # Show chat history
    for q, a, ctx in reversed(st.session_state.chat_history):
        with st.expander(f"Q: {q}", expanded=False):
            st.markdown(f"**Answer:** {a}")
            with st.expander("🔍 Context used"):
                st.markdown(ctx if ctx else "_No context retrieved._")


if __name__ == "__main__":
    main()
