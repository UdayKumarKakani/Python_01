from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
import os

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

Context: {context}
Question: {question}

Answer:"""
    )
    return (prompt | llm).invoke({"context": context, "question": question}).content
