import os
from typing import TypedDict, Optional, List, Dict, Any

from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage,BaseMessage
from langchain.agents import tool, create_tool_calling_agent
from langgraph.graph import StateGraph, END
from pinecone import Pinecone


# === ENV Setup ===
os.environ["TOGETHER_API_KEY"] = "tgp_v1_Gdl66OKThh1KsJjEym9JEgDMqFWqd6bXtlZhviYqf34"
os.environ["PINECONE_API_KEY"] = "pcsk_3875g1_PSfiVC6hgEBa7mPwUMFf6dbhmZa68JiueGaf5eSYDwKoyt8JABHRYsirkcLfRnm"
os.environ["PINECONE_ENVIRONMENT"] = "gcp-starter"  # Replace with your Pinecone environment


# === Define State ===
class AgentState(TypedDict):
    input: str
    agent_response: Optional[AIMessage]
    intermediate_steps: List
    tool_responses: List
    context_chunks: Optional[str]


# === Tools ===
@tool
def search_web(query: str) -> str:
    """Search the web for recent or factual information."""
    return DuckDuckGoSearchRun().run(query)


@tool
def search_vectorstore(query: str) -> str:
    """Search internal knowledge base using Pinecone and return most relevant chunks."""
    embedding = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
    query_vector = embedding.embed_documents([query])[0]

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("langchain-embeddings")

    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    matches = results.get('matches', [])
    docs = [m['metadata']['text'] for m in matches if 'text' in m.get('metadata', {})]

    return "\n\n".join(docs) if docs else "No relevant context found."


tools = [search_web, search_vectorstore]


# === Define LLM ===
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    together_api_key=os.environ["TOGETHER_API_KEY"],
    temperature=0.3,
    max_tokens=512,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant."),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}")
])

agent = create_tool_calling_agent(llm, tools, prompt)


# === LangGraph Nodes ===
def run_agent(state: AgentState):
    response = agent.invoke({
        "input": state["input"],
        "intermediate_steps": state["intermediate_steps"],
    })

    # Ensure it's a single message
    if isinstance(response, list):
        response = response[0]

    return {"agent_response": response}


def run_tools(state: AgentState):
    agent_msg = state["agent_response"]
    tool_calls = getattr(agent_msg, "tool_calls", [])

    if not tool_calls:
        return {"tool_responses": [], "intermediate_steps": state["intermediate_steps"]}

    tool_outputs = []
    for call in tool_calls:
        for tool in tools:
            if call["name"] == tool.name:
                output = tool.invoke(call["args"])
                tool_outputs.append((call, output))

    return {
        "tool_responses": tool_outputs,
        "intermediate_steps": state["intermediate_steps"] + tool_outputs
    }


# === LangGraph Structure ===
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("tool_runner", run_tools)

workflow.set_entry_point("agent")
workflow.add_edge("agent", "tool_runner")
workflow.add_conditional_edges(
    "tool_runner",
    lambda state: "agent" if state["tool_responses"] else END,
    {"agent": "agent", END: END}
)

graph_executor = workflow.compile()


# === CLI Runner ===
if __name__ == "__main__":
    print("🤖 LangGraph Chatbot Ready (type 'exit' to quit)")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        result = graph_executor.invoke({
            "input": query,
            "agent_response": None,
            "intermediate_steps": [],
            "tool_responses": [],
            "context_chunks": None,
        })

        final = result.get("agent_response")
        if isinstance(final, BaseMessage):
            print(f"\n🧠 Bot: {final.content}")
        else:
            print(f"\n🧠 Bot intermediate step: {final}")
