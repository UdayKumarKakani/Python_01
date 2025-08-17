from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service import retrieve_context, generate_answer, INDEX_NAME

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"message": "Together AI + Pinecone Chatbot is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        INDEX_NAME = "langchain-embeddings"
        context = retrieve_context(INDEX_NAME, request.question)
        answer = generate_answer(context, request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)