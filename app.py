from fastapi import FastAPI
from pydantic import BaseModel
from retriever import build_faiss_index, query_rag  # Your existing functions

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    answer = query_rag(request.question)
    return {"answer": answer}
