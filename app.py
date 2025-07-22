from fastapi import FastAPI
from pydantic import BaseModel
from retriever import build_faiss_index
from rag_chain import query_rag  # Your existing functions
import os
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    answer = query_rag(request.question)
    return {"answer": answer}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default 8000 locally
    uvicorn.run("app:app", host="0.0.0.0", port=port)
