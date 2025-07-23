from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import query_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],  # Or ["POST"] specifically
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    return query_rag(request.question)
