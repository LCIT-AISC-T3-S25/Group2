from fastapi import FastAPI, Request
from pydantic import BaseModel
from .rag_model import rag_answer

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/chat")
def get_answer(q: Query):
    answer = rag_answer(q.query)
    return {"response": answer}
