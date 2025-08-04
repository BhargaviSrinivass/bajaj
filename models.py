from pydantic import BaseModel
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    documents: str  # Blob URL for PDF
    questions: List[str]

class Clause(BaseModel):
    text: str
    location: str

class QueryAnswer(BaseModel):
    answer: str
    supporting_clauses: List[Dict]
    rationale: str

class QueryResponse(BaseModel):
    answers: List[QueryAnswer]
