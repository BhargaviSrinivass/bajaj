from fastapi import FastAPI, Header, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from models import QueryRequest, QueryResponse
from utils import download_file, parse_pdf, init_pinecone, upsert_clauses, semantic_search, get_llm_answer

app = FastAPI()

# Token for Authorization matching sample in your problem statement
CORRECT_TOKEN = "Bearer 1523a0e0ff64599779aae9611b2f4b33cc3e1b346a23b082b437e84e28c73151"

# Add FastAPI security for Authorization header, so you get Swagger's Authorize button
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def check_token(api_key: str = Security(api_key_header)):
    if api_key != CORRECT_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(
    payload: QueryRequest,
    api_key: str = Depends(check_token)
):
    # 1. Download and parse document
    pdf_path = download_file(payload.documents)
    clauses = parse_pdf(pdf_path)
    # 2. Pinecone: upsert and search
    pinecone_index = init_pinecone()
    upsert_clauses(clauses, pinecone_index)
    answers = []
    for question in payload.questions:
        top_clauses = semantic_search(question, pinecone_index)
        answer_text = get_llm_answer(question, top_clauses)
        # For demonstration: rationale is part of answer (could parse/extract with LLM or prompt engineering)
        answers.append({
            "answer": answer_text,
            "supporting_clauses": top_clauses,
            "rationale": answer_text  # Instruct LLM to include rationale in answer string above. Refine as needed.
        })
    return {"answers": answers}
