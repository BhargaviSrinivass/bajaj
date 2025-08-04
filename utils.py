import os
os.environ["USE_TF"] = "0"  # Force transformers to use PyTorch backend

import requests
import pdfplumber
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Local embedding model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# HuggingFace local LLM for answer generation (remove max_length here)
qa_model = pipeline("text-generation", model="distilgpt2")

def download_file(url, out_path='data/input.pdf'):
    r = requests.get(url)
    r.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def parse_pdf(pdf_path):
    clauses = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                for clause in text.split('\n'):
                    if clause.strip():
                        clauses.append({
                            "text": clause.strip(),
                            "location": f"Page {page_num+1}"
                        })
    return clauses

def get_embedding(text):
    return sentence_model.encode(text).tolist()

def init_pinecone(index_name='policy-index'):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")    # e.g., 'us-east-1'
    pc = Pinecone(api_key=pinecone_api_key)
    index_names = pc.list_indexes().names()
    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=384,   # all-MiniLM-L6-v2 outputs 384-d vectors
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=pinecone_env
            )
        )
    return pc.Index(index_name)

def upsert_clauses(clauses, index):
    items = []
    for i, clause in enumerate(clauses):
        emb = get_embedding(clause["text"])
        items.append(
            {
                "id": f"clause-{i}",
                "values": emb,
                "metadata": {"text": clause["text"], "location": clause["location"]}
            }
        )
    index.upsert(items)

def semantic_search(query, index, top_k=5):
    query_emb = get_embedding(query)
    result = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )
    return [match['metadata'] for match in result['matches']]

def get_llm_answer(query: str, clauses: list):
    context = "\n".join([f"[{c.get('location','')}] {c['text']}" for c in clauses])
    prompt = f"Answer as a helpful assistant. Question: {query}\nRelevant clauses:\n{context}\nAnswer:"
    output = qa_model(prompt, max_new_tokens=50)[0]['generated_text']  # Use max_new_tokens here!
    answer = output.replace(prompt, "").strip()
    return answer
