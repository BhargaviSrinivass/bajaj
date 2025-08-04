# LLM-Powered Intelligent Query–Retrieval System

## Setup

1. **Clone the repo.**
2. Create a `.env` file and add your Pinecone credentials:
    ```
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENVIRONMENT=us-east-1
    ```
3. **Install requirements:**
    ```
    pip install -r requirements.txt
    ```
4. **Start the API:**
    ```
    uvicorn main:app --reload
    ```
5. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and use the `/api/v1/hackrx/run` endpoint.

## Folder Structure

- **main.py** — FastAPI app
- **models.py** — Pydantic data models
- **utils.py** — Parsing, embeddings, Pinecone, and local LLM functions
- **.env** — Environment variables (**do not commit**)
- **requirements.txt** — Dependency list
- **data/** — Temp files

## Example Request Body

{
"documents": "https://www.dropbox.com/scl/fi/1ienabocwdylzur96ej1x/Arogya-Sanjeevani-Policy-CIN-U10200WB1906GOI001713-1-1.pdf?rlkey=c21w8qzkkdozvqbjqqezszfbr&st=79n1jb8y&dl=1",
"questions": [
"What is the grace period for payment of the premium?"
]
}

## Notes

- **Embedding model:** all-MiniLM-L6-v2 (runs locally, 384 dimensions)
- **Answer LLM:** distilgpt2 (runs locally, works for small/medium answers)
- **Pinecone index:** dimension 384, metric cosine
- **No OpenAI API needed!** (zero-cost, no external API limits)
- First run will download HuggingFace models; subsequent runs are fast.

## Example requirements.txt

fastapi
uvicorn
pinecone-client
sentence-transformers
transformers
torch
pdfplumber
python-dotenv

## Tips

- For best results, try questions that directly match your policy text.
- To improve answer quality, chunk the text into a few sentences per clause.
- Want a stronger LLM? Try `gpt2` or other HuggingFace models (check hardware requirements).

---

**Enjoy building and testing through `/docs`!**
