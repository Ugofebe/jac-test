# app.py
import os
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from jac_functions import answer_research_question
from langchain_openai import ChatOpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize FastAPI
app = FastAPI(title="RAG Jacmate API", version="1.0")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm_gpt = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

llm_groq = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=GROQ_API_KEY
        )


# Pydantic model for input
class QueryRequest(BaseModel):
    question: str

# Endpoint: Ask a research question
@app.post("/ask")
def ask_question(payload: QueryRequest):
    """
    Run a research query against the RAG system.
    """
    try:
        answer, sources = answer_research_question(
            payload.question, collection, embeddings, llm_gpt
        )
        return {
            "question": payload.question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/")
def home():
    return {"status": "ok", "message": "RAG API is running "}
