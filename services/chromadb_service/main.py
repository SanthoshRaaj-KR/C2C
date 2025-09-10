import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Load .env variables
# --------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ Warning: GROQ_API_KEY not set in environment variables")
else:
    print("✅ GROQ_API_KEY loaded successfully")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
INPUT_JSON_DIR = ROOT_DIR / "input_json"
CHROMA_DB_PATH = ROOT_DIR / "chroma_db"
PROCESSING_STATUS_FILE = ROOT_DIR / "vector_processing_status.json"
COLLECTION_NAME = "code_chunks"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7

class QueryResponse(BaseModel):
    query: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    processing_time: float

class DirectIngestRequest(BaseModel):
    chunks: List[Dict[str, Any]]

# --------------------------------------------------------------------------
# Vector Database Manager
# --------------------------------------------------------------------------
class VectorDBManager:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.groq_client = None
        self.setup_clients()

    def setup_clients(self):
        print("🔧 Initializing Vector DB Manager...")

        try:
            print("📥 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise

        try:
            print("🗄️ Initializing ChromaDB Client...")
            CHROMA_DB_PATH.mkdir(exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            print("✅ ChromaDB client initialized.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise

        if GROQ_API_KEY:
            try:
                self.groq_client = groq.Groq(api_key=GROQ_API_KEY)
                print("✅ Groq client initialized")
            except Exception as e:
                print(f"❌ Error initializing Groq client: {e}")
                self.groq_client = None
        else:
            print("⚠️ Groq client not initialized - API key missing")

    def _get_collection(self):
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            return self.collection
        except Exception:
            raise HTTPException(
                status_code=404,
                detail="Collection not found. Please ingest chunks first."
            )

    def generate_chunk_id(self, chunk: Dict[str, Any]) -> str:
        content = f"{chunk.get('file_path', '')}{chunk.get('function_name', '')}{chunk.get('class_name', '')}{chunk.get('chunk_no', 1)}"
        return hashlib.md5(content.encode()).hexdigest()

    def prepare_chunk_text(self, chunk: Dict[str, Any]) -> str:
        parts = []
        if chunk.get('file_path'): parts.append(f"File: {chunk['file_path']}")
        if chunk.get('language'): parts.append(f"Language: {chunk['language']}")
        if chunk.get('class_name'): parts.append(f"Class: {chunk['class_name']}")
        if chunk.get('function_name'): parts.append(f"Function: {chunk['function_name']}")
        if chunk.get('type'): parts.append(f"Type: {chunk['type']}")
        if chunk.get('code'): parts.append(f"Code:\n{chunk['code']}")
        return "\n".join(parts)

    def search_similar_chunks(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )

            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                formatted_results.append({
                    'document': doc,
                    'metadata': meta,
                    'similarity_score': 1 - results['distances'][0][i],
                    'chunk_info': {
                        'file': meta.get('file_path', ''),
                        'function': meta.get('function_name', ''),
                        'class': meta.get('class_name', ''),
                        'type': meta.get('type', ''),
                        'language': meta.get('language', '')
                    }
                })
            return formatted_results
        except Exception as e:
            print(f"❌ Error searching chunks: {e}")
            return []

    def generate_answer_with_groq(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        if not self.groq_client:
            return "Groq client not available. Please set GROQ_API_KEY."
        if not relevant_chunks:
            return "No relevant code chunks found."

        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:3]):
            ci = chunk['chunk_info']
            context_parts.append(f"""
Code Chunk {i+1}:
File: {ci['file']}
Language: {ci['language']}
Function/Class: {ci['function'] or ci['class'] or 'N/A'}
Similarity: {chunk['similarity_score']:.3f}
Code:
{chunk['document']}
""")
        context = "\n" + "="*50 + "\n".join(context_parts) + "="*50
        prompt = f"""You are a helpful coding assistant.
User Question: {query}
Relevant Code Context:{context}
Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error generating answer with Groq: {e}")
            return f"Error generating answer: {str(e)}"

# --------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------
vector_db = None

def get_vector_db():
    global vector_db
    if vector_db is None:
        vector_db = VectorDBManager()
    return vector_db

def update_processing_status(status: str, details: Dict[str, Any] = None):
    with open(PROCESSING_STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'status': status, 'timestamp': time.time(), 'details': details or {}}, f, indent=2)

def get_processing_status() -> Dict[str, Any]:
    if PROCESSING_STATUS_FILE.exists():
        with open(PROCESSING_STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'status': 'idle', 'timestamp': 0, 'details': {}}

# --------------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------------
app = FastAPI(title="RAG Code Assistant", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    INPUT_JSON_DIR.mkdir(exist_ok=True)
    CHROMA_DB_PATH.mkdir(exist_ok=True)
    get_vector_db()
    update_processing_status('ready', {'message': 'Service ready for ingestion.'})

@app.get("/")
async def root():
    db = get_vector_db()
    return {
        "service": "RAG Code Assistant",
        "version": "2.0.0",
        "status": "running",
        "groq_available": db.groq_client is not None
    }

@app.post("/vectordb/ingest")
async def ingest_chunks(request: DirectIngestRequest, background_tasks: BackgroundTasks):
    status = get_processing_status()
    if status.get('status') == 'processing':
        return JSONResponse(409, {"error": "Ingestion already in progress.", "status": status})

    chunks = request.chunks  # avoid closure issues

    def task():
        try:
            update_processing_status('processing', {'total_chunks': len(chunks)})
            db = get_vector_db()
            try:
                db.chroma_client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass
            db.collection = db.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Code chunks for RAG"}
            )

            processed_count, failed_count = 0, 0
            for chunk in chunks:
                try:
                    chunk_id = db.generate_chunk_id(chunk)
                    chunk_text = db.prepare_chunk_text(chunk)
                    embedding = db.embedding_model.encode(chunk_text).tolist()
                    metadata = {k: v for k, v in chunk.items() if k != "code" and v is not None}
                    db.collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk_text],
                        metadatas=[metadata]
                    )
                    processed_count += 1
                except Exception as e:
                    print(f"⚠️ Failed on chunk: {e}")
                    failed_count += 1

            result = {
                "status": "completed",
                "total_chunks": len(chunks),
                "processed_count": processed_count,
                "failed_count": failed_count,
                "collection_size": db.collection.count()
            }
            update_processing_status('completed', result)
        except Exception as e:
            update_processing_status('error', {'message': str(e)})

    background_tasks.add_task(task)
    return {"message": "Direct ingestion started.", "status": "initiated"}

@app.get("/status")
async def get_status():
    return get_processing_status()

@app.post("/query")
async def query_code(request: QueryRequest) -> QueryResponse:
    start_time = time.time()
    db = get_vector_db()
    relevant_chunks = db.search_similar_chunks(request.query, max_results=request.max_results)
    if request.similarity_threshold:
        relevant_chunks = [c for c in relevant_chunks if c['similarity_score'] >= request.similarity_threshold]
    answer = db.generate_answer_with_groq(request.query, relevant_chunks)
    return QueryResponse(
        query=request.query,
        answer=answer,
        relevant_chunks=relevant_chunks,
        processing_time=round(time.time() - start_time, 3)
    )

@app.get("/collection/info")
async def get_collection_info():
    db = get_vector_db()
    collection = db._get_collection()
    return {"collection_name": COLLECTION_NAME, "total_chunks": collection.count()}

@app.delete("/collection")
async def clear_collection():
    db = get_vector_db()
    try:
        db.chroma_client.delete_collection(name=COLLECTION_NAME)
        update_processing_status('cleared', {'message': 'Collection cleared.'})
        return {"message": "Collection cleared successfully"}
    except Exception:
        return {"message": "Collection did not exist or could not be cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # <<< pick the right port
