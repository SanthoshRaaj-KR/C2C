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
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Load .env variables
# --------------------------------------------------------------------------
load_dotenv()

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
    search_by_id: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    relevant_chunks: List[Dict[str, Any]]
    processing_time: float
    total_results: int

class DirectIngestRequest(BaseModel):
    chunks: List[Dict[str, Any]]

class ChunkByIdRequest(BaseModel):
    chunk_ids: List[str]

class BatchQueryRequest(BaseModel):
    queries: List[str]
    max_results_per_query: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7

# --------------------------------------------------------------------------
# Vector Database Manager
# --------------------------------------------------------------------------
class VectorDBManager:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
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
        """
        FIXED: Creates a descriptive paragraph for each chunk to improve semantic search.
        """
        file_path = chunk.get('file_path', 'an unknown file')
        language = chunk.get('language', 'an unspecified language')
        chunk_type = chunk.get('type', 'code segment')
        
        description = f"This is a '{chunk_type}' from the file '{file_path}', written in {language}. "
        
        if chunk.get('class_name'):
            description += f"It is part of the class '{chunk['class_name']}'. "
        if chunk.get('function_name'):
            description += f"Specifically, it is within the function '{chunk['function_name']}'. "
            
        description += f"The content is as follows:\n\n```\n{chunk.get('code', '')}\n```"
        return description

    def search_similar_chunks(self, query: str, max_results: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
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
                distance = results['distances'][0][i]
                similarity_score = 1 - distance
                
                # Apply similarity threshold
                if similarity_score < similarity_threshold:
                    continue
                    
                meta = results['metadatas'][0][i]
                formatted_results.append({
                    'document': doc,
                    'metadata': meta,
                    'similarity_score': similarity_score,
                    'chunk_info': {
                        'file': meta.get('file_path', ''),
                        'function': meta.get('function_name', ''),
                        'class': meta.get('class_name', ''),
                        'type': meta.get('type', ''),
                        'language': meta.get('language', ''),
                        'id': meta.get('id', '')
                    }
                })
            return formatted_results
        except Exception as e:
            print(f"❌ Error searching chunks: {e}")
            return []

    def search_by_id(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Search for a chunk by its exact ID"""
        collection = self._get_collection()
        
        if collection.count() == 0:
            return []

        try:
            # Try to get by exact ID first
            try:
                results = collection.get(ids=[chunk_id], include=['documents', 'metadatas'])
                if results['documents']:
                    doc = results['documents'][0]
                    meta = results['metadatas'][0]
                    return [{
                        'document': doc,
                        'metadata': meta,
                        'similarity_score': 1.0,  # Perfect match
                        'chunk_info': {
                            'file': meta.get('file_path', ''),
                            'function': meta.get('function_name', ''),
                            'class': meta.get('class_name', ''),
                            'type': meta.get('type', ''),
                            'language': meta.get('language', ''),
                            'id': meta.get('id', '')
                        }
                    }]
            except:
                pass
            
            # If exact ID search fails, try semantic search with the ID as query
            return self.search_similar_chunks(chunk_id, max_results=3, similarity_threshold=0.3)
            
        except Exception as e:
            print(f"❌ Error searching by ID {chunk_id}: {e}")
            return []

    def search_multiple_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Search for multiple chunks by their IDs"""
        all_results = []
        for chunk_id in chunk_ids:
            results = self.search_by_id(chunk_id)
            all_results.extend(results)
        return all_results

    def batch_search(self, queries: List[str], max_results_per_query: int = 5, similarity_threshold: float = 0.7) -> Dict[str, List[Dict[str, Any]]]:
        """Perform multiple searches in batch"""
        results = {}
        for query in queries:
            results[query] = self.search_similar_chunks(query, max_results_per_query, similarity_threshold)
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection = self._get_collection()
            total_count = collection.count()
            
            # Get sample of metadata to understand the data
            sample_data = collection.peek(limit=min(100, total_count))
            languages = set()
            types = set()
            files = set()
            
            for meta in sample_data.get('metadatas', []):
                if meta.get('language'):
                    languages.add(meta['language'])
                if meta.get('type'):
                    types.add(meta['type'])
                if meta.get('file_path'):
                    files.add(meta['file_path'])
            
            return {
                'total_chunks': total_count,
                'languages': list(languages),
                'types': list(types),
                'sample_files': list(files)[:10],  # First 10 files
                'collection_name': COLLECTION_NAME
            }
        except Exception as e:
            return {'error': str(e)}

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
app = FastAPI(title="VectorDB Service", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    INPUT_JSON_DIR.mkdir(exist_ok=True)
    CHROMA_DB_PATH.mkdir(exist_ok=True)
    get_vector_db()
    update_processing_status('ready', {'message': 'Service ready for ingestion.'})

@app.get("/")
async def root():
    return {
        "service": "VectorDB Service",
        "version": "2.0.0",
        "status": "running",
        "collection_name": COLLECTION_NAME
    }

# --------------------------------------------------------------------------
# Query Endpoints (for Orchestrator)
# --------------------------------------------------------------------------
@app.post("/query")
async def query_chunks(request: QueryRequest) -> QueryResponse:
    """Main query endpoint for orchestrator"""
    start_time = time.time()
    
    try:
        db = get_vector_db()
        
        if request.search_by_id:
            # Search by chunk ID
            relevant_chunks = db.search_by_id(request.query)
        else:
            # Normal similarity search
            relevant_chunks = db.search_similar_chunks(
                request.query,
                request.max_results,
                request.similarity_threshold
            )
        
        return QueryResponse(
            query=request.query,
            relevant_chunks=relevant_chunks,
            processing_time=round(time.time() - start_time, 3),
            total_results=len(relevant_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/query/batch")
async def batch_query(request: BatchQueryRequest):
    """Batch query endpoint for multiple queries"""
    start_time = time.time()
    
    try:
        db = get_vector_db()
        results = db.batch_search(
            request.queries,
            request.max_results_per_query,
            request.similarity_threshold
        )
        
        return {
            "batch_results": results,
            "processing_time": round(time.time() - start_time, 3),
            "total_queries": len(request.queries)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch query failed: {str(e)}")

@app.post("/query/by_ids")
async def query_by_ids(request: ChunkByIdRequest):
    """Query multiple chunks by their IDs"""
    start_time = time.time()
    
    try:
        db = get_vector_db()
        results = db.search_multiple_ids(request.chunk_ids)
        
        return {
            "chunks": results,
            "processing_time": round(time.time() - start_time, 3),
            "requested_ids": request.chunk_ids,
            "found_chunks": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ID query failed: {str(e)}")

@app.post("/query/filtered")
async def filtered_query(request: dict):
    """Query with metadata filters"""
    start_time = time.time()
    
    try:
        db = get_vector_db()
        query = request.get("query", "")
        filters = request.get("filters", {})
        max_results = request.get("max_results", 5)
        
        # For now, just do regular search - can be enhanced later
        results = db.search_similar_chunks(query, max_results)
        
        return {
            "query": query,
            "filters": filters,
            "results": results,
            "processing_time": round(time.time() - start_time, 3),
            "total_results": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filtered query failed: {str(e)}")

# --------------------------------------------------------------------------
# Management Endpoints
# --------------------------------------------------------------------------
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
                    # Add the ID to metadata for easier retrieval
                    metadata['id'] = chunk_id
                    
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

@app.get("/collection/info")
async def get_collection_info():
    db = get_vector_db()
    collection = db._get_collection()
    return {"collection_name": COLLECTION_NAME, "total_chunks": collection.count()}

@app.get("/collection/stats")
async def get_collection_stats():
    """Get collection statistics"""
    try:
        db = get_vector_db()
        stats = db.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.delete("/collection")
async def clear_collection():
    db = get_vector_db()
    try:
        db.chroma_client.delete_collection(name=COLLECTION_NAME)
        update_processing_status('cleared', {'message': 'Collection cleared.'})
        return {"message": "Collection cleared successfully"}
    except Exception:
        return {"message": "Collection did not exist or could not be cleared."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = get_vector_db()
        collection = db._get_collection()
        count = collection.count()
        return {
            "status": "healthy",
            "collection_exists": True,
            "total_chunks": count,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "collection_exists": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)