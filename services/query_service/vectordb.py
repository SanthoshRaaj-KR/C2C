import os
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
CHROMA_DB_PATH = ROOT_DIR / "query_chroma_db"
COLLECTION_NAME = "code_chunks"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class VectorQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7
    search_by_id: Optional[bool] = False  # New field for ID-based search

class VectorQueryResponse(BaseModel):
    query: str
    relevant_chunks: List[Dict[str, Any]]
    processing_time: float
    total_results: int

class BatchQueryRequest(BaseModel):
    queries: List[str]
    max_results_per_query: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7

class ChunkByIdRequest(BaseModel):
    chunk_ids: List[str]

# --------------------------------------------------------------------------
# Enhanced Vector Query Manager
# --------------------------------------------------------------------------
class VectorQueryManager:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.setup_clients()

    def setup_clients(self):
        """Initialize embedding model and ChromaDB client"""
        print("🔧 Initializing Vector Query Manager...")

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
        """Get the collection or raise error if not found"""
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            return self.collection
        except Exception:
            raise HTTPException(
                status_code=404,
                detail="Collection not found. Please run ingestion first."
            )

    def search_by_similarity(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity"""
        collection = self._get_collection()
        
        if collection.count() == 0:
            return []

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
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
            return self.search_by_similarity(chunk_id, max_results=3, similarity_threshold=0.3)
            
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

    def search_with_filters(self, query: str, filters: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
        """Search with additional metadata filters"""
        collection = self._get_collection()
        
        if collection.count() == 0:
            return []

        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build where clause for filtering
            where_clause = {}
            if filters.get('language'):
                where_clause['language'] = filters['language']
            if filters.get('file_path'):
                where_clause['file_path'] = {"$contains": filters['file_path']}
            if filters.get('type'):
                where_clause['type'] = filters['type']

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_clause if where_clause else None,
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
                        'language': meta.get('language', ''),
                        'id': meta.get('id', '')
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching with filters: {e}")
            return []

    def batch_search(self, queries: List[str], max_results_per_query: int = 5, similarity_threshold: float = 0.7) -> Dict[str, List[Dict[str, Any]]]:
        """Perform multiple searches in batch"""
        results = {}
        for query in queries:
            results[query] = self.search_by_similarity(query, max_results_per_query, similarity_threshold)
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
# FastAPI App
# --------------------------------------------------------------------------
app = FastAPI(title="Vector Query Service", version="1.0.0")

# Initialize the query manager
query_manager = VectorQueryManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    CHROMA_DB_PATH.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {
        "service": "Vector Query Service",
        "version": "1.0.0",
        "status": "running",
        "collection_name": COLLECTION_NAME
    }

@app.post("/query")
async def query_vectors(request: VectorQueryRequest) -> VectorQueryResponse:
    """Main vector query endpoint"""
    start_time = time.time()
    
    try:
        if request.search_by_id:
            # Search by chunk ID
            relevant_chunks = query_manager.search_by_id(request.query)
        else:
            # Normal similarity search
            relevant_chunks = query_manager.search_by_similarity(
                request.query,
                request.max_results,
                request.similarity_threshold
            )
        
        return VectorQueryResponse(
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
        results = query_manager.batch_search(
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
        results = query_manager.search_multiple_ids(request.chunk_ids)
        
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
        query = request.get("query", "")
        filters = request.get("filters", {})
        max_results = request.get("max_results", 5)
        
        results = query_manager.search_with_filters(query, filters, max_results)
        
        return {
            "query": query,
            "filters": filters,
            "results": results,
            "processing_time": round(time.time() - start_time, 3),
            "total_results": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filtered query failed: {str(e)}")

@app.get("/collection/stats")
async def get_collection_stats():
    """Get collection statistics"""
    try:
        stats = query_manager.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        collection = query_manager._get_collection()
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
    uvicorn.run(app, host="0.0.0.0", port=8004)