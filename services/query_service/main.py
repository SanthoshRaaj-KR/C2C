import os
import json
import time
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
VECTORDB_URL = "http://localhost:8001"
NEO4J_URL = "http://localhost:8002"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    include_neighbors: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.5 # Lowered for better raw query results

class QueryResponse(BaseModel):
    query: str
    preprocessed_query: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    neighbor_chunks: List[Dict[str, Any]]
    processing_time: float
    workflow_steps: List[str]

# FIX: Added back the missing model definition
class BatchQueryRequest(BaseModel):
    queries: List[str]
    max_results_per_query: Optional[int] = 3
    include_neighbors: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.5

# --------------------------------------------------------------------------
# Query Orchestrator Service
# --------------------------------------------------------------------------
class QueryOrchestrator:
    def __init__(self):
        self.groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        if self.groq_client:
            print("✅ Groq client initialized for answer generation")
        else:
            print("⚠️ Groq API key not found. Final answer generation will be disabled.")
    
    def query_vectordb(self, query: str, max_results: int, threshold: float) -> List[Dict[str, Any]]:
        """Query the vector database for relevant chunks"""
        try:
            payload = {"query": query, "max_results": max_results, "similarity_threshold": threshold}
            print(f"📡 Querying VectorDB with: {payload}")
            response = requests.post(f"{VECTORDB_URL}/query", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            chunks = data.get("relevant_chunks", [])
            print(f"✅ VectorDB returned {len(chunks)} relevant chunks.")
            return chunks
        except requests.RequestException as e:
            print(f"❌ Error querying VectorDB: {e}")
            return []

    def extract_chunk_ids(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extracts the 'id' field reliably from each chunk dictionary."""
        ids = [chunk['id'] for chunk in chunks if chunk.get('id')]
        print(f"🔑 Extracted {len(ids)} chunk IDs: {ids}")
        return ids

    def find_neighbors(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Find neighbors for given chunk IDs using Neo4j"""
        if not chunk_ids: return []
        try:
            payload = {"chunk_ids": chunk_ids, "search_types": ["downstream", "upstream"], "max_depth": 2, "limit": 5}
            print(f"📡 Querying Neo4j for neighbors of {len(chunk_ids)} IDs")
            response = requests.post(f"{NEO4J_URL}/neighbors/batch", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            batch_results = data.get("batch_results", {})
            
            all_neighbors = []
            for chunk_id, neighbors_dict in batch_results.items():
                for rel_type in ["downstream", "upstream"]:
                    for neighbor in neighbors_dict.get(rel_type, []):
                        all_neighbors.append({**neighbor, "relationship": "calls" if rel_type == "downstream" else "called_by", "source_chunk": chunk_id})
            
            print(f"✅ Neo4j returned {len(all_neighbors)} neighbor relationships.")
            return all_neighbors
        except requests.RequestException as e:
            print(f"❌ Error finding neighbors: {e}")
            return []

    def get_neighbor_code(self, neighbor_ids: List[str]) -> List[Dict[str, Any]]:
        """Get the actual code for neighbor chunks from VectorDB"""
        if not neighbor_ids: return []
        try:
            print(f"📡 Retrieving code for {len(neighbor_ids)} neighbors.")
            response = requests.post(f"{VECTORDB_URL}/query/by_ids", json={"chunk_ids": neighbor_ids}, timeout=30)
            response.raise_for_status()
            data = response.json()
            chunks = data.get("chunks", [])
            print(f"✅ Retrieved {len(chunks)} neighbor code snippets.")
            return chunks
        except requests.RequestException as e:
            print(f"❌ Error getting neighbor code: {e}")
            return []

    def generate_final_answer(self, user_query: str, relevant_chunks: List[Dict], neighbor_chunks: List[Dict]) -> str:
        """Generate final answer using all gathered context"""
        if not self.groq_client:
            return "Groq client not configured. Cannot generate a final answer."
        if not relevant_chunks:
            return "No relevant code chunks were found for your query. The codebase may not contain the information, or the query could be rephrased."

        context_parts = ["=== PRIMARY RELEVANT CODE ==="]
        for chunk in relevant_chunks:
            context_parts.append(chunk.get('document', ''))
        
        if neighbor_chunks:
            context_parts.append("\n=== RELATED/CONNECTED CODE ===")
            for chunk in neighbor_chunks:
                context_parts.append(chunk.get('document', ''))
        
        context = "\n\n---\n\n".join(context_parts)
        prompt = f"You are an expert code analyst. Answer the user's question based ONLY on the provided code context.\n\nUser Question: {user_query}\n\nCode Context:\n{context}\n\nAnswer:"
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error generating final answer: {e}")
            return f"Error during answer generation: {str(e)}"

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main orchestration logic for processing queries without LLM preprocessing."""
        start_time = time.time()
        workflow_steps = []
        
        print(f"🚀 Processing query directly: '{request.query}'")
        
        query_for_retrieval = request.query
        workflow_steps.append("Skipped LLM query preprocessing")
        
        relevant_chunks = self.query_vectordb(
            query_for_retrieval, 
            request.max_results, 
            request.similarity_threshold
        )
        workflow_steps.append(f"Found {len(relevant_chunks)} relevant chunks")
        
        neighbor_chunks = []
        if request.include_neighbors and relevant_chunks:
            chunk_ids = self.extract_chunk_ids(relevant_chunks)
            if chunk_ids:
                neighbors = self.find_neighbors(chunk_ids)
                workflow_steps.append(f"Found {len(neighbors)} neighbor relationships")
                
                neighbor_ids = list(set([n["id"] for n in neighbors if n.get("id")]))
                if neighbor_ids:
                    neighbor_chunks = self.get_neighbor_code(neighbor_ids)
                    workflow_steps.append(f"Retrieved code for {len(neighbor_chunks)} neighbors")
        
        final_answer = self.generate_final_answer(request.query, relevant_chunks, neighbor_chunks)
        workflow_steps.append("Generated final answer")
        
        return QueryResponse(
            query=request.query,
            preprocessed_query="N/A (Skipped)",
            answer=final_answer,
            relevant_chunks=relevant_chunks,
            neighbor_chunks=neighbor_chunks,
            processing_time=round(time.time() - start_time, 3),
            workflow_steps=workflow_steps
        )

    # FIX: Added back the missing batch processing method
    async def process_batch_queries(self, request: BatchQueryRequest) -> Dict[str, Any]:
        """Process multiple queries in batch"""
        start_time = time.time()
        print(f"🚀 Processing batch of {len(request.queries)} queries")
        
        batch_results = []
        for query in request.queries:
            query_request = QueryRequest(
                query=query,
                max_results=request.max_results_per_query,
                include_neighbors=request.include_neighbors,
                similarity_threshold=request.similarity_threshold
            )
            result = await self.process_query(query_request)
            batch_results.append({"query": query, "result": result.dict()})
        
        total_time = time.time() - start_time
        print(f"✅ Batch processing complete in {total_time:.3f}s")
        
        return {
            "batch_results": batch_results,
            "total_queries": len(request.queries),
            "total_processing_time": round(total_time, 3)
        }

# --------------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------------
app = FastAPI(title="Query Orchestrator", version="1.3.1 (No Preprocessing)")
orchestrator = QueryOrchestrator()

@app.get("/")
async def root():
    return {"service": "Query Orchestrator", "version": "1.3.1 (No Preprocessing)", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    """Main query processing endpoint (bypasses LLM preprocessing)."""
    return await orchestrator.process_query(request)

@app.post("/query/batch")
async def process_batch_queries(request: BatchQueryRequest):
    """Batch query processing endpoint"""
    return await orchestrator.process_batch_queries(request)


# --------------------------------------------------------------------------
# Debug Endpoints
# --------------------------------------------------------------------------
@app.get("/debug/chunk_structure")
async def debug_chunk_structure():
    """Debug endpoint to check chunk structure from VectorDB"""
    try:
        response = requests.post(
            f"{VECTORDB_URL}/query",
            json={"query": "function", "max_results": 1},
            timeout=10
        )
        
        if response.ok:
            data = response.json()
            chunks = data.get("relevant_chunks", [])
            if chunks:
                return {
                    "sample_chunk_keys": list(chunks[0].keys()),
                    "sample_chunk": chunks[0]
                }
        
        return {"error": "No chunks found or VectorDB unavailable"}
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

@app.post("/debug/extract_ids")
async def debug_extract_ids(request: dict):
    """Debug endpoint to test chunk ID extraction"""
    try:
        chunks = request.get("chunks", [])
        if not chunks:
            raise HTTPException(status_code=400, detail="Chunks array required")
        
        chunk_ids = orchestrator.extract_chunk_ids(chunks)
        
        return {
            "total_chunks": len(chunks),
            "extracted_ids": chunk_ids,
            "success_rate": f"{len(chunk_ids)}/{len(chunks)}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ID extraction failed: {str(e)}")

# --------------------------------------------------------------------------
# Direct Service Forwarding Endpoints
# --------------------------------------------------------------------------
@app.post("/vector/query")
async def forward_vector_query(request: dict):
    """Forward query directly to VectorDB service"""
    try:
        response = requests.post(f"{VECTORDB_URL}/query", json=request, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"VectorDB service unavailable: {str(e)}")

@app.post("/graph/query")
async def forward_graph_query(request: dict):
    """Forward query directly to Neo4j service"""
    try:
        response = requests.post(f"{NEO4J_URL}/query_graph", json=request, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Neo4j service unavailable: {str(e)}")

@app.get("/vector/stats")
async def get_vector_stats():
    """Get VectorDB statistics"""
    try:
        response = requests.get(f"{VECTORDB_URL}/collection/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"VectorDB service unavailable: {str(e)}")

@app.get("/graph/stats")
async def get_graph_stats():
    """Get Neo4j graph statistics"""
    try:
        response = requests.get(f"{NEO4J_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Neo4j service unavailable: {str(e)}")

# --------------------------------------------------------------------------
# Health Check Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check for all dependent services"""
    health_status = {
        "orchestrator": "healthy",
        "vectordb": "unknown",
        "neo4j": "unknown",
        "groq": "available" if orchestrator.groq_client else "unavailable"
    }
    
    # Check VectorDB
    try:
        response = requests.get(f"{VECTORDB_URL}/health", timeout=5)
        if response.ok:
            data = response.json()
            health_status["vectordb"] = {
                "status": "healthy",
                "total_chunks": data.get("total_chunks", 0)
            }
        else:
            health_status["vectordb"] = {"status": "unhealthy"}
    except:
        health_status["vectordb"] = {"status": "unreachable"}
    
    # Check Neo4j service
    try:
        response = requests.get(f"{NEO4J_URL}/health", timeout=5)
        if response.ok:
            data = response.json()
            health_status["neo4j"] = {
                "status": "healthy",
                "total_nodes": data.get("total_nodes", 0)
            }
        else:
            health_status["neo4j"] = {"status": "unhealthy"}
    except:
        health_status["neo4j"] = {"status": "unreachable"}
    
    return health_status

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service responses"""
    health_info = {
        "orchestrator": {
            "status": "healthy",
            "groq_available": orchestrator.groq_client is not None,
            "vectordb_url": VECTORDB_URL,
            "neo4j_url": NEO4J_URL
        },
        "services": {}
    }

    # Check VectorDB Service
    try:
        response = requests.get(f"{VECTORDB_URL}/health", timeout=5)
        response.raise_for_status()
        health_info["services"]["vectordb"] = {
            "status": "healthy",
            "details": response.json()
        }
    except requests.RequestException as e:
        health_info["services"]["vectordb"] = {
            "status": "unreachable",
            "error": str(e)
        }

    # Check Neo4j Service
    try:
        response = requests.get(f"{NEO4J_URL}/health", timeout=5)
        response.raise_for_status()
        health_info["services"]["neo4j"] = {
            "status": "healthy",
            "details": response.json()
        }
    except requests.RequestException as e:
        health_info["services"]["neo4j"] = {
            "status": "unreachable",
            "error": str(e)
        }
    
    return health_info