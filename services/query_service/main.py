import os
import json
import time
from typing import List, Dict, Any, Optional
import requests

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
    similarity_threshold: Optional[float] = 0.5

class QueryResponse(BaseModel):
    query: str
    preprocessed_query: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    neighbor_chunks: List[Dict[str, Any]]
    processing_time: float
    workflow_steps: List[str]

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
            print("✅ Groq client initialized")
        else:
            print("⚠️ Groq API key not found. LLM features will be disabled.")
    
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

    def _get_chunk_ids_from_vector_results(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extracts the 'id' field from the 'metadata' of each chunk from VectorDB results."""
        ids = []
        for chunk in chunks:
            chunk_id = chunk.get("metadata", {}).get("id")
            if chunk_id:
                ids.append(chunk_id)
        print(f"🔑 Extracted {len(ids)} chunk IDs from VectorDB results.")
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
            return "No relevant code chunks were found. The codebase may not contain the information, or the query could be rephrased for better results."

        context_parts = ["=== PRIMARY RELEVANT CODE ==="]
        for chunk in relevant_chunks:
            context_parts.append(chunk.get('document', ''))
        
        if neighbor_chunks:
            context_parts.append("\n=== RELATED/CONNECTED CODE (FROM CODE GRAPH) ===")
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
    
# In your QueryOrchestrator class, replace the whole function with this:

    def _transform_query_for_retrieval(self, user_query: str) -> str:
        """Uses an LLM to rewrite the user query for better semantic search retrieval."""
        if not self.groq_client:
            print("⚠️ Groq client not available, skipping query transformation.")
            return user_query

        prompt = f"""
        You are an expert software engineer who transforms user questions into descriptive paragraphs for a semantic search system.
        Your task is to rewrite the user's question as a rich, detailed paragraph that describes the ideal code chunk they are looking for.
        Do NOT use SQL, code syntax, or any other query language. The output must be a single block of natural language text.

        For example, if the user asks "how does file upload work", a good transformed query is:
        "A Python backend function that handles an HTTP file upload. It likely uses a library like FastAPI or Flask, receives a file as bytes, and includes logic for processing the image or saving the file to disk."

        User Question: "{user_query}"

        Transformed Paragraph:
        """
        
        try:
            print(f"🧠 Transforming query: '{user_query}'")
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0
            )
            transformed_query = response.choices[0].message.content.strip()
            print(f"✅ Transformed query: '{transformed_query}'")
            return transformed_query
        except Exception as e:
            print(f"❌ Error during query transformation: {e}")
            return user_query

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main orchestration logic with query transformation."""
        start_time = time.time()
        workflow_steps = []
        
        print(f"🚀 Processing query: '{request.query}'")
        
        query_for_retrieval = self._transform_query_for_retrieval(request.query)
        workflow_steps.append(f"Transformed query for better retrieval")
        
        relevant_chunks = self.query_vectordb(
            query_for_retrieval, 
            request.max_results, 
            request.similarity_threshold
        )
        workflow_steps.append(f"Found {len(relevant_chunks)} relevant chunks")
        
        neighbor_chunks = []
        if request.include_neighbors and relevant_chunks:
            chunk_ids = self._get_chunk_ids_from_vector_results(relevant_chunks)
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
            preprocessed_query=query_for_retrieval,
            answer=final_answer,
            relevant_chunks=relevant_chunks,
            neighbor_chunks=neighbor_chunks,
            processing_time=round(time.time() - start_time, 3),
            workflow_steps=workflow_steps
        )

    async def process_batch_queries(self, request: BatchQueryRequest) -> Dict[str, Any]:
        """Process multiple queries in batch"""
        # (Batch processing logic remains the same)
        pass

# --------------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------------
app = FastAPI(title="Query Orchestrator", version="2.0.0 (Optimized)")
orchestrator = QueryOrchestrator()

@app.get("/")
async def root():
    return {"service": "Query Orchestrator", "version": "2.0.0 (Optimized)", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    """Main query processing endpoint with LLM query transformation."""
    return await orchestrator.process_query(request)

# (Other endpoints like batch, debug, health, etc. remain the same)