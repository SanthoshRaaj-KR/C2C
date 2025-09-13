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
    similarity_threshold: Optional[float] = 0.7

class QueryResponse(BaseModel):
    query: str
    preprocessed_query: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    neighbor_chunks: List[Dict[str, Any]]
    processing_time: float
    workflow_steps: List[str]

@dataclass
class WorkflowStep:
    step: str
    status: str
    data: Any = None
    processing_time: float = 0.0

# --------------------------------------------------------------------------
# Query Orchestrator Service
# --------------------------------------------------------------------------
class QueryOrchestrator:
    def __init__(self):
        self.groq_client = None
        self.setup_groq()
    
    def setup_groq(self):
        """Initialize Groq client for query preprocessing"""
        if GROQ_API_KEY:
            try:
                self.groq_client = groq.Groq(api_key=GROQ_API_KEY)
                print("✅ Groq client initialized for query orchestrator")
            except Exception as e:
                print(f"❌ Error initializing Groq client: {e}")
                self.groq_client = None
        else:
            print("⚠️ Groq API key not found")

    def preprocess_query(self, user_query: str) -> str:
        """
        Use LLM to preprocess and enhance the user query for better code search
        """
        if not self.groq_client:
            return user_query
        
        preprocessing_prompt = f"""
You are a code search assistant. Your job is to transform user queries into more specific, technical queries that will work better with code search systems.

User Query: "{user_query}"

Transform this query to:
1. Include relevant technical terms and programming concepts
2. Specify what type of code elements to look for (functions, classes, methods, etc.)
3. Add context about programming patterns or architectures if relevant
4. Make it more specific for code retrieval

Return ONLY the enhanced query, nothing else.

Examples:
User: "How does user authentication work?"
Enhanced: "user authentication login logout session management password validation security middleware functions classes"

User: "Show me database operations"
Enhanced: "database operations CRUD create read update delete SQL queries ORM models database connection functions"

Enhanced Query:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "user", "content": preprocessing_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            enhanced_query = response.choices[0].message.content.strip()
            print(f"🔍 Query enhanced: '{user_query}' → '{enhanced_query}'")
            return enhanced_query
        except Exception as e:
            print(f"❌ Error preprocessing query: {e}")
            return user_query

    def query_vectordb(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Query the vector database for relevant chunks"""
        try:
            response = requests.post(
                f"{VECTORDB_URL}/query",
                json={
                    "query": query,
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Error querying VectorDB: {e}")
            return {"relevant_chunks": [], "error": str(e)}

    def find_neighbors(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Find neighbors for given chunk IDs using Neo4j"""
        try:
            # Get neighbors for each chunk ID
            all_neighbors = []
            for chunk_id in chunk_ids:
                # Query downstream dependencies
                downstream_response = requests.post(
                    f"{NEO4J_URL}/query_graph",
                    json={
                        "query_type": "find_downstream_dependencies",
                        "target_id": chunk_id
                    },
                    timeout=30
                )
                
                # Query upstream dependencies
                upstream_response = requests.post(
                    f"{NEO4J_URL}/query_graph",
                    json={
                        "query_type": "find_upstream_dependencies", 
                        "target_id": chunk_id
                    },
                    timeout=30
                )
                
                if downstream_response.ok:
                    downstream_data = downstream_response.json()
                    for result in downstream_data.get("results", []):
                        all_neighbors.append({
                            "id": result.get("id"),
                            "name": result.get("name"),
                            "relationship": "calls",
                            "source_chunk": chunk_id
                        })
                
                if upstream_response.ok:
                    upstream_data = upstream_response.json()
                    for result in upstream_data.get("results", []):
                        all_neighbors.append({
                            "id": result.get("id"),
                            "name": result.get("name"),
                            "relationship": "called_by",
                            "source_chunk": chunk_id
                        })
            
            return {"neighbors": all_neighbors}
        except requests.RequestException as e:
            print(f"❌ Error finding neighbors: {e}")
            return {"neighbors": [], "error": str(e)}

    def get_neighbor_code(self, neighbor_ids: List[str]) -> List[Dict[str, Any]]:
        """Get the actual code for neighbor chunks from VectorDB"""
        if not neighbor_ids:
            return []
        
        try:
            # Query VectorDB for each neighbor ID
            neighbor_chunks = []
            for neighbor_id in neighbor_ids:
                response = requests.post(
                    f"{VECTORDB_URL}/query",
                    json={
                        "query": neighbor_id,  # Search by chunk ID
                        "max_results": 1,
                        "similarity_threshold": 0.1  # Low threshold since we're searching by ID
                    },
                    timeout=30
                )
                
                if response.ok:
                    data = response.json()
                    chunks = data.get("relevant_chunks", [])
                    if chunks:
                        neighbor_chunks.extend(chunks)
            
            return neighbor_chunks
        except requests.RequestException as e:
            print(f"❌ Error getting neighbor code: {e}")
            return []

    def generate_final_answer(self, user_query: str, relevant_chunks: List[Dict], neighbor_chunks: List[Dict]) -> str:
        """Generate final answer using all gathered context"""
        if not self.groq_client:
            return "Groq client not available for answer generation."
        
        # Prepare context from relevant chunks
        context_parts = []
        context_parts.append("=== PRIMARY RELEVANT CODE ===")
        for i, chunk in enumerate(relevant_chunks[:3]):
            ci = chunk.get('chunk_info', {})
            context_parts.append(f"""
Code Chunk {i+1} (Similarity: {chunk.get('similarity_score', 0):.3f}):
File: {ci.get('file', 'Unknown')}
Language: {ci.get('language', 'Unknown')}
Function/Class: {ci.get('function') or ci.get('class') or 'N/A'}
Type: {ci.get('type', 'Unknown')}

Code:
{chunk.get('document', '')}
""")
        
        # Add neighbor context if available
        if neighbor_chunks:
            context_parts.append("\n=== RELATED/CONNECTED CODE ===")
            for i, chunk in enumerate(neighbor_chunks[:5]):
                ci = chunk.get('chunk_info', {})
                context_parts.append(f"""
Related Code {i+1}:
File: {ci.get('file', 'Unknown')}
Function/Class: {ci.get('function') or ci.get('class') or 'N/A'}

Code:
{chunk.get('document', '')}
""")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert code analyst. Analyze the provided code context and answer the user's question comprehensively.

User Question: {user_query}

Code Context:
{context}

Instructions:
1. Provide a clear, detailed answer based on the code context
2. Explain how different parts of the code work together
3. Mention specific file names, functions, and classes when relevant
4. If you see patterns or architectural decisions, explain them
5. Include code snippets in your explanation when helpful
6. If there are any limitations in the provided context, mention them

Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert code analyst and software architect."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error generating final answer: {e}")
            return f"Error generating answer: {str(e)}"

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main orchestration logic for processing queries"""
        start_time = time.time()
        workflow_steps = []
        
        # Step 1: Preprocess query
        step_start = time.time()
        preprocessed_query = self.preprocess_query(request.query)
        workflow_steps.append(f"Query preprocessing ({time.time() - step_start:.2f}s)")
        
        # Step 2: Query VectorDB for relevant chunks
        step_start = time.time()
        vectordb_response = self.query_vectordb(
            preprocessed_query, 
            request.max_results, 
            request.similarity_threshold
        )
        relevant_chunks = vectordb_response.get("relevant_chunks", [])
        workflow_steps.append(f"VectorDB query - {len(relevant_chunks)} chunks ({time.time() - step_start:.2f}s)")
        
        neighbor_chunks = []
        
        # Step 3: Find neighbors if requested and chunks found
        if request.include_neighbors and relevant_chunks:
            step_start = time.time()
            
            # Extract chunk IDs from relevant chunks
            chunk_ids = []
            for chunk in relevant_chunks:
                metadata = chunk.get('metadata', {})
                chunk_id = metadata.get('id') or f"{metadata.get('file_path', '')}::{metadata.get('function_name', 'None')}"
                if chunk_id:
                    chunk_ids.append(chunk_id)
            
            if chunk_ids:
                neighbors_response = self.find_neighbors(chunk_ids)
                neighbors = neighbors_response.get("neighbors", [])
                workflow_steps.append(f"Neo4j neighbor search - {len(neighbors)} neighbors ({time.time() - step_start:.2f}s)")
                
                # Step 4: Get code for neighbors
                if neighbors:
                    step_start = time.time()
                    neighbor_ids = [n["id"] for n in neighbors if n.get("id")]
                    neighbor_chunks = self.get_neighbor_code(neighbor_ids)
                    workflow_steps.append(f"Neighbor code retrieval - {len(neighbor_chunks)} chunks ({time.time() - step_start:.2f}s)")
            else:
                workflow_steps.append("Neo4j neighbor search - skipped (no chunk IDs)")
        
        # Step 5: Generate final answer
        step_start = time.time()
        final_answer = self.generate_final_answer(request.query, relevant_chunks, neighbor_chunks)
        workflow_steps.append(f"Final answer generation ({time.time() - step_start:.2f}s)")
        
        return QueryResponse(
            query=request.query,
            preprocessed_query=preprocessed_query,
            answer=final_answer,
            relevant_chunks=relevant_chunks,
            neighbor_chunks=neighbor_chunks,
            processing_time=round(time.time() - start_time, 3),
            workflow_steps=workflow_steps
        )

# --------------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------------
app = FastAPI(title="Query Orchestrator", version="1.0.0")
orchestrator = QueryOrchestrator()

@app.get("/")
async def root():
    return {
        "service": "Query Orchestrator",
        "version": "1.0.0", 
        "status": "running",
        "groq_available": orchestrator.groq_client is not None
    }

@app.post("/query")
async def process_query(request: QueryRequest) -> QueryResponse:
    """Main query processing endpoint"""
    try:
        return await orchestrator.process_query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

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
        response = requests.get(f"{VECTORDB_URL}/", timeout=5)
        health_status["vectordb"] = "healthy" if response.ok else "unhealthy"
    except:
        health_status["vectordb"] = "unreachable"
    
    # Check Neo4j service
    try:
        response = requests.get(f"{NEO4J_URL}/", timeout=5) 
        health_status["neo4j"] = "healthy" if response.ok else "unhealthy"
    except:
        health_status["neo4j"] = "unreachable"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)