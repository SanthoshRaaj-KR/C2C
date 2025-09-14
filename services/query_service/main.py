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

class BatchQueryRequest(BaseModel):
    queries: List[str]
    max_results_per_query: Optional[int] = 3
    include_neighbors: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.7

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
                model="llama-3.1-8b-instant",
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

    def query_vectordb(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Query the vector database for relevant chunks"""
        try:
            payload = {
                "query": query,
                "max_results": max_results,
                "similarity_threshold": similarity_threshold,
                "search_by_id": False
            }
            
            print(f"📡 Querying VectorDB with payload: {payload}")
            
            response = requests.post(
                f"{VECTORDB_URL}/query",
                json=payload,
                timeout=30
            )
            
            if not response.ok:
                print(f"❌ VectorDB query failed: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            print(f"✅ VectorDB response received: {len(data.get('relevant_chunks', []))} chunks")
            
            # Debug: Print structure of first chunk
            chunks = data.get("relevant_chunks", [])
            if chunks:
                print(f"🔍 Sample chunk structure: {list(chunks[0].keys())}")
            
            return chunks
            
        except requests.RequestException as e:
            print(f"❌ Error querying VectorDB: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error in VectorDB query: {e}")
            return []

    def extract_chunk_ids(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract chunk IDs from VectorDB response chunks.
        Handles multiple possible chunk structures from ingestion.
        """
        chunk_ids = []
        
        for chunk in chunks:
            chunk_id = None
            
            # Method 1: Direct ID field
            if 'id' in chunk:
                chunk_id = chunk['id']
            
            # Method 2: Check metadata
            elif 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                metadata = chunk['metadata']
                chunk_id = metadata.get('id') or metadata.get('chunk_id')
            
            # Method 3: Check chunk_info (from ingestion format)
            elif 'chunk_info' in chunk and isinstance(chunk['chunk_info'], dict):
                chunk_info = chunk['chunk_info']
                chunk_id = chunk_info.get('id') or chunk_info.get('chunk_id')
            
            # Method 4: Construct from file path and function name
            if not chunk_id:
                file_path = None
                function_name = None
                
                # Try different locations for file_path and function
                for location in [chunk, chunk.get('metadata', {}), chunk.get('chunk_info', {})]:
                    if isinstance(location, dict):
                        file_path = file_path or location.get('file_path') or location.get('file')
                        function_name = function_name or location.get('function_name') or location.get('function')
                
                if file_path:
                    if function_name:
                        chunk_id = f"{file_path}::{function_name}"
                    else:
                        chunk_id = f"{file_path}::unknown"
            
            if chunk_id:
                chunk_ids.append(str(chunk_id))
                print(f"✅ Extracted chunk ID: {chunk_id}")
            else:
                print(f"⚠️ Could not extract ID from chunk: {list(chunk.keys())}")
        
        print(f"📋 Total chunk IDs extracted: {len(chunk_ids)}")
        return chunk_ids

    def find_neighbors(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Find neighbors for given chunk IDs using Neo4j"""
        if not chunk_ids:
            print("⚠️ No chunk IDs provided for neighbor search")
            return []
            
        try:
            payload = {
                "chunk_ids": chunk_ids,
                "search_types": ["downstream", "upstream"],
                "max_depth": 2,
                "limit": 5
            }
            
            print(f"📡 Querying Neo4j neighbors with {len(chunk_ids)} chunk IDs")
            
            response = requests.post(
                f"{NEO4J_URL}/neighbors/batch",
                json=payload,
                timeout=30
            )
            
            if not response.ok:
                print(f"❌ Neo4j neighbor search failed: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            batch_results = data.get("batch_results", {})
            
            print(f"✅ Neo4j neighbor response: {len(batch_results)} batch results")
            
            # Flatten the results
            all_neighbors = []
            for chunk_id, neighbors_dict in batch_results.items():
                # Add downstream neighbors
                downstream = neighbors_dict.get("downstream", [])
                for neighbor in downstream:
                    all_neighbors.append({
                        "id": neighbor.get("id"),
                        "name": neighbor.get("name", "Unknown"),
                        "file_path": neighbor.get("file_path", "Unknown"),
                        "relationship": "calls",
                        "source_chunk": chunk_id
                    })
                
                # Add upstream neighbors  
                upstream = neighbors_dict.get("upstream", [])
                for neighbor in upstream:
                    all_neighbors.append({
                        "id": neighbor.get("id"),
                        "name": neighbor.get("name", "Unknown"),
                        "file_path": neighbor.get("file_path", "Unknown"),
                        "relationship": "called_by",
                        "source_chunk": chunk_id
                    })
            
            print(f"📋 Total neighbors found: {len(all_neighbors)}")
            return all_neighbors
                
        except requests.RequestException as e:
            print(f"❌ Error finding neighbors: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error in neighbor search: {e}")
            return []

    def get_neighbor_code(self, neighbor_ids: List[str]) -> List[Dict[str, Any]]:
        """Get the actual code for neighbor chunks from VectorDB"""
        if not neighbor_ids:
            print("⚠️ No neighbor IDs provided for code retrieval")
            return []
        
        try:
            payload = {"chunk_ids": neighbor_ids}
            print(f"📡 Retrieving code for {len(neighbor_ids)} neighbor chunks")
            
            response = requests.post(
                f"{VECTORDB_URL}/query/by_ids",
                json=payload,
                timeout=30
            )
            
            if not response.ok:
                print(f"❌ VectorDB chunk retrieval failed: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            chunks = data.get("chunks", [])
            
            print(f"✅ Retrieved code for {len(chunks)} neighbor chunks")
            return chunks
                
        except requests.RequestException as e:
            print(f"❌ Error getting neighbor code: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error in neighbor code retrieval: {e}")
            return []

    def generate_final_answer(self, user_query: str, relevant_chunks: List[Dict], neighbor_chunks: List[Dict]) -> str:
        """Generate final answer using all gathered context"""
        if not self.groq_client:
            return "Groq client not available for answer generation."
        
        # Prepare context from relevant chunks
        context_parts = []
        context_parts.append("=== PRIMARY RELEVANT CODE ===")
        
        for i, chunk in enumerate(relevant_chunks[:3]):
            # Handle different chunk structures
            similarity = chunk.get('similarity_score', chunk.get('score', 0))
            
            # Extract document/code content
            document = chunk.get('document') or chunk.get('code') or chunk.get('content', '')
            
            # Extract chunk info from multiple possible locations
            chunk_info = {}
            for info_location in [chunk.get('chunk_info', {}), chunk.get('metadata', {}), chunk]:
                if isinstance(info_location, dict):
                    chunk_info.update(info_location)
            
            file_path = chunk_info.get('file') or chunk_info.get('file_path', 'Unknown')
            language = chunk_info.get('language', 'Unknown')
            function_name = chunk_info.get('function') or chunk_info.get('function_name')
            class_name = chunk_info.get('class') or chunk_info.get('class_name')
            chunk_type = chunk_info.get('type', 'Unknown')
            
            context_parts.append(f"""
Code Chunk {i+1} (Similarity: {similarity:.3f}):
File: {file_path}
Language: {language}
Function/Class: {function_name or class_name or 'N/A'}
Type: {chunk_type}

Code:
{document}
""")
        
        # Add neighbor context if available
        if neighbor_chunks:
            context_parts.append("\n=== RELATED/CONNECTED CODE ===")
            for i, chunk in enumerate(neighbor_chunks[:5]):
                # Extract document/code content
                document = chunk.get('document') or chunk.get('code') or chunk.get('content', '')
                
                # Extract chunk info from multiple possible locations
                chunk_info = {}
                for info_location in [chunk.get('chunk_info', {}), chunk.get('metadata', {}), chunk]:
                    if isinstance(info_location, dict):
                        chunk_info.update(info_location)
                
                file_path = chunk_info.get('file') or chunk_info.get('file_path', 'Unknown')
                function_name = chunk_info.get('function') or chunk_info.get('function_name')
                class_name = chunk_info.get('class') or chunk_info.get('class_name')
                
                context_parts.append(f"""
Related Code {i+1}:
File: {file_path}
Function/Class: {function_name or class_name or 'N/A'}

Code:
{document}
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
                model="llama-3.1-8b-instant",
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
        
        print(f"🚀 Processing query: '{request.query}'")
        
        # Step 1: Preprocess query
        step_start = time.time()
        preprocessed_query = self.preprocess_query(request.query)
        step_time = time.time() - step_start
        workflow_steps.append(f"Query preprocessing ({step_time:.2f}s)")
        
        # Step 2: Query VectorDB for relevant chunks
        step_start = time.time()
        relevant_chunks = self.query_vectordb(
            preprocessed_query, 
            request.max_results, 
            request.similarity_threshold
        )
        step_time = time.time() - step_start
        workflow_steps.append(f"VectorDB query - {len(relevant_chunks)} chunks ({step_time:.2f}s)")
        
        neighbor_chunks = []
        
        # Step 3: Find neighbors if requested and chunks found
        if request.include_neighbors and relevant_chunks:
            step_start = time.time()
            
            # Extract chunk IDs with improved logic
            chunk_ids = self.extract_chunk_ids(relevant_chunks)
            
            if chunk_ids:
                neighbors = self.find_neighbors(chunk_ids)
                step_time = time.time() - step_start
                workflow_steps.append(f"Neo4j neighbor search - {len(neighbors)} neighbors ({step_time:.2f}s)")
                
                # Step 4: Get code for neighbors
                if neighbors:
                    step_start = time.time()
                    neighbor_ids = [n["id"] for n in neighbors if n.get("id")]
                    neighbor_chunks = self.get_neighbor_code(neighbor_ids)
                    step_time = time.time() - step_start
                    workflow_steps.append(f"Neighbor code retrieval - {len(neighbor_chunks)} chunks ({step_time:.2f}s)")
            else:
                workflow_steps.append("Neo4j neighbor search - skipped (no valid chunk IDs)")
        elif request.include_neighbors:
            workflow_steps.append("Neo4j neighbor search - skipped (no relevant chunks)")
        
        # Step 5: Generate final answer
        step_start = time.time()
        final_answer = self.generate_final_answer(request.query, relevant_chunks, neighbor_chunks)
        step_time = time.time() - step_start
        workflow_steps.append(f"Final answer generation ({step_time:.2f}s)")
        
        total_time = time.time() - start_time
        
        print(f"✅ Query processing complete in {total_time:.3f}s")
        
        return QueryResponse(
            query=request.query,
            preprocessed_query=preprocessed_query,
            answer=final_answer,
            relevant_chunks=relevant_chunks,
            neighbor_chunks=neighbor_chunks,
            processing_time=round(total_time, 3),
            workflow_steps=workflow_steps
        )

    async def process_batch_queries(self, request: BatchQueryRequest) -> Dict[str, Any]:
        """Process multiple queries in batch"""
        start_time = time.time()
        
        print(f"🚀 Processing batch of {len(request.queries)} queries")
        
        batch_results = []
        for i, query in enumerate(request.queries):
            print(f"📋 Processing query {i+1}/{len(request.queries)}: '{query}'")
            
            query_request = QueryRequest(
                query=query,
                max_results=request.max_results_per_query,
                include_neighbors=request.include_neighbors,
                similarity_threshold=request.similarity_threshold
            )
            
            result = await self.process_query(query_request)
            batch_results.append({
                "query": query,
                "result": result.dict()
            })
        
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
app = FastAPI(title="Query Orchestrator", version="1.1.0")
orchestrator = QueryOrchestrator()

@app.get("/")
async def root():
    return {
        "service": "Query Orchestrator",
        "version": "1.1.0", 
        "status": "running",
        "groq_available": orchestrator.groq_client is not None,
        "vectordb_url": VECTORDB_URL,
        "neo4j_url": NEO4J_URL
    }

@app.post("/query")
async def process_query(request: QueryRequest) -> QueryResponse:
    """Main query processing endpoint"""
    try:
        return await orchestrator.process_query(request)
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/query/batch")
async def process_batch_queries(request: BatchQueryRequest):
    """Batch query processing endpoint"""
    try:
        return await orchestrator.process_batch_queries(request)
    except Exception as e:
        print(f"❌ Batch query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch query processing failed: {str(e)}")

@app.post("/preprocess")
async def preprocess_query_endpoint(request: dict):
    """Standalone query preprocessing endpoint"""
    try:
        user_query = request.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        enhanced_query = orchestrator.preprocess_query(user_query)
        
        return {
            "original_query": user_query,
            "enhanced_query": enhanced_query,
            "groq_available": orchestrator.groq_client is not None
        }
    except Exception as e:
        print(f"❌ Query preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query preprocessing failed: {str(e)}")

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