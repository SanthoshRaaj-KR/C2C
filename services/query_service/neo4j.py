import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class GraphQueryRequest(BaseModel):
    query_type: str
    target_id: str
    max_depth: Optional[int] = 2
    limit: Optional[int] = 10

class NeighborSearchRequest(BaseModel):
    chunk_ids: List[str]
    search_types: Optional[List[str]] = ["downstream", "upstream"]
    max_depth: Optional[int] = 2
    limit: Optional[int] = 10

class PathQueryRequest(BaseModel):
    source_id: str
    target_id: str
    max_path_length: Optional[int] = 5

class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_relationships: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]

class NeighborResponse(BaseModel):
    query_chunk_id: str
    neighbors: List[Dict[str, Any]]
    relationship_type: str
    processing_time: float

# --------------------------------------------------------------------------
# Database Lifespan Management
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔗 Connecting to Neo4j...")
    try:
        app.state.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        app.state.driver.verify_connectivity()
        print("✅ Connection to Neo4j established.")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        raise
    
    yield
    
    print("🔌 Closing Neo4j connection...")
    if hasattr(app.state, 'driver'):
        app.state.driver.close()
    print("✅ Neo4j connection closed.")

# --------------------------------------------------------------------------
# Enhanced Graph Query Manager
# --------------------------------------------------------------------------
class GraphQueryManager:
    def __init__(self, driver: Driver):
        self.driver = driver

    def find_downstream_dependencies(self, target_id: str, max_depth: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """Find all functions/classes that this target calls (downstream dependencies)"""
        cypher_query = f"""
        MATCH path = (source:Function {{id: $target_id}})-[:CALLS*1..{max_depth}]->(target:Function)
        RETURN DISTINCT target.id AS id, 
               target.name AS name, 
               target.file_path AS file_path,
               target.type AS type,
               length(path) AS distance,
               [node in nodes(path) | node.name] AS path_names
        ORDER BY distance ASC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, target_id=target_id, limit=limit)
                return [
                    {
                        "id": record["id"],
                        "name": record["name"], 
                        "file_path": record["file_path"],
                        "type": record["type"],
                        "distance": record["distance"],
                        "path": record["path_names"],
                        "relationship": "calls"
                    }
                    for record in result
                ]
        except Exception as e:
            print(f"❌ Error finding downstream dependencies: {e}")
            return []

    def find_upstream_dependencies(self, target_id: str, max_depth: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """Find all functions/classes that call this target (upstream dependencies)"""
        cypher_query = f"""
        MATCH path = (source:Function)-[:CALLS*1..{max_depth}]->(target:Function {{id: $target_id}})
        RETURN DISTINCT source.id AS id, 
               source.name AS name, 
               source.file_path AS file_path,
               source.type AS type,
               length(path) AS distance,
               [node in nodes(path) | node.name] AS path_names
        ORDER BY distance ASC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, target_id=target_id, limit=limit)
                return [
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "file_path": record["file_path"], 
                        "type": record["type"],
                        "distance": record["distance"],
                        "path": record["path_names"],
                        "relationship": "called_by"
                    }
                    for record in result
                ]
        except Exception as e:
            print(f"❌ Error finding upstream dependencies: {e}")
            return []

    def find_direct_neighbors(self, target_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find direct neighbors (both directions) more efficiently"""
        cypher_query = """
        MATCH (target:Function {id: $target_id})
        OPTIONAL MATCH (target)-[:CALLS]->(downstream:Function)
        OPTIONAL MATCH (upstream:Function)-[:CALLS]->(target)
        RETURN 
            collect(DISTINCT {
                id: downstream.id, 
                name: downstream.name, 
                file_path: downstream.file_path,
                type: downstream.type,
                relationship: "calls"
            }) AS downstream_neighbors,
            collect(DISTINCT {
                id: upstream.id, 
                name: upstream.name, 
                file_path: upstream.file_path,
                type: upstream.type,
                relationship: "called_by"
            }) AS upstream_neighbors
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, target_id=target_id)
                record = result.single()
                if record:
                    downstream = [n for n in record["downstream_neighbors"] if n["id"]]
                    upstream = [n for n in record["upstream_neighbors"] if n["id"]]
                    return {
                        "downstream": downstream,
                        "upstream": upstream
                    }
                return {"downstream": [], "upstream": []}
        except Exception as e:
            print(f"❌ Error finding direct neighbors: {e}")
            return {"downstream": [], "upstream": []}

    def find_path_between_nodes(self, source_id: str, target_id: str, max_length: int = 5) -> List[Dict[str, Any]]:
        """Find shortest path between two nodes"""
        cypher_query = f"""
        MATCH path = shortestPath((source:Function {{id: $source_id}})-[:CALLS*1..{max_length}]->(target:Function {{id: $target_id}}))
        RETURN [node in nodes(path) | {{
            id: node.id, 
            name: node.name, 
            file_path: node.file_path,
            type: node.type
        }}] AS path_nodes,
        length(path) AS path_length
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, source_id=source_id, target_id=target_id)
                record = result.single()
                if record:
                    return {
                        "path": record["path_nodes"],
                        "length": record["path_length"],
                        "exists": True
                    }
                return {"path": [], "length": 0, "exists": False}
        except Exception as e:
            print(f"❌ Error finding path: {e}")
            return {"path": [], "length": 0, "exists": False}

    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific node"""
        cypher_query = """
        MATCH (n:Function {id: $node_id})
        OPTIONAL MATCH (n)-[:CALLS]->(out:Function)
        OPTIONAL MATCH (in:Function)-[:CALLS]->(n)
        RETURN n.id AS id, n.name AS name, n.file_path AS file_path, n.type AS type,
               count(DISTINCT out) AS outgoing_calls,
               count(DISTINCT in) AS incoming_calls,
               collect(DISTINCT out.name) AS calls_functions,
               collect(DISTINCT in.name) AS called_by_functions
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, node_id=node_id)
                record = result.single()
                if record:
                    return {
                        "id": record["id"],
                        "name": record["name"],
                        "file_path": record["file_path"],
                        "type": record["type"],
                        "outgoing_calls": record["outgoing_calls"],
                        "incoming_calls": record["incoming_calls"],
                        "calls_functions": [f for f in record["calls_functions"] if f],
                        "called_by_functions": [f for f in record["called_by_functions"] if f]
                    }
                return {}
        except Exception as e:
            print(f"❌ Error getting node info: {e}")
            return {}

    def find_central_nodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find most connected nodes (high degree centrality)"""
        cypher_query = """
        MATCH (n:Function)
        OPTIONAL MATCH (n)-[:CALLS]->(out)
        OPTIONAL MATCH (in)-[:CALLS]->(n)
        WITH n, count(DISTINCT out) AS outgoing, count(DISTINCT in) AS incoming
        RETURN n.id AS id, n.name AS name, n.file_path AS file_path,
               outgoing, incoming, (outgoing + incoming) AS total_connections
        ORDER BY total_connections DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, limit=limit)
                return [
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "file_path": record["file_path"],
                        "outgoing_calls": record["outgoing"],
                        "incoming_calls": record["incoming"],
                        "total_connections": record["total_connections"]
                    }
                    for record in result
                ]
        except Exception as e:
            print(f"❌ Error finding central nodes: {e}")
            return []

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        stats_query = """
        MATCH (n:Function)
        OPTIONAL MATCH ()-[r:CALLS]->()
        RETURN count(DISTINCT n) AS total_nodes,
               count(r) AS total_relationships,
               collect(DISTINCT n.type) AS node_types,
               collect(DISTINCT n.language) AS languages
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(stats_query)
                record = result.single()
                if record:
                    return {
                        "total_nodes": record["total_nodes"],
                        "total_relationships": record["total_relationships"],
                        "node_types": [t for t in record["node_types"] if t],
                        "languages": [l for l in record["languages"] if l]
                    }
                return {"total_nodes": 0, "total_relationships": 0, "node_types": [], "languages": []}
        except Exception as e:
            print(f"❌ Error getting graph stats: {e}")
            return {"error": str(e)}

    def batch_neighbor_search(self, chunk_ids: List[str], search_types: List[str] = ["downstream", "upstream"], max_depth: int = 2, limit: int = 10) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Efficiently search for neighbors of multiple chunks"""
        results = {}
        
        for chunk_id in chunk_ids:
            chunk_neighbors = {}
            
            if "downstream" in search_types:
                chunk_neighbors["downstream"] = self.find_downstream_dependencies(chunk_id, max_depth, limit)
            
            if "upstream" in search_types:
                chunk_neighbors["upstream"] = self.find_upstream_dependencies(chunk_id, max_depth, limit)
            
            if "direct" in search_types:
                direct_neighbors = self.find_direct_neighbors(chunk_id)
                chunk_neighbors.update(direct_neighbors)
            
            results[chunk_id] = chunk_neighbors
        
        return results

# --------------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------------
app = FastAPI(title="Graph Query Service", lifespan=lifespan, version="1.0.0")

@app.get("/")
async def root():
    return {
        "service": "Graph Query Service",
        "version": "1.0.0",
        "status": "running",
        "neo4j_uri": NEO4J_URI.replace(NEO4J_PASSWORD, "***") if NEO4J_PASSWORD in NEO4J_URI else NEO4J_URI
    }

@app.post("/query_graph")
async def query_graph(request: GraphQueryRequest):
    """Main graph query endpoint"""
    start_time = time.time()
    
    try:
        query_manager = GraphQueryManager(app.state.driver)
        results = []
        
        if request.query_type == "find_downstream_dependencies":
            results = query_manager.find_downstream_dependencies(
                request.target_id, 
                request.max_depth, 
                request.limit
            )
        elif request.query_type == "find_upstream_dependencies":
            results = query_manager.find_upstream_dependencies(
                request.target_id, 
                request.max_depth, 
                request.limit
            )
        elif request.query_type == "find_direct_neighbors":
            neighbor_results = query_manager.find_direct_neighbors(request.target_id)
            results = neighbor_results.get("downstream", []) + neighbor_results.get("upstream", [])
        elif request.query_type == "get_node_info":
            results = [query_manager.get_node_info(request.target_id)]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown query type: {request.query_type}")
        
        return {
            "query_type": request.query_type,
            "target_id": request.target_id,
            "results": results,
            "total_results": len(results),
            "processing_time": round(time.time() - start_time, 3)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")

@app.post("/neighbors/batch")
async def batch_neighbor_search(request: NeighborSearchRequest):
    """Batch search for neighbors of multiple chunks"""
    start_time = time.time()
    
    try:
        query_manager = GraphQueryManager(app.state.driver)
        results = query_manager.batch_neighbor_search(
            request.chunk_ids,
            request.search_types,
            request.max_depth,
            request.limit
        )
        
        return {
            "batch_results": results,
            "total_chunks_queried": len(request.chunk_ids),
            "search_types": request.search_types,
            "processing_time": round(time.time() - start_time, 3)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch neighbor search failed: {str(e)}")

@app.post("/path")
async def find_path(request: PathQueryRequest):
    """Find path between two nodes"""
    start_time = time.time()
    
    try:
        query_manager = GraphQueryManager(app.state.driver)
        path_result = query_manager.find_path_between_nodes(
            request.source_id,
            request.target_id,
            request.max_path_length
        )
        
        return {
            "source_id": request.source_id,
            "target_id": request.target_id,
            "path_result": path_result,
            "processing_time": round(time.time() - start_time, 3)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path search failed: {str(e)}")

@app.get("/stats")
async def get_graph_statistics():
    """Get graph statistics"""
    try:
        query_manager = GraphQueryManager(app.state.driver)
        stats = query_manager.get_graph_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.get("/central_nodes")
async def get_central_nodes(limit: int = 10):
    """Get most connected nodes"""
    try:
        query_manager = GraphQueryManager(app.state.driver)
        central_nodes = query_manager.find_central_nodes(limit)
        return {
            "central_nodes": central_nodes,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Central nodes retrieval failed: {str(e)}")

@app.get("/node/{node_id}")
async def get_node_details(node_id: str):
    """Get detailed information about a specific node"""
    try:
        query_manager = GraphQueryManager(app.state.driver)
        node_info = query_manager.get_node_info(node_id)
        
        if not node_info:
            raise HTTPException(status_code=404, detail=f"Node with id '{node_id}' not found")
        
        return node_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node details retrieval failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with app.state.driver.session() as session:
            result = session.run("RETURN 1 as health_check")
            record = result.single()
            
            # Get basic stats for health check
            stats_result = session.run("MATCH (n:Function) RETURN count(n) as node_count")
            stats_record = stats_result.single()
            
            return {
                "status": "healthy",
                "database_connected": True,
                "total_nodes": stats_record["node_count"] if stats_record else 0,
                "neo4j_version": "Connected"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 