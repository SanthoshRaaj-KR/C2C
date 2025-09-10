import os
import json
from contextlib import asynccontextmanager
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# --------------------------------------------------------------------------
# DATABASE LIFESPAN
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to Neo4j...")
    app.state.driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    app.state.driver.verify_connectivity()
    print("Connection to Neo4j established.")
    yield
    print("Closing Neo4j connection...")
    app.state.driver.close()
    print("Connection closed.")

# --------------------------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------------------------
app = FastAPI(title="C2C Neo4j Service", lifespan=lifespan)

# --------------------------------------------------------------------------
# Pydantic Models
# --------------------------------------------------------------------------
class Chunk(BaseModel):
    id: str
    code: str

class Neo4jIngestRequest(BaseModel):
    chunks: List[Chunk]

class GraphQueryRequest(BaseModel):
    query_type: str
    target_id: str

# --------------------------------------------------------------------------
# CORE LOGIC
# --------------------------------------------------------------------------
def execute_cypher_write(driver: Driver, queries: List[str]) -> dict:
    def tx_work(tx, queries_to_run):
        for q in queries_to_run:
            tx.run(q)
    try:
        with driver.session() as session:
            session.execute_write(tx_work, queries)
        return {"status": "Success", "queries_executed": len(queries)}
    except Exception as e:
        return {"status": "Error", "error_message": str(e)}

def build_relationships(chunks: List[Chunk]) -> List[str]:
    """
    Build Cypher queries for CALLS relationships.
    This currently assumes calls between functions with matching names in the same codebase.
    """
    queries = []
    name_to_id_map = {c.id.split("::")[-1]: c.id for c in chunks}
    for c in chunks:
        code_lines = c.code.splitlines()
        for name, uid in name_to_id_map.items():
            if name in c.code and name != c.id.split("::")[-1]:
                queries.append(
                    f"""
                    MATCH (caller:Function {{id: '{c.id}'}}), 
                          (callee:Function {{id: '{uid}'}})
                    MERGE (caller)-[:CALLS]->(callee)
                    """
                )
    return queries

# --------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------
@app.post("/neo4j/ingest")
async def ingest_chunks(request: Neo4jIngestRequest, http_request: Request):
    driver = http_request.app.state.driver
    chunks = request.chunks

    # 1️⃣ Create nodes
    node_queries = [
        f"""
        MERGE (f:Function {{id: '{c.id}'}})
        SET f.name = '{c.id.split('::')[-1]}',
            f.file_path = '{c.id.split('::')[0]}',
            f.code = $code_{i}
        """
        for i, c in enumerate(chunks)
    ]
    # Run nodes with parameters to avoid Cypher injection
    def tx_nodes(tx):
        for i, c in enumerate(chunks):
            tx.run(
                node_queries[i],
                code_=c.code
            )
    try:
        with driver.session() as session:
            session.execute_write(tx_nodes)
        node_summary = {"status": "Success", "nodes_created": len(chunks)}
    except Exception as e:
        node_summary = {"status": "Error", "error_message": str(e)}

    # 2️⃣ Create relationships
    relationship_queries = build_relationships(chunks)
    rel_summary = execute_cypher_write(driver, relationship_queries)

    return {
        "message": "Neo4j ingestion complete.",
        "node_summary": node_summary,
        "rel_summary": rel_summary
    }

@app.post("/query_graph")
async def query_graph(request: GraphQueryRequest, http_request: Request):
    driver = http_request.app.state.driver
    cypher_query = ""
    if request.query_type == "find_downstream_dependencies":
        cypher_query = """
        MATCH (f:Function {id: $target_id})-[:CALLS]->(callee:Function)
        RETURN callee.name AS name, callee.id AS id
        """
    elif request.query_type == "find_upstream_dependencies":
        cypher_query = """
        MATCH (caller:Function)-[:CALLS]->(f:Function {id: $target_id})
        RETURN caller.name AS name, caller.id AS id
        """
    results = []
    if cypher_query:
        with driver.session() as session:
            records = session.execute_read(lambda tx: tx.run(cypher_query, target_id=request.target_id).data())
            results = records
    return {"query_type": request.query_type, "target_id": request.target_id, "results": results}
