import os
import json
from contextlib import asynccontextmanager
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from neo4j import Driver, GraphDatabase
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------
# BLOCK 1: Database Connection Management (Lifespan)
# --------------------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to Neo4j...")
    app.state.driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    app.state.driver.verify_connectivity()
    print("Connection to Neo4j established.")
    yield
    print("Closing connection to Neo4j...")
    app.state.driver.close()
    print("Connection closed.")


# --------------------------------------------------------------------------
# BLOCK 2: FastAPI App, Models, and Groq Client Setup
# --------------------------------------------------------------------------
app = FastAPI(title="C2C Neo4j Service", lifespan=lifespan)

class CodeChunksRequest(BaseModel):
    chunks: Dict[str, str]

class GraphQueryRequest(BaseModel):
    query_type: str
    target_id: str

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# --------------------------------------------------------------------------
# BLOCK 3: Core Logic (Author & Researcher Functions)
# --------------------------------------------------------------------------
def get_analysis_from_llm(code_chunk: str) -> dict:
    prompt = f"""
    Analyze the following Python code and identify the function calls.
    Respond with a valid JSON object containing a single key "calls" which is a list of strings.
    CRITICAL INSTRUCTIONS:
    1.  Only identify calls to user-defined functions within the project.
    2.  You MUST IGNORE calls to built-in Python methods or standard library functions (e.g., .upper(), str(), print()).
    Code:
    ---
    {code_chunk}
    ---
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {"calls": []}


def execute_cypher_write(driver: Driver, queries: List[str]) -> dict:
    def transaction_work(tx, queries_to_run):
        for query in queries_to_run:
            tx.run(query)
    try:
        with driver.session() as session:
            session.execute_write(transaction_work, queries)
            return {"status": "Success", "queries_executed": len(queries)}
    except Exception as e:
        print(f"Transaction failed. Error: {e}")
        return {"status": "Error", "error_message": str(e)}

# --------------------------------------------------------------------------
# FIX IS HERE vvvv
# --------------------------------------------------------------------------
def run_graph_query(driver: Driver, query_type: str, target_id: str) -> List[dict]:
    """
    Selects and runs a Cypher query based on the requested query_type.
    """
    cypher_query = ""
    if query_type == "find_downstream_dependencies":
        cypher_query = """
        MATCH (f:Function {id: $target_id})-[:CALLS]->(callee:Function)
        RETURN callee.name AS name, callee.id AS id
        """
    elif query_type == "find_upstream_dependencies":
        cypher_query = """
        MATCH (caller:Function)-[:CALLS]->(f:Function {id: $target_id})
        RETURN caller.name AS name, caller.id AS id
        """

    results = []
    if cypher_query:
        with driver.session() as session:
            # Corrected line: session.execute_read returns only one value (the records)
            records = session.execute_read(
                lambda tx: tx.run(cypher_query, target_id=target_id).data()
            )
            results = records
    return results
# --------------------------------------------------------------------------
# FIX IS HERE ^^^^
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# BLOCK 4: API Endpoints (Author & Researcher)
# --------------------------------------------------------------------------
@app.post("/build_graph")
async def build_graph_endpoint(request: CodeChunksRequest, http_request: Request):
    # ... (This endpoint is unchanged)
    driver = http_request.app.state.driver
    chunks = request.chunks
    print("Pass 1: Creating all function nodes...")
    node_queries = [f"MERGE (f:Function {{id: '{uid}'}}) SET f.name = '{uid.split('::')[-1]}', f.file_path = '{uid.split('::')[0]}'" for uid in chunks]
    node_summary = execute_cypher_write(driver, node_queries)
    print(f"Pass 1 Complete. Summary: {node_summary}")
    print("Pass 2: Analyzing code and creating relationships...")
    relationship_queries = []
    name_to_id_map = {uid.split("::")[-1]: uid for uid in chunks}
    for uid, code in chunks.items():
        analysis = get_analysis_from_llm(code)
        for called_name in analysis.get("calls", []):
            called_id = name_to_id_map.get(called_name)
            if called_id:
                relationship_queries.append(f"MATCH (c:Function {{id: '{uid}'}}), (d:Function {{id: '{called_id}'}}) MERGE (c)-[:CALLS]->(d)")
    rel_summary = execute_cypher_write(driver, relationship_queries)
    print(f"Pass 2 Complete. Summary: {rel_summary}")
    return {"message": "Graph update complete.", "node_summary": node_summary, "rel_summary": rel_summary}


@app.post("/query_graph", response_model=dict)
async def query_graph_endpoint(request: GraphQueryRequest, http_request: Request):
    # ... (This endpoint is unchanged)
    driver = http_request.app.state.driver
    results = run_graph_query(driver, request.query_type, request.target_id)
    return {"query_type": request.query_type, "target_id": request.target_id, "results": results}