import os
import shutil
import tempfile
import git
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import requests

# --------------------------------------------------------------------------
# BLOCK 1: The Smart File Walker (The Gatekeeper)
# --------------------------------------------------------------------------
def discover_code_files(root_dir: str):
    """
    Recursively finds all relevant code files in a directory,
    intelligently ignoring specified directories, files, and extensions.
    """
    # Configuration for files/folders to ignore.
    IGNORE_DIRS = {
        'node_modules', '.git', '_pycache_', 'venv', '.venv',
        'dist', 'build', '.vscode', '.idea'
    }
    IGNORE_FILES = {
        '.gitignore', 'package-lock.json', 'yarn.lock',
        'docker-compose.yml', 'Dockerfile' # Ignoring as per our plan
    }
    IGNORE_EXTENSIONS = {
        '.log', '.tmp', '.env', '.DS_Store', '.png', '.jpg', '.jpeg',
        '.gif', '.svg', '.ico', '.pdf', '.zip', '.gz', '.woff', '.ttf'
    }

    for root, dirs, files in os.walk(root_dir):
        # This line efficiently stops the walker from entering ignored folders
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for filename in files:
            if filename in IGNORE_FILES:
                continue
            
            file_ext = os.path.splitext(filename)[1]
            if file_ext in IGNORE_EXTENSIONS:
                continue
            
            # If a file passes all checks, we yield its full path
            yield os.path.join(root, filename)

# --------------------------------------------------------------------------
# BLOCK 2: The Core Ingestion Logic (The Background Worker)
# --------------------------------------------------------------------------
def process_repository(repo_url: str):
    """
    The main background task that clones, parses, and delegates.
    """
    # Use a system-wide temporary directory to avoid triggering the server reloader
    temp_clone_dir = os.path.join(tempfile.gettempdir(), "c2c_temp_repo")
    print(f"Starting ingestion for {repo_url}...")

    try:
        # Step 1: Clone the repo cleanly
        if os.path.exists(temp_clone_dir):
            shutil.rmtree(temp_clone_dir)
        print(f"Cloning repository into {temp_clone_dir}...")
        git.Repo.clone_from(repo_url, temp_clone_dir)
        print("Clone successful.")

        # Step 2: Discover all relevant files using our smart walker
        all_chunks = {}
        print("Discovering relevant code files...")
        
        for file_path in discover_code_files(temp_clone_dir):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # --- THIS IS WHERE YOUR CUSTOM CHUNKING LOGIC WILL GO ---
                    # For now, we treat the whole file as one chunk.
                    unique_id = file_path 
                    all_chunks[unique_id] = content
            except Exception as e:
                print(f"Could not read or chunk file {file_path}: {e}")
        
        print(f"Found {len(all_chunks)} files to process.")

        # Step 3: Delegate to specialist services
        print("Delegating to ChromaDB service...")
        # requests.post("http://localhost:8002/index", json=all_chunks)
        
        print("Delegating to Neo4j service...")
        # requests.post("http://localhost:8003/build_graph", json=all_chunks)

        print("--- Ingestion process complete! ---")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
    finally:
        # Step 4: Guaranteed cleanup of the temporary folder
        if os.path.exists(temp_clone_dir):
            shutil.rmtree(temp_clone_dir)
        print("Cleanup complete.")

# --------------------------------------------------------------------------
# BLOCK 3: The FastAPI Application (The Public Interface)
# --------------------------------------------------------------------------
app = FastAPI(title="C2C Orchestrator Service")

# Defines the structure of the incoming request from the frontend
class RepoRequest(BaseModel):
    repo_url: str

@app.post("/ingest")
async def ingest_repository(request: RepoRequest, background_tasks: BackgroundTasks):
    """
    Receives the repo URL and starts the processing in the background.
    """
    # This immediately schedules our main function to run without making the user wait
    background_tasks.add_task(process_repository, request.repo_url)
    return {"message": "Repository ingestion started in the background."}