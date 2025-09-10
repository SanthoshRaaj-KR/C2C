import os
import json
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import your enhanced chunking logic
import chunking

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
CODE_BASE_DIR = os.path.join(ROOT_DIR, "code_base")   
OUTPUT_JSON = os.path.join(ROOT_DIR, "repo_chunks.json")
PROCESSING_STATUS_FILE = os.path.join(ROOT_DIR, "processing_status.json")

# --------------------------------------------------------------------------
# BLOCK 1: Enhanced Smart File Walker
# --------------------------------------------------------------------------
def discover_code_files(root_dir: str) -> List[str]:
    """
    Discover all relevant code files in a directory.
    Returns a list of absolute file paths.
    """
    IGNORE_DIRS = {
        'node_modules', '.git', '__pycache__', 'venv', '.venv', 'env',
        'dist', 'build', '.vscode', '.idea', 'target', 'out',
        'coverage', '.nyc_output', '.next', '.nuxt', 'vendor',
        'bower_components', '.sass-cache', '.cache'
    }
    
    IGNORE_FILES = {
        '.gitignore', '.env', '.env.local', '.env.production',
        'package-lock.json', 'yarn.lock', 'composer.lock',
        'docker-compose.yml', 'Dockerfile', '.dockerignore',
        'README.md', 'LICENSE', 'CHANGELOG.md'
    }
    
    IGNORE_EXTENSIONS = {
        '.log', '.tmp', '.temp', '.DS_Store', '.pyc', '.pyo',
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
        '.pdf', '.zip', '.gz', '.tar', '.rar', '.7z',
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        '.mp4', '.mp3', '.wav', '.avi', '.mov',
        '.exe', '.dll', '.so', '.dylib',
        '.min.js', '.min.css'  
    }
    
    SUPPORTED_EXTENSIONS = {
        '.py', '.java', '.js', '.jsx', '.ts', '.tsx', 
        '.mjs', '.cjs', '.vue'  # Added Vue support
    }

    discovered_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]

        for filename in files:
            # Skip ignored files
            if filename in IGNORE_FILES or filename.startswith('.'):
                continue
                
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Skip ignored extensions
            if file_ext in IGNORE_EXTENSIONS:
                continue
                
            # Only include supported code files
            if file_ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, filename)
                discovered_files.append(file_path)
    
    return discovered_files

# --------------------------------------------------------------------------
# BLOCK 2: Status Management
# --------------------------------------------------------------------------
def update_processing_status(status: str, progress: Dict[str, Any] = None):
    """Update the processing status file"""
    status_data = {
        'status': status,
        'timestamp': time.time(),
        'progress': progress or {}
    }
    
    try:
        with open(PROCESSING_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"Error updating status: {e}")

def get_processing_status() -> Dict[str, Any]:
    """Get the current processing status"""
    try:
        if os.path.exists(PROCESSING_STATUS_FILE):
            with open(PROCESSING_STATUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading status: {e}")
    
    return {'status': 'idle', 'timestamp': 0, 'progress': {}}

# --------------------------------------------------------------------------
# BLOCK 3: Enhanced Core Ingestion Logic
# --------------------------------------------------------------------------
def process_local_codebase():
    """
    Process files inside the local code_base folder with enhanced error handling and progress tracking.
    """
    start_time = time.time()
    print(f"🚀 Starting ingestion for local folder: {CODE_BASE_DIR}")
    
    update_processing_status('starting', {'message': 'Initializing codebase processing'})

    if not os.path.exists(CODE_BASE_DIR):
        error_msg = f"❌ code_base folder not found at: {CODE_BASE_DIR}"
        print(error_msg)
        update_processing_status('error', {'message': error_msg})
        return

    all_chunks = []
    processing_stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'total_chunks': 0,
        'languages': {},
        'errors': []
    }
    
    print("🔍 Discovering relevant code files...")
    update_processing_status('discovering', {'message': 'Scanning for code files'})
    
    try:
        discovered_files = discover_code_files(CODE_BASE_DIR)
        processing_stats['total_files'] = len(discovered_files)
        
        print(f"📁 Found {len(discovered_files)} code files to process")
        
        if not discovered_files:
            update_processing_status('completed', {
                'message': 'No supported code files found',
                'stats': processing_stats
            })
            return
        
        update_processing_status('processing', {
            'message': f'Processing {len(discovered_files)} files',
            'progress': processing_stats
        })
        
        # Process each file
        for i, file_path in enumerate(discovered_files):
            try:
                print(f"📝 Processing ({i+1}/{len(discovered_files)}): {os.path.relpath(file_path, CODE_BASE_DIR)}")
                
                chunks = chunking.parse_file(file_path, base_dir=CODE_BASE_DIR)
                
                if chunks:
                    all_chunks.extend(chunks)
                    processing_stats['total_chunks'] += len(chunks)
                    
                    # Track language stats
                    for chunk in chunks:
                        lang = chunk.get('language', 'unknown')
                        processing_stats['languages'][lang] = processing_stats['languages'].get(lang, 0) + 1
                
                processing_stats['processed_files'] += 1
                
                # Update progress every 10 files or on the last file
                if (i + 1) % 10 == 0 or i == len(discovered_files) - 1:
                    progress_percent = round((i + 1) / len(discovered_files) * 100, 1)
                    update_processing_status('processing', {
                        'message': f'Processing files: {progress_percent}% complete',
                        'progress': processing_stats,
                        'current_file': os.path.relpath(file_path, CODE_BASE_DIR)
                    })
                
            except Exception as e:
                error_msg = f"Could not process {os.path.relpath(file_path, CODE_BASE_DIR)}: {str(e)}"
                print(f"⚠️  {error_msg}")
                processing_stats['failed_files'] += 1
                processing_stats['errors'].append({
                    'file': os.path.relpath(file_path, CODE_BASE_DIR),
                    'error': str(e)
                })

        # Save results
        print(f"💾 Saving {len(all_chunks)} chunks to {OUTPUT_JSON}")
        update_processing_status('saving', {
            'message': f'Saving {len(all_chunks)} chunks to file',
            'progress': processing_stats
        })
        
        # Create a comprehensive output structure
        output_data = {
            'metadata': {
                'generated_at': time.time(),
                'processing_time_seconds': round(time.time() - start_time, 2),
                'codebase_path': CODE_BASE_DIR,
                'stats': processing_stats
            },
            'chunks': all_chunks
        }
        
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Final status update
        final_stats = processing_stats.copy()
        final_stats['processing_time'] = round(time.time() - start_time, 2)
        
        update_processing_status('completed', {
            'message': f'Successfully processed {processing_stats["processed_files"]} files, generated {len(all_chunks)} chunks',
            'stats': final_stats,
            'output_file': OUTPUT_JSON
        })
        
        print(f"✅ Ingestion complete!")
        print(f"📊 Stats: {final_stats}")
        print(f"⏱️  Processing time: {final_stats['processing_time']} seconds")
        print(f"💾 Output saved to: {OUTPUT_JSON}")

    except Exception as e:
        error_msg = f"Critical error during processing: {str(e)}"
        print(f"💥 {error_msg}")
        processing_stats['errors'].append({'type': 'critical', 'error': str(e)})
        update_processing_status('error', {
            'message': error_msg,
            'stats': processing_stats
        })

# --------------------------------------------------------------------------
# BLOCK 4: Enhanced FastAPI Application
# --------------------------------------------------------------------------
app = FastAPI(
    title="Code-to-Chunks (C2C) Orchestrator Service",
    description="Service for parsing and chunking codebases into manageable pieces",
    version="2.0.0"
)

class IngestRequest(BaseModel):
    trigger: str = "local"
    max_chunk_lines: Optional[int] = 50

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    timestamp: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Code-to-Chunks Orchestrator", 
        "version": "2.0.0",
        "status": "running",
        "supported_languages": ["Python", "Java", "JavaScript", "TypeScript"]
    }

@app.post("/ingest")
async def ingest_repository(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Kicks off ingestion of the local code_base folder.
    """
    current_status = get_processing_status()
    
    # Check if already processing
    if current_status.get('status') == 'processing':
        return JSONResponse(
            status_code=409,
            content={
                "error": "Processing already in progress",
                "current_status": current_status
            }
        )
    
    # Update chunk size if specified
    if request.max_chunk_lines and request.max_chunk_lines > 0:
        chunking.MAX_CHUNK_LINES = request.max_chunk_lines
    
    background_tasks.add_task(process_local_codebase)
    
    return {
        "message": "Local code_base ingestion started in the background",
        "status": "initiated",
        "config": {
            "codebase_path": CODE_BASE_DIR,
            "output_file": OUTPUT_JSON,
            "max_chunk_lines": chunking.MAX_CHUNK_LINES
        }
    }

@app.get("/status")
async def get_status():
    """
    Get the current processing status
    """
    status_data = get_processing_status()
    return StatusResponse(**status_data)

@app.get("/results")
async def get_results():
    """
    Get the latest processing results
    """
    if not os.path.exists(OUTPUT_JSON):
        raise HTTPException(
            status_code=404, 
            detail="No results found. Run /ingest first."
        )
    
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading results: {str(e)}"
        )

@app.get("/results/summary")
async def get_results_summary():
    """
    Get a summary of the latest processing results
    """
    if not os.path.exists(OUTPUT_JSON):
        raise HTTPException(
            status_code=404, 
            detail="No results found. Run /ingest first."
        )
    
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        chunks = data.get('chunks', [])
        
        return {
            "total_chunks": len(chunks),
            "metadata": metadata,
            "sample_chunks": chunks[:3] if chunks else []  # First 3 chunks as sample
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading results: {str(e)}"
        )

@app.delete("/results")
async def clear_results():
    """
    Clear the results and status files
    """
    files_to_clear = [OUTPUT_JSON, PROCESSING_STATUS_FILE]
    cleared = []
    
    for file_path in files_to_clear:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                cleared.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    return {
        "message": "Results cleared",
        "cleared_files": cleared
    }

# --------------------------------------------------------------------------
# BLOCK 5: Application startup
# --------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print(f"🚀 C2C Orchestrator Service starting...")
    print(f"📁 Codebase directory: {CODE_BASE_DIR}")
    print(f"💾 Output file: {OUTPUT_JSON}")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # Initialize status
    update_processing_status('idle', {'message': 'Service started and ready'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)