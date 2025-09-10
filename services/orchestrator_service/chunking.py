import os
import ast
import re
import json
from typing import List, Dict, Any, Optional

try:
    import javalang
    JAVA_SUPPORT = True
except ImportError:
    JAVA_SUPPORT = False
    print("Warning: javalang not installed. Java parsing will be disabled.")

# -------------------------------
# Config
# -------------------------------
MAX_CHUNK_LINES = 50

# -------------------------------
# 1️⃣ Enhanced Language Detection
# -------------------------------
def get_language_from_filename(filename: str) -> Optional[str]:
    """Detect programming language from file extension"""
    ext_to_lang = {
        '.py': 'python',
        '.java': 'java',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
    }
    _, ext = os.path.splitext(filename)
    return ext_to_lang.get(ext.lower())

# -------------------------------
# 2️⃣ Enhanced Chunk Splitting
# -------------------------------
def split_chunk(code: str, max_lines: int, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Splits a chunk if it exceeds max_lines.
    Returns a list of chunk dicts with metadata.
    """
    lines = code.splitlines()
    chunks = []
    
    for i in range(0, len(lines), max_lines):
        sub_lines = lines[i:i+max_lines]
        chunk_no = (i // max_lines) + 1
        chunk_meta = base_meta.copy()
        chunk_meta['chunk_no'] = chunk_no
        chunk_meta['code'] = "\n".join(sub_lines)
        chunk_meta['line_count'] = len(sub_lines)
        
        # Generate unique ID with relative path
        chunk_meta['id'] = f"{chunk_meta['file_path']}::{chunk_meta.get('class_name','None')}::{chunk_meta.get('function_name','None')}::{chunk_no}"
        chunks.append(chunk_meta)
    
    return chunks

# -------------------------------
# 3️⃣ Python Chunking (Enhanced)
# -------------------------------
def chunk_python(file_path: str, file_name: str, code: str) -> List[Dict[str, Any]]:
    """Parse Python code into chunks"""
    chunks = []
    
    try:
        tree = ast.parse(code)
        code_lines = code.splitlines()
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                
                # Find the actual end line by looking for the last statement
                if hasattr(node, 'body') and node.body:
                    end_line = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else node.body[-1].lineno
                else:
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                
                # Extract the code snippet
                snippet = "\n".join(code_lines[start_line:end_line])
                
                base_meta = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'language': 'python',
                    'class_name': node.name if isinstance(node, ast.ClassDef) else None,
                    'function_name': node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                    'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                    'start_line': start_line + 1,
                    'end_line': end_line,
                    'is_async': isinstance(node, ast.AsyncFunctionDef)
                }
                
                # Split if too long
                sub_chunks = split_chunk(snippet, MAX_CHUNK_LINES, base_meta)
                chunks.extend(sub_chunks)
                
    except SyntaxError as e:
        print(f"Python syntax error in {file_path}: {e}")
    except Exception as e:
        print(f"Error parsing Python file {file_path}: {e}")
    
    return chunks

# -------------------------------
# 4️⃣ Java Chunking (Enhanced)
# -------------------------------
def chunk_java(file_path: str, file_name: str, code: str) -> List[Dict[str, Any]]:
    """Parse Java code into chunks"""
    if not JAVA_SUPPORT:
        print(f"Skipping Java file {file_path} - javalang not installed")
        return []
    
    chunks = []
    try:
        tree = javalang.parse.parse(code)
        code_lines = code.splitlines()
        
        for path, node in tree.filter((javalang.tree.ClassDeclaration, javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
            start_line = (node.position.line - 1) if node.position else 0
            
            # Estimate end line (Java parsing doesn't give us exact end positions easily)
            end_line = min(start_line + 100, len(code_lines))  # Default estimate
            snippet = "\n".join(code_lines[start_line:end_line])
            
            base_meta = {
                'file_path': file_path,
                'file_name': file_name,
                'language': 'java',
                'class_name': node.name if isinstance(node, javalang.tree.ClassDeclaration) else None,
                'function_name': node.name if isinstance(node, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)) else None,
                'type': 'class' if isinstance(node, javalang.tree.ClassDeclaration) else ('constructor' if isinstance(node, javalang.tree.ConstructorDeclaration) else 'method'),
                'start_line': start_line + 1,
                'end_line': end_line
            }
            
            sub_chunks = split_chunk(snippet, MAX_CHUNK_LINES, base_meta)
            chunks.extend(sub_chunks)
            
    except Exception as e:
        print(f"Error parsing Java file {file_path}: {e}")
    
    return chunks

# -------------------------------
# 5️⃣ JavaScript/TypeScript Chunking (NEW)
# -------------------------------
def chunk_javascript(file_path: str, file_name: str, code: str) -> List[Dict[str, Any]]:
    """Parse JavaScript/TypeScript code into chunks using regex patterns"""
    chunks = []
    code_lines = code.splitlines()
    
    # Patterns for different JavaScript constructs
    patterns = [
        # Function declarations
        (r'^(\s*)(function\s+(\w+)\s*\([^)]*\)\s*\{)', 'function'),
        # Arrow functions assigned to variables
        (r'^(\s*)(const|let|var)\s+(\w+)\s*=\s*(\([^)]*\)\s*=>\s*\{|[^=]*=>\s*\{)', 'function'),
        # Class declarations
        (r'^(\s*)(class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{)', 'class'),
        # Method definitions inside classes
        (r'^(\s*)(\w+)\s*\([^)]*\)\s*\{', 'method'),
        # Export function
        (r'^(\s*)(export\s+(?:default\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{)', 'function'),
        # Export const functions
        (r'^(\s*)(export\s+const\s+(\w+)\s*=\s*(\([^)]*\)\s*=>\s*\{|[^=]*=>\s*\{))', 'function'),
    ]
    
    i = 0
    while i < len(code_lines):
        line = code_lines[i]
        matched = False
        
        for pattern, construct_type in patterns:
            match = re.match(pattern, line)
            if match:
                start_line = i
                indent_level = len(match.group(1)) if match.group(1) else 0
                
                # Extract function/class name
                name = None
                if len(match.groups()) >= 3 and match.group(3):
                    name = match.group(3)
                elif construct_type == 'method':
                    method_match = re.match(r'^(\s*)(\w+)\s*\(', line)
                    if method_match:
                        name = method_match.group(2)
                
                # Find the end of this construct by tracking braces
                brace_count = line.count('{') - line.count('}')
                end_line = start_line
                
                for j in range(start_line + 1, len(code_lines)):
                    current_line = code_lines[j]
                    brace_count += current_line.count('{') - current_line.count('}')
                    end_line = j
                    if brace_count <= 0:
                        break
                
                # Extract the code snippet
                snippet = "\n".join(code_lines[start_line:end_line + 1])
                
                base_meta = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'language': 'typescript' if file_path.endswith(('.ts', '.tsx')) else 'javascript',
                    'class_name': name if construct_type == 'class' else None,
                    'function_name': name if construct_type in ['function', 'method'] else None,
                    'type': construct_type,
                    'start_line': start_line + 1,
                    'end_line': end_line + 1,
                    'is_export': 'export' in line
                }
                
                sub_chunks = split_chunk(snippet, MAX_CHUNK_LINES, base_meta)
                chunks.extend(sub_chunks)
                
                i = end_line + 1
                matched = True
                break
        
        if not matched:
            i += 1
    
    return chunks

# -------------------------------
# 6️⃣ Enhanced Main Parser
# -------------------------------
def parse_file(file_path: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse a code file and return chunks with metadata.
    
    Args:
        file_path: Absolute path to the file
        base_dir: Base directory for calculating relative paths
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    file_name = os.path.basename(file_path)
    
    # Calculate relative path
    if base_dir:
        try:
            relative_path = os.path.relpath(file_path, base_dir)
        except ValueError:
            # Handle case where paths are on different drives (Windows)
            relative_path = file_path
    else:
        relative_path = os.path.relpath(file_path, os.getcwd())
    
    # Normalize path separators for consistency
    relative_path = relative_path.replace(os.path.sep, '/')
    
    lang = get_language_from_filename(file_name)
    if not lang:
        print(f"Unsupported file type: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    chunks = []
    if lang == 'python':
        chunks = chunk_python(relative_path, file_name, code)
    elif lang == 'java':
        chunks = chunk_java(relative_path, file_name, code)
    elif lang in ['javascript', 'typescript']:
        chunks = chunk_javascript(relative_path, file_name, code)

    return chunks

# -------------------------------
# 7️⃣ Utility Functions
# -------------------------------
def get_file_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about processed chunks"""
    if not chunks:
        return {}
    
    languages = {}
    types = {}
    total_lines = 0
    
    for chunk in chunks:
        lang = chunk.get('language', 'unknown')
        chunk_type = chunk.get('type', 'unknown')
        lines = chunk.get('line_count', 0)
        
        languages[lang] = languages.get(lang, 0) + 1
        types[chunk_type] = types.get(chunk_type, 0) + 1
        total_lines += lines
    
    return {
        'total_chunks': len(chunks),
        'total_lines': total_lines,
        'languages': languages,
        'types': types
    }

# -------------------------------
# 8️⃣ Test Runner
# -------------------------------
if __name__ == "__main__":
    import sys
    
    test_file = sys.argv[1] if len(sys.argv) > 1 else "test.py"
    all_chunks = parse_file(test_file)

    if all_chunks:
        output_file = "chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4)
        
        stats = get_file_stats(all_chunks)
        print(f"\n✅ Generated {len(all_chunks)} chunks")
        print(f"📊 Stats: {stats}")
        print(f"💾 Stored in '{output_file}'")
    else:
        print("❌ No chunks generated")