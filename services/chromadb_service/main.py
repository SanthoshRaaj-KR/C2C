import os
import ast
import javalang
import json

# -------------------------------
# Config
# -------------------------------
MAX_CHUNK_LINES = 50
OUTPUT_JSON = "chunks.json"

# -------------------------------
# 1️⃣ Language detection
# -------------------------------
def get_language_from_filename(filename):
    ext_to_lang = {
        '.py': 'python',
        '.java': 'java',
    }
    _, ext = os.path.splitext(filename)
    return ext_to_lang.get(ext.lower())

# -------------------------------
# 2️⃣ Chunk splitting
# -------------------------------
def split_chunk(code, max_lines, base_meta):
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
        # Generate unique ID
        chunk_meta['id'] = f"{chunk_meta['file_path']}::{chunk_meta.get('class_name','None')}::{chunk_meta.get('function_name','None')}::{chunk_no}"
        chunks.append(chunk_meta)
    return chunks

# -------------------------------
# 3️⃣ Python chunking
# -------------------------------
def chunk_python(file_path, file_name, code):
    chunks = []
    tree = ast.parse(code)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1
            end_line = node.body[-1].lineno if node.body else node.lineno
            snippet = "\n".join(code.splitlines()[start_line:end_line])

            base_meta = {
                'file_path': file_path,
                'file_name': file_name,
                'class_name': node.name if isinstance(node, ast.ClassDef) else None,
                'function_name': node.name if isinstance(node, ast.FunctionDef) else None,
                'type': 'class' if isinstance(node, ast.ClassDef) else 'function'
            }

            # Split if too long
            sub_chunks = split_chunk(snippet, MAX_CHUNK_LINES, base_meta)
            chunks.extend(sub_chunks)
    return chunks

# -------------------------------
# 4️⃣ Java chunking
# -------------------------------
def chunk_java(file_path, file_name, code):
    chunks = []
    try:
        tree = javalang.parse.parse(code)
        for path, node in tree.filter((javalang.tree.ClassDeclaration, javalang.tree.MethodDeclaration)):
            start_line = node.position.line - 1 if node.position else 0
            snippet = "\n".join(code.splitlines()[start_line:start_line+50])  # initial max 50 lines
            base_meta = {
                'file_path': file_path,
                'file_name': file_name,
                'class_name': node.name if isinstance(node, javalang.tree.ClassDeclaration) else None,
                'function_name': node.name if isinstance(node, javalang.tree.MethodDeclaration) else None,
                'type': 'class' if isinstance(node, javalang.tree.ClassDeclaration) else 'method'
            }
            sub_chunks = split_chunk(snippet, MAX_CHUNK_LINES, base_meta)
            chunks.extend(sub_chunks)
    except:
        pass
    return chunks

# -------------------------------
# 5️⃣ Main parser
# -------------------------------
def parse_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    file_name = os.path.basename(file_path)
    lang = get_language_from_filename(file_path)
    if not lang:
        print(f"Unsupported file type: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    if lang == 'python':
        chunks = chunk_python(file_path, file_name, code)
    elif lang == 'java':
        chunks = chunk_java(file_path, file_name, code)
    else:
        chunks = []

    return chunks

# -------------------------------
# 6️⃣ Run example & store in JSON
# -------------------------------
if __name__ == "__main__":
    test_file = "test.py" 
    all_chunks = parse_file(test_file)

    if all_chunks:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4)
        print(f"\n✅ All chunks with metadata stored in '{OUTPUT_JSON}'")
