# Chunk.py - Split prompt into chunks of given size
# Used in RLM (Recursive Language Model) for processing long inputs in pieces

args = args.get("args", {})
repl_state = args.get("_repl_state", {})

prompt = repl_state.get("prompt_var", "")
size = args.get("size", 2000)
overlap = args.get("overlap", 200)

if not prompt:
    result = {"error": "No prompt in repl_state", "chunks": [], "count": 0}
elif size <= 0 or overlap < 0 or overlap >= size:
    result = {"error": "size must be > overlap >= 0", "chunks": [], "count": 0}
else:
    chunks = []
    i = 0
    while i < len(prompt):
        chunk_text = prompt[i:i+size]
        chunks.append({
            "index": len(chunks),
            "start": i,
            "end": i + len(chunk_text),
            "text": chunk_text,
            "length": len(chunk_text)
        })
        i += size - overlap
    
    result = {"chunks": chunks, "count": len(chunks)}

