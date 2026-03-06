# Peek.py - View a slice of the prompt by character position
# Used in RLM (Recursive Language Model) for examining long inputs

args = args.get("args", {})
repl_state = args.get("_repl_state", {})

prompt = repl_state.get("prompt_var", "")
start = args.get("start", 0)
end = args.get("end", min(1000, len(prompt)))

# Clamp to valid range
start = max(0, min(start, len(prompt)))
end = max(start, min(end, len(prompt)))

result = prompt[start:end]

