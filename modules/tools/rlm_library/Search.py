# Search.py - Find regex matches in the prompt
# Used in RLM (Recursive Language Model) for searching through long inputs

import re

args = args.get("args", {})
repl_state = args.get("_repl_state", {})

prompt = repl_state.get("prompt_var", "")
pattern = args.get("pattern", "")
max_results = args.get("max_results", 20)

if not pattern:
    result = {"error": "No pattern provided"}
else:
    matches = []
    try:
        for m in re.finditer(pattern, prompt, re.IGNORECASE):
            if len(matches) >= max_results:
                break
            matches.append({
                "match": m.group(),
                "start": m.start(),
                "end": m.end(),
                "context": prompt[max(0, m.start()-50):min(len(prompt), m.end()+50)]
            })
        result = {
            "matches": matches,
            "count": len(matches),
            "total_in_text": len(re.findall(pattern, prompt, re.IGNORECASE))
        }
    except re.error as e:
        result = {"error": f"Invalid regex pattern: {str(e)}"}

