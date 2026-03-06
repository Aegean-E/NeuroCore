# SetVariable.py - Store intermediate results in REPL state
# Used in RLM (Recursive Language Model) for accumulating results across iterations

args = args.get("args", {})
repl_state = args.get("_repl_state", {})

name = args.get("name", "")
value = args.get("value", "")

if not name:
    result = {"error": "No variable name provided"}
else:
    variables = repl_state.get("variables", {})
    variables[name] = value
    repl_state["variables"] = variables
    
    result = {
        "success": True,
        "name": name,
        "value_type": type(value).__name__,
        "stored_keys": list(variables.keys())
    }

