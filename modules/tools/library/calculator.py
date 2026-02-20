import math

expression = str(args.get('expression', ''))

# Create a safe dictionary of allowed functions and constants
safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
safe_dict.update({
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
})

try:
    # Basic security check
    if not expression:
        result = "Error: No expression provided."
    elif any(forbidden in expression for forbidden in ["__", "import", "lambda", "exec", "eval", "compile"]):
        result = "Error: Unsafe expression detected."
    else:
        # Handle power operator ^ -> ** (common LLM mistake)
        expression = expression.replace('^', '**')
        # Evaluate the expression in the restricted scope
        calc_result = eval(expression, {"__builtins__": {}}, safe_dict)
        result = f"{calc_result}"
except Exception as e:
    result = f"Error evaluating expression: {str(e)}"