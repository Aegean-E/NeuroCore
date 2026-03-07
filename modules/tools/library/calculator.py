import ast
import math
import operator

expression = str(args.get('expression', ''))

# Allowed operators for safe evaluation
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Safe constants and functions
safe_constants = {
    "pi": math.pi,
    "e": math.e,
    "inf": float('inf'),
}

safe_functions = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


class SafeEvaluator(ast.NodeVisitor):
    """Safe AST-based expression evaluator."""
    
    def __init__(self):
        self._variables = {}
    
    def visit_Name(self, node):
        if node.id in safe_constants:
            return safe_constants[node.id]
        if node.id in safe_functions:
            return safe_functions[node.id]
        if node.id in self._variables:
            return self._variables[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    
    def visit_Constant(self, node):
        return node.value
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return ALLOWED_OPERATORS[op_type](left, right)
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return ALLOWED_OPERATORS[op_type](operand)
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_type = type(op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            if not ALLOWED_OPERATORS[op_type](left, right):
                return False
            left = right
        return True
    
    def visit_Call(self, node):
        # Only allow calls to safe functions
        if isinstance(node.func, ast.Name) and node.func.id in safe_functions:
            args = [self.visit(arg) for arg in node.args]
            return safe_functions[node.func.id](*args)
        raise ValueError(f"Function call not allowed: {node.func.id if isinstance(node.func, ast.Name) else 'unknown'}")
    
    def generic_visit(self, node):
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


try:
    # Basic security check
    if not expression:
        result = "Error: No expression provided."
    elif any(forbidden in expression for forbidden in ["__", "import", "lambda", "exec", "eval", "compile", "class", "def"]):
        result = "Error: Unsafe expression detected."
    else:
        # Handle power operator ^ -> ** (common LLM mistake)
        expression = expression.replace('^', '**')
        
        # Parse and evaluate safely using AST
        try:
            tree = ast.parse(expression, mode='eval')
            evaluator = SafeEvaluator()
            calc_result = evaluator.visit(tree.body)
            result = f"{calc_result}"
        except SyntaxError:
            result = "Error: Invalid expression syntax."
        except ValueError as e:
            result = f"Error: {str(e)}"
except Exception as e:
    result = f"Error evaluating expression: {str(e)}"
