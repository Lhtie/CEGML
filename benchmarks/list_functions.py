"""Author List Functions task interface; CE search remains a project extension."""
import ast
from functools import lru_cache
import re

PROMPT = '''Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers.

{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```

Include the complete function definition, without a function call, inside the requested <ans> tags.
{feedback}'''


def extract_program(response):
    response = response or ''
    sources = re.findall(r'<ans>\s*(.*?)\s*</ans>', response, re.S)
    sources = sources or [response.split('assistantfinal')[-1].strip()]
    for source in reversed(sources):
        fences = re.findall(r'```(?:python)?\s*\n(.*?)```', source, re.S)
        source = (fences[-1] if fences else source).strip()
        try:
            program_function(source)
        except (ValueError, SyntaxError):
            continue
        return source
    return None


def program_function(source):
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    name = 'fn' if 'fn' in functions else 'transform'  # historical log compatibility
    if name not in functions:
        raise ValueError('Expected Python function fn(x)')
    function = functions[name]
    if len(function.args.posonlyargs) + len(function.args.args) != 1:
        raise ValueError('Expected a function taking one positional input')
    if function.decorator_list or function.args.defaults or function.args.kw_defaults:
        raise ValueError('Expected an undecorated function without default arguments')
    return tree, name


@lru_cache(maxsize=128)
def compiled_code(source):
    tree, name = program_function(source)
    return compile(tree, '<list-function-candidate>', 'exec'), name


def execute_program(source, x):
    """Normal Python execution, like the author executor; caller enforces timeout.

    No task-language allowlist. This is not a security sandbox. Create a fresh
    namespace on each call to avoid state leaking between independent examples.
    """
    code, name = compiled_code(source)
    namespace = {}
    exec(code, namespace)
    return namespace[name](x)


def execute(source, x):
    output = execute_program(source, x)
    if not isinstance(output, list) or any(type(v) is not int for v in output):
        raise ValueError('fn(x) must return a list of integers')
    return output
