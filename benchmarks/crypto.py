"""Paper-style Cryptography: blind Python program induction, not cipher selection.

Task/prompt reference: HKUST-KnowComp/Robust-Rule-Induction,
  prompts/crypto_prompt.py and tasks/base.py (n_train=10, n_test=10).
The CE method and bounded oracle are this project's extensions.
"""
from modeling.response import all_tags
import ast
import builtins
from functools import lru_cache
import re
import string

PAPER_PROMPT = '''Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a string. The output is a string.

{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```

Include the complete function definition, without a function call, inside the requested <ans> tags.
{feedback}'''


def paper_examples(item, ood_ratio=0.):
    """Select ten correctly labeled observations; never use the noise pool."""
    if not 0 <= ood_ratio <= 1 or abs(10 * ood_ratio - round(10 * ood_ratio)) > 1e-8:
        raise ValueError('Crypto OOD ratio must be in [0,1] in steps of 0.1')
    ood = int(round(10 * ood_ratio))
    normal = 10 - ood
    for kind, count in [('normal', normal), ('ood', ood)]:
        if len(item['train'].get(kind, [])) < count:
            raise ValueError(f'Insufficient published {kind} examples: requested {count}')
    examples = item['train']['normal'][:normal] + item['train'].get('ood', [])[:ood]
    return [(e['input'], e['output']) for e in examples]


def extract_program(response):
    """Accept the author's fenced/bare fn(x) output, or our explicit ans wrapper."""
    response = response or ''
    answers = all_tags(response, 'ans')
    choices = answers[-1:] or re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.S)
    if not choices:
        # vLLM may include a separate analysis channel before the final code.
        choices = [response.split('assistantfinal')[-1].strip()]
    for source in reversed(choices):
        fences = re.findall(r'```(?:python)?\s*\n(.*?)```', source, re.S)
        source = (fences[-1] if fences else source).strip()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name == 'fn' for n in tree.body):
            return source
    return None


@lru_cache(maxsize=128)
def compile_program(source):
    """Bounded, allowlisted execution of common string-transformation programs.

    Python is not an OS security sandbox: keep imports/calls restricted rather
    than enabling unrestricted execution of model-generated code.
    """
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or funcs[0].name != 'fn':
        raise ValueError('Expected one Python function named fn')
    if len(funcs[0].args.args) != 1 or funcs[0].args.args[0].arg != 'x':
        raise ValueError('Expected fn(x)')
    if any(not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body):
        raise ValueError('Only the function definition and optional string imports are supported')
    calls = {'ord', 'chr', 'len', 'range', 'str', 'list', 'dict', 'tuple', 'set', 'zip',
             'enumerate', 'int', 'bool', 'abs', 'min', 'max', 'sum', 'reversed', 'sorted', 'all', 'any'}
    methods = {'append', 'extend', 'join', 'index', 'find', 'lower', 'upper', 'isupper', 'islower',
               'isalpha', 'isdigit', 'isalnum', 'isspace', 'replace', 'get', 'maketrans', 'translate',
               'split', 'strip', 'startswith', 'endswith', 'items', 'keys', 'values',
               'ascii_lowercase', 'ascii_uppercase', 'ascii_letters', 'digits', 'punctuation'}
    local = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    blocked = (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda,
               ast.Try, ast.With, ast.Raise, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name != 'string' for a in node.names):
                raise ValueError('Only import string is supported')
        if isinstance(node, ast.ImportFrom):
            if node.module != 'string' or node.level or any(a.name not in methods for a in node.names):
                raise ValueError('Only string constants may be imported')
        if isinstance(node, blocked) or (isinstance(node, ast.Name) and node.id.startswith('__')):
            raise ValueError('Unsupported Python syntax')
        if isinstance(node, ast.Attribute) and node.attr not in methods:
            raise ValueError('Unsupported string operation')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in calls | local:
                raise ValueError('Unsupported function call')
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError('Unsupported call')
    def import_string(name, globals=None, locals=None, fromlist=(), level=0):
        if name != 'string' or level:
            raise ValueError('Only string imports are supported')
        return string
    safe = {name: getattr(builtins, name) for name in calls}
    safe['__import__'] = import_string
    scope = {'__builtins__': safe}
    exec(compile(tree, '<crypto-candidate>', 'exec'), scope)
    return scope['fn']
