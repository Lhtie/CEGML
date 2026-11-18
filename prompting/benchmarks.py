"""Prompts for the four input/output program induction benchmarks."""
from benchmarks.list_functions import PROMPT as LIST_FUNCTIONS_PROMPT
from benchmarks.ilp import PROMPT as ILP_PROMPT

PROMPTS = {
    "ilp": ILP_PROMPT,
    "cryptography": """Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a string. The output is a string.
{feedback}
{examples}
Return a Python function named fn(x), without the function call.
""",
    "list_functions": LIST_FUNCTIONS_PROMPT,
    "instruction_induction": """Infer the natural-language instruction that maps each input to its output. Return one concise instruction in <ans> tags.\n{feedback}\nExamples (JSON input => JSON output):\n{examples}\n""",
    "scan": """Infer a compositional interpreter for SCAN commands. Write Python function transform(x) taking a command string and returning its action sequence as a space-separated string. Use Python standard syntax only. Wrap the complete function in <ans> tags.\n{feedback}\nExamples (JSON input => JSON output):\n{examples}\n""",
}

REGULARIZATION = """REGULARIZATION
Prefer a concise rule or program that explains all examples. Avoid unnecessary special cases.

"""

# The grounded teacher prompt is built with official annotations and induction-only
# reference coverage. Keep the public constant as the shared policy text.
from benchmarks.instruction_oracle import BASE as INSTRUCTION_ORACLE


ICL_TASKS = {
    "ilp": "Infer the Boolean truth of target ground atoms given relational background knowledge.",
    "cryptography": "Infer the same character substitution used in the examples. Preserve letter case.",
    "list_functions": "Infer the same transformation of integer lists used in the examples.",
    "instruction_induction": "Infer the same instruction or mapping used in the examples.",
    "scan": "Translate each SCAN command into its compositional sequence of actions.",
}

ICL_PROMPT = """{task}\n{regularization}\nTraining examples (one JSON input => JSON output per line):\n{examples}\n\nPredict the output for each input below, in exactly the same order. Return one JSON array of outputs inside <ans>...</ans>. The outer array must have exactly {count} elements. Do not put prose inside <ans>.\nInputs (one JSON value per line):\n{inputs}\n"""
