"""Prompt templates for propositional-logic inference."""

TASK_INSTR = """TASK
You will be given truth-table rows and must infer one propositional-logic formula that matches every row.

"""

INPUT_INSTR = """INPUT FORMAT
- Each row is "{{p=0, q=1, ...}}, label".
- 0 means False and 1 means True.
- The value after the closing brace is the formula's output.
- The available variables are: {variables}.

"""

SYNTAX_INSTR = """PROPOSITIONAL-LOGIC SYNTAX
- Negation: !p
- Conjunction: p & q
- Disjunction: p | q
- Implication: p -> q
- Biconditional: p <-> q
- Constants: True and False
- Parentheses may be used freely.
- Use only the variables listed in the input instructions.

"""

INFERENCE_STRATEGY = """INFERENCE STRATEGY
1) Identify assignments that make the target True and False.
2) Look for variables that are irrelevant or always decisive.
3) Test compact conjunctions, disjunctions, implications, and equivalences.
4) Before answering, verify the formula against every supplied row.

"""

REGULARIZATION = """CONSTRAINTS
- Prefer the shortest clear formula consistent with all supplied rows.
- Avoid redundant terms such as p & p, p | False, or double negation.

"""

OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- First provide 1-3 concise sentences of reasoning wrapped in <reasoning> and </reasoning>.
- Then output the formula wrapped in <ans> and </ans>.
- Do not place prose inside <ans>.

"""

DIRECT_OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- Output ONLY the final formula wrapped in <ans> and </ans>.

"""

TRAINING_DATA_INSTR = """Training Data:
{0}
"""

AGENTIC_REFLECTION_INSTR = """
AGENTIC REFLECTION UPDATE
- You will receive the previous reasoning and formula plus new counterexamples.
- Briefly explain what failed, then produce an updated formula consistent with all rows.

"""

AGENTIC_REPAIR_INSTR = """
AGENTIC REPAIR UPDATE
- Repair the previous formula using the supplied feedback and examples.
- Ensure the result parses under the required syntax and fixes every highlighted assignment.
- Preserve correct parts of the previous formula where possible.

"""

PROMPT_TEMPLATE = (
    TASK_INSTR
    + INPUT_INSTR
    + SYNTAX_INSTR
    + INFERENCE_STRATEGY
    + "{regularization_instr}\n{agentic_reflection_instr}\n"
    + OUTPUT_FORMAT_INSTR
    + TRAINING_DATA_INSTR
)
