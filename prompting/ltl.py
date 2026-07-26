"""Prompt templates for LTL inference from labeled lasso traces."""

TASK_INSTR = """TASK
Infer one infinite-trace Linear Temporal Logic (LTL) formula consistent with every labeled lasso trace.

"""

INPUT_INSTR = """INPUT FORMAT
- Each example is "states=[{{...}}, {{...}}, ...], loop=i, label".
- State values use 0=False and 1=True.
- After the last listed state, execution returns to state i forever.
- The available atomic propositions are: {variables}.

"""

SYNTAX_INSTR = """LTL SYNTAX
- Boolean: ! (not), & (and), | (or), -> (implication), <-> (equivalence)
- Temporal unary: X phi (next), F phi (eventually), G phi (globally)
- Temporal binary: phi U psi (until), phi R psi (release)
- Constants: True and False
- Use parentheses liberally and only the listed propositions.
- Formulas use infinite lasso-trace semantics.

"""

INFERENCE_STRATEGY = """INFERENCE STRATEGY
1) Compare positive and negative traces at the initial state.
2) Check safety patterns with G, reachability patterns with F, and immediate transitions with X.
3) Use U/R only when the examples require an ordering or persistence condition.
4) Verify the candidate against every supplied lasso, including its infinite loop.

"""

REGULARIZATION = """CONSTRAINTS
- Prefer the shortest clear LTL formula consistent with all examples.
- Avoid redundant Boolean or temporal subformulas.

"""

OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- Give 1-3 concise sentences in <reasoning>...</reasoning>.
- Return the final LTL formula in <ans>...</ans>.
- Do not put prose inside <ans>.

"""

DIRECT_OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- Output ONLY the final LTL formula wrapped in <ans> and </ans>.

"""

TRAINING_DATA_INSTR = """Training Data:
{0}
"""

AGENTIC_REFLECTION_INSTR = """
AGENTIC REFLECTION UPDATE
- Revise the previous reasoning and formula using the new counterexample lassos.
- Explain the failed temporal condition briefly, then return a corrected formula.

"""

AGENTIC_REPAIR_INSTR = """
AGENTIC REPAIR UPDATE
- Repair the previous LTL formula so that it parses and classifies every highlighted lasso correctly.
- Preserve correct temporal structure where possible.

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

