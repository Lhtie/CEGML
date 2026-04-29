# Prompting templates for regex inference via RL.


#########################################################General prompt template #################################################################
DIRECT_OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- Output ONLY the final regex wrapped in <ans> and </ans>.

"""

AGENTIC_REFLECTION_INSTR = """
AGENTIC REFLECTION UPDATE
- You will receive the previous attempt's reasoning and regex, plus new counterexamples.
- First, briefly revise the previous reasoning to explain what failed and what should be changed.
- Then output an updated regex consistent with all training data and the counterexamples.
- Keep reasoning concise (1-3 sentences) and directly tied to the regex revision.

"""

AGENTIC_REPAIR_INSTR = """
AGENTIC REPAIR UPDATE
- You are repairing the previous attempt using the failure feedback and repair examples below.
- Repair goals:
  1) Produce a regex that compiles under the required regex syntax.
  2) Fix the specific mistakes exposed by the counterexamples, disagreement witness, and any reported errors.
  3) Preserve the parts of the previous solution that still fit the training data.
- What to do:
  - If the previous regex is invalid, first correct the syntax or unsupported constructs.
  - If a witness or repair example is rejected/accepted incorrectly, revise the regex so that string gets the correct label.
  - Keep the reasoning concise and focused on what changed from the previous attempt.
  - Return the repaired regex in the required output format.

"""

#############################################################Simplyrx prompt template and instructions#############################################################
SIMPLYRX_TASK_INSTR = """TASK
You will be given labeled strings and must infer a single regular language that matches all positives (label 1) and rejects all negatives (label 0). Output a concise regex in pyformlang.regular_expression.Regex syntax.

"""

SIMPLYRX_INPUT_INSTR = """INPUT FORMAT
- You receive a block titled “Training Data (Each line has one input-output pair separated by comma):”.
- Each line is "<string>, <label>" where label ∈ {{1, 0}}. The string may be empty; an empty string appears as nothing before the comma (", 1") and represents epsilon.
- The alphabet is exactly the set of characters appearing in the data (typically a, b, c). Do not introduce other symbols.
{clustered_ce_instr}
"""

SIMPLYRX_INFERENCE_STRATEGY = """INFERENCE STRATEGY
1) Start/end constraints:
   - Check if all positives start with a specific letter or set (e.g., all non-empty positives start with c). If so, encode a mandatory prefix, e.g., "c ..." or "(b + c) ...".
   - Check for a forced suffix or final-block restriction (e.g., must end with b or a specific 2-letter tail). Place this outside any repeating block when needed.

2) Length/modular and block structure:
   - Look for fixed-length blocks repeated via "*".
   - More generally:
     - Use a star over a union of allowed blocks when strings can mix block types: "((block1) + (block2))*".
     - If internal blocks allow more endings than the final block, use: "(InternalBlockUnion)* FinalRestrictedBlock".
   - If a singleton positive (e.g., "b") exists alongside block-based strings, include it via a top-level union only if it cannot be captured by a prefix plus star (e.g., "c (...) *" already accepts "c" because the star can be epsilon).

3) Union design: star-of-union vs union-of-stars
   - If strings mix different block types within one string, prefer a star over a union of blocks: "((...)+(...))*".
   - If each positive is formed by repeating exactly one fixed block with no mixing, a compact union of stars can be better: "(a b)* + (a c)* + (b c)*".

4) Compactness tactics:
   - Factor repeated substrings (e.g., "(a+b+c) a b c (...)").
   - Use per-position unions like "(a+b)" or "(a+b+c)" instead of enumerating full strings.
   - Factor common prefixes/suffixes within unions: "(a b c a b + a b c c b)" instead of duplicating.

5) Handling epsilon:
   - Accept epsilon only if explicitly required by the data.
   - Prefer to obtain needed epsilon through an existing Kleene star (e.g., "c (block)*" accepts "c"; "(block)*" accepts epsilon). Use "epsilon +" only when unavoidable at top level (e.g., when the empty string is positive but cannot be included via a star elsewhere).

6) Avoid over-generalization:
   - Do not allow arbitrary middles like "(a+b+c)*" unless strictly supported by all positives and necessary to exclude negatives.
   - Do not invent constraints not universally implied by positives.

7) Quality checks before finalizing:
   - Verify your regex accepts every 1-labeled string and rejects every 0-labeled string.
   - Sanity-check near-misses from negatives (e.g., wrong start letter, wrong modular length, incomplete final block, mixing vs non-mixing).
   - Re-check syntax: unions around multi-symbol sequences, spaces everywhere in concatenation, and only allowed symbols.

"""

SIMPLYRX_SYNTAX_INSTR = """PYFORMLANG REGEX SYNTAX
- Union: +
- Concatenation: space-separated symbols (each symbol is a single character from the alphabet or the literal epsilon).
- Kleene star: *
- Parentheses are allowed for grouping; use them whenever you union multi-symbol sequences or need precedence control.
- Spacing rules:
  - Concatenation uses spaces between every symbol: "a b", not "ab".
  - To union sequences, group them: "(a b c + a c c)".
- Epsilon handling: Use the literal epsilon when needed; prefer satisfying epsilon via an existing Kleene star rather than "epsilon + ...", unless epsilon is explicitly required at the top level.
- Do NOT use: | . ? [] {{}} anchors/lookarounds, multi-character tokens, or any symbol not present in the training data.

"""

SIMPLYRX_OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- First, provide 1-3 concise sentences explaining the observed structure (mandatory prefix/set, block size/pattern, modular length, final-block restriction, epsilon/singleton handling), wrapped in <reasoning> and </reasoning>, e.g.:
  <reasoning>All positives start with c and then repeat 2-char blocks from a restricted set.</reasoning>
- Then output ONLY the final regex wrapped in <ans> and </ans>, e.g.:
  <ans>(a a* b)*</ans>

"""

SIMPLYRX_OUTPUT_INSTR = {
    "syntax": SIMPLYRX_SYNTAX_INSTR,
    "output_format": SIMPLYRX_OUTPUT_FORMAT_INSTR,
    "direct_output_format": DIRECT_OUTPUT_FORMAT_INSTR,
}

TRAINING_DATA_INSTR = """Training Data (Each line has one input-output pair separated by comma):
{0}
"""

SIMPLYRX_PROMPT_TEMPLATE = (
    SIMPLYRX_TASK_INSTR
    + SIMPLYRX_INPUT_INSTR
    + SIMPLYRX_OUTPUT_INSTR["syntax"]
    + SIMPLYRX_INFERENCE_STRATEGY
    + "{regularization_instr}\n{agentic_reflection_instr}\n"
    + SIMPLYRX_OUTPUT_INSTR["output_format"]
    + TRAINING_DATA_INSTR
)

SIMPLYRX_REGULARIZATION = """
CONSTRAINTS
- Prefer simpler, more general regexes while staying consistent with all datapoints.
- Total regex length (ignoring spaces) must be ≤ 60 characters.
- Nesting depth of Kleene stars must be ≤ 4.
- Use only symbols that appear in the training data (eg. a, b, c, epsilon).

"""

SIMPLYRX_CLUSTRED_CE_INSTR = """
- The strings may contain grouped class of characters, e.g., [abc] for letter a or b or c etc.
- Each character class only represent one possible character in the string, e.g., "a[a-c]c" can represent "abc" but not "abcc".

"""

###############################################################Extrx prompt template and instructions#############################################################
EXTRX_SIGMA = "[A-Za-z0-9#]"
EXTRX_TASK_INSTR = """TASK
You will be given labeled strings and must infer a single regular language that matches all positives (label 1) and rejects all negatives (label 0). Output a concise regex in our specified syntax (extended from pyformlang.regular_expression.PythonRegex).

"""

EXTRX_INPUT_INSTR = """INPUT FORMAT
- You receive a block titled “Training Data (Each line has one input-output pair separated by comma):”.
- Each line is "<string>, <label>" where label ∈ {{1, 0}}. The string may be empty; an empty string appears as nothing before the comma (", 1") and represents epsilon.
- The alphabet is fixed. Do not introduce other symbols.
{clustered_ce_instr}
"""

EXTRX_INFERENCE_STRATEGY = """INFERENCE STRATEGY

1) Start/end constraints:
   - Check if all positives start with a specific character or set.
     If so, encode a mandatory prefix (example: "^c.*" becomes "c.*" since anchors are implicit).
   - Check for a forced suffix or ending pattern.
     Place this outside repeating blocks when required.

2) Length/modular and block structure:
   - Look for fixed substrings repeated via "*".
   - If strings mix multiple allowed blocks internally, prefer:
       (block1|block2)*
   - If internal repetitions are freer than endings, use:
       (InternalBlockUnion)* FinalRestrictedBlock
   - If a singleton positive exists (example: "b"), include it using union only if it cannot be captured via star behavior.

3) Union design: star-of-union vs union-of-stars
   - If block types mix inside strings: (A|B|C)*
   - If each string repeats exactly one block type: A*|B*|C*

4) Compactness tactics:
   - Factor common prefixes/suffixes.
   - Use character classes when appropriate:
       [ab] instead of (a|b)
   - Factor repeated substrings inside unions:
       ab(c|d) instead of abc|abd

5) Handling epsilon:
   - Accept epsilon only if explicitly required.
   - Prefer x* instead of (|x)* or (x|).
   - Only use empty alternation (|) when unavoidable.

6) Avoid over-generalization:
   - Do NOT allow patterns contradicted by negatives.

7) Quality checks before finalizing:
   - Verify acceptance of all label-1 strings.
   - Verify rejection of all label-0 strings.
   - Check boundary cases (short strings, empty string).
   - Re-check syntax correctness and grouping.

"""

EXTRX_SYNTAX_INSTR = """EXT REGEX SYNTAX (Extended PythonRegex)

Alphabet
- The alphabet is fixed: Σ = {sigma}
- No other characters may appear anywhere in the regex.
- No escape sequences are supported. Do not use '\\' at all.
Atomic forms
1) Literal character: Any single symbol in Σ
2) Character class:
   - Syntax: [ ... ]
   - Contents may use only:
     * ranges like A-Z, 0-9
     * an individual literal symbol
Core operators (* are extended operators beyond PythonRegex)
- Concatenation: implicit by adjacency
  Example: ab means 'a' followed by 'b'
- Union (OR): |
  Example: a|b means 'a' or 'b'
- Grouping: ( ... )
  Parentheses define scope and precedence.
- *Conjunction / intersection: &
  Semantics: L(R1 & R2) = L(R1) ∩ L(R2)
- *Negation / complement: ~(R)
  Semantics: L(~(R)) = Σ* \\ L(R)
  Negation must always be written with parentheses: ~( ... )
Quantifiers
Quantifiers apply to the immediately preceding atom or parenthesized group.
- * : zero or more
- + : one or more
- ? : zero or one
- {{n}} : exactly n repetitions (n is a nonnegative integer)
- {{n,m}}: between n and m repetitions inclusive (0 <= n <= m)
- {{n,}} : at least n repetitions, equivalent to “(E){{n}}(E)*”
Associativity
- Concatenation, &, and | are left-associative.
- Parenthesize whenever there is ambiguity.
Priority (from highest to lowest): Quantifiers, ~, Concatenation, &, |
Prohibited constructs (must not appear)
- Do not use '.' (dot). Use [A-Za-z0-9#] explicitly when you need Σ.
- Do not use negated character classes [^...].
- Do not use anchors ^ or $.
- Do not use word boundary \\b.
- Do not use lookarounds or backreferences.

"""

EXTRX_OUTPUT_FORMAT_INSTR = """OUTPUT FORMAT
- First, provide 1-3 concise sentences explaining the observed structure (mandatory prefix/set, block size/pattern, modular length, final-block restriction, epsilon/singleton handling), wrapped in <reasoning> and </reasoning>, e.g.:
  <reasoning>Accepted strings are length-5 alphanumerics that must not contain vowels.</reasoning>
- Then output ONLY the final regex wrapped in <ans> and </ans>, e.g.:
  <ans>(a*([A-Z])|(b))*</ans>

"""

EXTRX_OUTPUT_INSTR = {
    "syntax": EXTRX_SYNTAX_INSTR,
    "output_format": EXTRX_OUTPUT_FORMAT_INSTR,
    "direct_output_format": DIRECT_OUTPUT_FORMAT_INSTR,
}

EXTRX_PROMPT_TEMPLATE = (
    EXTRX_TASK_INSTR
    + EXTRX_INPUT_INSTR
    + EXTRX_OUTPUT_INSTR["syntax"]
    + EXTRX_INFERENCE_STRATEGY
    + "{regularization_instr}\n{agentic_reflection_instr}\n"
    + EXTRX_OUTPUT_INSTR["output_format"]
    + TRAINING_DATA_INSTR
)

EXTRX_REGULARIZATION = """
CONSTRAINTS
- Prefer simpler, more general regexes while staying consistent with all datapoints.
- Total regex length (ignoring spaces) must be ≤ 150 characters.
- Nesting depth of Kleene stars (*, +, ?) must be ≤ 2.
- Use only symbols that appear in the alphabet (except metacharacters such as (), |, *, +, ?, []).
"""

EXTRX_CLUSTRED_CE_INSTR = """
- The strings may contain grouped class of characters, e.g., [A-Z] for uppercase letters, [^0-9] for non-digits, etc.
- Each character class only represent one possible character in the string, e.g., "a[A-Z]c" can represent "aBc" but not "aBCc".

"""
