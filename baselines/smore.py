"""Smore-style, type-erased sketch synthesis for SimplyRx.

This is an adaptation, not the original Smore implementation.  Smore's
semantic types are replaced by one inert ``Default`` type.  An LLM proposes a
regex sketch, a bounded symbolic search fills its holes, and failed training
examples are used to ask for a repaired sketch.  Exact DFA equivalence is used
only to evaluate the final candidate; equivalence witnesses are never exposed
to the synthesizer.
"""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from baselines.regexulator import accuracy, validate_simplyrx_regex
from tasks.utils import dfa_accepts_ex
from teacher import _equivalent_dfa_and_witness


GenerateFn = Callable[[str, float], dict[str, Any]]
HOLE_RE = re.compile(r"\{\{\s*(H\d+)\s*(?::\s*[A-Za-z_][A-Za-z0-9_]*)?\s*\}\}")


@dataclass
class SketchAttempt:
    iteration: int
    sketch: Optional[str]
    normalized_sketch: Optional[str]
    holes: list[str]
    valid_sketch: bool
    error: Optional[str]
    combinations_tried: int
    consistent: bool
    candidate_regex: Optional[str]
    train_accuracy: float
    eval_accuracy: float
    equivalent: bool
    witness: Optional[str]
    prompt_tokens: int
    response_tokens: int
    elapsed_seconds: float


def _count_tokens(tokenizer, text: Optional[str]) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(str(text).split())
    try:
        return len(tokenizer.encode(str(text), add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(str(text)))


def extract_sketch(response: Optional[str]) -> Optional[str]:
    if not response:
        return None
    # Reasoning models often quote the requested format first, e.g.
    # ``<sketch>...</sketch>``, and put the actual answer in a later tag.  The
    # last tagged block is therefore the answer, not the first one.
    matches = re.findall(
        r"<sketch>\s*(.*?)\s*</sketch>", response, re.DOTALL | re.IGNORECASE
    )
    if not matches:
        return None
    sketch = matches[-1].strip()
    if sketch.startswith("```") and sketch.endswith("```"):
        sketch = re.sub(r"^```[^\n]*\n?", "", sketch)
        sketch = re.sub(r"\n?```$", "", sketch).strip()
    return sketch


def normalize_sketch(sketch: str) -> tuple[str, list[str]]:
    """Erase hole types and return holes in first-occurrence order."""
    holes: list[str] = []

    # Check the source before replacing valid holes with their canonical form.
    residue = HOLE_RE.sub("", sketch)
    if "{{" in residue or "}}" in residue:
        raise ValueError("Malformed hole; use {{H0:Default}}, {{H1:Default}}, ...")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in holes:
            holes.append(name)
        return "{{" + name + "}}"

    normalized = HOLE_RE.sub(replace, sketch.strip())
    return normalized, holes


def _complexity(regex: str) -> tuple[int, int, str]:
    return (regex.count("(") + regex.count("*") + regex.count("+"), len(regex), regex)


def completion_library(alphabet: list[str], max_candidates: int = 192) -> list[str]:
    """Build a deterministic bounded library for type-erased holes.

    The sketch is expected to carry most high-level structure.  The library
    covers symbols, unions, stars, and short concatenations without attempting
    unbounded whole-regex enumeration.
    """
    symbols = sorted(set(alphabet))
    candidates = {"epsilon", *symbols}
    unions: set[str] = set()
    for size in range(2, len(symbols) + 1):
        for subset in itertools.combinations(symbols, size):
            unions.add("(" + "+".join(subset) + ")")
    candidates.update(unions)
    star_bases = symbols + sorted(unions)
    candidates.update(f"({base})*" for base in star_bases)

    concat_atoms = symbols + [f"({base})*" for base in star_bases]
    for length in (2, 3):
        for parts in itertools.product(concat_atoms, repeat=length):
            candidates.add("(" + " ".join(parts) + ")")

    ranked = sorted(candidates, key=_complexity)
    return ranked[:max_candidates]


def instantiate_sketch(sketch: str, assignment: dict[str, str]) -> str:
    result = sketch
    for hole, completion in assignment.items():
        result = result.replace("{{" + hole + "}}", "(" + completion + ")")
    return result


def validate_candidate_syntax(regex: str, alphabet: list[str]) -> None:
    """Validate the restricted SimplyRx grammar (where ``+`` is only union)."""
    validate_simplyrx_regex(regex, alphabet)
    tokens = re.findall(r"epsilon|[()+*]|[^\s]", regex)
    position = 0

    def peek() -> Optional[str]:
        return tokens[position] if position < len(tokens) else None

    def parse_atom() -> None:
        nonlocal position
        token = peek()
        if token in set(alphabet) | {"epsilon"}:
            position += 1
            return
        if token == "(":
            position += 1
            parse_union()
            if peek() != ")":
                raise ValueError("Unmatched '(' in SimplyRx expression")
            position += 1
            return
        raise ValueError(f"Expected a SimplyRx atom, found {token!r}")

    def parse_repeat() -> None:
        nonlocal position
        parse_atom()
        if peek() == "*":
            position += 1

    def starts_atom() -> bool:
        return peek() in set(alphabet) | {"epsilon", "("}

    def parse_concat() -> None:
        if not starts_atom():
            raise ValueError(f"Expected a concatenation operand, found {peek()!r}")
        while starts_atom():
            parse_repeat()

    def parse_union() -> None:
        nonlocal position
        parse_concat()
        while peek() == "+":
            position += 1
            parse_concat()

    parse_union()
    if position != len(tokens):
        raise ValueError(f"Unexpected SimplyRx token {tokens[position]!r}")


def _mistakes(dfa, examples: list[str], labels: list[int], limit: int = 16) -> list[tuple[str, int, int]]:
    result = []
    for example, label in zip(examples, labels):
        prediction = int(dfa_accepts_ex(dfa, example))
        if prediction != int(label):
            result.append((example, int(label), prediction))
            if len(result) >= limit:
                break
    return result


def _format_examples(examples: list[str], labels: list[int], limit: int) -> str:
    pairs = list(zip(examples, labels))[:limit]
    return "\n".join(f"{s if s else 'epsilon'}, {int(y)}" for s, y in pairs)


def initial_prompt(alphabet: list[str], examples: list[str], labels: list[int], limit: int) -> str:
    return f"""You are the neural sketch generator in a type-erased adaptation of Smore.
Infer a SimplyRx regular expression from labeled whole-string examples.

SYNTAX
- Alphabet: {', '.join(alphabet)}
- Union is +; concatenation MUST contain spaces; Kleene star is *.
- Parentheses group expressions; epsilon is the empty string.
- Do not use |, ., ?, brackets, braces, anchors, or new alphabet symbols.

SKETCH HOLES
- Use holes such as {{{{H0:Default}}}}, {{{{H1:Default}}}}.
- Every hole and every terminal ({', '.join(alphabet)}) has the same inert
  Default type. Types are ignored during symbolic completion.
- Put the high-level union/concatenation/star structure in the sketch and use
  holes for small subexpressions. Use at most 3 distinct holes.
- A complete regex with no holes is also allowed.

Return a short explanation and exactly one sketch in <sketch>...</sketch>.

Examples (string, label):
{_format_examples(examples, labels, limit)}"""


def repair_prompt(
    alphabet: list[str],
    previous_sketch: Optional[str],
    previous_candidate: Optional[str],
    mistakes: list[tuple[str, int, int]],
    error: Optional[str],
    examples: list[str],
    labels: list[int],
    example_limit: int,
) -> str:
    feedback = "\n".join(
        f"{s if s else 'epsilon'}: expected {expected}, got {actual}"
        for s, expected, actual in mistakes
    )
    if not feedback:
        feedback = error or "No completion of the sketch was consistent with all examples."
    return f"""Repair a type-erased Smore sketch for a SimplyRx language.
Alphabet: {', '.join(alphabet)}. Union is +, concatenation uses spaces, and
Kleene star is *. All holes have the same ignored Default type. Use at most 3
distinct holes and output exactly one <sketch>...</sketch>.

Previous sketch: {previous_sketch}
Best symbolic completion: {previous_candidate}
Failure information:
{feedback}

Original labeled examples (string, label):
{_format_examples(examples, labels, example_limit)}

Change the high-level structure so bounded symbolic hole completion can satisfy
the examples. Holes must use exactly the form {{{{H0:Default}}}}. Do not use
ellipsis, question-mark holes, semantic predicates, or operators outside
SimplyRx."""


def synthesize_completion(
    *,
    task,
    sketch: str,
    holes: list[str],
    examples: list[str],
    labels: list[int],
    library: list[str],
    max_combinations: int,
    deadline: Optional[float] = None,
) -> tuple[Optional[str], Any, float, int, Optional[str]]:
    """Enumerate hole assignments and return a consistent or best candidate."""
    if len(holes) > 3:
        return None, None, 0.0, 0, "Sketch has more than 3 distinct holes"
    assignments = itertools.product(library, repeat=len(holes)) if holes else [()]
    best_regex = None
    best_dfa = None
    best_score = -1.0
    tried = 0
    last_error = None
    alphabet = sorted(str(symbol.value) for symbol in task.sigma)
    for values in assignments:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if tried >= max_combinations:
            break
        tried += 1
        candidate = instantiate_sketch(sketch, dict(zip(holes, values)))
        try:
            validate_candidate_syntax(candidate, alphabet)
            dfa = task.regex_to_dfa(candidate)
            candidate_alphabet = {
                str(symbol.value) for symbol in dfa.symbols
            }
            unexpected = candidate_alphabet - set(alphabet)
            if unexpected:
                raise ValueError(
                    "Candidate introduced symbols outside the character "
                    f"alphabet: {sorted(unexpected)}. Put spaces between "
                    "concatenated letters."
                )
            score = accuracy(dfa, examples, labels)
        except Exception as exc:
            last_error = str(exc)
            continue
        if score > best_score or (score == best_score and _complexity(candidate) < _complexity(best_regex or "z" * 10000)):
            best_regex, best_dfa, best_score = candidate, dfa, score
        if score == 1.0:
            return candidate, dfa, score, tried, None
    error = None if best_dfa is not None else (last_error or "No compilable completion")
    return best_regex, best_dfa, max(best_score, 0.0), tried, error


def run_smore_search(
    *,
    task,
    train_examples: list[str],
    train_labels: list[int],
    eval_examples: list[str],
    eval_labels: list[int],
    generate: GenerateFn,
    tokenizer=None,
    max_iterations: int = 5,
    max_prompt_examples: int = 80,
    max_hole_candidates: int = 192,
    max_combinations: int = 50000,
    temperature: float = 0.0,
    time_limit_seconds: Optional[float] = 200.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = (
        started + time_limit_seconds
        if time_limit_seconds is not None and time_limit_seconds > 0
        else None
    )
    alphabet = sorted(str(symbol.value) for symbol in task.sigma)
    library = completion_library(alphabet, max_hole_candidates)
    attempts: list[SketchAttempt] = []
    generations: list[dict[str, Any]] = []
    prompt_tokens = response_tokens = 0
    previous_sketch = previous_candidate = previous_error = None
    previous_mistakes: list[tuple[str, int, int]] = []
    best: Optional[SketchAttempt] = None

    for iteration in range(max_iterations):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        prompt = (
            initial_prompt(alphabet, train_examples, train_labels, max_prompt_examples)
            if iteration == 0
            else repair_prompt(
                alphabet,
                previous_sketch,
                previous_candidate,
                previous_mistakes,
                previous_error,
                train_examples,
                train_labels,
                max_prompt_examples,
            )
        )
        call_started = time.perf_counter()
        message = generate(prompt, temperature)
        response = message.get("Response")
        sketch = message.get("Sketch") or extract_sketch(response)
        pt = _count_tokens(tokenizer, prompt)
        rt = _count_tokens(tokenizer, response)
        prompt_tokens += pt
        response_tokens += rt
        normalized = None
        holes: list[str] = []
        valid_sketch = False
        error = None
        candidate = None
        dfa = None
        score = 0.0
        tried = 0
        try:
            if not sketch:
                raise ValueError("No <sketch>...</sketch> was extracted")
            normalized, holes = normalize_sketch(sketch)
            candidate, dfa, score, tried, error = synthesize_completion(
                task=task,
                sketch=normalized,
                holes=holes,
                examples=train_examples,
                labels=train_labels,
                library=library,
                max_combinations=max_combinations,
                deadline=deadline,
            )
            valid_sketch = dfa is not None
        except Exception as exc:
            error = str(exc)

        consistent = dfa is not None and score == 1.0
        eval_score = accuracy(dfa, eval_examples, eval_labels) if dfa is not None else 0.0
        equivalent, witness = _equivalent_dfa_and_witness(task.dfa, dfa) if dfa is not None else (False, None)
        attempt = SketchAttempt(
            iteration=iteration,
            sketch=sketch,
            normalized_sketch=normalized,
            holes=holes,
            valid_sketch=valid_sketch,
            error=error,
            combinations_tried=tried,
            consistent=consistent,
            candidate_regex=candidate,
            train_accuracy=score,
            eval_accuracy=eval_score,
            equivalent=equivalent,
            witness=witness,
            prompt_tokens=pt,
            response_tokens=rt,
            elapsed_seconds=time.perf_counter() - call_started,
        )
        attempts.append(attempt)
        generations.append({"iteration": iteration, "prompt": prompt, "response": response})
        # Candidate selection must not inspect held-out evaluation labels.
        if best is None or attempt.train_accuracy > best.train_accuracy:
            best = attempt
        if consistent:
            break
        previous_sketch = sketch
        previous_candidate = candidate
        previous_error = error
        previous_mistakes = _mistakes(dfa, train_examples, train_labels) if dfa is not None else []

    selected = best
    timed_out = deadline is not None and time.perf_counter() >= deadline
    if timed_out:
        stop_reason = "time_limit"
    elif selected and selected.consistent:
        stop_reason = "training_consistent"
    else:
        stop_reason = "iteration_limit"
    return {
        "equivalent": bool(selected and selected.equivalent),
        "consistent": bool(selected and selected.consistent),
        "selected_regex": selected.candidate_regex if selected else None,
        "selected_sketch": selected.sketch if selected else None,
        "selected_iteration": selected.iteration if selected else None,
        "train_accuracy": selected.train_accuracy if selected else 0.0,
        "eval_accuracy": selected.eval_accuracy if selected else 0.0,
        "witness": selected.witness if selected else None,
        "iterations": len(attempts),
        "timed_out": timed_out,
        "stop_reason": stop_reason,
        "completion_library_size": len(library),
        "total_combinations_tried": sum(a.combinations_tried for a in attempts),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "wall_time_seconds": time.perf_counter() - started,
        "attempts": [asdict(a) for a in attempts],
        "generations": generations,
    }
