"""Regexulator-style validation-guided tree search for SimplyRX.

The search is model-backend agnostic: callers provide a generation function.
This keeps model loading and benchmark orchestration out of the algorithm.
"""

from __future__ import annotations

import heapq
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from tasks.utils import dfa_accepts_ex
from teacher import _equivalent_dfa_and_witness


GenerateFn = Callable[[str, float], dict[str, Any]]


@dataclass
class SearchNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    sibling_index: int
    kind: str
    regex: Optional[str]
    reasoning: Optional[str]
    valid: bool
    error: Optional[str]
    train_accuracy: float
    validation_accuracy: float
    equivalent: bool
    witness: Optional[str]
    generation_call: int
    prompt_tokens: int
    response_tokens: int
    elapsed_seconds: float


def accuracy(dfa, examples: list[str], labels: list[int]) -> float:
    if not examples:
        return 0.0
    correct = sum(
        int(int(dfa_accepts_ex(dfa, example)) == int(label))
        for example, label in zip(examples, labels)
    )
    return correct / len(examples)


def categorized_examples(
    dfa,
    examples: list[str],
    labels: list[int],
) -> dict[str, list[tuple[str, int]]]:
    groups = {"true_positive": [], "false_positive": [], "false_negative": []}
    for example, label in zip(examples, labels):
        prediction = int(dfa_accepts_ex(dfa, example))
        if prediction == 1 and int(label) == 1:
            groups["true_positive"].append((example, 1))
        elif prediction == 1 and int(label) == 0:
            groups["false_positive"].append((example, 0))
        elif prediction == 0 and int(label) == 1:
            groups["false_negative"].append((example, 1))
    return groups


def _format_examples(examples: list[tuple[str, int]]) -> str:
    return "\n".join(f"{example}, {label}" for example, label in examples)


def _base_prompt(alphabet: list[str]) -> str:
    return f"""TASK
Infer a regular language from labeled whole-string examples. A label of 1
means the complete string must match; 0 means it must be rejected.

PYFORMLANG REGEX SYNTAX
- Alphabet: {", ".join(alphabet)}
- Union uses +.
- Concatenation uses spaces, for example "a b", never "ab".
- Kleene star uses *.
- Use parentheses for grouping and the literal epsilon for the empty string.
- Do not use |, ., ?, brackets, braces, anchors, lookarounds, or new symbols.

OUTPUT
Give a concise explanation in <reasoning>...</reasoning>, then exactly one
complete regex in <ans>...</ans>.
"""


def start_prompt(alphabet: list[str], positives: list[tuple[str, int]]) -> str:
    return (
        _base_prompt(alphabet)
        + "\nCreate an initial regex from these positive examples. Prefer a "
        "compact structural generalization.\n\nPositive examples:\n"
        + _format_examples(positives)
    )


def improve_prompt(
    alphabet: list[str],
    current_regex: str,
    validation_accuracy: float,
    examples: dict[str, list[tuple[str, int]]],
) -> str:
    sections = []
    labels = (
        ("true_positive", "Correctly accepted examples to preserve"),
        ("false_positive", "False positives that must be rejected"),
        ("false_negative", "False negatives that must be accepted"),
    )
    for key, title in labels:
        if examples[key]:
            sections.append(f"{title}:\n{_format_examples(examples[key])}")
    feedback = "\n\n".join(sections) or "No sampled mistakes were available."
    return (
        _base_prompt(alphabet)
        + f"\nImprove this regex:\n<current>{current_regex}</current>\n"
        + f"Its validation accuracy is {validation_accuracy:.6f}.\n\n"
        + feedback
        + "\n\nRevise the regex to fix the mistakes while preserving correct behavior."
    )


def repair_prompt(
    alphabet: list[str],
    invalid_regex: Optional[str],
    error: str,
) -> str:
    return (
        _base_prompt(alphabet)
        + "\nThe previous regex did not compile. Repair only its syntax while "
        "preserving its intended language.\n"
        + f"Previous regex: {invalid_regex}\nCompiler error: {error}"
    )


def _sample(
    items: list[tuple[str, int]],
    count: int,
    rng: random.Random,
) -> list[tuple[str, int]]:
    if len(items) <= count:
        result = list(items)
        rng.shuffle(result)
        return result
    return rng.sample(items, count)


def validate_simplyrx_regex(regex: str, alphabet: list[str]) -> None:
    if not regex.strip():
        raise ValueError("Regex is empty")
    tokens = re.findall(r"epsilon|[()+*]|[^\s]", regex)
    allowed = set(alphabet) | {"epsilon", "(", ")", "+", "*"}
    unsupported = sorted({token for token in tokens if token not in allowed})
    if unsupported:
        raise ValueError(f"Unsupported SimplyRX token(s): {unsupported}")


def select_feedback(
    groups: dict[str, list[tuple[str, int]]],
    max_examples: int,
    rng: random.Random,
) -> dict[str, list[tuple[str, int]]]:
    selected = {key: [] for key in groups}
    order = ["false_positive", "false_negative", "true_positive"]
    while sum(len(values) for values in selected.values()) < max_examples:
        changed = False
        for key in order:
            remaining = [item for item in groups[key] if item not in selected[key]]
            if remaining:
                selected[key].append(rng.choice(remaining))
                changed = True
                if sum(len(values) for values in selected.values()) >= max_examples:
                    break
        if not changed:
            break
    return selected


def run_regexulator_search(
    *,
    task,
    train_examples: list[str],
    train_labels: list[int],
    validation_examples: list[str],
    validation_labels: list[int],
    generate: GenerateFn,
    tokenizer,
    initial_splits: int = 4,
    max_depth: int = 3,
    branching_factor: int = 2,
    max_generation_calls: int = 16,
    time_limit_seconds: Optional[float] = 200.0,
    start_examples: int = 10,
    improve_examples: int = 5,
    max_compile_repairs: int = 1,
    depth_base: float = 0.9,
    sibling_base: float = 0.95,
    temperature: float = 0.7,
    seed: int = 0,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    deadline = (
        started_at + time_limit_seconds
        if time_limit_seconds is not None and time_limit_seconds > 0
        else None
    )
    rng = random.Random(seed)
    alphabet = sorted(str(symbol.value) for symbol in task.sigma)
    target_dfa = task.dfa
    positives = [
        (example, 1)
        for example, label in zip(train_examples, train_labels)
        if int(label) == 1
    ]
    if not positives:
        raise ValueError("Regexulator requires at least one positive training example")

    nodes: list[SearchNode] = []
    raw_generations: list[dict[str, Any]] = []
    frontier: list[tuple[float, int, int]] = []
    compiled_dfas: dict[int, Any] = {}
    generation_calls = 0
    total_prompt_tokens = 0
    total_response_tokens = 0
    best_node_id: Optional[int] = None
    successful_node_id: Optional[int] = None

    def time_limit_reached() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    def can_generate() -> bool:
        return (
            generation_calls < max_generation_calls
            and not time_limit_reached()
        )

    def count_tokens(text: Optional[str]) -> int:
        if not text:
            return 0
        if tokenizer is None:
            return len(str(text).split())
        try:
            return len(tokenizer.encode(str(text), add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(str(text)))

    def generate_node(
        prompt: str,
        *,
        parent_id: Optional[int],
        depth: int,
        sibling_index: int,
        kind: str,
    ) -> Optional[int]:
        nonlocal generation_calls, total_prompt_tokens, total_response_tokens
        nonlocal best_node_id, successful_node_id
        if not can_generate():
            return None

        call_started_at = time.perf_counter()
        generation_calls += 1
        message = generate(prompt, temperature)
        prediction = message.get("Prediction")
        reasoning = message.get("Reasoning")
        response = message.get("Response")
        prompt_token_count = count_tokens(prompt)
        response_token_count = count_tokens(response)
        total_prompt_tokens += prompt_token_count
        total_response_tokens += response_token_count

        valid = False
        error = None
        learned_dfa = None
        try:
            if not prediction:
                raise ValueError("No <ans> regex was extracted")
            validate_simplyrx_regex(prediction, alphabet)
            learned_dfa = task.regex_to_dfa(prediction)
            candidate_alphabet = {
                str(symbol.value) for symbol in learned_dfa.symbols
            }
            unexpected_symbols = candidate_alphabet - set(alphabet)
            if unexpected_symbols:
                raise ValueError(
                    f"Regex introduced symbols outside the alphabet: "
                    f"{sorted(unexpected_symbols)}"
                )
            valid = True
        except Exception as exc:
            error = str(exc)

        train_score = (
            accuracy(learned_dfa, train_examples, train_labels) if valid else 0.0
        )
        validation_score = (
            accuracy(learned_dfa, validation_examples, validation_labels)
            if valid
            else 0.0
        )
        equivalent, witness = (
            _equivalent_dfa_and_witness(target_dfa, learned_dfa)
            if valid
            else (False, None)
        )
        node_id = len(nodes)
        node = SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            sibling_index=sibling_index,
            kind=kind,
            regex=prediction,
            reasoning=reasoning,
            valid=valid,
            error=error,
            train_accuracy=train_score,
            validation_accuracy=validation_score,
            equivalent=equivalent,
            witness=witness,
            generation_call=generation_calls,
            prompt_tokens=prompt_token_count,
            response_tokens=response_token_count,
            elapsed_seconds=time.perf_counter() - call_started_at,
        )
        nodes.append(node)
        raw_generations.append(
            {
                "node_id": node_id,
                "prompt": prompt,
                "response": response,
            }
        )
        if valid:
            compiled_dfas[node_id] = learned_dfa
            if (
                best_node_id is None
                or validation_score > nodes[best_node_id].validation_accuracy
            ):
                best_node_id = node_id
            priority = -(
                validation_score
                * (depth_base ** depth)
                * (sibling_base ** sibling_index)
            )
            heapq.heappush(frontier, (priority, generation_calls, node_id))
        if equivalent:
            successful_node_id = node_id
        return node_id

    def repair_invalid(node_id: Optional[int]) -> Optional[int]:
        repairs = 0
        current_id = node_id
        while (
            current_id is not None
            and not nodes[current_id].valid
            and repairs < max_compile_repairs
            and can_generate()
        ):
            invalid = nodes[current_id]
            current_id = generate_node(
                repair_prompt(alphabet, invalid.regex, invalid.error or "unknown"),
                parent_id=current_id,
                depth=invalid.depth,
                sibling_index=invalid.sibling_index,
                kind="compile_repair",
            )
            repairs += 1
        return current_id

    for split_index in range(initial_splits):
        sampled = _sample(positives, start_examples, rng)
        node_id = generate_node(
            start_prompt(alphabet, sampled),
            parent_id=None,
            depth=0,
            sibling_index=split_index,
            kind="start",
        )
        repair_invalid(node_id)
        if successful_node_id is not None or node_id is None:
            break

    while (
        frontier
        and can_generate()
        and successful_node_id is None
    ):
        _, _, parent_id = heapq.heappop(frontier)
        parent = nodes[parent_id]
        if parent.depth >= max_depth:
            continue
        parent_dfa = compiled_dfas[parent_id]
        groups = categorized_examples(parent_dfa, train_examples, train_labels)
        for sibling_index in range(branching_factor):
            feedback = select_feedback(groups, improve_examples, rng)
            child_id = generate_node(
                improve_prompt(
                    alphabet,
                    parent.regex or "",
                    parent.validation_accuracy,
                    feedback,
                ),
                parent_id=parent_id,
                depth=parent.depth + 1,
                sibling_index=sibling_index,
                kind="improve",
            )
            if child_id is None or successful_node_id is not None:
                break
            repair_invalid(child_id)
            if successful_node_id is not None:
                break

    selected_id = successful_node_id if successful_node_id is not None else best_node_id
    selected = nodes[selected_id] if selected_id is not None else None
    timed_out = time_limit_reached()
    if successful_node_id is not None:
        stop_reason = "equivalent"
    elif timed_out:
        stop_reason = "time_limit"
    elif generation_calls >= max_generation_calls:
        stop_reason = "call_limit"
    else:
        stop_reason = "frontier_exhausted"
    return {
        "equivalent": successful_node_id is not None,
        "timed_out": timed_out,
        "stop_reason": stop_reason,
        "successful_call": (
            nodes[successful_node_id].generation_call
            if successful_node_id is not None
            else None
        ),
        "selected_node_id": selected_id,
        "selected_regex": selected.regex if selected else None,
        "selected_train_accuracy": selected.train_accuracy if selected else 0.0,
        "selected_validation_accuracy": (
            selected.validation_accuracy if selected else 0.0
        ),
        "witness": selected.witness if selected else None,
        "generation_calls": generation_calls,
        "valid_candidates": sum(node.valid for node in nodes),
        "invalid_candidates": sum(not node.valid for node in nodes),
        "prompt_tokens": total_prompt_tokens,
        "response_tokens": total_response_tokens,
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "wall_time_seconds": time.perf_counter() - started_at,
        "nodes": [asdict(node) for node in nodes],
        "generations": raw_generations,
    }
