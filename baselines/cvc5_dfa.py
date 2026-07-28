#!/usr/bin/env python3
"""Synthesize a DFA consistent with labeled examples using CVC5."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from cvc5.pythonic import (
    Bool,
    Implies,
    Not,
    Or,
    Solver,
    is_true,
    sat,
    unknown,
)

from baselines.state_merging import LearnedDFA


@dataclass
class PrefixTree:
    prefixes: list[str]
    parent: list[Optional[int]]
    symbol: list[Optional[str]]
    endpoint_labels: dict[int, int]


def build_prefix_tree(
    examples: Sequence[str],
    labels: Sequence[int],
) -> PrefixTree:
    if len(examples) != len(labels):
        raise ValueError("examples and labels must have the same length")

    prefixes = [""]
    parent: list[Optional[int]] = [None]
    symbols: list[Optional[str]] = [None]
    prefix_to_index = {"": 0}
    endpoint_labels: dict[int, int] = {}

    for example, raw_label in zip(examples, labels):
        label = int(raw_label)
        if label not in {0, 1}:
            raise ValueError(f"Expected binary label, got {raw_label!r}")
        prefix = ""
        parent_index = 0
        for char in example:
            prefix += char
            if prefix not in prefix_to_index:
                prefix_to_index[prefix] = len(prefixes)
                prefixes.append(prefix)
                parent.append(parent_index)
                symbols.append(char)
            parent_index = prefix_to_index[prefix]
        previous = endpoint_labels.get(parent_index)
        if previous is not None and previous != label:
            raise ValueError(f"Contradictory labels for example {example!r}")
        endpoint_labels[parent_index] = label

    return PrefixTree(prefixes, parent, symbols, endpoint_labels)


def _solve_for_num_states(
    tree: PrefixTree,
    alphabet: list[str],
    num_states: int,
    timeout_ms: Optional[int],
) -> tuple[Optional[LearnedDFA], dict[str, Any]]:
    solver = Solver()
    if timeout_ms is not None:
        solver.set("tlimit-per", timeout_ms)

    transitions = {
        symbol: [
            [
                Bool(f"delta_{num_states}_{symbol}_{source}_{target}")
                for target in range(num_states)
            ]
            for source in range(num_states)
        ]
        for symbol in alphabet
    }
    accepting = [
        Bool(f"accepting_{num_states}_{state}")
        for state in range(num_states)
    ]
    prefix_states = [
        [
            Bool(f"prefix_state_{num_states}_{index}_{state}")
            for state in range(num_states)
        ]
        for index in range(len(tree.prefixes))
    ]

    def exactly_one(terms):
        solver.add(Or(*terms))
        for left in range(len(terms)):
            for right in range(left + 1, len(terms)):
                solver.add(Or(Not(terms[left]), Not(terms[right])))

    solver.add(prefix_states[0][0])
    for state in range(1, num_states):
        solver.add(Not(prefix_states[0][state]))
    for state_terms in prefix_states[1:]:
        exactly_one(state_terms)
    for symbol in alphabet:
        for source in range(num_states):
            exactly_one(transitions[symbol][source])

    # Break a large class of state-renaming symmetries: every non-initial
    # state must first be reachable from a lower-numbered state.
    for state in range(1, num_states):
        solver.add(
            Or(
                *[
                    transitions[symbol][source][state]
                    for source in range(state)
                    for symbol in alphabet
                ]
            )
        )

    for index in range(1, len(tree.prefixes)):
        parent_index = tree.parent[index]
        symbol = tree.symbol[index]
        for source in range(num_states):
            for target in range(num_states):
                solver.add(
                    Implies(
                        prefix_states[parent_index][source],
                        (
                            transitions[symbol][source][target]
                            == prefix_states[index][target]
                        ),
                    )
                )

    for endpoint, label in tree.endpoint_labels.items():
        for state in range(num_states):
            solver.add(
                Implies(
                    prefix_states[endpoint][state],
                    accepting[state] if label else Not(accepting[state]),
                )
            )

    started_at = time.perf_counter()
    result = solver.check()
    solve_time = time.perf_counter() - started_at
    attempt = {
        "num_states": num_states,
        "status": str(result),
        "solve_time_seconds": solve_time,
    }
    if result != sat:
        if result == unknown:
            attempt["unknown_reason"] = str(solver.reason_unknown())
        return None, attempt

    model = solver.model()
    learned = LearnedDFA()
    learned.start = 0
    learned.next_state = num_states
    learned.transitions = {state: {} for state in range(num_states)}
    learned.prefixes = {state: "" for state in range(num_states)}
    learned.accepting = set()
    learned.rejecting = set()

    for state in range(num_states):
        if is_true(model.eval(accepting[state], model_completion=True)):
            learned.accepting.add(state)
        else:
            learned.rejecting.add(state)
        for symbol in alphabet:
            target = next(
                candidate
                for candidate in range(num_states)
                if is_true(
                    model.eval(
                        transitions[symbol][state][candidate],
                        model_completion=True,
                    )
                )
            )
            learned.transitions[state][symbol] = target

    return learned, attempt


def learn_cvc5_dfa(
    examples: Sequence[str],
    labels: Sequence[int],
    alphabet: Iterable[str],
    *,
    min_states: int = 1,
    max_states: int = 12,
    timeout_ms_per_round: Optional[int] = None,
) -> tuple[Optional[LearnedDFA], dict[str, Any]]:
    """Find the smallest bounded DFA consistent with all labeled examples."""

    alphabet = sorted(set(alphabet))
    tree = build_prefix_tree(examples, labels)
    attempts = []
    learned = None
    started_at = time.perf_counter()

    for num_states in range(min_states, max_states + 1):
        learned, attempt = _solve_for_num_states(
            tree,
            alphabet,
            num_states,
            timeout_ms_per_round,
        )
        attempts.append(attempt)
        if learned is not None:
            break

    total_time = time.perf_counter() - started_at
    metrics = {
        "satisfiable": learned is not None,
        "solver_rounds": len(attempts),
        "successful_round": len(attempts) if learned is not None else None,
        "learned_num_states": len(learned.states) if learned is not None else None,
        "num_examples": len(examples),
        "num_unique_examples": len(set(examples)),
        "num_prefixes": len(tree.prefixes),
        "attempts": attempts,
        "total_solver_time_seconds": sum(
            attempt["solve_time_seconds"] for attempt in attempts
        ),
        "wall_time_seconds": total_time,
        "min_states": min_states,
        "max_states": max_states,
        "timeout_ms_per_round": timeout_ms_per_round,
    }
    return learned, metrics
