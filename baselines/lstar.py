#!/usr/bin/env python3
"""AALpy L* baseline for learning SimplyRx target languages."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Iterable, Optional

from aalpy.base import Oracle, SUL
from aalpy.learning_algs import run_Lstar
from pyformlang.finite_automaton import Symbol


def _target_next(target_dfa, state, symbol: str):
    if state is None:
        return None
    next_states = target_dfa._transition_function(state, Symbol(symbol))
    return next(iter(next_states), None)


class PyformlangDfaSUL(SUL):
    """Expose a pyformlang DFA as an AALpy membership-query SUL."""

    def __init__(self, target_dfa):
        super().__init__()
        self.target_dfa = target_dfa
        self.current_state = target_dfa.start_state

    def pre(self):
        self.current_state = self.target_dfa.start_state

    def post(self):
        pass

    def step(self, letter):
        if letter is not None:
            self.current_state = _target_next(
                self.target_dfa, self.current_state, str(letter)
            )
        return (
            self.current_state is not None
            and self.current_state in self.target_dfa.final_states
        )


class ExactDfaEqOracle(Oracle):
    """Return a shortest counterexample via an exact product-DFA search."""

    def __init__(self, alphabet: Iterable[str], sul: SUL, target_dfa):
        super().__init__(sorted(set(alphabet)), sul)
        self.target_dfa = target_dfa
        self.counterexamples: list[str] = []
        self.product_states_explored: list[int] = []

    def find_cex(self, hypothesis):
        self.num_queries += 1
        start = (hypothesis.initial_state, self.target_dfa.start_state, ())
        queue = deque([start])
        visited = {(hypothesis.initial_state, self.target_dfa.start_state)}
        explored = 0

        while queue:
            hypothesis_state, target_state, witness = queue.popleft()
            explored += 1
            hypothesis_accepts = bool(hypothesis_state.is_accepting)
            target_accepts = (
                target_state is not None
                and target_state in self.target_dfa.final_states
            )
            if hypothesis_accepts != target_accepts:
                counterexample = "".join(witness)
                self.counterexamples.append(counterexample)
                self.product_states_explored.append(explored)
                return witness

            for symbol in self.alphabet:
                self.num_steps += 1
                next_hypothesis = hypothesis_state.transitions[symbol]
                next_target = _target_next(self.target_dfa, target_state, symbol)
                pair = (next_hypothesis, next_target)
                if pair in visited:
                    continue
                visited.add(pair)
                queue.append((next_hypothesis, next_target, witness + (symbol,)))

        self.product_states_explored.append(explored)
        return None


def hypothesis_accepts(hypothesis, string: str) -> bool:
    state = hypothesis.initial_state
    for symbol in string:
        state = state.transitions[symbol]
    return bool(state.is_accepting)


def evaluate_hypothesis(
    hypothesis,
    examples: list[str],
    labels: list[int],
) -> float:
    if not examples:
        return 0.0
    correct = sum(
        hypothesis_accepts(hypothesis, example) == bool(label)
        for example, label in zip(examples, labels)
    )
    return correct / len(examples)


def learn_lstar(
    target_dfa,
    alphabet: Iterable[str],
    *,
    max_learning_rounds: Optional[int] = None,
    closing_strategy: str = "shortest_first",
    cex_processing: str = "rs",
    print_level: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Learn a target DFA and return the hypothesis plus detailed metrics."""

    alphabet = sorted(set(alphabet))
    target_sul = PyformlangDfaSUL(target_dfa)
    eq_oracle = ExactDfaEqOracle(alphabet, target_sul, target_dfa)

    started_at = time.perf_counter()
    hypothesis, aalpy_info = run_Lstar(
        alphabet,
        target_sul,
        eq_oracle,
        automaton_type="dfa",
        closing_strategy=closing_strategy,
        cex_processing=cex_processing,
        max_learning_rounds=max_learning_rounds,
        cache_and_non_det_check=True,
        return_data=True,
        print_level=print_level,
    )
    wall_time = time.perf_counter() - started_at

    final_counterexample = eq_oracle.find_cex(hypothesis)
    equivalent = final_counterexample is None
    final_verification_product_states = eq_oracle.product_states_explored.pop()
    if final_counterexample is not None:
        # The final verification call is diagnostic, not a learning round.
        eq_oracle.counterexamples.pop()

    metrics = {
        "equivalent": equivalent,
        "learning_rounds": int(aalpy_info["learning_rounds"]),
        "successful_round": (
            int(aalpy_info["learning_rounds"]) if equivalent else None
        ),
        "num_counterexamples": len(eq_oracle.counterexamples),
        "counterexamples": list(eq_oracle.counterexamples),
        "counterexample_lengths": [
            len(counterexample) for counterexample in eq_oracle.counterexamples
        ],
        "membership_queries": int(aalpy_info["queries_learning"]),
        "membership_steps": int(aalpy_info["steps_learning"]),
        "uncached_target_queries": int(target_sul.num_queries),
        "uncached_target_steps": int(target_sul.num_steps),
        "cache_saved_queries": int(aalpy_info.get("cache_saved", 0)),
        "equivalence_queries": int(aalpy_info["queries_eq_oracle"]),
        "equivalence_steps": int(aalpy_info["steps_eq_oracle"]),
        "product_states_explored_per_eq": list(eq_oracle.product_states_explored),
        "final_verification_product_states": final_verification_product_states,
        "learned_num_states": len(hypothesis.states),
        "learning_time_seconds": float(aalpy_info["learning_time"]),
        "equivalence_time_seconds": float(aalpy_info["eq_oracle_time"]),
        "aalpy_total_time_seconds": float(aalpy_info["total_time"]),
        "wall_time_seconds": wall_time,
        "closing_strategy": closing_strategy,
        "cex_processing": cex_processing,
    }
    return hypothesis, metrics
