#!/usr/bin/env python3
"""State-merging DFA baselines for labeled string examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pyformlang.finite_automaton import Symbol


SINK_STATE = -1


@dataclass
class LearnedDFA:
    start: int = 0
    transitions: Dict[int, Dict[str, int]] = field(default_factory=dict)
    accepting: Set[int] = field(default_factory=set)
    rejecting: Set[int] = field(default_factory=set)
    prefixes: Dict[int, str] = field(default_factory=dict)
    next_state: int = 1

    def copy(self) -> "LearnedDFA":
        return LearnedDFA(
            start=self.start,
            transitions={state: dict(edges) for state, edges in self.transitions.items()},
            accepting=set(self.accepting),
            rejecting=set(self.rejecting),
            prefixes=dict(self.prefixes),
            next_state=self.next_state,
        )

    @property
    def states(self) -> Set[int]:
        states = set(self.transitions)
        states.update(self.accepting)
        states.update(self.rejecting)
        states.update(self.prefixes)
        for edges in self.transitions.values():
            states.update(edges.values())
        return states

    def accepts(self, string: str) -> bool:
        state = self.start
        for char in string:
            state = self.transitions.get(state, {}).get(char, SINK_STATE)
            if state == SINK_STATE:
                return False
        return state in self.accepting


def build_pta(strings: Sequence[str], labels: Sequence[int]) -> LearnedDFA:
    if len(strings) != len(labels):
        raise ValueError("strings and labels must have the same length")

    dfa = LearnedDFA()
    dfa.transitions[0] = {}
    dfa.prefixes[0] = ""

    endpoint_labels: Dict[str, int] = {}
    for string, label in zip(strings, labels):
        label = int(label)
        if label not in {0, 1}:
            raise ValueError(f"Labels must be 0/1, got {label!r}")
        previous = endpoint_labels.get(string)
        if previous is not None and previous != label:
            raise ValueError(f"Contradictory labels for string {string!r}")
        endpoint_labels[string] = label

    for string, label in endpoint_labels.items():
        state = dfa.start
        prefix = ""
        for char in string:
            prefix += char
            edges = dfa.transitions.setdefault(state, {})
            if char not in edges:
                new_state = dfa.next_state
                dfa.next_state += 1
                edges[char] = new_state
                dfa.transitions[new_state] = {}
                dfa.prefixes[new_state] = prefix
            state = edges[char]
        if label == 1:
            dfa.accepting.add(state)
        else:
            dfa.rejecting.add(state)

    if dfa.accepting & dfa.rejecting:
        raise ValueError("PTA has a state marked as both accepting and rejecting")
    return dfa


def _state_order(dfa: LearnedDFA, state: int) -> Tuple[int, str, int]:
    prefix = dfa.prefixes.get(state, "")
    return (len(prefix), prefix, state)


def _redirect_state(dfa: LearnedDFA, src: int, dst: int) -> None:
    if dfa.start == src:
        dfa.start = dst
    for edges in dfa.transitions.values():
        for symbol, target in list(edges.items()):
            if target == src:
                edges[symbol] = dst


def _merge_into(
    dfa: LearnedDFA,
    dst: int,
    src: int,
    active: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[bool, int]:
    if dst == src:
        return True, 0
    if active is None:
        active = set()
    pair = (dst, src)
    if pair in active:
        return True, 0
    active.add(pair)

    if src not in dfa.states or dst not in dfa.states:
        active.remove(pair)
        return False, 0

    score = 0
    if dst in dfa.accepting and src in dfa.accepting:
        score += 1
    if dst in dfa.rejecting and src in dfa.rejecting:
        score += 1

    if src in dfa.accepting:
        dfa.accepting.add(dst)
    if src in dfa.rejecting:
        dfa.rejecting.add(dst)
    if dst in dfa.accepting and dst in dfa.rejecting:
        active.remove(pair)
        return False, score

    src_edges = dict(dfa.transitions.get(src, {}))
    _redirect_state(dfa, src, dst)

    for symbol, src_target in src_edges.items():
        if src_target == src:
            src_target = dst
        dst_edges = dfa.transitions.setdefault(dst, {})
        if symbol not in dst_edges:
            dst_edges[symbol] = src_target
            continue

        dst_target = dst_edges[symbol]
        if dst_target == src:
            dst_target = dst
            dst_edges[symbol] = dst
        if dst_target == src_target:
            continue

        score += 1
        ok, child_score = _merge_into(dfa, dst_target, src_target, active)
        score += child_score
        if not ok:
            active.remove(pair)
            return False, score

    dfa.transitions.pop(src, None)
    dfa.accepting.discard(src)
    dfa.rejecting.discard(src)
    dfa.prefixes.pop(src, None)
    active.remove(pair)
    return True, score


def try_merge(dfa: LearnedDFA, red: int, blue: int) -> Tuple[Optional[LearnedDFA], int]:
    candidate = dfa.copy()
    ok, score = _merge_into(candidate, red, blue)
    if not ok:
        return None, score
    if candidate.accepting & candidate.rejecting:
        return None, score
    return candidate, score


def _blue_states(dfa: LearnedDFA, red_states: Set[int]) -> List[int]:
    blue = set()
    live_states = dfa.states
    for red in red_states:
        for target in dfa.transitions.get(red, {}).values():
            if target in live_states and target not in red_states:
                blue.add(target)
    return sorted(blue, key=lambda state: _state_order(dfa, state))


def _normalize_red(red_states: Set[int], dfa: LearnedDFA) -> Set[int]:
    return {state for state in red_states if state in dfa.states}


def learn_rpni(strings: Sequence[str], labels: Sequence[int]) -> LearnedDFA:
    dfa = build_pta(strings, labels)
    red_states: Set[int] = {dfa.start}

    while True:
        red_states = _normalize_red(red_states, dfa)
        blue = _blue_states(dfa, red_states)
        if not blue:
            break

        blue_state = blue[0]
        merged = False
        for red_state in sorted(red_states, key=lambda state: _state_order(dfa, state)):
            candidate, _ = try_merge(dfa, red_state, blue_state)
            if candidate is None:
                continue
            dfa = candidate
            red_states.discard(blue_state)
            merged = True
            break

        if not merged:
            red_states.add(blue_state)

    return dfa


def learn_blue_fringe(strings: Sequence[str], labels: Sequence[int]) -> LearnedDFA:
    dfa = build_pta(strings, labels)
    red_states: Set[int] = {dfa.start}

    while True:
        red_states = _normalize_red(red_states, dfa)
        blue = _blue_states(dfa, red_states)
        if not blue:
            break

        best: Optional[Tuple[int, Tuple[int, int], LearnedDFA, int]] = None
        for blue_state in blue:
            for red_state in sorted(red_states, key=lambda state: _state_order(dfa, state)):
                candidate, score = try_merge(dfa, red_state, blue_state)
                if candidate is None:
                    continue
                tie_break = (_state_order(dfa, blue_state), _state_order(dfa, red_state))
                item = (score, tie_break, candidate, blue_state)
                if best is None or item[0] > best[0] or (item[0] == best[0] and item[1] < best[1]):
                    best = item

        if best is None:
            red_states.add(blue[0])
            continue

        _, _, dfa, merged_blue = best
        red_states.discard(merged_blue)

    return dfa


def accuracy(dfa: LearnedDFA, strings: Sequence[str], labels: Sequence[int]) -> float:
    if not strings:
        return 0.0
    correct = sum(int(dfa.accepts(string) == bool(label)) for string, label in zip(strings, labels))
    return correct / len(strings)


def _target_next(target_dfa, state, char: str):
    if state is None:
        return None
    next_states = target_dfa._transition_function(state, Symbol(char))
    if len(next_states) == 0:
        return None
    return list(next_states)[0]


def equivalent_to_target(
    learned_dfa: LearnedDFA,
    target_dfa,
    alphabet: Iterable[str],
) -> Tuple[bool, Optional[str]]:
    alphabet = sorted(set(alphabet))
    start_pair = (learned_dfa.start, target_dfa.start_state, "")
    queue = [start_pair]
    visited = {(learned_dfa.start, target_dfa.start_state)}

    while queue:
        learned_state, target_state, witness = queue.pop(0)
        learned_accepts = learned_state != SINK_STATE and learned_state in learned_dfa.accepting
        target_accepts = target_state is not None and target_state in target_dfa.final_states
        if learned_accepts != target_accepts:
            return False, witness

        for char in alphabet:
            next_learned = (
                SINK_STATE
                if learned_state == SINK_STATE
                else learned_dfa.transitions.get(learned_state, {}).get(char, SINK_STATE)
            )
            next_target = _target_next(target_dfa, target_state, char)
            pair = (next_learned, next_target)
            if pair in visited:
                continue
            visited.add(pair)
            queue.append((next_learned, next_target, witness + char))

    return True, None


LEARNERS = {
    "rpni": learn_rpni,
    "blue_fringe": learn_blue_fringe,
}
