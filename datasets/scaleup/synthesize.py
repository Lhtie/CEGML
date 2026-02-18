import argparse
import random
from typing import List, Sequence, Tuple
from tqdm import tqdm

from pyformlang.finite_automaton import Symbol
from pyformlang.regular_expression import Regex

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from tasks.utils import (
    _complement_dfa,
    _dfa_concat,
    _dfa_intersection,
    _dfa_star,
    _dfa_symbols,
    _dfa_union,
    build_dfa,
    collect_sigma,
    split_regex_into_atoms,
)
from tasks.rl import ExtRegularLanguage

BASE_REGEX_EXAMPLES: Tuple[str, ...] = (
    "a", "b", "c",               # Uniqueness: exactly one a/b/c
    "[bc]*a[abc]*",              # Existence(a): at least one a
    "[ac]*b[abc]*",              # Existence(b): at least one b
    "([bc]*a)?([bc]*a)?[bc]*",   # Bounded Existence(a<=2)
    "c*(a[abc]*)?",              # Precedence(a before b): no b before first a
    "[bc]*(a[ac]*b[bc]*)*",      # Response(a->b): every a is eventually followed by b
)

def _tokenize_regex(s: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            tokens.append(s[i : i + 2])
            i += 2
            continue
        if ch in {"(", ")", "|", "+", ".", "*", "?"}:
            tokens.append(ch)
            i += 1
            continue
        if ch.isalpha():
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        tokens.append(ch)
        i += 1
    return tokens

def _format_compact_regex(raw: str) -> str:
    """
    Convert explicit concatenation (a.b) to implicit concatenation (ab),
    and remove unnecessary parentheses by precedence-aware rendering.
    """
    # pyformlang may emit '$' as epsilon-like marker.
    # Clean epsilon-related explicit concatenations first:
    #   '$.x' -> 'x', 'x.$' -> 'x', and collapse repeated dots.
    tokens = _tokenize_regex(raw)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected=None):
        nonlocal pos
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end of regex")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected '{expected}' but got '{tok}'")
        pos += 1
        return tok

    def starts_primary(tok: str) -> bool:
        return tok not in {None, ")", "|", "+", ".", "*", "?"}

    def parse_primary():
        tok = peek()
        if tok == "(":
            consume("(")
            node = parse_union()
            consume(")")
            return node
        if tok is None:
            raise ValueError("Unexpected end while parsing primary")
        return ("lit", consume())

    def parse_unary():
        node = parse_primary()
        while peek() == "*":
            consume("*")
            node = ("star", node)
        return node

    def parse_concat():
        nodes = [parse_unary()]
        while True:
            tok = peek()
            if tok == ".":
                consume(".")
                nodes.append(parse_unary())
                continue
            if starts_primary(tok):
                nodes.append(parse_unary())
                continue
            break
        if len(nodes) == 1:
            return nodes[0]
        return ("concat", nodes)

    def parse_union():
        nodes = [parse_concat()]
        while peek() in {"|", "+"}:
            consume()
            nodes.append(parse_concat())
        if len(nodes) == 1:
            return nodes[0]
        return ("union", nodes)

    def precedence(node) -> int:
        kind = node[0]
        if kind == "union":
            return 1
        if kind == "concat":
            return 2
        if kind == "star":
            return 3
        return 4

    def render(node) -> str:
        kind = node[0]
        if kind == "lit":
            return node[1]
        if kind == "star":
            child = node[1]
            child_s = render(child)
            if precedence(child) < 3:
                child_s = f"({child_s})"
            return f"{child_s}*"
        if kind == "concat":
            out = []
            for child in node[1]:
                s = render(child)
                if precedence(child) < 2:
                    s = f"({s})"
                out.append(s)
            return "".join(out)
        if kind == "union":
            out = []
            for child in node[1]:
                s = render(child)
                if precedence(child) < 1:
                    s = f"({s})"
                out.append(s)
            return "+".join(out)
        raise ValueError(f"Unknown node kind: {kind}")

    try:
        ast = parse_union()
        if pos != len(tokens):
            raise ValueError("Trailing tokens remain")
        return render(ast)
    except Exception:
        # Keep output usable even if pyformlang prints an unhandled edge case.
        cleaned = raw.replace(" ", "").replace("$", "")
        while ".." in cleaned:
            cleaned = cleaned.replace("..", ".")
        cleaned = cleaned.replace(".", "")
        return cleaned

def _dfa_from_regex(rx: str):
    tree = split_regex_into_atoms(rx)
    sigma = collect_sigma(tree)
    dfa = build_dfa(tree, sigma).minimize()
    return dfa, sigma

def _to_symbol_set(raw_symbols: Sequence[Symbol]) -> set:
    return set(raw_symbols)

def _compose_once(left_dfa, left_sigma, right_dfa, right_sigma, op: str):
    sigma = _to_symbol_set(left_sigma) | _to_symbol_set(right_sigma)

    if op == "union":
        return _dfa_union(left_dfa, right_dfa, sigma).minimize(), sigma
    if op == "intersection":
        return _dfa_intersection(left_dfa, right_dfa, sigma).minimize(), sigma
    if op == "concat":
        return _dfa_concat(left_dfa, right_dfa).minimize(), sigma

    raise ValueError(f"Unsupported binary op: {op}")

def _apply_unary(dfa, sigma, op: str):
    sigma = _to_symbol_set(sigma)

    if op == "star":
        return _dfa_star(dfa).minimize(), sigma
    if op == "complement":
        return _complement_dfa(dfa, sigma).minimize(), sigma

    raise ValueError(f"Unsupported unary op: {op}")

def dfa_to_regex_string(dfa) -> str:
    """Convert DFA to a regex string via pyformlang."""
    if hasattr(dfa, "to_regex"):
        regex_obj = dfa.to_regex()
        raw = getattr(regex_obj, "regex_string", str(regex_obj))
        return _format_compact_regex(raw)

    if hasattr(dfa, "to_regular_expression"):
        regex_obj = dfa.to_regular_expression()
        raw = getattr(regex_obj, "regex_string", str(regex_obj))
        return _format_compact_regex(raw)

    raise RuntimeError("Current pyformlang DFA object has no regex-conversion method")

def synthesize_dfa(
    examples: Sequence[str],
    pick_k: int = 3,
    combine_steps: int = 4,
    seed: int = 42,
):
    random.seed(seed)
    if pick_k < 2:
        raise ValueError("pick_k must be >= 2")
    if combine_steps < 1:
        raise ValueError("combine_steps must be >= 1")

    picked = random.sample(list(examples), k=min(pick_k, len(examples)))
    pool: List[Tuple[str, object, set]] = []
    for rx in picked:
        dfa, sigma = _dfa_from_regex(rx)
        pool.append((rx, dfa, sigma))

    steps = []
    # binary_ops = ["union", "intersection", "concat"]
    # unary_ops = ["star", "complement"]
    binary_ops = ["union", "concat"]
    unary_ops = ["star"]

    for _ in range(combine_steps):
        can_binary = len(pool) >= 2
        use_binary = can_binary and random.random() < 0.7

        if use_binary:
            i, j = random.sample(range(len(pool)), 2)
            name_a, dfa_a, sigma_a = pool[i]
            name_b, dfa_b, sigma_b = pool[j]
            op = random.choice(binary_ops)
            new_dfa, new_sigma = _compose_once(dfa_a, sigma_a, dfa_b, sigma_b, op)
            new_name = f"({name_a} {op} {name_b})"
            steps.append(new_name)
            pool.append((new_name, new_dfa, new_sigma))
        else:
            i = random.randrange(len(pool))
            name, dfa, sigma = pool[i]
            op = random.choice(unary_ops)
            new_dfa, new_sigma = _apply_unary(dfa, sigma, op)
            new_name = f"{op}({name})"
            steps.append(new_name)
            pool.append((new_name, new_dfa, new_sigma))

    final_name, final_dfa, _ = pool[-1]
    final_regex = dfa_to_regex_string(final_dfa)

    return {
        "picked_examples": picked,
        "composition_steps": steps,
        "final_dfa_expr": final_name,
        "final_dfa_states": len(final_dfa.states),
        "final_dfa_symbols": sorted([s.value for s in _dfa_symbols(final_dfa)]),
        "final_regex": final_regex,
    }


def main(args, verbose: bool = False):
    result = synthesize_dfa(
        examples=BASE_REGEX_EXAMPLES,
        pick_k=args.pick_k,
        combine_steps=args.combine_steps,
        seed=args.seed,
    )

    if verbose:
        print("Picked examples:")
        for x in result["picked_examples"]:
            print(f"- {x}")

        print("\nComposition steps:")
        for i, step in enumerate(result["composition_steps"], start=1):
            print(f"{i}. {step}")

        print("\nFinal DFA summary:")
        print(f"- expr: {result['final_dfa_expr']}")
        print(f"- #states: {result['final_dfa_states']}")
        print(f"- symbols: {result['final_dfa_symbols']}")

        print("\nFinal regex:")
        print(result["final_regex"])
    
    return result["final_regex"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--pick_k", type=int, default=3)
    # parser.add_argument("--combine_steps", type=int, default=4)
    args = parser.parse_args()
    
    regex_list = set()
    for steps in tqdm(range(2, 30)):
        for k in range(50):
            args.pick_k = 3
            args.combine_steps = steps
            regex = main(args, verbose=False)
            regex_list.add(regex)
    paired_regex = []
    for regex in regex_list:
        re = Regex(regex)
        n = len(re.to_epsilon_nfa().to_deterministic().minimize().states)
        paired_regex.append((regex, n))
    paired_regex = sorted(paired_regex, key=lambda x: x[1])
    print("Generated regexes sorted by DFA state count:")
    for regex, state_count in paired_regex:
        print(f"DFA states: {state_count}\t{regex}")
    
    if False:
        gt = "(b([bc]*a)?([bc]*a)?[bc]*)"
        hp = "b(c|b)*|b(c|b)*a(c|b)*|b(c|b)*a(c|b)*a(c|b)*"
        sigma = "[abc]"
        task = ExtRegularLanguage(gt, 32, alphabet=sigma)
        dfa_gt, fst_gt, sig = task.regex_to_pynini_via_pyformlang(gt)
        dfa_hp, fst_hp, sig = task.regex_to_pynini_via_pyformlang(hp)
        print(task.equivalent_and_witness(fst_gt, fst_hp, sig))
