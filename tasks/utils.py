import json
import re

from pyformlang.regular_expression import PythonRegex
from collections import deque
from pyformlang.finite_automaton import State, Symbol, DeterministicFiniteAutomaton, EpsilonNFA, Epsilon

def expand_char_class(s):
    # s the same form as "[A-Za-z0-9#]"
    assert s.startswith("[") and s.endswith("]")
    body = s[1:-1]
    out = []
    i = 0
    while i < len(body):
        if i + 2 < len(body) and body[i+1] == "-" and body[i].isalnum() and body[i+2].isalnum():
            start, end = body[i], body[i+2]
            out.extend(chr(c) for c in range(ord(start), ord(end) + 1))
            i += 3
        else:
            out.append(body[i])
            i += 1
    return out

def _to_single_char_set(chars, err_msg):
    out = sorted(set(chars))
    if not out:
        raise ValueError(err_msg)
    if any(len(ch) != 1 for ch in out):
        raise ValueError("Character class only supports single-character symbols")
    return out

def _class_body_from_sorted_chars(chars):
    # Build compact class body such as "A-Za-z0-9#"
    runs = []
    i = 0
    while i < len(chars):
        j = i
        while j + 1 < len(chars) and ord(chars[j + 1]) == ord(chars[j]) + 1:
            j += 1
        runs.append((chars[i], chars[j]))
        i = j + 1

    body_parts = []
    for lo, hi in runs:
        run_len = ord(hi) - ord(lo) + 1
        if run_len >= 3:
            body_parts.append(f"{lo}-{hi}")
        else:
            for code in range(ord(lo), ord(hi) + 1):
                body_parts.append(chr(code))
    return "".join(body_parts)

def minimal_char_class(chars, universe_chars=None):
    """
    Build a compact character class token from a character set.
    Returns:
      - single character if len(chars) == 1
      - bracket class like "[A-Za-z0-9#]" otherwise
    """
    chars = _to_single_char_set(chars, "Cannot build character class from empty charset")
    if len(chars) == 1:
        return chars[0]

    pos = "[" + _class_body_from_sorted_chars(chars) + "]"
    if universe_chars is None:
        return pos

    universe = _to_single_char_set(universe_chars, "Cannot build complemented class without universe charset")
    char_set = set(chars)
    universe_set = set(universe)
    if not char_set.issubset(universe_set):
        return pos

    comp = sorted(universe_set - char_set)
    if not comp:
        return pos

    neg = "[^" + _class_body_from_sorted_chars(comp) + "]"
    return neg if len(neg) < len(pos) else pos

def dfa_edge_char_class(dfa: DeterministicFiniteAutomaton, state_a, state_b):
    """
    Return minimal character token for all symbols that transition state_a -> state_b.
    The token is either a single character or a bracket class.
    """
    chars = []
    universe = []
    for symbol in dfa.symbols:
        token = symbol.value if hasattr(symbol, "value") else str(symbol)
        universe.append(token)
        nxt = dfa._transition_function(state_a, symbol)
        if len(nxt) == 0:
            continue
        if list(nxt)[0] == state_b:
            chars.append(token)
    if not chars:
        raise ValueError(f"No symbol-carrying transition from state {state_a} to {state_b}")
    return minimal_char_class(chars, universe_chars=universe)

def _tokenize_ex(ex):
    tokens = []
    i = 0
    n = len(ex)
    while i < n:
        if ex[i] == "[":
            j = i + 1
            while j < n and ex[j] != "]":
                j += 1
            if j >= n:
                raise ValueError(f"Unmatched '[' in example: {ex}")
            tokens.append(ex[i:j + 1])
            i = j + 1
        else:
            tokens.append(ex[i])
            i += 1
    return tokens

def dfa_accepts_ex(dfa: DeterministicFiniteAutomaton, ex):
    """
    Check acceptance of an example string where each position can be:
      - a literal char, e.g. "a"
      - a char class token, e.g. "[A-Z]" or "[^0-9]"
    Semantics are universal over char classes: all expansions must be accepted.
    """
    universe_chars = _sigma_to_single_chars(dfa.symbols)

    cur_states = {dfa.start_state}
    for token in _tokenize_ex(ex):
        if token.startswith("[") and token.endswith("]"):
            if token.startswith("[^"):
                excluded = set(expand_char_class("[" + token[2:]))
                cands = [ch for ch in universe_chars if ch not in excluded]
            else:
                cands = expand_char_class(token)
        else:
            cands = [token]
        if not cands:
            return False
        next_states = set()
        saw_dead_branch = False
        for st in cur_states:
            for ch in cands:
                nxt = dfa._transition_function(st, Symbol(ch))
                if len(nxt) > 0:
                    next_states.add(list(nxt)[0])
                else:
                    saw_dead_branch = True
        if saw_dead_branch or not next_states:
            return False
        cur_states = next_states
    return all(st in dfa.final_states for st in cur_states)

def split_regex_into_atoms(regex):
    """
    Parse regex into a nesting tree by ().
    Returns a dict node:
      - group: {"type": "group", "children": [...], "ops": [...], "quant": "", "negated": False}
      - atom:  {"type": "atom", "value": "...", "quant": "", "negated": False}
    Operators are parsed by precedence: quantifier > concatenation > "&" > "|".
    Internal group nodes are built to preserve this precedence.
    Unary "~" is recorded as node["negated"] = True on the following atom/group.
    Quantifiers (* + ? {m,n}) are attached to the preceding atom/group node.
    """
    n = len(regex)
    i = 0

    def take_quant(i0):
        if i0 < n and regex[i0] in {"*", "+", "?"}:
            return regex[i0], i0 + 1
        if i0 < n and regex[i0] == "{":
            j = i0 + 1
            while j < n and regex[j] != "}":
                j += 1
            return regex[i0:j + 1], min(j + 1, n)
        return "", i0

    def make_nary_group(children, op):
        if len(children) == 1:
            return children[0]
        return {
            "type": "group",
            "children": children,
            "ops": [op] * (len(children) - 1),
            "quant": "",
            "negated": False,
        }

    def wrap_group(node):
        return {"type": "group", "children": [node], "ops": [], "quant": "", "negated": False}

    def toggle_neg(node):
        node = dict(node)
        node["negated"] = not node.get("negated", False)
        return node

    def parse_atom():
        nonlocal i
        if i >= n:
            raise ValueError("Unexpected end of regex while parsing atom")
        ch = regex[i]
        if ch == "[":
            j = i + 1
            while j < n:
                if regex[j] == "\\" and j + 1 < n:
                    j += 2
                else:
                    j += 1
                    if regex[j - 1] == "]":
                        break
            atom = regex[i:j]
            i = j
            return {"type": "atom", "value": atom, "quant": "", "negated": False}
        if ch == "\\" and i + 1 < n:
            atom = regex[i:i + 2]
            i += 2
            return {"type": "atom", "value": atom, "quant": "", "negated": False}
        if ch in {")", "|", "&"}:
            raise ValueError(f"Unexpected token {ch} at position {i}")
        i += 1
        return {"type": "atom", "value": ch, "quant": "", "negated": False}

    def parse_primary():
        nonlocal i
        if i < n and regex[i] == "(":
            i += 1
            node = parse_or()
            if i >= n or regex[i] != ")":
                raise ValueError("Unmatched '(' in regex")
            i += 1
            # Preserve explicit parentheses as a scope node.
            if node["type"] != "group":
                node = wrap_group(node)
        else:
            node = parse_atom()
        q, i2 = take_quant(i)
        i = i2
        node = dict(node)
        node["quant"] = q
        return node

    def parse_unary():
        nonlocal i
        neg = False
        while i < n and regex[i] == "~":
            neg = not neg
            i += 1
        node = parse_primary()
        return toggle_neg(node) if neg else node

    def starts_unary():
        if i >= n:
            return False
        ch = regex[i]
        return ch in {"~", "("} or ch == "[" or ch == "\\" or ch not in {"|", "&", ")"}

    def parse_concat():
        nodes = [parse_unary()]
        while starts_unary():
            nodes.append(parse_unary())
        return make_nary_group(nodes, ".")

    def parse_and():
        nonlocal i
        nodes = [parse_concat()]
        while i < n and regex[i] == "&":
            i += 1
            nodes.append(parse_concat())
        return make_nary_group(nodes, "&")

    def parse_or():
        nonlocal i
        nodes = [parse_and()]
        while i < n and regex[i] == "|":
            i += 1
            nodes.append(parse_and())
        return make_nary_group(nodes, "|")

    tree = parse_or()
    if i != n:
        raise ValueError(f"Unexpected trailing token at position {i}: {regex[i:]}")
    if tree["type"] == "group":
        return tree
    return {"type": "group", "children": [tree], "ops": [], "quant": "", "negated": False}

def tree_to_regex(node, par=None):
    """
    Serialize tree back to regex string. Uses "." as implicit concat.
    """
    neg = node.get("negated", False)
    if node["type"] == "atom":
        prefix = "~" if neg else ""
        return prefix + node["value"] + node.get("quant", "")

    children = node.get("children", [])
    ops = node.get("ops", [])

    parts = []
    for idx, child in enumerate(children):
        child_rx = tree_to_regex(child, node)
        parts.append(child_rx)
        if idx < len(ops) and ops[idx] != ".":
            parts.append(ops[idx])

    inner = "(" + "".join(parts) + ")"
    if neg:
        inner = "~" + inner
    if par is None and not neg and not node.get("quant", ""):
        inner = inner[1:-1]
    return inner + node.get("quant", "")

def _dfa_from_atom(atom_value: str):
    rx = PythonRegex(atom_value)
    nfa = rx.to_epsilon_nfa()
    return nfa.to_deterministic().minimize()

def _sigma_to_single_chars(sigma):
    chars = []
    for sym in sigma:
        ch = sym.value if hasattr(sym, "value") else str(sym)
        if len(ch) != 1:
            return None
        chars.append(ch)
    return sorted(set(chars))

def _normalize_negated_char_class_atom(atom_value: str, sigma):
    # Rewrite [^...] into an equivalent positive class over finite sigma.
    if sigma is None:
        return atom_value
    if not (atom_value.startswith("[^") and atom_value.endswith("]")):
        return atom_value

    universe = _sigma_to_single_chars(sigma)
    if not universe:
        return atom_value

    excluded = set(expand_char_class("[" + atom_value[2:]))
    keep = [ch for ch in universe if ch not in excluded]
    if not keep:
        # Empty language atom under this sigma.
        return None
    return minimal_char_class(keep)

def _dfa_empty():
    dfa = DeterministicFiniteAutomaton()
    s = State("empty")
    dfa.add_start_state(s)
    return dfa

def _dfa_symbols(dfa):
    return set(dfa.symbols)

def _dfa_next(dfa, state, symbol):
    nxt = dfa._transition_function(state, symbol)
    if len(nxt) == 0:
        return None
    return list(nxt)[0]

def _complete_dfa(dfa, sigma):
    if not sigma:
        return dfa
    new = DeterministicFiniteAutomaton()
    mapping = {s: State(("q", s)) for s in dfa.states}
    sink = State(("sink",))
    new.add_start_state(mapping[dfa.start_state])
    for s in dfa.final_states:
        new.add_final_state(mapping[s])
    for (q, a, t) in dfa._transition_function.get_edges():
        new.add_transition(mapping[q], a, mapping[t])
    for s in list(new.states):
        for sym in sigma:
            nxt = _dfa_next(new, s, sym)
            if nxt is None:
                new.add_transition(s, sym, sink)
    for sym in sigma:
        new.add_transition(sink, sym, sink)
    return new

def _complement_dfa(dfa, sigma):
    dfa = _complete_dfa(dfa, sigma)
    new = DeterministicFiniteAutomaton()
    new.add_start_state(dfa.start_state)
    for s in dfa.states:
        if s not in dfa.final_states:
            new.add_final_state(s)
    for (q, a, t) in dfa._transition_function.get_edges():
        new.add_transition(q, a, t)
    return new

def _product_dfa(dfa_a, dfa_b, sigma, accept_fn):
    dfa_a = _complete_dfa(dfa_a, sigma)
    dfa_b = _complete_dfa(dfa_b, sigma)
    new = DeterministicFiniteAutomaton()
    start = State((dfa_a.start_state, dfa_b.start_state))
    new.add_start_state(start)
    queue = deque([start])
    seen = {start}
    while queue:
        cur = queue.popleft()
        sa, sb = cur.value
        if accept_fn(sa in dfa_a.final_states, sb in dfa_b.final_states):
            new.add_final_state(cur)
        for sym in sigma:
            na = _dfa_next(dfa_a, sa, sym)
            nb = _dfa_next(dfa_b, sb, sym)
            nxt = State((na, nb))
            new.add_transition(cur, sym, nxt)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return new.minimize()

def _dfa_union(dfa_a, dfa_b, sigma):
    return _product_dfa(dfa_a, dfa_b, sigma, lambda fa, fb: fa or fb)

def _dfa_intersection(dfa_a, dfa_b, sigma):
    return _product_dfa(dfa_a, dfa_b, sigma, lambda fa, fb: fa and fb)

def _dfa_to_enfa(dfa, tag):
    enfa = EpsilonNFA()
    mapping = {s: State((tag, s)) for s in dfa.states}
    enfa.add_start_state(mapping[dfa.start_state])
    for s in dfa.final_states:
        enfa.add_final_state(mapping[s])
    for (q, a, t) in dfa._transition_function.get_edges():
        enfa.add_transition(mapping[q], a, mapping[t])
    return enfa, mapping

def _merge_enfa(enfa, other, add_starts=True, add_finals=True):
    if add_starts:
        for s in other.start_states:
            enfa.add_start_state(s)
    if add_finals:
        for s in other.final_states:
            enfa.add_final_state(s)
    for (q, a, t) in other._transition_function.get_edges():
        enfa.add_transition(q, a, t)

def _dfa_concat(dfa_a, dfa_b):
    enfa = EpsilonNFA()
    enfa_a, _ = _dfa_to_enfa(dfa_a, "a")
    enfa_b, _ = _dfa_to_enfa(dfa_b, "b")
    _merge_enfa(enfa, enfa_a, add_starts=True, add_finals=False)
    _merge_enfa(enfa, enfa_b, add_starts=False, add_finals=True)
    eps = Epsilon()
    for fa in enfa_a.final_states:
        for sb in enfa_b.start_states:
            enfa.add_transition(fa, eps, sb)
    return enfa.to_deterministic().minimize()

def _dfa_star(dfa):
    enfa = EpsilonNFA()
    enfa_a, _ = _dfa_to_enfa(dfa, "s")
    _merge_enfa(enfa, enfa_a, add_starts=True, add_finals=True)
    new_start = State(("star_start",))
    enfa.add_start_state(new_start)
    enfa.add_final_state(new_start)
    eps = Epsilon()
    for sb in enfa_a.start_states:
        enfa.add_transition(new_start, eps, sb)
    for fa in enfa_a.final_states:
        enfa.add_transition(fa, eps, new_start)
    return enfa.to_deterministic().minimize()

def _dfa_epsilon():
    dfa = DeterministicFiniteAutomaton()
    s = State("eps")
    dfa.add_start_state(s)
    dfa.add_final_state(s)
    return dfa

def _apply_quant(dfa, quant):
    if not quant:
        return dfa
    if quant == "*":
        return _dfa_star(dfa)
    if quant == "+":
        return _dfa_concat(dfa, _dfa_star(dfa))
    if quant == "?":
        return _dfa_union(dfa, _dfa_epsilon(), _dfa_symbols(dfa))
    m = re.fullmatch(r"\{\s*(\d+)\s*\}", quant)
    if m:
        k = int(m.group(1))
        if k == 0:
            return _dfa_epsilon()
        out = dfa
        for _ in range(k - 1):
            out = _dfa_concat(out, dfa)
        return out
    m = re.fullmatch(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", quant)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi < lo:
            return _dfa_epsilon()
        opts = []
        for k in range(lo, hi + 1):
            opts.append(_apply_quant(dfa, "{" + str(k) + "}"))
        out = opts[0]
        for opt in opts[1:]:
            out = _dfa_union(out, opt, _dfa_symbols(out) | _dfa_symbols(opt))
        return out
    m = re.fullmatch(r"\{\s*(\d+)\s*,\s*\}", quant)
    if m:
        lo = int(m.group(1))
        base = _apply_quant(dfa, "{" + str(lo) + "}")
        return _dfa_concat(base, _dfa_star(dfa))
    return dfa

def _dfa_summary(dfa):
    return f"states={len(dfa.states)} finals={len(dfa.final_states)} symbols={len(dfa.symbols)}"

def collect_sigma(node):
    if node["type"] == "atom":
        dfa = _dfa_from_atom(node["value"])
        return _dfa_symbols(dfa)
    sigma = set()
    for child in node.get("children", []):
        sigma |= collect_sigma(child)
    return sigma

def build_dfa(node, sigma=None, debug=False):
    if sigma is None:
        sigma = collect_sigma(node)
    
    if node["type"] == "atom":
        atom_value = _normalize_negated_char_class_atom(node["value"], sigma)
        dfa = _dfa_empty() if atom_value is None else _dfa_from_atom(atom_value)
        dfa = _apply_quant(dfa, node.get("quant", ""))
        if node.get("negated", False):
            dfa = _complement_dfa(dfa, sigma)
        if debug:
            print(f"[atom] {node['value']} -> {atom_value}{node.get('quant','')} neg={node.get('negated',False)} -> {_dfa_summary(dfa)}")
        return dfa

    children = node.get("children", [])
    ops = node.get("ops", [])
    if not children:
        dfa = _dfa_epsilon()
    else:
        dfa = build_dfa(children[0], sigma, debug)
        for op, child in zip(ops, children[1:]):
            right = build_dfa(child, sigma, debug)
            if op == ".":
                dfa = _dfa_concat(dfa, right)
            elif op == "|":
                dfa = _dfa_union(dfa, right, sigma)
            elif op == "&":
                dfa = _dfa_intersection(dfa, right, sigma)
            else:
                dfa = _dfa_concat(dfa, right)
    dfa = _apply_quant(dfa, node.get("quant", ""))
    if node.get("negated", False):
        dfa = _complement_dfa(dfa, sigma)
    if debug:
        print(f"[group] ops={ops} quant={node.get('quant','')} neg={node.get('negated',False)} -> {_dfa_summary(dfa)}")
    return dfa.minimize()
