import torch
import random
import pynini
import re
import numpy as np

from pyformlang.regular_expression import Regex, PythonRegex
from collections import deque
from pyformlang.finite_automaton import State, Symbol, DeterministicFiniteAutomaton, EpsilonNFA, Epsilon

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tasks.utils import split_regex_into_atoms, expand_char_class

class RegularLanguage:
    def __init__(self, regex_str, max_length):
        self.regex_str = regex_str
        self.max_length = max_length
    
    def _generate_string_to_state(self, dfa, target_state, max_depth):
        strings, labels = [], []
        visited = set()

        queue = deque([(dfa.start_state, [], 0)])
        visited.add((dfa.start_state, 0))
        alphabet = list(dfa.symbols)

        while queue:
            current_state, path, depth = queue.popleft()
            if depth > max_depth:
                continue
            if current_state == target_state:
                string = "".join(path)
                strings.append(string)
                labels.append(int(current_state in dfa.final_states))
            elif current_state == None:
                continue

            random.shuffle(alphabet)
            for symbol in alphabet:
                next_state = dfa._transition_function(current_state, symbol)
                if len(next_state) > 0:
                    next_state = next_state[0]
                else:
                    next_state = None
                if (next_state, depth + len(symbol.value)) not in visited:
                    queue.append((next_state, path + [symbol.value], depth + len(symbol.value)))
                    visited.add((next_state, depth + len(symbol.value)))

        assert len(strings) > 0, "Cannot generate string to target state"
        return random.choice(strings)
    
    def generate_random_strings_balanced(self, m, n, rate=0.5):
        strings = []
        for _ in range(m):
            accepted = random.random() < rate
            if accepted:
                target_state = random.choice(list(self.dfa.final_states))
            else:
                target_state = random.choice(list(self.dfa.states - self.dfa.final_states))
            while True:
                try:
                    s = self._generate_string_to_state(self.dfa, target_state, n)
                    break
                except AssertionError:
                    continue
            strings.append(s)
        return strings

    def accepts(self, str):
        return self.dfa.accepts(str)
    
    def sigma_from_chars(self, chars):
        st = pynini.SymbolTable()
        for i, ch in enumerate(chars, start=1):
            st.add_symbol(ch, i)
        return st

    def dfa_to_pynini_fst(self, dfa: DeterministicFiniteAutomaton, sigma: pynini.SymbolTable) -> pynini.Fst:
        f = pynini.Fst()
        idmap = {}
        for q in dfa.states:
            idmap[q] = f.add_state()
        f.set_start(idmap[dfa.start_state])
        for q in dfa.final_states:
            f.set_final(idmap[q])
        for q, a, t in dfa._transition_function.get_edges():
            il = sigma.find(a.value) if hasattr(a, "value") else sigma.find(a)
            if il == -1:
                raise ValueError(f"symbol {a} not in sigma")
            f.add_arc(idmap[q], pynini.Arc(il, il, pynini.Weight.one(f.weight_type()), idmap[t]))
        f.set_input_symbols(sigma); f.set_output_symbols(sigma)
        return pynini.minimize(pynini.determinize(f)).optimize()

    def equivalent_and_witness(self, A: pynini.Fst, B: pynini.Fst, sigma: pynini.SymbolTable):
        """
        Returns (equivalent: bool, witness: str or None).
        We compute symmetric difference and test emptiness.
        If nonempty, we extract a shortest witness string using shortestpath.
        """
        # Symmetric difference = (A\B) ∪ (B\A)
        symdiff = (pynini.difference(A, B) | pynini.difference(B, A)).optimize()

        # Empty FSA has no valid start
        if symdiff.start() == pynini.NO_STATE_ID or symdiff.num_states() == 0:
            return True, None

        # Get a shortest string in the difference (witness)
        sp = pynini.shortestpath(symdiff).optimize()
        # Convert path to string using the symbol table
        sp.set_input_symbols(sigma)
        sp.set_output_symbols(sigma)
        s = pynini.shortestpath(sp).string(token_type=sigma)  # or rewrite.lattice_to_string(sp)
        return False, s
    
    def k_witnesses(self, A: pynini.Fst, B: pynini.Fst, sigma: pynini.SymbolTable, k=10):
        """Return up to k shortest disagreement strings."""
        symdiff = (pynini.difference(A, B)).optimize()

        if symdiff.start() == pynini.NO_STATE_ID or symdiff.num_states() == 0:
            return []

        symdiff = pynini.project(symdiff, "input")
        symdiff = pynini.rmepsilon(symdiff).optimize()
        symdiff.set_input_symbols(sigma)
        symdiff.set_output_symbols(sigma)

        # Extract strings
        out = []
        for _ in range(k):
            sp = pynini.shortestpath(symdiff).optimize()  # single shortest path
            if sp.start() == pynini.NO_STATE_ID or sp.num_states() == 0:
                break  # empty language
            s = sp.string(token_type=sigma)             # shortest string
            out.append(s.replace(" ", ""))              # remove spaces
            # Remove that exact string from the language and continue
            symdiff = pynini.project(symdiff, "input").optimize()
            symdiff = pynini.difference(symdiff, pynini.accep(s, token_type=sigma)).optimize()
        return out
    
    def count_strings_of_length(self, fst: pynini.Fst, sigma: pynini.SymbolTable, length: int) -> int:
        """ Count number of strings of exactly given length accepted by fst. """
        if fst.start() == pynini.NO_STATE_ID or fst.num_states() == 0:
            return 0

        # Initialize DP table
        dp = {state: 0 for state in fst.states()}
        dp[fst.start()] = 1

        for _ in range(length):
            next_dp = {state: 0 for state in fst.states()}
            for state in fst.states():
                for arc in fst.arcs(state):
                    next_dp[arc.nextstate] += dp[state]
            dp = next_dp

        # Sum counts of final states
        count = sum(dp[state] for state in fst.states() if fst.final(state) != pynini.Weight.zero(fst.weight_type()))
        return count
    
    def diff_ratio(self, A: pynini.Fst, B: pynini.Fst, sigma: pynini.SymbolTable, k=8):
        """ Return ratio of disagreement strings of up to length k. """
        symdiff = (pynini.difference(A, B) | pynini.difference(B, A)).optimize()
        
        if symdiff.start() == pynini.NO_STATE_ID or symdiff.num_states() == 0:
            return 0.0
        
        symdiff = pynini.project(symdiff, "input")
        symdiff = pynini.rmepsilon(symdiff).optimize()
        symdiff.set_input_symbols(sigma)
        symdiff.set_output_symbols(sigma)

        total, diff = 0, 0
        for length in range(0, k + 1):
            total += sigma.num_symbols() ** length
            diff += self.count_strings_of_length(symdiff, sigma, length)
            
        return diff / total if total > 0 else 0.0
    
class SimplyRegularLanguage(RegularLanguage):
    def __init__(self, regex_str, max_length):
        super().__init__(regex_str, max_length)

        self.regex = Regex(regex_str)

        self.nfa = self.regex.to_epsilon_nfa()
        self.dfa = self.nfa.to_deterministic().minimize()
        self.num_alphabets = len(self.dfa.symbols)
        self.num_categories = 2     # positive | negative

    def to_tensor(self, batched_str):
        lengths = torch.tensor([len(s) for s in batched_str], dtype=torch.long)
        max_length = lengths.max().item()
        tensor = torch.zeros((len(batched_str), max_length + 1, self.num_alphabets), dtype=torch.float)
        for i, s in enumerate(batched_str):
            for j, c in enumerate(s):
                tensor[i, j, ord(c) - ord('a')] = 1
            tensor[i, len(s), :] = 1
        return tensor, lengths
    
    def generate_random_strings_uniform(self, m, n):
        strings = []
        alphabet = [chr(c + ord('a')) for c in range(self.num_alphabets)]
        for _ in range(m):
            length = random.randint(1, n)
            s = ''.join(random.choices(alphabet, k=length))
            strings.append(s)
        return strings

    def generate_random_strings_beta(self, m, n, alpha=None):
        alphabet = [chr(ord('a') + i) for i in range(self.num_alphabets)]
        if isinstance(alpha, (int, float)) or alpha is None:
            alpha = [alpha if alpha else 1.0] * self.num_alphabets
        strings = []

        for _ in range(m):
            probs = np.random.dirichlet(alpha)
            
            indices = np.random.choice(self.num_alphabets, size=n, p=probs)
            strings.append(''.join(alphabet[i] for i in indices))
        return strings
    
    def regex_to_pynini_via_pyformlang(self, rx: str, sigma=None):
        re = Regex(rx)
        nfa = re.to_epsilon_nfa()
        dfa = nfa.to_deterministic().minimize()
        if sigma is None:
            sigma = self.sigma_from_chars([s.value for s in dfa.symbols])
        fst = self.dfa_to_pynini_fst(dfa, sigma)
        return dfa, fst, sigma
    
class PythonRegularLanguage(RegularLanguage):
    def __init__(self, regex_str, max_length):
        super().__init__(regex_str, max_length)

        self.regex = PythonRegex(regex_str)

        self.nfa = self.regex.to_epsilon_nfa()
        self.dfa = self.nfa.to_deterministic().minimize()
        self.num_alphabets = len(self.dfa.symbols)
        self.num_categories = 2     # positive | negative
    
    def regex_to_pynini_via_pyformlang(self, rx: str, sigma=None):
        re = PythonRegex(rx)
        nfa = re.to_epsilon_nfa()
        dfa = nfa.to_deterministic().minimize()
        if sigma is None:
            sigma = self.sigma_from_chars([s.value for s in dfa.symbols])
        fst = self.dfa_to_pynini_fst(dfa, sigma)
        return dfa, fst, sigma
    
class NLRXRegularLanguage(RegularLanguage):
    def __init__(self, regex_str, max_length, alphabet=None, debug=False):
        """
        alphabet: String (re format e.g., '[A-Z]' and '[^a-z]') | List of characters
        """
        super().__init__(regex_str, max_length)

        self.regex_tree = split_regex_into_atoms(regex_str)
        if alphabet is not None and isinstance(alphabet, str):
            alphabet = expand_char_class(alphabet)
        sigma = {Symbol(c) for c in alphabet} if alphabet else None

        def _dfa_from_atom(atom_value: str):
            rx = PythonRegex(atom_value)
            nfa = rx.to_epsilon_nfa()
            return nfa.to_deterministic().minimize()

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

        def _collect_sigma(node):
            if node["type"] == "atom":
                dfa = _dfa_from_atom(node["value"])
                return _dfa_symbols(dfa)
            sigma = set()
            for child in node.get("children", []):
                sigma |= _collect_sigma(child)
            return sigma

        if sigma is None: sigma = _collect_sigma(self.regex_tree)

        def _dfa_summary(dfa):
            return f"states={len(dfa.states)} finals={len(dfa.final_states)} symbols={len(dfa.symbols)}"

        def _build_dfa(node):
            if node["type"] == "atom":
                dfa = _dfa_from_atom(node["value"])
                dfa = _apply_quant(dfa, node.get("quant", ""))
                if node.get("negated", False):
                    dfa = _complement_dfa(dfa, sigma)
                if debug:
                    print(f"[atom] {node['value']}{node.get('quant','')} neg={node.get('negated',False)} -> {_dfa_summary(dfa)}")
                return dfa

            children = node.get("children", [])
            ops = node.get("ops", [])
            if not children:
                dfa = _dfa_epsilon()
            else:
                dfa = _build_dfa(children[0])
                for op, child in zip(ops, children[1:]):
                    right = _build_dfa(child)
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
            return dfa

        self.dfa = _build_dfa(self.regex_tree).minimize()
        self.sigma = sigma
        self.num_alphabets = len(sigma)
        self.num_categories = 2

if __name__ == "__main__":
    nl = NLRXRegularLanguage(
        "[A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z0-9#]*spoon))[A-Za-z0-9#]*",
        max_length=10,
        alphabet="[A-Za-z0-9#]",
        debug=True,
    )
