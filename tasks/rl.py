import torch
import random
import pynini
import re
import numpy as np

from pyformlang.regular_expression import Regex, PythonRegex
from collections import deque
from pyformlang.finite_automaton import Symbol, DeterministicFiniteAutomaton

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tasks.utils import split_regex_into_atoms, expand_char_class
from tasks.utils import collect_sigma, build_dfa, tree_to_regex, _complement_dfa

class RegularLanguage:
    def __init__(self, regex_str, max_length):
        self.regex_str = regex_str
        self.max_length = max_length
    
    def _generate_string_to_state(self, dfa, target_states, max_depth):
        strings, labels = [], []
        visited = set()

        queue = deque([(dfa.start_state, [], 0)])
        visited.add((dfa.start_state, 0))
        alphabet = list(dfa.symbols)

        while queue:
            current_state, path, depth = queue.popleft()
            if depth > max_depth:
                continue
            if current_state in target_states:
                string = "".join(path)
                strings.append(string)
                labels.append(int(current_state in dfa.final_states))
            elif current_state == None:
                continue

            # Group outgoing symbols by next-state, then pick one random symbol per next-state.
            next_state_to_symbols = {}
            for symbol in alphabet:
                next_states = dfa._transition_function(current_state, symbol)
                if len(next_states) == 0:
                    continue
                next_state = list(next_states)[0]
                next_state_to_symbols.setdefault(next_state, []).append(symbol)

            next_states = list(next_state_to_symbols.keys())
            random.shuffle(next_states)
            for next_state in next_states:
                symbol = random.choice(next_state_to_symbols[next_state])
                step = len(symbol.value)
                if (next_state, depth + step) not in visited:
                    queue.append((next_state, path + [symbol.value], depth + step))
                    visited.add((next_state, depth + step))

        assert len(strings) > 0, "Cannot generate string to target state"
        return random.choice(strings)
    
    def generate_random_strings_balanced(self, m, n, rate=0.5):
        if not hasattr(self, "complement_dfa") or self.complement_dfa is None:
            sigma = self.sigma if hasattr(self, "sigma") else self.dfa.symbols
            self.complement_dfa = _complement_dfa(self.dfa, sigma).minimize()

        strings = []
        for _ in range(m):
            accepted = random.random() < rate
            source_dfa = self.dfa if accepted else self.complement_dfa
            target_states = list(source_dfa.final_states)
            if not target_states:
                raise ValueError("Source DFA has no final states for sampling")

            attempts = 0
            while attempts < 256:
                try:
                    s = self._generate_string_to_state(source_dfa, target_states, n)
                    break
                except AssertionError:
                    attempts += 1
                    continue
            if attempts >= 256:
                raise RuntimeError("Failed to sample string from DFA within attempt budget")
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
        return False, s.replace(" ", "")  # remove spaces from witness string
    
    def k_witnesses(self, dfa_a: DeterministicFiniteAutomaton, dfa_b: DeterministicFiniteAutomaton, k=10):
        """Return up to k disagreement strings from L(dfa_a) \\ L(dfa_b) with diverse accept states."""
        diff_dfa = dfa_a.get_difference(dfa_b).minimize()
        final_states = list(diff_dfa.final_states)
        if not final_states:
            return []

        # Round-robin target final states to keep acceptance states as balanced as possible.
        random.shuffle(final_states)
        targets = [final_states[i % len(final_states)] for i in range(k)]
        random.shuffle(targets)

        out = []
        seen = set()
        for target_state in targets:
            attempts = 0
            while attempts < 16:
                try:
                    s = self._generate_string_to_state(diff_dfa, [target_state], self.max_length)
                except AssertionError:
                    break
                if s not in seen:
                    seen.add(s)
                    out.append(s)
                    break
                attempts += 1
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
    
    def regex_to_pynini_via_pyformlang(self, rx: str, sigma: pynini.SymbolTable=None):
        re = PythonRegex(rx)
        nfa = re.to_epsilon_nfa()
        dfa = nfa.to_deterministic().minimize()
        if sigma is None:
            sigma = self.sigma_from_chars([s.value for s in dfa.symbols])
        fst = self.dfa_to_pynini_fst(dfa, sigma)
        return dfa, fst, sigma
    
class ExtRegularLanguage(RegularLanguage):
    def __init__(self, regex_str, max_length, alphabet=None, debug=False):
        """
        alphabet: String (re format e.g., '[A-Z]' and '[^a-z]') | List of characters
        """
        super().__init__(regex_str, max_length)

        self.regex_tree = split_regex_into_atoms(regex_str)
        if alphabet is not None and isinstance(alphabet, str):
            alphabet = expand_char_class(alphabet)
        sigma = {Symbol(c) for c in alphabet} if alphabet else collect_sigma(self.regex_tree)

        self.dfa = build_dfa(self.regex_tree, sigma, debug)
        self.sigma = sigma
        self.num_alphabets = len(sigma)
        self.num_categories = 2
        
    def regex_to_pynini_via_pyformlang(self, rx: str, sigma: pynini.SymbolTable=None):
        regex_tree = split_regex_into_atoms(rx)
        dfa = build_dfa(regex_tree, self.sigma)
        if sigma is None:
            sigma = self.sigma_from_chars([s.value for s in self.sigma])
        fst = self.dfa_to_pynini_fst(dfa, sigma)
        return dfa, fst, sigma

if __name__ == "__main__":
    nl = ExtRegularLanguage(
        "([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){5,}",
        max_length=32,
        alphabet="[A-Za-z0-9#]",
        debug=True,
    )
    print(tree_to_regex(nl.regex_tree))
    datasets = nl.generate_random_strings_balanced(20, 32, rate=0.5)
    print(list(zip(datasets, [nl.accepts(s) for s in datasets])))
