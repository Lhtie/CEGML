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
from tasks.utils import dfa_edge_char_class, dfa_accepts_ex

class RegularLanguage:
    def __init__(self, regex_str, max_length):
        self.regex_str = regex_str
        self.max_length = max_length
    
    def _generate_string_to_state(self, dfa, target_states, max_depth, clustered: bool=False):
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

            # Group outgoing symbols by next-state.
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
                if clustered:
                    token = dfa_edge_char_class(dfa, current_state, next_state)
                else:
                    symbol = random.choice(next_state_to_symbols[next_state])
                    token = symbol.value
                if (next_state, depth + 1) not in visited:
                    queue.append((next_state, path + [token], depth + 1))
                    visited.add((next_state, depth + 1))

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
        return dfa_accepts_ex(self.dfa, str)
    
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
    
    def k_witnesses_sample(self, dfa_a: DeterministicFiniteAutomaton, dfa_b: DeterministicFiniteAutomaton, 
                    k: int=10, clustered: bool=False):
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
                    s = self._generate_string_to_state(
                        diff_dfa, [target_state], self.max_length, clustered=clustered
                    )
                except AssertionError:
                    break
                if s not in seen:
                    seen.add(s)
                    out.append(s)
                    break
                attempts += 1
        return out
    
    def k_witnesses_dfs(
        self,
        dfa_a: DeterministicFiniteAutomaton,
        dfa_b: DeterministicFiniteAutomaton,
        k: int = 10,
        clustered: bool = True,
    ):
        """Return up to k disagreement witnesses from loop-free state paths."""
        diff_dfa = dfa_a.get_difference(dfa_b).minimize()
        if not diff_dfa.final_states or diff_dfa.start_state is None:
            return []

        # In clustered mode, symbols reaching the same next state are merged into
        # one char-class token. Otherwise, every symbol is a separate frontier.
        transitions_cache = {}
        token_cache = {}

        def outgoing_transitions(state):
            if state in transitions_cache:
                return transitions_cache[state]

            if clustered:
                next_states = set()
                for symbol in diff_dfa.symbols:
                    nxt = diff_dfa._transition_function(state, symbol)
                    if len(nxt) > 0:
                        next_states.add(list(nxt)[0])
                transitions = [
                    (nxt, edge_token(state, nxt))
                    for nxt in sorted(next_states, key=lambda s: str(s))
                ]
                transitions_cache[state] = transitions
                return transitions

            transitions = []
            for symbol in diff_dfa.symbols:
                nxt = diff_dfa._transition_function(state, symbol)
                if len(nxt) == 0:
                    continue
                token = symbol.value if hasattr(symbol, "value") else str(symbol)
                transitions.append((list(nxt)[0], token))
            transitions.sort(key=lambda item: (str(item[0]), item[1]))
            transitions_cache[state] = transitions
            return transitions

        def edge_token(state_a, state_b):
            key = (state_a, state_b)
            if key not in token_cache:
                token_cache[key] = dfa_edge_char_class(diff_dfa, state_a, state_b)
            return token_cache[key]

        # Keep traversal time bounded so clustered witness generation cannot
        # wander through an enormous difference DFA forever.
        max_frontier = max(256, 8 * k, 4 * len(diff_dfa.states))
        max_expansions = max_frontier * max(4, self.max_length)
        witnesses = []
        # Stack item: (state, token_path, visited_states)
        stack = [(diff_dfa.start_state, [], {diff_dfa.start_state})]
        expansions = 0
        while stack:
            cur, token_path, visited_states = stack.pop()

            if cur in diff_dfa.final_states:
                witnesses.append("".join(token_path))
                if len(witnesses) >= k:
                    return witnesses

            if len(token_path) >= self.max_length:
                continue

            expansions += 1
            if expansions > max_expansions:
                break

            # Expand loop-free paths only (no repeated states on current path).
            for nxt, token in outgoing_transitions(cur):
                if nxt in visited_states:
                    continue
                stack.append((nxt, token_path + [token], visited_states | {nxt}))
                if len(stack) >= max_frontier:
                    break
        return witnesses

    def k_witnesses_bfs(
        self,
        dfa_a: DeterministicFiniteAutomaton,
        dfa_b: DeterministicFiniteAutomaton,
        k: int = 10,
        clustered: bool = False,
    ):
        """Return up to k shortest disagreement witnesses using breadth-first search."""
        diff_dfa = dfa_a.get_difference(dfa_b).minimize()
        if not diff_dfa.final_states or diff_dfa.start_state is None:
            return []

        transitions_cache = {}

        def outgoing_transitions(state):
            if state in transitions_cache:
                return transitions_cache[state]

            next_state_to_symbols = {}
            for symbol in diff_dfa.symbols:
                nxt = diff_dfa._transition_function(state, symbol)
                if len(nxt) == 0:
                    continue
                next_state = list(nxt)[0]
                token = symbol.value if hasattr(symbol, "value") else str(symbol)
                next_state_to_symbols.setdefault(next_state, []).append(token)

            transitions = []
            for next_state in sorted(next_state_to_symbols, key=lambda s: str(s)):
                if clustered:
                    token = dfa_edge_char_class(diff_dfa, state, next_state)
                    transitions.append((next_state, token))
                else:
                    for token in sorted(next_state_to_symbols[next_state]):
                        transitions.append((next_state, token))
            transitions_cache[state] = transitions
            return transitions

        max_frontier = max(256, 8 * k, 4 * len(diff_dfa.states))
        max_expansions = max_frontier * max(4, self.max_length)
        witnesses = []
        frontier = deque([(diff_dfa.start_state, [])])
        expansions = 0

        while frontier:
            cur, token_path = frontier.popleft()

            if cur in diff_dfa.final_states:
                witnesses.append("".join(token_path))
                if len(witnesses) >= k:
                    return witnesses

            if len(token_path) >= self.max_length:
                continue

            expansions += 1
            if expansions > max_expansions:
                break

            for nxt, token in outgoing_transitions(cur):
                if len(frontier) >= max_frontier:
                    break
                frontier.append((nxt, token_path + [token]))

        return witnesses

    def k_witnesses_shortest(
        self,
        dfa_a: DeterministicFiniteAutomaton,
        dfa_b: DeterministicFiniteAutomaton,
        k: int = 10,
        clustered: bool = False,
    ):
        """Return all disagreement witnesses at the shortest discovered length.

        This is a BFS over the difference DFA. Once the first accepting state is
        reached, its path length is the shortest counterexample length. We keep
        collecting witnesses at exactly that length, and stop as soon as BFS
        reaches a longer accepting witness.
        """
        diff_dfa = dfa_a.get_difference(dfa_b).minimize()
        if not diff_dfa.final_states or diff_dfa.start_state is None:
            return []

        transitions_cache = {}

        def outgoing_transitions(state):
            if state in transitions_cache:
                return transitions_cache[state]

            next_state_to_symbols = {}
            for symbol in diff_dfa.symbols:
                nxt = diff_dfa._transition_function(state, symbol)
                if len(nxt) == 0:
                    continue
                next_state = list(nxt)[0]
                token = symbol.value if hasattr(symbol, "value") else str(symbol)
                next_state_to_symbols.setdefault(next_state, []).append(token)

            transitions = []
            for next_state in sorted(next_state_to_symbols, key=lambda s: str(s)):
                if clustered:
                    token = dfa_edge_char_class(diff_dfa, state, next_state)
                    transitions.append((next_state, token))
                else:
                    for token in sorted(next_state_to_symbols[next_state]):
                        transitions.append((next_state, token))
            transitions_cache[state] = transitions
            return transitions

        # "All shortest" can still be large for a non-clustered large alphabet,
        # so keep the same bounded-search spirit as dfs/bfs witness generation.
        max_frontier = max(1024, 16 * k, 8 * len(diff_dfa.states))
        max_expansions = max_frontier * max(4, self.max_length)
        witnesses = []
        seen = set()
        shortest_len = None
        frontier = deque([(diff_dfa.start_state, [])])
        expansions = 0

        while frontier:
            cur, token_path = frontier.popleft()
            path_len = len(token_path)

            if shortest_len is not None and path_len > shortest_len:
                break

            if cur in diff_dfa.final_states:
                witness = "".join(token_path)
                if shortest_len is None:
                    shortest_len = path_len
                if path_len == shortest_len and witness not in seen:
                    seen.add(witness)
                    witnesses.append(witness)
                continue

            if path_len >= self.max_length:
                continue

            expansions += 1
            if expansions > max_expansions:
                break

            for nxt, token in outgoing_transitions(cur):
                if len(frontier) >= max_frontier:
                    break
                frontier.append((nxt, token_path + [token]))

        return witnesses
    
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
        union = (A | B).optimize()
        
        if symdiff.start() == pynini.NO_STATE_ID or symdiff.num_states() == 0:
            return 0.0
        if union.start() == pynini.NO_STATE_ID or union.num_states() == 0:
            return 0.0
        
        def project_and_optimize(fst):
            fst = pynini.project(fst, "input")
            fst = pynini.rmepsilon(fst).optimize()
            fst.set_input_symbols(sigma)
            fst.set_output_symbols(sigma)
            return fst
        
        symdiff = project_and_optimize(symdiff)
        union = project_and_optimize(union)

        total, diff = 0, 0
        for length in range(0, k + 1):
            total += self.count_strings_of_length(union, sigma, length)
            diff += self.count_strings_of_length(symdiff, sigma, length)
            
        return diff / total if total > 0 else 0.0
    
class SimplyRegularLanguage(RegularLanguage):
    def __init__(self, regex_str, max_length):
        super().__init__(regex_str, max_length)

        self.regex = Regex(regex_str)

        self.nfa = self.regex.to_epsilon_nfa()
        self.dfa = self.nfa.to_deterministic().minimize()
        self.sigma = {Symbol(c.value) for c in self.dfa.symbols}
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
            sigma = self.sigma_from_chars([s.value for s in self.sigma])
        fst = self.dfa_to_pynini_fst(dfa, sigma)
        return dfa, fst, sigma
    
class PythonRegularLanguage(RegularLanguage):
    def __init__(self, regex_str, max_length):
        super().__init__(regex_str, max_length)

        self.regex = PythonRegex(regex_str)

        self.nfa = self.regex.to_epsilon_nfa()
        self.dfa = self.nfa.to_deterministic().minimize()
        self.sigma = {Symbol(c.value) for c in self.dfa.symbols}
        self.num_alphabets = len(self.dfa.symbols)
        self.num_categories = 2     # positive | negative
    
    def regex_to_pynini_via_pyformlang(self, rx: str, sigma: pynini.SymbolTable=None):
        re = PythonRegex(rx)
        nfa = re.to_epsilon_nfa()
        dfa = nfa.to_deterministic().minimize()
        if sigma is None:
            sigma = self.sigma_from_chars([s.value for s in self.sigma])
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
        "([A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z]{0,5}))[A-Za-z0-9#]*){5,}",
        max_length=32,
        alphabet="[A-Za-z0-9#]",
        debug=True,
    )
    print(tree_to_regex(nl.regex_tree))
    dfa = nl.dfa
    state_map = {s: i for i, s in enumerate(dfa.states)}
    print("Start state:", state_map[dfa.start_state])
    print("Final states:", [state_map[s] for s in dfa.final_states])
    print("Transitions:")
    for state in dfa.states:
        for symbol in dfa.symbols:
            next_states = dfa._transition_function(state, symbol)
            if len(next_states) > 0:
                print(f"  {state_map[state]} --{symbol}--> {list(state_map[s] for s in next_states)}")
                
    from teacher import Teacher
    teacher = Teacher(nl)
    msg = {}
    msg["Prediction"] = "([A-Za-z0-9#]{20,})&~([A-Za-z0-9#]*#[A-Za-z0-9#]*#[A-Za-z0-9#]*#[A-Za-z0-9#]*)"
    _, fst_gt, _ = nl.regex_to_pynini_via_pyformlang(nl.regex_str)
    msg = teacher.judge_regex(
        msg=msg,
        fst_gt=fst_gt,
        train_ex=[], train_labels=[], eval_ex=[], eval_labels=[]
    )
    print(msg)
    train_ex, train_labels = teacher.generate_counterexamples(
        bs=250,
        regex_gt=nl.regex_str,
        regex_gen=msg["Prediction"],
        clustered=True
    )
    print(f"Generated {len(train_ex)} counterexamples:")
    for ex in train_ex[:20]:
        print(f"  {ex} (accepted by GT: {nl.accepts(ex)})")
