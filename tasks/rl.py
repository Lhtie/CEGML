import torch
import random
import pynini
import numpy as np

from pyformlang.regular_expression import Regex
from collections import deque
from pyformlang.finite_automaton import State, Symbol, DeterministicFiniteAutomaton

class SimplyRegularLanguage:
    def __init__(self, regex_str, max_length):
        self.regex_str = regex_str
        self.max_length = max_length

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
    
    def generate_random_strings_balanced(self, m, n, rate=0.5, len_gen=None):
        strings = []
        alphabet = [chr(c + ord('a')) for c in range(self.num_alphabets)]
        for _ in range(m):
            accepted = random.random() < rate
            if accepted:
                while True:
                    try:
                        s = self._generate_string_to_state(self.dfa, random.choice(list(self.dfa.final_states)), n)
                        break
                    except AssertionError:
                        continue
                strings.append(s)
            else:
                length = random.randint(1, n)
                while True:
                    s = ''.join(random.choices(alphabet, k=length))
                    if not self.accepts(s):
                        strings.append(s)
                        break
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

    def regex_to_pynini_via_pyformlang(self, rx: str, sigma=None):
        re = Regex(rx)
        nfa = re.to_epsilon_nfa()
        dfa = nfa.to_deterministic().minimize()
        if sigma is None:
            sigma = self.sigma_from_chars([s.value for s in dfa.symbols])
        fst = self.dfa_to_pynini_fst(dfa, sigma)
        return dfa, fst, sigma

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