import torch
import random
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
            if current_state == target_state and depth > 0:
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