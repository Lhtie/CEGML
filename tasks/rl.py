import torch
import random
import numpy as np
from pyformlang.regular_expression import Regex

class SimplyRegularLanguage:
    def __init__(self, regex_str, max_length):
        self.regex_str = regex_str
        self.max_length = max_length

        self.regex = Regex(regex_str)

        self.nfa = self.regex.to_epsilon_nfa()
        self.dfa = self.nfa.to_deterministic().minimize()
        # print(len(self.dfa.states))
        self.num_alphabets = 2      # a, b
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
    
    def generate_random_strings_balanced(self, m, n):
        strings = []
        alphabet = [chr(c + ord('a')) for c in range(self.num_alphabets)]
        for _ in range(m):
            length = random.randint(1, n)
            accepted = random.choice([True, False])
            for _ in range(10):
                s = ''.join(random.choices(alphabet, k=length))
                if (accepted and self.accepts(s)) or (not accepted and not self.accepts(s)):
                    strings.append(s)
                    # print(f"String {s} accepted {accepted}")
                    break
        return strings

    def accepts(self, str):
        return self.dfa.accepts(str)