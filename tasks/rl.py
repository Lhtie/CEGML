import torch
import random
from pyformlang.regular_expression import Regex

class SimplyRegularLanguage:
    def __init__(self, regex_str, max_length):
        self.regex_str = regex_str
        self.max_length = max_length

        self.regex = Regex(regex_str)

        self.nfa = self.regex.to_epsilon_nfa()
        self.dfa = self.nfa.to_deterministic().minimize()
        self.num_alphabets = len({s.value for s in self.dfa.symbols})
        self.num_categories = 2     # positive | negative

    def to_tensor(self, batched_str):
        lengths = torch.tensor([len(s) + 1 for s in batched_str], dtype=torch.long)
        max_length = lengths.max().item()
        tensor = torch.zeros((len(batched_str), max_length, self.num_alphabets), dtype=torch.float)
        for i, s in enumerate(batched_str):
            for j, c in enumerate(s):
                tensor[i, j, ord(c) - ord('a')] = 1
            tensor[i, len(s), :] = 1
        return tensor, lengths
    
    def generate_random_strings(self, m, n):
        strings = []
        alphabet = [chr(c + ord('a')) for c in range(self.num_alphabets)]
        for _ in range(m):
            length = random.randint(1, n)
            s = ''.join(random.choices(alphabet, k=length))
            strings.append(s)
        return strings

    def accepts(self, str):
        return self.dfa.accepts(str)