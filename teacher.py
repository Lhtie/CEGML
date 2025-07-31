import random
from collections import deque
from pyformlang.finite_automaton import State, Symbol, DeterministicFiniteAutomaton

class Teacher:
    def __init__(self, task):
        self.task = task

    def _get_final_state(self, dfa, string):
        current_state = dfa.start_state
        for char in string:
            sym = Symbol(char)
            next = dfa._transition_function(current_state, sym)
            if len(next) > 0:
                current_state = next[0]
            else:
                current_state = None
                break
        return current_state

    def _generate_strings_to_state(self, dfa, target_state, max_depth, num_samples):
        strings, labels = [], []
        queue = deque([(dfa.start_state, [], 0)])

        while queue:
            current_state, path, depth = queue.popleft()
            if current_state == target_state:
                string = "".join(path)
                strings.append(string)
                labels.append(int(current_state in dfa.final_states))
            if depth > max_depth or current_state is None:
                continue

            for symbol in dfa.symbols:
                next_state = dfa._transition_function(current_state, symbol)
                if len(next_state) > 0:
                    next_state = next_state[0]
                else:
                    next_state = None
                queue.append((next_state, path + [symbol.value], depth + 1))

        paired = list(zip(strings, labels))
        sampled = random.sample(paired, num_samples)
        strings, labels = zip(*sampled)
        return list(strings), list(labels)

    def _gen_from_ex(self, ex, n):
        state = self._get_final_state(self.task.dfa, ex)
        return self._generate_strings_to_state(
            self.task.dfa, 
            state, 
            max_depth=self.task.max_length,
            num_samples=n
        )

    def generate_counterexamples(self, n, neg_ex, pos_ex):
        ce_x, ce_y = [], []
        for ex in neg_ex:
            gt = self.task.accepts(ex)
            if gt:
                x, y = self._gen_from_ex(ex, n)
                ce_x += x
                ce_y += y
        for ex in pos_ex:
            gt = self.task.accepts(ex)
            if not gt:
                x, y = self._gen_from_ex(ex, n)
                ce_x += x
                ce_y += y
        return ce_x, ce_y
    
    def _generate_random_ab_strings(self, m, n):
        strings = []
        alphabet = [chr(c + ord('a')) for c in range(self.task.num_alphabets)]
        for _ in range(m):
            length = random.randint(0, n)
            s = ''.join(random.choices(alphabet, k=length))
            strings.append(s)
        return strings
    
    def judge(self, classifier, n, batch_size):
        inputs = self._generate_random_ab_strings(n, self.task.max_length)
        labels = [int(self.task.accepts(i)) for i in inputs]

        pred = classifier(inputs, batch_size)
        acc = sum([int(x == y) for x, y in zip(pred, labels)]) / len(pred)
        return acc

