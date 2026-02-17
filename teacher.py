import random
import numpy as np
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
            if depth > max_depth:
                continue
            if current_state == target_state:
                string = "".join(path)
                strings.append(string)
                labels.append(int(current_state in dfa.final_states))
            elif current_state == None:
                continue

            for symbol in dfa.symbols:
                next_state = dfa._transition_function(current_state, symbol)
                if len(next_state) > 0:
                    next_state = next_state[0]
                else:
                    next_state = None
                queue.append((next_state, path + [symbol.value], depth + len(symbol.value)))

        paired = list(zip(strings, labels))
        sampled = random.choices(paired, k=num_samples)
        if num_samples == 0:
            return [], []
        else:
            strings, labels = zip(*sampled)
            return list(strings), list(labels)

    def _gen_from_ex(self, ex, n, mode):
        if mode == "dfa_state":
            state = self._get_final_state(self.task.dfa, ex)
            return self._generate_strings_to_state(
                self.task.dfa, 
                state, 
                max_depth=self.task.max_length,
                num_samples=n
            )
        elif mode == "random":
            x = self.task.generate_random_strings_uniform(n, self.task.max_length)
            y = [int(self.task.accepts(s)) for s in x]
            return x, y
        elif mode == "repeat":
            x = [ex] * n
            y = [int(self.task.accepts(ex))] * n
            return x, y

    def generate_counterexamples(self, n, neg_ex, pos_ex, mode="dfa_state"):
        ce_x, ce_y = [], []
        for ex in neg_ex:
            gt = self.task.accepts(ex)
            if gt:
                x, y = self._gen_from_ex(ex, n - 1, mode)
                ce_x += [ex] + x
                ce_y += [int(gt)] + y
        for ex in pos_ex:
            gt = self.task.accepts(ex)
            if not gt:
                x, y = self._gen_from_ex(ex, n - 1, mode)
                ce_x += [ex] + x
                ce_y += [int(gt)] + y
        return ce_x, ce_y
    
    def generate_counterexamples(self, bs, regex_gt, regex_gen):
        dfa_gt, fst_gt, sigma = self.task.regex_to_pynini_via_pyformlang(regex_gt)
        dfa_gen, fst_gen, _ = self.task.regex_to_pynini_via_pyformlang(regex_gen, sigma)

        rate = self.task.diff_ratio(fst_gt, fst_gen, sigma, k=self.task.max_length)
        n = np.ceil(bs * rate / 2).astype(int)

        ce_x = self.task.k_witnesses(dfa_gt, dfa_gen, k=n) + \
                self.task.k_witnesses(dfa_gen, dfa_gt, k=n)
        ce_y = [int(self.task.accepts(x)) for x in ce_x]
        return ce_x, ce_y
    
    def generate_posexamples(self, n, seq_len):
        final_states = list(self.task.dfa.final_states)
        xs, ys = [], []
        nums = [0] * len(final_states)
        for x in range(n):
            bucket_id = random.randrange(len(final_states))
            nums[bucket_id] += 1

        for i, num in enumerate(nums):
            x, y = self._generate_strings_to_state(
                self.task.dfa,
                final_states[i],
                max_depth=seq_len,
                num_samples=num
            )
            xs += x
            ys += y
        return xs, ys
    
    def judge(self, classifier, n, batch_size, seq_len):
        # pos_x, pos_y = self.generate_posexamples(int(n * 0.4), seq_len)
        # inputs = pos_x + self.task.generate_random_strings_uniform(n - int(n * 0.4), seq_len)
        inputs = self.task.generate_random_strings_balanced(n, seq_len)
        labels = [int(self.task.accepts(i)) for i in inputs]

        pred = classifier(inputs, batch_size)
        acc = sum([int(x == y) for x, y in zip(pred, labels)]) / len(pred)
        return acc
