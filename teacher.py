import random
import numpy as np
import signal
import multiprocessing as mp
from contextlib import contextmanager
from collections import deque
from pyformlang.finite_automaton import State, Symbol, DeterministicFiniteAutomaton
from tasks.utils import dfa_accepts_ex


class JudgeTimeoutError(TimeoutError):
    pass


@contextmanager
def time_limit(seconds):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise JudgeTimeoutError(f"judge_regex timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _judge_regex_worker(
    queue,
    task,
    msg,
    fst_gt,
    train_ex,
    train_labels,
    eval_ex,
    eval_labels,
    sigma,
):
    def score_examples(dfa_pred, examples, labels):
        if len(examples) == 0:
            return None
        return sum(
            [int(int(dfa_accepts_ex(dfa_pred, ex)) == label) for ex, label in zip(examples, labels)]
        ) / len(examples)

    pred = msg.get("Prediction")
    try:
        if sigma is None:
            dfa_pred, fst_pred, sigma_cur = task.regex_to_pynini_via_pyformlang(pred)
        else:
            dfa_pred, fst_pred, sigma_cur = task.regex_to_pynini_via_pyformlang(pred, sigma)

        eq, witness = task.equivalent_and_witness(fst_gt, fst_pred, sigma_cur)
        result = dict(msg)
        result["Equivalent"] = eq
        result["Witness"] = witness
        result["scoreTrainSet"] = score_examples(dfa_pred, train_ex, train_labels)
        result["scoreEvalSet"] = score_examples(dfa_pred, eval_ex, eval_labels)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", str(e)))


def _generate_counterexamples_worker(
    queue,
    task,
    bs,
    regex_gt,
    regex_gen,
    clustered,
    generation_mode,
):
    try:
        dfa_gt, fst_gt, sigma = task.regex_to_pynini_via_pyformlang(regex_gt)
        dfa_gen, fst_gen, _ = task.regex_to_pynini_via_pyformlang(regex_gen, sigma)

        if generation_mode == "dfs":
            ce_pos = task.k_witnesses_dfs(
                dfa_gt, dfa_gen, bs // 2, clustered=clustered
            )
            ce_neg = task.k_witnesses_dfs(
                dfa_gen, dfa_gt, bs // 2, clustered=clustered
            )
        elif generation_mode == "random":
            ce_pos = task.k_witnesses_sample(
                dfa_gt, dfa_gen, bs // 2, clustered=clustered
            )
            ce_neg = task.k_witnesses_sample(
                dfa_gen, dfa_gt, bs // 2, clustered=clustered
            )
        elif generation_mode == "bfs":
            ce_pos = task.k_witnesses_bfs(
                dfa_gt, dfa_gen, bs // 2, clustered=clustered
            )
            ce_neg = task.k_witnesses_bfs(
                dfa_gen, dfa_gt, bs // 2, clustered=clustered
            )
        else:
            raise ValueError(f"Unknown counterexample generation mode: {generation_mode}")

        ce_x = ce_pos + ce_neg
        ce_y = [1] * len(ce_pos) + [0] * len(ce_neg)
        queue.put(("ok", (ce_x, ce_y)))
    except Exception as e:
        queue.put(("error", str(e)))

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
    
    def generate_counterexamples(
        self,
        bs,
        regex_gt,
        regex_gen,
        clustered=False,
        generation_mode="random",
        timeout_seconds=10,
    ):
        if timeout_seconds is not None and timeout_seconds > 0:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = None

            if ctx is not None:
                queue = ctx.Queue()
                proc = ctx.Process(
                    target=_generate_counterexamples_worker,
                    args=(
                        queue,
                        self.task,
                        bs,
                        regex_gt,
                        regex_gen,
                        clustered,
                        generation_mode,
                    ),
                )
                proc.start()
                proc.join(timeout_seconds)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    raise JudgeTimeoutError(f"generate_counterexamples timed out after {timeout_seconds}s")
                if not queue.empty():
                    status, payload = queue.get()
                    if status == "ok":
                        return payload
                    raise RuntimeError(payload)
                if proc.exitcode not in (0, None):
                    raise RuntimeError(f"generate_counterexamples worker exited with code {proc.exitcode}")
                raise RuntimeError("generate_counterexamples worker exited without returning a result")

        with time_limit(timeout_seconds):
            inline_queue = _ReturnQueue()
            _generate_counterexamples_worker(
                inline_queue,
                self.task,
                bs,
                regex_gt,
                regex_gen,
                clustered,
                generation_mode,
            )
            return inline_queue.value
    
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

    def judge_regex(
        self, msg, fst_gt,
        train_ex, train_labels, eval_ex, eval_labels,
        sigma=None,
        timeout_seconds=10,
    ):
        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                try:
                    ctx = mp.get_context("fork")
                except ValueError:
                    ctx = None

                if ctx is not None:
                    queue = ctx.Queue()
                    proc = ctx.Process(
                        target=_judge_regex_worker,
                        args=(
                            queue,
                            self.task, msg, fst_gt,
                            train_ex, train_labels,
                            eval_ex, eval_labels,
                            sigma,
                        ),
                    )
                    proc.start()
                    proc.join(timeout_seconds)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                        raise JudgeTimeoutError(f"judge_regex timed out after {timeout_seconds}s")
                    if not queue.empty():
                        status, payload = queue.get()
                        if status == "ok":
                            msg.update(payload)
                        else:
                            raise RuntimeError(payload)
                    elif proc.exitcode not in (0, None):
                        raise RuntimeError(f"judge_regex worker exited with code {proc.exitcode}")
                    return msg

            with time_limit(timeout_seconds):
                _judge_regex_worker(
                    queue=_InlineQueue(msg),
                    task=self.task,
                    msg=msg,
                    fst_gt=fst_gt,
                    train_ex=train_ex,
                    train_labels=train_labels,
                    eval_ex=eval_ex,
                    eval_labels=eval_labels,
                    sigma=sigma,
                )
        except Exception as e:
            msg["Error"] = f"Error compiling regex: {e}"
            print(msg["Error"])

        return msg


class _InlineQueue:
    def __init__(self, target_msg):
        self.target_msg = target_msg

    def put(self, item):
        status, payload = item
        if status == "ok":
            self.target_msg.update(payload)
        else:
            raise RuntimeError(payload)


class _ReturnQueue:
    def __init__(self):
        self.value = None

    def put(self, item):
        status, payload = item
        if status == "ok":
            self.value = payload
        else:
            raise RuntimeError(payload)
