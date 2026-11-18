"""Dataset adapters and counterexample oracles for program induction tasks."""

import ast
import json
import random
import re
import signal
import time
from copy import deepcopy
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "datasets"


@dataclass(frozen=True)
class OracleResult:
    """Shared verdict contract; backend-specific evidence is opaque to the runner."""
    equivalent: bool | None
    counterexamples: list
    details: dict | None = None
    response: str | None = None
    error: str | None = None

    def log_fields(self):
        status = ("error" if self.error else "accepted" if self.equivalent is True
                  else "rejected" if self.equivalent is False else "inconclusive")
        fields = {"Equivalent": self.equivalent,
                  "OracleCounterexamples": deepcopy(self.counterexamples),
                  "OracleStatus": status}
        if self.details is not None:
            fields["OracleDetails"] = deepcopy(self.details)
        if self.response is not None:
            fields["OracleResponse"] = self.response
        if self.error is not None:
            fields["OracleError"] = self.error
            fields["Error"] = self.error
        return fields


def _pairs(items):
    return [(item["input"], item["output"]) for item in items]


def _instruction_pairs(items, name):
    from benchmarks.instruction_oracle import instruction_pairs
    return instruction_pairs(items, name)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _smt_solver(script):
    """Load an SMT query through cvc5's Python API, without a CLI process."""
    import cvc5

    solver = cvc5.Solver()
    solver.setOption("incremental", "true")
    solver.setOption("produce-models", "true")
    solver.setOption("tlimit-per", "10000")
    parser = cvc5.InputParser(solver)
    parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6,
                          "\n".join(script), "benchmark")
    symbols = parser.getSymbolManager()
    while True:
        command = parser.nextCommand()
        if command.isNull():
            break
        output = command.invoke(solver, symbols)
        if output and output.lstrip().startswith("(error"):
            raise RuntimeError("cvc5 query failed: " + output)
    terms = {str(term): term for term in symbols.getDeclaredTerms()}
    return solver, terms


def _smt_choice(disagreements, limit):
    """Ask cvc5 for indices of disagreements in an explicitly bounded domain."""
    if not disagreements or limit <= 0:
        return []
    from cvc5 import Kind

    count = max(disagreements) + 1
    solver, terms = _smt_solver([
        "(set-logic QF_LIA)", "(declare-const i Int)",
        f"(assert (and (<= 0 i) (< i {count})))",
        "(assert (or " + " ".join(f"(= i {j})" for j in disagreements) + "))",
    ])
    found = []
    for _ in range(min(limit, len(disagreements))):
        result = solver.checkSat()
        if result.isUnsat():
            break
        if not result.isSat():
            raise RuntimeError("cvc5 returned unknown: " + str(result))
        index = solver.getValue(terms["i"]).getIntegerValue()
        found.append(index)
        solver.assertFormula(solver.mkTerm(
            Kind.DISTINCT, terms["i"], solver.mkInteger(index)))
    return found


def _crypto_smt(reference, candidate, limit):
    """Prove or refute equality of two finite alphabet substitutions with QF_LIA."""
    from cvc5 import Kind

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def mapping(spec):
        return [ord(BenchmarkTask._apply_cipher(spec, ch)) for ch in alphabet]
    left, right = mapping(reference), mapping(candidate)
    def expression(values):
        expr = str(values[-1])
        for i in range(len(values) - 2, -1, -1):
            expr = f"(ite (= i {i}) {values[i]} {expr})"
        return expr
    solver, terms = _smt_solver([
        "(set-logic QF_LIA)", "(declare-const i Int)",
        f"(assert (and (<= 0 i) (< i {len(alphabet)})))",
        f"(assert (not (= {expression(left)} {expression(right)})))",
    ])
    witnesses = []
    for _ in range(limit):
        result = solver.checkSat()
        if result.isUnsat():
            return witnesses, not witnesses
        if not result.isSat():
            raise RuntimeError("cvc5 returned unknown: " + str(result))
        index = solver.getValue(terms["i"]).getIntegerValue()
        witnesses.append(alphabet[index])
        solver.assertFormula(solver.mkTerm(
            Kind.DISTINCT, terms["i"], solver.mkInteger(index)))
    return witnesses, False if witnesses else None


def _seq_expr(node):
    """Translate a useful subset of list expressions to SMT sequence theory."""
    if isinstance(node, ast.Name) and node.id == "x":
        return "x"
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(- {_seq_expr(node.operand)})"
    if isinstance(node, ast.List):
        if not node.elts:
            return "(as seq.empty (Seq Int))"
        parts = [f"(seq.unit {_seq_expr(item)})" for item in node.elts]
        return parts[0] if len(parts) == 1 else "(seq.++ " + " ".join(parts) + ")"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return f"(seq.++ {_seq_expr(node.left)} {_seq_expr(node.right)})"
    if isinstance(node, ast.Subscript):
        base = _seq_expr(node.value)
        sub = node.slice
        if isinstance(sub, ast.Slice):
            if sub.step is not None:
                if isinstance(sub.step, ast.UnaryOp) and isinstance(sub.step.op, ast.USub) and isinstance(sub.step.operand, ast.Constant) and sub.step.operand.value == 1 and sub.lower is None and sub.upper is None:
                    return f"(seq.rev {base})"
                raise ValueError("Unsupported slice step")
            def pos(value, default):
                if value is None:
                    return default
                v = _seq_expr(value)
                return f"(+ (seq.len {base}) {v})" if v.startswith("(- ") else v
            lo = pos(sub.lower, "0")
            hi = pos(sub.upper, f"(seq.len {base})")
            return f"(seq.extract {base} {lo} (- {hi} {lo}))"
        index = _seq_expr(sub)
        if index.startswith("(- "):
            index = f"(+ (seq.len {base}) {index})"
        return f"(seq.nth {base} {index})"
    raise ValueError("Function is outside the SMT sequence subset")


def _return_seq(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function = next((node for node in functions if node.name == "fn"),
                    next((node for node in functions if node.name == "transform"), functions[0]))
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        raise ValueError("Function is outside the SMT sequence subset")
    return _seq_expr(function.body[0].value)


def _list_sequence_smt(reference, candidate):
    """Search bounded lengths with symbolic list elements in cvc5 sequence theory."""
    left, right = _return_seq(reference), _return_seq(candidate)
    for length in range(3, 9):
        names = [f"v{i}" for i in range(length)]
        xexpr = "(seq.++ " + " ".join(f"(seq.unit {v})" for v in names) + ")"
        script = ["(set-logic ALL)"]
        script += [f"(declare-const {v} Int)" for v in names]
        script += [f"(assert (and (<= 0 {v}) (<= {v} 100)))" for v in names]
        script += [f"(define-fun x () (Seq Int) {xexpr})",
                   f"(assert (not (= {left} {right})))"]
        solver, terms = _smt_solver(script)
        result = solver.checkSat()
        if result.isSat():
            return [solver.getValue(terms[name]).getIntegerValue() for name in names]
        if not result.isUnsat():
            raise RuntimeError("cvc5 returned unknown: " + str(result))
    return None


def _validate_function(source):
    """Accept a small Python subset; never import or expose process APIs to model code."""
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "transform":
        raise ValueError("Expected one function named transform")
    if len(tree.body[0].args.args) != 1 or tree.body[0].args.args[0].arg != "x":
        raise ValueError("Expected transform(x)")
    blocked = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
               ast.AsyncFunctionDef, ast.Lambda, ast.Try, ast.With, ast.Raise,
               ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.While)
    allowed_calls = {"len", "range", "sum", "min", "max", "sorted", "reversed", "list", "int",
                     "str", "abs", "enumerate", "zip", "all", "any", "set", "tuple"}
    local_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    allowed_methods = {"append", "extend", "insert", "count", "index", "copy", "sort", "reverse",
                       "split", "join", "strip", "lower", "upper", "startswith", "endswith", "get"}
    for node in ast.walk(tree):
        if isinstance(node, blocked) or (isinstance(node, ast.Name) and node.id.startswith("__")):
            raise ValueError("Unsupported Python syntax")
        if isinstance(node, ast.Attribute) and (node.attr not in allowed_methods or node.attr.startswith("__")):
            raise ValueError("Unsupported method")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in allowed_calls | local_functions:
                raise ValueError("Unsupported function call")
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError("Unsupported call")
    import builtins
    safe = {name: getattr(builtins, name) for name in allowed_calls if hasattr(builtins, name)}
    scope = {"__builtins__": safe}
    exec(compile(tree, "<candidate>", "exec"), scope)
    return scope["transform"]


@contextmanager
def _time_limit(seconds=1):
    def timeout(_signum, _frame):
        raise TimeoutError("Candidate execution timed out")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class BenchmarkTask:
    def __init__(self, kind, benchmark_id, seed=43, eval_size=32, *, crypto_setting="paper",
                 crypto_ood_ratio=0.):
        self.kind, self.benchmark_id = kind, benchmark_id
        self.rng = random.Random(seed)
        self.oracle_name = {"cryptography": "cvc5_QF_LIA", "list_functions": "cvc5_sequence_or_bounded_Python",
                            "instruction_induction": "LLM", "scan": "exhaustive_enumeration", "ilp": "finite_Horn_forward_chaining"}[kind]
        if kind == "cryptography":
            family, row = benchmark_id.split(":") if ":" in benchmark_id else (benchmark_id, "0")
            if family not in {"caesar", "atbash", "keyboard"}:
                raise ValueError("Unknown cipher family")
            item = _read_jsonl(ROOT / "cryptography/raw" / f"{family}.jsonl")[int(row)]
            self.family = family
            self.crypto_setting = crypto_setting
            if crypto_setting not in {"paper", "legacy"}:
                raise ValueError("crypto_setting must be paper or legacy")
            self.clean_train = _pairs(item["train"]["normal"])
            self.shift = (ord(self.clean_train[0][1][0].lower()) - ord(self.clean_train[0][0][0].lower())) % 26 if family == "caesar" else None
            self.eval = _pairs(item["test"])[:eval_size]
            if crypto_setting == "paper":
                from benchmarks.crypto import PAPER_PROMPT, paper_examples
                self.train = paper_examples(item, ood_ratio=crypto_ood_ratio)
                self.prompt_template = PAPER_PROMPT
                self.episode_defaults = {"start_size": 10, "tot_train_size": 10, "ce_start_size": 10}
                self.shuffle_training = False
                self.allow_regularization = False
                self.oracle_name = "cvc5_bounded_symbolic_Python"
                # Retain the existing clean-log suffix for path compatibility.
                self.log_suffix = f"__paper_noise0_ood{crypto_ood_ratio:g}_posappend"
                self.task_settings = {"crypto_setting": "paper", "n_train": 10, "n_test": len(self.eval),
                                      "noise_ratio": 0.0, "ood_ratio": crypto_ood_ratio, "hypothesis": "Python_fn_x",
                                      "paper_task_protocol": len(self.eval) == 10, "method": "shared_CE_pipeline_not_paper_SRR",
                                      "regularization_applied": False,
                                      "prompt_variant": "base_with_reasoning_and_ans",
                                      "oracle_policy": "bounded_symbolic_python_crypto_v1",
                                      "oracle_domain": {"alphabet": "ASCII A-Z/a-z", "min_length": 0, "max_length": 10}}
            else:
                self.train = self.clean_train
                self.task_settings = {"crypto_setting": "legacy", "hypothesis": "closed_cipher_JSON"}
        elif kind == "list_functions":
            items = _read_jsonl(ROOT / "list_functions/raw/list_functions.jsonl")
            item = next((v for v in items if v["id"] == benchmark_id), None)
            if item is None:
                raise ValueError(f"Unknown list function: {benchmark_id}")
            self.train = _pairs(item["train"]["normal"])
            self.eval = _pairs(item["test"])[:eval_size]
            self.reference_source = item["function"]
            self.reference_name = re.search(r"def (\w+)\(", self.reference_source).group(1)
            scope = {}
            exec(self.reference_source, scope)
            self.reference = scope[self.reference_name]
            from benchmarks.list_functions import PROMPT
            self.prompt_template = PROMPT
            self.episode_defaults = {"start_size": 10, "tot_train_size": 10, "ce_start_size": 10}
            self.shuffle_training = False
            self.allow_regularization = False
            self.task_settings = {"list_setting": "paper_clean", "n_train": len(self.train),
                                  "n_test": len(self.eval), "noise_ratio": 0.0,
                                  "hypothesis": "Python_fn_x", "python_syntax": "standard_Python",
                                  "prompt_variant": "base_with_reasoning_and_ans",
                                  "method": "shared_CE_pipeline_not_paper_SRR"}
        elif kind == "ilp":
            from benchmarks.ilp import ILPProblem
            self.ilp = ILPProblem(benchmark_id)
            self.train, self.eval = self.ilp.train, self.ilp.eval
            self.prompt_template = self.ilp.prompt
            self.icl_instruction = self.ilp.icl_instruction
            self.episode_defaults = {"ce_start_size": 3}
            self.shuffle_training = False
            self.task_settings = self.ilp.settings
        elif kind == "instruction_induction":
            base = ROOT / "instruction_induction/raw"
            induce = json.loads((base / "induce" / f"{benchmark_id}.json").read_text())
            execute = json.loads((base / "execute" / f"{benchmark_id}.json").read_text())
            self.train = _instruction_pairs(induce["examples"].values(), benchmark_id)
            self.eval = _instruction_pairs(execute["examples"].values(), benchmark_id)[:eval_size]
            self.task_settings = {"oracle_policy": "grounded_instruction_oracle_v3",
                                  "oracle_groundtruth": "official_human_annotations",
                                  "oracle_reference_split": "induce_only",
                                  "oracle_verdict_semantics": "task_specification_not_execution_accuracy",
                                  "input_adapter": "official_cause_selection_v2" if benchmark_id == "cause_and_effect" else "raw_original"}
        elif kind == "scan":
            splits = {"simple", "length", "addprim_jump", "addprim_turn_left",
                      "template_around_right", "template_jump_around_right",
                      "template_opposite_right", "template_right"}
            if benchmark_id not in splits:
                raise ValueError(f"Unknown SCAN split: {benchmark_id}")
            base = ROOT / "scan/raw"
            def load(name):
                return [tuple(line.strip().split(" OUT: ")) for line in (base / name).read_text().splitlines()]
            self.train = [(x.removeprefix("IN: "), y) for x, y in load(f"tasks_train_{benchmark_id}.txt")]
            self.eval = [(x.removeprefix("IN: "), y) for x, y in load(f"tasks_test_{benchmark_id}.txt")]
            self.all_examples = [(x.removeprefix("IN: "), y) for x, y in load("tasks.txt")]
            self.task_settings = {"scan_setting": "official_split_program_induction",
                                  "split": benchmark_id, "n_train_available": len(self.train),
                                  "n_test": len(self.eval), "hypothesis": "Python_transform_x",
                                  "python_syntax": "standard_Python", "evaluation": "full_split_exact_match",
                                  "oracle_domain": "all_public_commands_including_test",
                                  "oracle_domain_size": len(self.all_examples),
                                  "method": "shared_CE_pipeline_not_original_seq2seq"}
        else:
            raise ValueError(f"Unknown benchmark: {kind}")

    def extract_candidate(self, response, default_extractor, multiple=False):
        if self.kind == "ilp":
            from benchmarks.ilp import extract_candidate
            return extract_candidate(response, default_extractor, multiple)
        if self.kind in {"list_functions", "scan"}:
            from benchmarks.list_functions import extract_program
        elif self.kind == "cryptography" and self.crypto_setting == "paper":
            from benchmarks.crypto import extract_program
        else:
            return default_extractor(response)
        if not multiple:
            return extract_program(response)
        sources = re.findall(r"<ans>\s*(.*?)\s*</ans>", response or "", re.S)
        if not sources:
            sources = re.findall(r"```(?:python)?\s*\n(.*?)```", response or "", re.S)
        return [program for source in sources if (program := extract_program(source)) is not None]

    def format_pair(self, x, y):
        if self.kind == "list_functions" or (self.kind == "cryptography" and self.crypto_setting == "paper"):
            return f"Input: {x}\nOutput: {y}"
        return f"{json.dumps(x, ensure_ascii=False)} => {json.dumps(y, ensure_ascii=False)}"

    def predict(self, candidate, x):
        if self.kind == "ilp":
            return self.ilp.predict(candidate, x)
        if self.kind == "cryptography":
            if self.crypto_setting == "legacy":
                spec = json.loads(candidate)
                return "".join(self._apply_cipher(spec, ch) for ch in x)
            from benchmarks.crypto import compile_program
            with _time_limit():
                output = compile_program(candidate)(x)
            if not isinstance(output, str):
                raise ValueError("fn(x) must return a string")
            return output
        if self.kind == "list_functions":
            from benchmarks.list_functions import execute
            with _time_limit(10):
                return execute(candidate, deepcopy(x))
        if self.kind == "scan":
            from benchmarks.list_functions import execute_program
            if getattr(self, "_scan_candidate", None) != candidate:
                self._scan_candidate, self._scan_cache = candidate, {}
            if x not in self._scan_cache:
                try:
                    with _time_limit(10):
                        output = execute_program(candidate, x)
                    if not isinstance(output, str):
                        raise ValueError("transform(x) must return an action sequence string")
                    self._scan_cache[x] = (True, output)
                except Exception as exc:
                    self._scan_cache[x] = (False, str(exc))
            valid, output = self._scan_cache[x]
            if not valid:
                raise RuntimeError(output)
            return output
        raise ValueError("Instruction predictions are judged by the LLM oracle")

    @staticmethod
    def _apply_cipher(spec, ch):
        alpha = "abcdefghijklmnopqrstuvwxyz"
        if ch.lower() not in alpha:
            return ch
        index = alpha.index(ch.lower())
        family = spec["cipher"]
        if family == "caesar":
            out = alpha[(index + int(spec["shift"])) % 26]
        elif family == "atbash":
            out = alpha[25-index]
        elif family == "keyboard":
            out = "qwertyuiopasdfghjklzxcvbnm"[index]
        else:
            raise ValueError("Unknown cipher")
        return out.upper() if ch.isupper() else out

    def reference_output(self, x):
        if self.kind == "ilp":
            return self.ilp.reference_output(x)
        if self.kind == "cryptography":
            spec = {"cipher": self.family}
            if self.shift is not None:
                spec["shift"] = self.shift
            return "".join(self._apply_cipher(spec, ch) for ch in x)
        if self.kind == "list_functions":
            return self.reference(deepcopy(x))
        return dict(self.all_examples)[x]

    def _list_pool(self, size=512):
        pool = [x for x, _ in self.train + self.eval]
        for _ in range(size):
            length = self.rng.randint(3, 8)
            pool.append([self.rng.randint(0, 100) for _ in range(length)])
        return pool

    def validate_candidate(self, candidate, example):
        """Task-specific validity check exposed through the same runner interface."""
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("Expected a nonempty candidate")
        if self.kind != "instruction_induction":
            self.predict(candidate, example)

    def training_errors(self, candidate, pairs):
        """Executable training disagreements, or None for semantic instructions."""
        if self.kind == "instruction_induction":
            return None
        errors = []
        started = time.monotonic()
        self.last_training_details = None
        for index, (x, expected) in enumerate(pairs):
            if self.kind == "scan" and time.monotonic() - started >= 30:
                self.last_training_details = {"examples_checked": index, "training_complete": False,
                                              "stop_reason": "time_budget", "time_budget_seconds": 30}
                return None
            try:
                correct = self.predict(candidate, x) == expected
            except Exception:
                correct = False
            if not correct:
                errors.append((deepcopy(x), deepcopy(expected)))
        return errors

    def initial_examples(self, limit):
        return deepcopy(self.train[:limit])

    def supplemental_examples(self, seen, limit):
        """Fresh labeled inputs when an oracle is inconclusive, not counterexamples."""
        if self.kind != "list_functions":
            return deepcopy([pair for pair in self.train if pair not in seen][:limit])
        additions = []
        for x in self._list_pool(limit):
            try:
                pair = (deepcopy(x), self.reference_output(x))
            except Exception:
                continue
            if pair not in seen and pair not in additions:
                additions.append(pair)
            if len(additions) >= limit:
                break
        return additions

    def verify(self, candidate, limit=8, llm_call=None):
        """Run one oracle check; preserve unknown/errors as distinct from acceptance."""
        self.last_oracle_details = {
            "backend": self.oracle_name,
            "guarantee": ("equivalent_on_bounded_domain_only"
                          if self.kind == "cryptography" and self.crypto_setting == "paper"
                          else "heuristic_not_global_proof"
                          if self.kind in {"list_functions", "instruction_induction"}
                          else "finite_domain_equivalence"),
        }
        self.last_oracle_response = None
        try:
            pairs, equivalent = self.counterexamples(candidate, limit, llm_call=llm_call)
            # Actual failure evidence always takes precedence over acceptance.
            return OracleResult(False if pairs else equivalent, deepcopy(pairs),
                                deepcopy(self.last_oracle_details), self.last_oracle_response)
        except Exception as exc:
            return OracleResult(None, [], deepcopy(self.last_oracle_details),
                                self.last_oracle_response, str(exc))

    def counterexamples(self, candidate, limit=8, llm_call=None):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return [], None
        if candidate is None:
            return self.train[:limit], None
        if self.kind == "ilp":
            pairs, equivalent, details = self.ilp.counterexamples(candidate, limit)
            self.last_oracle_details = details
            return pairs, equivalent
        if self.kind == "instruction_induction":
            if llm_call is None:
                raise ValueError("LLM oracle requires llm_call")
            from benchmarks.instruction_oracle import build_prompt, adjudicate
            prompt = build_prompt(self.benchmark_id, candidate, limit)
            response = llm_call(prompt)
            self.last_oracle_response = response
            pairs, equivalent, details = adjudicate(self.benchmark_id, response, limit)
            details["prompt_version"] = 3
            details["oracle_prompt"] = prompt
            self.last_oracle_details = details
            return pairs, equivalent
        if self.kind == "cryptography" and self.crypto_setting == "paper":
            from benchmarks.crypto_smt import search
            reference = {"cipher": self.family, "shift": self.shift}

            def replay(word):
                try:
                    return self.predict(candidate, word) != self.reference_output(word)
                except Exception:
                    return True

            words, equivalent, details = search(candidate, reference, limit=limit, replay=replay)
            self.last_oracle_details = details
            return [(word, self.reference_output(word)) for word in words], equivalent
        if self.kind == "cryptography":
            reference = {"cipher": self.family}
            if self.shift is not None:
                reference["shift"] = self.shift
            letters, equivalent = _crypto_smt(reference, json.loads(candidate), limit)
            return [(ch, self.reference_output(ch)) for ch in letters], equivalent
        elif self.kind == "list_functions":
            from benchmarks.list_functions import program_function
            program_function(candidate)
            symbolic_pairs = []
            symbolic_status = "no_witness"
            symbolic_error = None
            try:
                witness = _list_sequence_smt(self.reference_source, candidate)
                if witness is not None:
                    symbolic_pairs = [(witness, self.reference_output(witness))]
                    symbolic_status = "witness"
            except ValueError:
                # Unsupported translation uses the concrete-search oracle.
                symbolic_status = "unsupported"
            except RuntimeError as exc:
                symbolic_status = "unknown_or_error"
                symbolic_error = str(exc)
            pool = []
            for x in self._list_pool():
                try:
                    pool.append((x, self.reference_output(x)))
                except Exception:
                    continue
        else:
            return self._scan_counterexamples(candidate, limit)
        disagreements = []
        for i, (x, y) in enumerate(pool):
            try:
                if self.predict(candidate, x) != y:
                    disagreements.append(i)
            except Exception:
                disagreements.append(i)
        indices = _smt_choice(disagreements, limit)
        # Operational, noisy equivalence: accept after a completed search
        # finds no witness, not a proof about arbitrary Python inputs.
        equivalent = (True if pool and not disagreements and symbolic_error is None
                      else None)
        self.last_oracle_details = {
            "policy": "bounded_search_no_counterexample_v1",
            "guarantee": "heuristic_not_global_proof",
            "symbolic_status": symbolic_status,
            "symbolic_domain": {"length_min": 3, "length_max": 8,
                                "element_min": 0, "element_max": 100},
            "concrete_examples_checked": len(pool),
            "concrete_disagreements": len(disagreements),
            "symbolic_error": symbolic_error,
        }
        pairs = symbolic_pairs if self.kind == "list_functions" else []
        for index in indices:
            if pool[index] not in pairs:
                pairs.append(pool[index])
            if len(pairs) >= limit:
                break
        pairs = pairs[:limit]
        return pairs, False if pairs else equivalent

    def _scan_counterexamples(self, candidate, limit):
        started = time.monotonic()
        pairs, checked = [], 0
        print(f"[SCAN oracle] checking {len(self.all_examples)} commands", flush=True)
        for x, expected in self.all_examples:
            if time.monotonic() - started >= 30:
                break
            try:
                wrong = self.predict(candidate, x) != expected
            except Exception:
                wrong = True
            checked += 1
            if wrong and (x, expected) not in pairs:
                pairs.append((x, expected))
            if len(pairs) >= limit:
                break
        complete = checked == len(self.all_examples)
        verdict = False if pairs else True if complete else None
        self.last_oracle_details = {
            "backend": "exhaustive_enumeration", "policy": "scan_all_public_commands_v2",
            "guarantee": "finite_domain_equivalence", "domain": "all_public_commands_including_test",
            "domain_size": len(self.all_examples), "examples_checked": checked,
            "counterexamples_found": len(pairs), "search_complete": complete,
            "stop_reason": "counterexample_limit" if len(pairs) >= limit else "complete" if complete else "time_budget",
            "time_budget_seconds": 30, "elapsed_seconds": time.monotonic() - started,
        }
        print(f"[SCAN oracle] checked={checked}, CE={len(pairs)}, equivalent={verdict}", flush=True)
        return pairs, verdict

    def score(self, candidate, limit=32):
        self.last_evaluation_details = None
        if self.kind == "instruction_induction":
            return None
        examples = self.eval if self.kind in {"scan", "ilp"} else self.eval[:limit]
        if not examples:
            return None
        correct, checked = 0, 0
        started = time.monotonic()
        for x, y in examples:
            if self.kind == "scan" and time.monotonic() - started >= 30:
                self.last_evaluation_details = {"test_examples": len(examples), "examples_checked": checked,
                                                "correct": correct, "evaluation_complete": False,
                                                "stop_reason": "time_budget", "time_budget_seconds": 30}
                print(f"[SCAN score] incomplete: {checked}/{len(examples)}; score=null", flush=True)
                return None
            try:
                correct += self.predict(candidate, x) == y
            except Exception:
                pass
            checked += 1
        self.last_evaluation_details = {"test_examples": len(examples), "correct": correct,
                                        "sample_accuracy": correct / len(examples),
                                        "all_test_correct": correct == len(examples)}
        if self.kind == "ilp":
            mse = (len(examples)-correct) / len(examples)
            self.last_evaluation_details.update(mse=mse, paper_success_threshold=1e-4,
                                                paper_symbolic_success=mse < 1e-4)
        return correct / len(examples)
