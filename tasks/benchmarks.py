"""Dataset adapters and counterexample oracles for program induction tasks."""

import ast
import json
import random
import re
import signal
from contextlib import contextmanager
from pathlib import Path

from prompting.benchmarks import INSTRUCTION_ORACLE

ROOT = Path(__file__).resolve().parents[1] / "datasets"


def _pairs(items):
    return [(item["input"], item["output"]) for item in items]


def _instruction_pairs(items, name):
    if name == "cause_and_effect":
        return [(item["cause"], item["effect"]) for item in items]
    if name == "common_concept":
        return [(item["items"], item["concept"]) for item in items]
    return _pairs(items)


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
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
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
    def __init__(self, kind, benchmark_id, seed=43, eval_size=32):
        self.kind, self.benchmark_id = kind, benchmark_id
        self.rng = random.Random(seed)
        self.oracle_name = {"cryptography": "cvc5_QF_LIA", "list_functions": "cvc5_sequence_or_bounded_Python",
                            "instruction_induction": "LLM", "scan": "exhaustive_enumeration"}[kind]
        if kind == "cryptography":
            family, row = benchmark_id.split(":") if ":" in benchmark_id else (benchmark_id, "0")
            if family not in {"caesar", "atbash", "keyboard"}:
                raise ValueError("Unknown cipher family")
            item = _read_jsonl(ROOT / "cryptography/raw" / f"{family}.jsonl")[int(row)]
            self.family = family
            self.train = _pairs(item["train"]["normal"])
            self.eval = _pairs(item["test"])[:eval_size]
            self.shift = (ord(self.train[0][1][0].lower()) - ord(self.train[0][0][0].lower())) % 26 if family == "caesar" else None
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
        elif kind == "instruction_induction":
            base = ROOT / "instruction_induction/raw"
            induce = json.loads((base / "induce" / f"{benchmark_id}.json").read_text())
            execute = json.loads((base / "execute" / f"{benchmark_id}.json").read_text())
            self.train = _instruction_pairs(induce["examples"].values(), benchmark_id)
            self.eval = _instruction_pairs(execute["examples"].values(), benchmark_id)[:eval_size]
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
        else:
            raise ValueError(f"Unknown benchmark: {kind}")

    def format_pair(self, x, y):
        return f"{json.dumps(x, ensure_ascii=False)} => {json.dumps(y, ensure_ascii=False)}"

    def predict(self, candidate, x):
        if self.kind == "cryptography":
            spec = json.loads(candidate)
            return "".join(self._apply_cipher(spec, ch) for ch in x)
        if self.kind in {"list_functions", "scan"}:
            with _time_limit():
                return _validate_function(candidate)(x)
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
        if self.kind == "cryptography":
            spec = {"cipher": self.family}
            if self.shift is not None:
                spec["shift"] = self.shift
            return self.predict(json.dumps(spec), x)
        if self.kind == "list_functions":
            return self.reference(x)
        return dict(self.all_examples)[x]

    def _list_pool(self, size=512):
        pool = [x for x, _ in self.train + self.eval]
        for _ in range(size):
            length = self.rng.randint(3, 8)
            pool.append([self.rng.randint(0, 100) for _ in range(length)])
        return pool

    def counterexamples(self, candidate, limit=8, llm_call=None):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return [], None
        if candidate is None:
            return self.train[:limit], None
        if self.kind == "instruction_induction":
            if llm_call is None:
                raise ValueError("LLM oracle requires llm_call")
            examples = "\n".join(self.format_pair(x, y) for x, y in self.train[:20])
            response = llm_call(INSTRUCTION_ORACLE.replace("{examples}", examples).replace("{candidate}", candidate).replace("{count}", str(limit)))
            self.last_oracle_response = response
            try:
                verdict = json.loads(re.search(r"\{.*\}", response, re.S).group())
            except (ValueError, AttributeError, TypeError):
                return [], None
            proposals = verdict.get("counterexamples")
            if proposals is None and verdict.get("input") is not None:
                proposals = [verdict]
            pairs = []
            known = {json.dumps(x, sort_keys=True): y for x, y in self.train + self.eval}
            for proposal in proposals or []:
                if not isinstance(proposal, dict) or proposal.get("input") is None or proposal.get("expected_output") is None:
                    continue
                pair = (proposal["input"], proposal["expected_output"])
                key = json.dumps(pair[0], sort_keys=True)
                if key in known and known[key] != pair[1]:
                    continue
                if pair not in pairs:
                    pairs.append(pair)
                if len(pairs) >= limit:
                    break
            if pairs:
                return pairs, False
            return [], True if verdict.get("equivalent") is True else None
        if self.kind == "cryptography":
            reference = {"cipher": self.family}
            if self.shift is not None:
                reference["shift"] = self.shift
            letters, equivalent = _crypto_smt(reference, json.loads(candidate), limit)
            return [(ch, self.reference_output(ch)) for ch in letters], equivalent
        elif self.kind == "list_functions":
            _validate_function(candidate)
            symbolic_pairs = []
            try:
                witness = _list_sequence_smt(self.reference_source, candidate)
                if witness is not None:
                    symbolic_pairs = [(witness, self.reference_output(witness))]
            except (ValueError, RuntimeError):
                pass
            pool = []
            for x in self._list_pool():
                try:
                    pool.append((x, self.reference_output(x)))
                except Exception:
                    continue
        else:
            pool = self.all_examples
        disagreements = []
        for i, (x, y) in enumerate(pool):
            try:
                if self.predict(candidate, x) != y:
                    disagreements.append(i)
            except Exception:
                disagreements.append(i)
        if self.kind == "scan":
            indices = disagreements[:limit]
            equivalent = not disagreements
        else:
            indices = _smt_choice(disagreements, limit)
            equivalent = None
        pairs = symbolic_pairs if self.kind == "list_functions" else []
        for index in indices:
            if pool[index] not in pairs:
                pairs.append(pool[index])
            if len(pairs) >= limit:
                break
        pairs = pairs[:limit]
        return pairs, False if pairs else equivalent

    def score(self, candidate, limit=32):
        if self.kind == "instruction_induction":
            return None
        examples = self.eval[:limit]
        if not examples:
            return None
        correct = 0
        for x, y in examples:
            try:
                correct += self.predict(candidate, x) == y
            except Exception:
                pass
        return correct / len(examples)
