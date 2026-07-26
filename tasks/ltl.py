"""LTL parsing, lasso-trace evaluation, and BLACK counterexamples."""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


_TOKEN_PATTERN = re.compile(
    r"\s*(<->|->|&&|\|\||[()!&|]|[XFGUR]|True|False|"
    r"[A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass(frozen=True)
class LTLNode:
    operator: str
    left: "LTLNode | None" = None
    right: "LTLNode | None" = None
    name: str | None = None
    constant: bool | None = None

    @property
    def variables(self) -> set[str]:
        if self.operator == "variable":
            assert self.name is not None
            return {self.name}
        if self.operator == "constant":
            return set()
        assert self.left is not None
        result = set(self.left.variables)
        if self.right is not None:
            result.update(self.right.variables)
        return result


class LTLParser:
    """Parser for !, X, F, G, U, R, &, |, ->, and <->."""

    def __init__(self, formula: str):
        self.formula = formula
        self.tokens = self._tokenize(formula)
        self.position = 0

    @staticmethod
    def _tokenize(formula: str) -> list[str]:
        tokens = []
        position = 0
        while position < len(formula):
            match = _TOKEN_PATTERN.match(formula, position)
            if match is None:
                if not formula[position:].strip():
                    break
                raise ValueError(
                    f"Invalid token at position {position}: "
                    f"{formula[position:position + 12]!r}"
                )
            token = match.group(1)
            tokens.append({"&&": "&", "||": "|"}.get(token, token))
            position = match.end()
        if not tokens:
            raise ValueError("Formula cannot be empty")
        return tokens

    def parse(self) -> LTLNode:
        node = self._parse_disjunction()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token: {self._peek()!r}")
        return node

    def _peek(self) -> str | None:
        return (
            self.tokens[self.position]
            if self.position < len(self.tokens)
            else None
        )

    def _consume(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of formula")
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.position += 1
        return token

    # BLACK gives temporal connectives higher precedence than implication,
    # implication/biconditional higher precedence than &, and & higher than |.
    def _parse_disjunction(self) -> LTLNode:
        node = self._parse_conjunction()
        while self._peek() == "|":
            node = LTLNode(self._consume(), node, self._parse_conjunction())
        return node

    def _parse_conjunction(self) -> LTLNode:
        node = self._parse_implication()
        while self._peek() == "&":
            node = LTLNode(self._consume(), node, self._parse_implication())
        return node

    def _parse_implication(self) -> LTLNode:
        node = self._parse_temporal_binary()
        if self._peek() in {"->", "<->"}:
            operator = self._consume()
            return LTLNode(operator, node, self._parse_implication())
        return node

    def _parse_temporal_binary(self) -> LTLNode:
        node = self._parse_unary()
        while self._peek() in {"U", "R"}:
            node = LTLNode(self._consume(), node, self._parse_unary())
        return node

    def _parse_unary(self) -> LTLNode:
        if self._peek() in {"!", "X", "F", "G"}:
            operator = self._consume()
            return LTLNode(operator, self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> LTLNode:
        token = self._peek()
        if token == "(":
            self._consume("(")
            node = self._parse_disjunction()
            self._consume(")")
            return node
        if token is None:
            raise ValueError("Expected a proposition or parenthesized formula")
        token = self._consume()
        if token.lower() == "true":
            return LTLNode("constant", constant=True)
        if token.lower() == "false":
            return LTLNode("constant", constant=False)
        if token in {"!", "X", "F", "G", "U", "R", "&", "|", "->", "<->", ")"}:
            raise ValueError(f"Expected a proposition, got {token!r}")
        return LTLNode("variable", name=token)


@dataclass
class LassoTrace:
    states: list[dict[str, bool]]
    loop: int

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("A lasso trace must contain at least one state")
        if not 0 <= self.loop < len(self.states):
            raise ValueError("loop must index a state in the trace")

    def to_dict(self) -> dict:
        return {"states": self.states, "loop": self.loop}

    @classmethod
    def from_dict(cls, value: Mapping) -> "LassoTrace":
        return cls(
            states=[
                {name: bool(truth) for name, truth in state.items()}
                for state in value["states"]
            ],
            loop=int(value["loop"]),
        )


class LTLTask:
    def __init__(
        self,
        formula: str,
        variables: Sequence[str] | None = None,
        seed: int | None = None,
        min_trace_length: int = 1,
        max_trace_length: int = 8,
    ):
        self.formula = formula
        self.formula_tree = LTLParser(formula).parse()
        formula_variables = self.formula_tree.variables
        self.variables = tuple(
            sorted(formula_variables) if variables is None else variables
        )
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variables must not contain duplicates")
        if not formula_variables.issubset(set(self.variables)):
            raise ValueError(
                f"variables is missing: {sorted(formula_variables - set(self.variables))}"
            )
        if min_trace_length < 1 or max_trace_length < min_trace_length:
            raise ValueError("Invalid trace length bounds")
        self.min_trace_length = min_trace_length
        self.max_trace_length = max_trace_length
        self._rng = random.Random(seed)

    @staticmethod
    def parse(formula: str) -> LTLNode:
        return LTLParser(formula).parse()

    def validate_formula(self, formula: str | LTLNode) -> LTLNode:
        tree = self.parse(formula) if isinstance(formula, str) else formula
        unknown = tree.variables - set(self.variables)
        if unknown:
            raise ValueError(
                f"Formula contains propositions outside the vocabulary: {sorted(unknown)}"
            )
        return tree

    def random_trace(self) -> LassoTrace:
        length = self._rng.randint(
            self.min_trace_length, self.max_trace_length
        )
        states = [
            {
                variable: bool(self._rng.getrandbits(1))
                for variable in self.variables
            }
            for _ in range(length)
        ]
        return LassoTrace(states, self._rng.randrange(length))

    def evaluate(
        self,
        trace: LassoTrace | Mapping,
        formula: str | LTLNode | None = None,
    ) -> bool:
        if not isinstance(trace, LassoTrace):
            trace = LassoTrace.from_dict(trace)
        tree = (
            self.formula_tree
            if formula is None
            else self.validate_formula(formula)
        )
        return self._evaluate_tree(trace, tree)[0]

    def _evaluate_tree(self, trace: LassoTrace, root: LTLNode) -> list[bool]:
        size = len(trace.states)
        successor = list(range(1, size)) + [trace.loop]
        cache: dict[LTLNode, list[bool]] = {}

        def values(node: LTLNode) -> list[bool]:
            if node in cache:
                return cache[node]
            if node.operator == "variable":
                assert node.name is not None
                result = [
                    bool(state.get(node.name, False)) for state in trace.states
                ]
            elif node.operator == "constant":
                result = [bool(node.constant)] * size
            elif node.operator == "!":
                assert node.left is not None
                result = [not value for value in values(node.left)]
            elif node.operator == "X":
                assert node.left is not None
                child = values(node.left)
                result = [child[successor[index]] for index in range(size)]
            elif node.operator in {"F", "G"}:
                assert node.left is not None
                child = values(node.left)
                least = node.operator == "F"
                result = [False if least else True] * size
                while True:
                    updated = [
                        (
                            child[index] or result[successor[index]]
                            if least
                            else child[index] and result[successor[index]]
                        )
                        for index in range(size)
                    ]
                    if updated == result:
                        break
                    result = updated
            else:
                assert node.left is not None and node.right is not None
                left, right = values(node.left), values(node.right)
                if node.operator == "&":
                    result = [a and b for a, b in zip(left, right)]
                elif node.operator == "|":
                    result = [a or b for a, b in zip(left, right)]
                elif node.operator == "->":
                    result = [(not a) or b for a, b in zip(left, right)]
                elif node.operator == "<->":
                    result = [a == b for a, b in zip(left, right)]
                elif node.operator in {"U", "R"}:
                    least = node.operator == "U"
                    result = [False if least else True] * size
                    while True:
                        updated = [
                            (
                                right[index]
                                or (
                                    left[index]
                                    and result[successor[index]]
                                )
                                if least
                                else right[index]
                                and (
                                    left[index]
                                    or result[successor[index]]
                                )
                            )
                            for index in range(size)
                        ]
                        if updated == result:
                            break
                        result = updated
                else:
                    raise ValueError(f"Unknown operator: {node.operator}")
            cache[node] = result
            return result

        return values(root)

    def generate_random_data(
        self, count: int, balanced: bool = False, max_attempts: int = 10000
    ) -> tuple[list[dict], list[int]]:
        if count < 0:
            raise ValueError("count must be non-negative")
        traces: list[dict] = []
        labels: list[int] = []
        targets = [index % 2 for index in range(count)] if balanced else None
        if targets is not None:
            self._rng.shuffle(targets)
        attempts = 0
        while len(traces) < count and attempts < max_attempts:
            attempts += 1
            trace = self.random_trace()
            label = int(self.evaluate(trace))
            if targets is not None and label != targets[len(traces)]:
                continue
            traces.append(trace.to_dict())
            labels.append(label)
        if len(traces) < count:
            raise RuntimeError(
                f"Could generate only {len(traces)} requested traces; "
                "the formula may be a tautology or contradiction"
            )
        return traces, labels


class BlackSolver:
    """Subprocess adapter for BLACK's JSON model output."""

    def __init__(
        self,
        binary: str = "black",
        timeout_seconds: float = 30,
        bound: int | None = None,
    ):
        self.binary = binary
        self.timeout_seconds = timeout_seconds
        self.bound = bound

    def ensure_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"BLACK executable '{self.binary}' was not found. "
                "Install BLACK and/or pass --black_binary."
            )

    def solve(self, formula: str, model: bool = True) -> dict:
        self.ensure_available()
        command = [self.binary, "solve", "-o", "json"]
        if model:
            command.append("-m")
        if self.bound is not None:
            command.extend(["-k", str(self.bound)])
        command.extend(["-f", formula])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"BLACK failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Could not parse BLACK JSON output: {completed.stdout!r}"
            ) from error
        return result

    @staticmethod
    def trace_from_model(result: Mapping, variables: Sequence[str]) -> LassoTrace:
        model = result.get("model")
        if not model:
            raise ValueError("BLACK result does not contain a model")
        states = []
        for raw_state in model["states"]:
            states.append(
                {
                    variable: str(raw_state.get(variable, "false")).lower()
                    == "true"
                    for variable in variables
                }
            )
        return LassoTrace(states=states, loop=int(model["loop"]))

    @staticmethod
    def block_prefix(trace: LassoTrace, variables: Sequence[str]) -> str:
        state_terms = []
        for state in trace.states:
            literals = [
                variable if state[variable] else f"!{variable}"
                for variable in variables
            ]
            state_terms.append("(" + " & ".join(literals) + ")")
        prefix = state_terms[-1]
        for term in reversed(state_terms[:-1]):
            prefix = f"({term} & X ({prefix}))"
        return f"!({prefix})"

    def counterexamples(
        self,
        ground_truth: str,
        hypothesis: str,
        variables: Sequence[str],
        count: int = 1,
    ) -> tuple[list[dict], list[int]]:
        if count < 0:
            raise ValueError("count must be non-negative")
        examples: list[dict] = []
        labels: list[int] = []
        directions = [
            (f"(({ground_truth}) & !({hypothesis}))", 1),
            (f"(!({ground_truth}) & ({hypothesis}))", 0),
        ]
        blocks: list[str] = []
        while len(examples) < count:
            found = False
            for difference, label in directions:
                query = difference
                if blocks:
                    query += " & " + " & ".join(f"({block})" for block in blocks)
                result = self.solve(query, model=True)
                if str(result.get("result", "")).upper() != "SAT":
                    continue
                trace = self.trace_from_model(result, variables)
                examples.append(trace.to_dict())
                labels.append(label)
                blocks.append(self.block_prefix(trace, variables))
                found = True
                if len(examples) >= count:
                    break
            if not found:
                break
        return examples, labels

    def equivalent(
        self, ground_truth: str, hypothesis: str, variables: Sequence[str]
    ) -> tuple[bool, dict | None]:
        examples, _ = self.counterexamples(
            ground_truth, hypothesis, variables, count=1
        )
        return (not examples, None if not examples else examples[0])

