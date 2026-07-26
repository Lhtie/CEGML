"""Propositional-logic task utilities.

Supported syntax:

    !p                  negation
    p & q               conjunction
    p | q               disjunction
    p -> q              implication
    p <-> q             biconditional

Parentheses may be used freely. Operator precedence, from high to low, is
``!``, ``&``, ``|``, ``->``, ``<->``.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


_TOKEN_PATTERN = re.compile(
    r"\s*(<->|->|[()!&|]|True|False|true|false|[A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass(frozen=True)
class FormulaNode:
    operator: str
    left: "FormulaNode | None" = None
    right: "FormulaNode | None" = None
    name: str | None = None
    constant: bool | None = None

    def evaluate(self, assignment: Mapping[str, bool]) -> bool:
        if self.operator == "variable":
            assert self.name is not None
            if self.name not in assignment:
                raise ValueError(f"Assignment is missing variable '{self.name}'")
            return bool(assignment[self.name])
        if self.operator == "constant":
            assert self.constant is not None
            return self.constant
        if self.operator == "!":
            assert self.left is not None
            return not self.left.evaluate(assignment)

        assert self.left is not None and self.right is not None
        left = self.left.evaluate(assignment)
        right = self.right.evaluate(assignment)
        if self.operator == "&":
            return left and right
        if self.operator == "|":
            return left or right
        if self.operator == "->":
            return (not left) or right
        if self.operator == "<->":
            return left == right
        raise ValueError(f"Unknown operator: {self.operator}")

    @property
    def variables(self) -> set[str]:
        if self.operator == "variable":
            assert self.name is not None
            return {self.name}
        if self.operator == "constant":
            return set()
        assert self.left is not None
        variables = set(self.left.variables)
        if self.right is not None:
            variables.update(self.right.variables)
        return variables


class FormulaParser:
    """Recursive-descent parser for the supported propositional syntax."""

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
                if formula[position:].strip() == "":
                    break
                raise ValueError(
                    f"Invalid token at position {position}: {formula[position:position + 12]!r}"
                )
            tokens.append(match.group(1))
            position = match.end()
        if not tokens:
            raise ValueError("Formula cannot be empty")
        return tokens

    def parse(self) -> FormulaNode:
        node = self._parse_biconditional()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token: {self._peek()!r}")
        return node

    def _peek(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def _consume(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of formula")
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.position += 1
        return token

    def _parse_biconditional(self) -> FormulaNode:
        node = self._parse_implication()
        while self._peek() == "<->":
            operator = self._consume()
            node = FormulaNode(operator, node, self._parse_implication())
        return node

    def _parse_implication(self) -> FormulaNode:
        # Implication is right-associative: p -> q -> r == p -> (q -> r).
        node = self._parse_disjunction()
        if self._peek() == "->":
            operator = self._consume()
            return FormulaNode(operator, node, self._parse_implication())
        return node

    def _parse_disjunction(self) -> FormulaNode:
        node = self._parse_conjunction()
        while self._peek() == "|":
            operator = self._consume()
            node = FormulaNode(operator, node, self._parse_conjunction())
        return node

    def _parse_conjunction(self) -> FormulaNode:
        node = self._parse_unary()
        while self._peek() == "&":
            operator = self._consume()
            node = FormulaNode(operator, node, self._parse_unary())
        return node

    def _parse_unary(self) -> FormulaNode:
        if self._peek() == "!":
            self._consume("!")
            return FormulaNode("!", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> FormulaNode:
        token = self._peek()
        if token == "(":
            self._consume("(")
            node = self._parse_biconditional()
            self._consume(")")
            return node
        if token is None:
            raise ValueError("Expected a variable or parenthesized formula")

        token = self._consume()
        if token.lower() == "true":
            return FormulaNode("constant", constant=True)
        if token.lower() == "false":
            return FormulaNode("constant", constant=False)
        if token in {"!", "&", "|", "->", "<->", ")"}:
            raise ValueError(f"Expected a variable, got {token!r}")
        return FormulaNode("variable", name=token)


class PropositionalLogic:
    """A truth-table task backed by a ground-truth propositional formula."""

    def __init__(
        self,
        formula: str,
        variables: Sequence[str] | None = None,
        seed: int | None = None,
    ):
        self.formula = formula
        self.formula_tree = FormulaParser(formula).parse()
        formula_variables = self.formula_tree.variables

        if variables is None:
            variables = sorted(formula_variables)
        if len(set(variables)) != len(variables):
            raise ValueError("variables must not contain duplicates")
        if not formula_variables.issubset(set(variables)):
            missing = sorted(formula_variables - set(variables))
            raise ValueError(f"variables is missing formula variables: {missing}")

        self.variables = tuple(variables)
        self._rng = random.Random(seed)

    @staticmethod
    def parse(formula: str) -> FormulaNode:
        return FormulaParser(formula).parse()

    def evaluate(
        self,
        assignment: Mapping[str, bool],
        formula: str | FormulaNode | None = None,
    ) -> bool:
        if formula is None:
            tree = self.formula_tree
        elif isinstance(formula, str):
            tree = self.parse(formula)
        else:
            tree = formula
        unknown = tree.variables - set(self.variables)
        if unknown:
            raise ValueError(
                f"Formula contains variables outside the task vocabulary: {sorted(unknown)}"
            )
        return tree.evaluate(assignment)

    def all_assignments(self) -> list[dict[str, bool]]:
        return [
            dict(zip(self.variables, values))
            for values in itertools.product((False, True), repeat=len(self.variables))
        ]

    def generate_data(self) -> tuple[list[dict[str, bool]], list[int]]:
        """Return the complete truth table and its ground-truth labels."""
        assignments = self.all_assignments()
        labels = [int(self.evaluate(assignment)) for assignment in assignments]
        return assignments, labels

    def generate_random_data(
        self,
        n: int,
        balanced: bool = False,
    ) -> tuple[list[dict[str, bool]], list[int]]:
        """Sample assignments, with replacement when ``n`` exceeds table size."""
        if n < 0:
            raise ValueError("n must be non-negative")
        assignments, labels = self.generate_data()
        if n == 0:
            return [], []

        if balanced:
            buckets = {
                0: [x for x, y in zip(assignments, labels) if y == 0],
                1: [x for x, y in zip(assignments, labels) if y == 1],
            }
            if not buckets[0] or not buckets[1]:
                raise ValueError(
                    "Balanced sampling requires both true and false assignments"
                )
            sampled_labels = [index % 2 for index in range(n)]
            self._rng.shuffle(sampled_labels)
            sampled = [dict(self._rng.choice(buckets[label])) for label in sampled_labels]
            return sampled, sampled_labels

        indices = list(range(len(assignments)))
        if n <= len(indices):
            chosen = self._rng.sample(indices, n)
        else:
            chosen = [self._rng.choice(indices) for _ in range(n)]
        return [dict(assignments[index]) for index in chosen], [
            labels[index] for index in chosen
        ]

    def equivalent_and_witness(
        self, hypothesis: str
    ) -> tuple[bool, dict[str, bool] | None]:
        hypothesis_tree = self.parse(hypothesis)
        self._validate_hypothesis_variables(hypothesis_tree)
        for assignment in self.all_assignments():
            if self.evaluate(assignment) != hypothesis_tree.evaluate(assignment):
                return False, assignment
        return True, None

    def generate_counterexamples(
        self,
        hypothesis: str,
        k: int | None = None,
        shuffle: bool = False,
    ) -> tuple[list[dict[str, bool]], list[int]]:
        """Return assignments where ``hypothesis`` disagrees with ground truth."""
        if k is not None and k < 0:
            raise ValueError("k must be non-negative or None")
        hypothesis_tree = self.parse(hypothesis)
        self._validate_hypothesis_variables(hypothesis_tree)

        counterexamples = [
            assignment
            for assignment in self.all_assignments()
            if self.evaluate(assignment) != hypothesis_tree.evaluate(assignment)
        ]
        if shuffle:
            self._rng.shuffle(counterexamples)
        if k is not None:
            counterexamples = counterexamples[:k]
        labels = [int(self.evaluate(assignment)) for assignment in counterexamples]
        return counterexamples, labels

    def counterexample_records(
        self,
        hypothesis: str,
        k: int | None = None,
        shuffle: bool = False,
    ) -> list[dict]:
        """Return counterexamples with both expected and hypothesis labels."""
        hypothesis_tree = self.parse(hypothesis)
        assignments, expected_labels = self.generate_counterexamples(
            hypothesis, k=k, shuffle=shuffle
        )
        return [
            {
                "assignment": assignment,
                "expected": expected,
                "hypothesis": int(hypothesis_tree.evaluate(assignment)),
            }
            for assignment, expected in zip(assignments, expected_labels)
        ]

    def _validate_hypothesis_variables(self, hypothesis: FormulaNode) -> None:
        unknown = hypothesis.variables - set(self.variables)
        if unknown:
            raise ValueError(
                "Hypothesis contains variables outside the task vocabulary: "
                f"{sorted(unknown)}"
            )

