#!/usr/bin/env python3
"""Randomly generate propositional-logic formulas from a small grammar.

Grammar (parentheses are emitted explicitly):

    formula ::= variable
              | !formula
              | (formula & formula)
              | (formula | formula)
              | (formula -> formula)
              | (formula <-> formula)

The generator groups formulas by exact syntactic depth, which makes it easy to
construct scale-up experiments similar to the regex datasets.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BINARY_OPERATORS = ("&", "|", "->", "<->")


@dataclass(frozen=True)
class Formula:
    operator: str | None = None
    left: "Formula | None" = None
    right: "Formula | None" = None
    variable: str | None = None

    @classmethod
    def atom(cls, variable: str) -> "Formula":
        return cls(variable=variable)

    @classmethod
    def unary(cls, child: "Formula") -> "Formula":
        return cls(operator="!", left=child)

    @classmethod
    def binary(
        cls, operator: str, left: "Formula", right: "Formula"
    ) -> "Formula":
        return cls(operator=operator, left=left, right=right)

    @property
    def depth(self) -> int:
        if self.variable is not None:
            return 0
        if self.operator == "!":
            assert self.left is not None
            return 1 + self.left.depth
        assert self.left is not None and self.right is not None
        return 1 + max(self.left.depth, self.right.depth)

    @property
    def size(self) -> int:
        if self.variable is not None:
            return 1
        if self.operator == "!":
            assert self.left is not None
            return 1 + self.left.size
        assert self.left is not None and self.right is not None
        return 1 + self.left.size + self.right.size

    @property
    def variables(self) -> set[str]:
        if self.variable is not None:
            return {self.variable}
        assert self.left is not None
        variables = set(self.left.variables)
        if self.right is not None:
            variables.update(self.right.variables)
        return variables

    def render(self) -> str:
        if self.variable is not None:
            return self.variable
        if self.operator == "!":
            assert self.left is not None
            return f"!{self.left.render()}"
        assert self.left is not None and self.right is not None
        return f"({self.left.render()} {self.operator} {self.right.render()})"


def generate_formula(
    rng: random.Random,
    variables: Sequence[str],
    exact_depth: int,
    unary_probability: float,
    binary_operators: Sequence[str] = BINARY_OPERATORS,
) -> Formula:
    """Generate a formula whose syntactic depth is exactly ``exact_depth``."""
    if exact_depth < 0:
        raise ValueError("exact_depth must be non-negative")
    if not variables:
        raise ValueError("At least one variable is required")
    if not 0.0 <= unary_probability <= 1.0:
        raise ValueError("unary_probability must be between 0 and 1")

    if exact_depth == 0:
        return Formula.atom(rng.choice(variables))

    if rng.random() < unary_probability:
        return Formula.unary(
            generate_formula(
                rng,
                variables,
                exact_depth - 1,
                unary_probability,
                binary_operators,
            )
        )

    # At least one child must have depth exact_depth - 1. The other child is
    # allowed to be shallower, producing more varied, unbalanced formulas.
    deep_child = generate_formula(
        rng,
        variables,
        exact_depth - 1,
        unary_probability,
        binary_operators,
    )
    other_child = generate_formula(
        rng,
        variables,
        rng.randint(0, exact_depth - 1),
        unary_probability,
        binary_operators,
    )
    if rng.random() < 0.5:
        left, right = deep_child, other_child
    else:
        left, right = other_child, deep_child
    return Formula.binary(rng.choice(binary_operators), left, right)


def generate_dataset(
    *,
    variables: Sequence[str],
    min_depth: int,
    max_depth: int,
    formulas_per_depth: int,
    unary_probability: float,
    seed: int,
    require_all_variables: bool = True,
    max_attempt_factor: int = 100,
) -> dict:
    if min_depth < 0:
        raise ValueError("min_depth must be non-negative")
    if max_depth < min_depth:
        raise ValueError("max_depth must be >= min_depth")
    if formulas_per_depth <= 0:
        raise ValueError("formulas_per_depth must be positive")

    rng = random.Random(seed)
    seen: set[str] = set()
    depth_groups = []

    for depth in range(min_depth, max_depth + 1):
        records = []
        attempts = 0
        max_attempts = max_attempt_factor * formulas_per_depth
        while len(records) < formulas_per_depth and attempts < max_attempts:
            attempts += 1
            formula = generate_formula(
                rng,
                variables,
                exact_depth=depth,
                unary_probability=unary_probability,
            )
            if require_all_variables and formula.variables != set(variables):
                continue
            rendered = formula.render()
            if rendered in seen:
                continue
            seen.add(rendered)
            records.append(
                {
                    "formula": rendered,
                    "depth": formula.depth,
                    "size": formula.size,
                    "variables": sorted(formula.variables),
                }
            )

        if len(records) < formulas_per_depth:
            raise RuntimeError(
                f"Could generate only {len(records)} unique formulas at depth "
                f"{depth}; requested {formulas_per_depth}. Increase the number "
                "of variables or reduce formulas_per_depth."
            )
        depth_groups.append({"depth": depth, "formulas": records})

    return {
        "grammar": {
            "formula": [
                "variable",
                "!formula",
                "(formula & formula)",
                "(formula | formula)",
                "(formula -> formula)",
                "(formula <-> formula)",
            ],
            "variables": list(variables),
        },
        "seed": seed,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "formulas_per_depth": formulas_per_depth,
        "require_all_variables": require_all_variables,
        "num_formulas": sum(len(group["formulas"]) for group in depth_groups),
        "formula_groups": depth_groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random propositional-logic formulas by grammar depth."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["p", "q", "r", "s", "t"],
        help="Atomic proposition names.",
    )
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--formulas-per-depth", type=int, default=25)
    parser.add_argument(
        "--unary-probability",
        type=float,
        default=0.25,
        help="Probability of choosing negation at a non-atomic grammar node.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-variable-subsets",
        action="store_true",
        help="Allow formulas that do not contain every configured variable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("formulas.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(
        variables=args.variables,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        formulas_per_depth=args.formulas_per_depth,
        unary_probability=args.unary_probability,
        seed=args.seed,
        require_all_variables=not args.allow_variable_subsets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Wrote {dataset['num_formulas']} formulas to {args.output}")


if __name__ == "__main__":
    main()
