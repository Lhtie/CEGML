#!/usr/bin/env python3
"""Generate random LTL formulas grouped by variable count and syntax depth."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


UNARY_OPERATORS = ("!", "X", "F", "G")
BINARY_OPERATORS = ("&", "|", "->", "<->", "U", "R")


@dataclass(frozen=True)
class Formula:
    operator: str | None = None
    left: "Formula | None" = None
    right: "Formula | None" = None
    variable: str | None = None

    @property
    def variables(self) -> set[str]:
        if self.variable is not None:
            return {self.variable}
        assert self.left is not None
        result = set(self.left.variables)
        if self.right is not None:
            result.update(self.right.variables)
        return result

    @property
    def depth(self) -> int:
        if self.variable is not None:
            return 0
        assert self.left is not None
        return 1 + max(
            self.left.depth, self.right.depth if self.right is not None else 0
        )

    @property
    def size(self) -> int:
        if self.variable is not None:
            return 1
        assert self.left is not None
        return 1 + self.left.size + (
            self.right.size if self.right is not None else 0
        )

    def render(self) -> str:
        if self.variable is not None:
            return self.variable
        assert self.left is not None and self.operator is not None
        if self.right is None:
            return f"{self.operator}({self.left.render()})"
        return (
            f"({self.left.render()} {self.operator} {self.right.render()})"
        )


def generate_formula(
    rng: random.Random,
    variables: Sequence[str],
    exact_depth: int,
    unary_probability: float,
) -> Formula:
    if exact_depth == 0:
        return Formula(variable=rng.choice(variables))
    if rng.random() < unary_probability:
        return Formula(
            operator=rng.choice(UNARY_OPERATORS),
            left=generate_formula(
                rng, variables, exact_depth - 1, unary_probability
            ),
        )
    deep = generate_formula(
        rng, variables, exact_depth - 1, unary_probability
    )
    other = generate_formula(
        rng, variables, rng.randint(0, exact_depth - 1), unary_probability
    )
    left, right = (deep, other) if rng.random() < 0.5 else (other, deep)
    return Formula(rng.choice(BINARY_OPERATORS), left, right)


def generate_dataset(
    variables: Sequence[str],
    min_variables: int,
    max_variables: int,
    min_depth: int,
    max_depth: int,
    formulas_per_depth: int,
    seed: int,
    unary_probability: float = 0.3,
    require_all_variables: bool = True,
) -> dict:
    rng = random.Random(seed)
    seen = set()
    variable_groups = []
    for num_variables in range(min_variables, max_variables + 1):
        group_variables = list(variables[:num_variables])
        depth_groups = []
        for depth in range(min_depth, max_depth + 1):
            records = []
            attempts = 0
            while len(records) < formulas_per_depth:
                attempts += 1
                if attempts > formulas_per_depth * 1000:
                    raise RuntimeError(
                        "Could not generate enough formulas with "
                        f"{num_variables} variables at depth {depth}"
                    )
                formula = generate_formula(
                    rng, group_variables, depth, unary_probability
                )
                if (
                    require_all_variables
                    and formula.variables != set(group_variables)
                ):
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
                        "num_variables": len(formula.variables),
                        "variables": sorted(formula.variables),
                    }
                )
            depth_groups.append({"depth": depth, "formulas": records})
        variable_groups.append(
            {
                "num_variables": num_variables,
                "variables": group_variables,
                "depth_groups": depth_groups,
            }
        )
    return {
        "grammar": {
            "unary": list(UNARY_OPERATORS),
            "binary": list(BINARY_OPERATORS),
            "variables": list(variables[:max_variables]),
        },
        "semantics": "infinite LTL over ultimately-periodic lasso traces",
        "seed": seed,
        "require_all_variables": require_all_variables,
        "num_formulas": sum(
            len(depth_group["formulas"])
            for variable_group in variable_groups
            for depth_group in variable_group["depth_groups"]
        ),
        "variable_groups": variable_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variables", nargs="+", default=["p", "q", "r", "s"]
    )
    parser.add_argument("--min-variables", type=int, default=1)
    parser.add_argument("--max-variables", type=int, default=4)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--formulas-per-depth", type=int, default=25)
    parser.add_argument("--unary-probability", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-variable-subsets", action="store_true"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("formula_list.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.min_variables <= args.max_variables:
        parser.error("require 1 <= --min-variables <= --max-variables")
    if args.max_variables > len(args.variables):
        parser.error("--max-variables exceeds the supplied variable names")
    data = generate_dataset(
        args.variables,
        args.min_variables,
        args.max_variables,
        args.min_depth,
        args.max_depth,
        args.formulas_per_depth,
        args.seed,
        args.unary_probability,
        not args.allow_variable_subsets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    print(f"Wrote {data['num_formulas']} formulas to {args.output}")


if __name__ == "__main__":
    main()
