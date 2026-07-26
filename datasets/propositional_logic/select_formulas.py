#!/usr/bin/env python3
"""Select formulas as evenly as possible across syntax-depth groups."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def select_formulas(path: Path, count: int, seed: int) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    groups = [group for group in data["formula_groups"] if group["formulas"]]
    if not groups:
        raise ValueError("formula_list contains no non-empty depth groups")
    available = sum(len(group["formulas"]) for group in groups)
    if count > available:
        raise ValueError(f"Requested {count} formulas, but only {available} exist")

    rng = random.Random(seed)
    base, remainder = divmod(count, len(groups))
    selected = []
    unused_by_depth = {}

    for index, group in enumerate(groups):
        target = base + int(index < remainder)
        formulas = list(group["formulas"])
        rng.shuffle(formulas)
        take = min(target, len(formulas))
        selected.extend(formulas[:take])
        unused_by_depth[group["depth"]] = formulas[take:]

    # Fill any deficit caused by a small depth group from the remaining pool.
    deficit = count - len(selected)
    if deficit:
        remaining = [
            formula
            for formulas in unused_by_depth.values()
            for formula in formulas
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[:deficit])

    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("formula_list.json"),
    )
    parser.add_argument("--count", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file recording the selected formula metadata.",
    )
    parser.add_argument(
        "--formulas-only",
        action="store_true",
        help="Print one formula per line for consumption by a shell script.",
    )
    args = parser.parse_args()

    selected = select_formulas(args.input, args.count, args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "source": str(args.input),
                    "seed": args.seed,
                    "count": len(selected),
                    "formulas": selected,
                },
                file,
                indent=2,
            )
            file.write("\n")

    if args.formulas_only:
        for item in selected:
            print(item["formula"])
    elif args.output is not None:
        print(f"Wrote {len(selected)} selected formulas to {args.output}")
    else:
        print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
