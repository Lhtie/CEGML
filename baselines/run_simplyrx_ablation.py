#!/usr/bin/env python3
"""Run state-merging baselines on the SimplyRx ablation regex set."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.run_state_merging import dataset_path_for_regex, run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("datasets/scaleup/run_scripts/simplyrx_ablation.sh"),
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("datasets/scaleup/regex_datasets"),
    )
    parser.add_argument("--methods", default="rpni,blue_fringe")
    parser.add_argument("--max_train_examples", type=int, default=1500)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/summary/state_merging_simplyrx_ablation_train1500.json"),
    )
    return parser.parse_args()


def regexes_from_script(path: Path) -> list[str]:
    regexes: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("python "):
            continue
        tokens = shlex.split(stripped)
        if "--task_type" not in tokens or tokens[tokens.index("--task_type") + 1] != "simplyrx":
            continue
        if "--regex" not in tokens:
            continue
        regex = tokens[tokens.index("--regex") + 1]
        if regex not in seen:
            seen.add(regex)
            regexes.append(regex)
    return regexes


def write_checkpoint(
    path: Path,
    regexes: list[str],
    results: list[dict],
    methods: list[str],
    max_train_examples: int,
    seed: int,
) -> None:
    payload = {
        "task_type": "simplyrx",
        "dataset": "ablation",
        "max_train_examples": max_train_examples,
        "seed": seed,
        "methods": methods,
        "num_datasets": len(regexes),
        "num_completed": len(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    regexes = regexes_from_script(args.script)
    if not regexes:
        raise SystemExit(f"No SimplyRx regexes found in {args.script}")

    results: list[dict] = []
    for index, regex in enumerate(regexes, start=1):
        print(f"[{index}/{len(regexes)}] {regex}", flush=True)
        results.append(
            run_one(
                dataset_path_for_regex(args.dataset_dir, regex),
                regex,
                "simplyrx",
                methods,
                args.max_length,
                args.max_train_examples,
                args.seed,
            )
        )
        write_checkpoint(
            args.out,
            regexes,
            results,
            methods,
            args.max_train_examples,
            args.seed,
        )
        print(f"checkpoint: {args.out}", flush=True)


if __name__ == "__main__":
    main()
