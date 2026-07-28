#!/usr/bin/env python3
"""Run state-merging baselines on scaleup regex datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.state_merging import LEARNERS, accuracy, equivalent_to_target
from tasks.rl import ExtRegularLanguage, SimplyRegularLanguage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", choices=["simplyrx", "extrx"], default="simplyrx")
    parser.add_argument("--regex", type=str, default=None, help="Target regex to run.")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Explicit dataset JSON path. Overrides --regex path construction.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("datasets/scaleup/regex_datasets"),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="rpni,blue_fringe",
        help="Comma-separated methods: rpni,blue_fringe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many datasets when --regex/--dataset_path is omitted.",
    )
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional stratified cap on training examples for faster baseline runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def dataset_path_for_regex(dataset_dir: Path, regex: str) -> Path:
    return dataset_dir / f"regex={regex}_trainMaxLen=32_evalMaxLen=32.json"


def regex_from_dataset_path(path: Path) -> str:
    name = path.name
    prefix = "regex="
    suffix = "_trainMaxLen=32_evalMaxLen=32.json"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Cannot parse regex from dataset filename: {path}")
    return name[len(prefix):-len(suffix)]


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"train_ex", "train_labels", "eval_ex", "eval_labels"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"{path} missing keys: {sorted(missing)}")
    return data


def cap_training_examples(data: Dict[str, Any], max_train_examples: Optional[int], seed: int) -> Dict[str, Any]:
    if max_train_examples is None or max_train_examples >= len(data["train_ex"]):
        return data

    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for idx, label in enumerate(data["train_labels"]):
        by_label[int(label)].append(idx)
    for indices in by_label.values():
        rng.shuffle(indices)

    half = max_train_examples // 2
    selected = by_label[1][:half] + by_label[0][: max_train_examples - half]
    if len(selected) < max_train_examples:
        remaining = [
            idx
            for indices in by_label.values()
            for idx in indices
            if idx not in selected
        ]
        selected.extend(remaining[: max_train_examples - len(selected)])
    selected = sorted(selected)

    capped = dict(data)
    capped["train_ex"] = [data["train_ex"][idx] for idx in selected]
    capped["train_labels"] = [data["train_labels"][idx] for idx in selected]
    return capped


def make_task(task_type: str, regex: str, max_length: int):
    if task_type == "extrx":
        return ExtRegularLanguage(regex, max_length=max_length, alphabet="[A-Za-z0-9#]")
    return SimplyRegularLanguage(regex, max_length=max_length)


def task_alphabet(task) -> List[str]:
    return sorted(sym.value if hasattr(sym, "value") else str(sym) for sym in task.sigma)


def run_one(
    path: Path,
    regex: str,
    task_type: str,
    methods: List[str],
    max_length: int,
    max_train_examples: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    data = load_dataset(path)
    original_train_size = len(data["train_ex"])
    data = cap_training_examples(data, max_train_examples, seed)
    task = make_task(task_type, regex, max_length)
    target_dfa = task.dfa
    alphabet = task_alphabet(task)

    method_results = {}
    for method in methods:
        learner = LEARNERS[method]
        start_time = time.time()
        learned = learner(data["train_ex"], data["train_labels"])
        runtime = time.time() - start_time
        equivalent, witness = equivalent_to_target(learned, target_dfa, alphabet)
        train_accuracy = accuracy(learned, data["train_ex"], data["train_labels"])
        eval_accuracy = accuracy(learned, data["eval_ex"], data["eval_labels"])
        method_results[method] = {
            "equivalent": equivalent,
            "witness": witness,
            "num_states": len(learned.states),
            "num_accepting_states": len(learned.accepting),
            "train_accuracy": train_accuracy,
            "eval_accuracy": eval_accuracy,
            "runtime_seconds": runtime,
        }

    return {
        "regex": regex,
        "task_type": task_type,
        "dataset_path": str(path),
        "target_num_states": len(target_dfa.states),
        "alphabet_size": len(alphabet),
        "train_size": len(data["train_ex"]),
        "original_train_size": original_train_size,
        "methods": method_results,
    }


def collect_jobs(args: argparse.Namespace) -> List[tuple[Path, str]]:
    if args.dataset_path is not None:
        path = args.dataset_path
        return [(path, args.regex or regex_from_dataset_path(path))]
    if args.regex is not None:
        return [(dataset_path_for_regex(args.dataset_dir, args.regex), args.regex)]

    jobs = []
    for path in sorted(args.dataset_dir.glob("regex=*_trainMaxLen=32_evalMaxLen=32.json")):
        regex = regex_from_dataset_path(path)
        jobs.append((path, regex))
        if args.limit is not None and len(jobs) >= args.limit:
            break
    return jobs


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = [method for method in methods if method not in LEARNERS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {sorted(LEARNERS)}")

    results = []
    for path, regex in collect_jobs(args):
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Running {args.task_type}: {regex}", flush=True)
        results.append(
            run_one(
                path,
                regex,
                args.task_type,
                methods,
                args.max_length,
                args.max_train_examples,
                args.seed,
            )
        )

    summary = {
        "task_type": args.task_type,
        "num_datasets": len(results),
        "methods": {},
        "results": results,
    }
    for method in methods:
        method_rows = [row["methods"][method] for row in results]
        solved = sum(1 for row in method_rows if row["equivalent"])
        summary["methods"][method] = {
            "equivalent_count": solved,
            "total": len(method_rows),
            "equivalent_rate": solved / len(method_rows) if method_rows else 0.0,
            "mean_eval_accuracy": (
                sum(row["eval_accuracy"] for row in method_rows) / len(method_rows)
                if method_rows
                else 0.0
            ),
            "mean_runtime_seconds": (
                sum(row["runtime_seconds"] for row in method_rows) / len(method_rows)
                if method_rows
                else 0.0
            ),
        }

    print(json.dumps(summary["methods"], indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
