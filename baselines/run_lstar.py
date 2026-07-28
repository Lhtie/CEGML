#!/usr/bin/env python3
"""Run the AALpy L* baseline on SimplyRx regexes or the ablation benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.lstar import evaluate_hypothesis, learn_lstar
from baselines.run_simplyrx_ablation import regexes_from_script
from baselines.run_state_merging import dataset_path_for_regex, load_dataset
from tasks.rl import SimplyRegularLanguage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", default=None)
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
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_learning_rounds", type=int, default=None)
    parser.add_argument("--closing_strategy", default="shortest_first")
    parser.add_argument("--cex_processing", default="rs")
    parser.add_argument("--print_level", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/summary/lstar_simplyrx_ablation.json"),
    )
    return parser.parse_args()


def run_one(regex: str, args: argparse.Namespace) -> dict[str, Any]:
    benchmark_started_at = time.perf_counter()
    task = SimplyRegularLanguage(regex, max_length=args.max_length)
    alphabet = sorted(str(symbol.value) for symbol in task.sigma)
    dataset_path = dataset_path_for_regex(args.dataset_dir, regex)
    dataset = load_dataset(dataset_path)

    hypothesis, metrics = learn_lstar(
        task.dfa,
        alphabet,
        max_learning_rounds=args.max_learning_rounds,
        closing_strategy=args.closing_strategy,
        cex_processing=args.cex_processing,
        print_level=args.print_level,
    )
    metrics["train_accuracy"] = evaluate_hypothesis(
        hypothesis, dataset["train_ex"], dataset["train_labels"]
    )
    metrics["eval_accuracy"] = evaluate_hypothesis(
        hypothesis, dataset["eval_ex"], dataset["eval_labels"]
    )
    metrics["benchmark_time_seconds"] = time.perf_counter() - benchmark_started_at
    return {
        "regex": regex,
        "dataset_path": str(dataset_path),
        "target_num_states": len(task.dfa.states),
        "alphabet": alphabet,
        "train_size": len(dataset["train_ex"]),
        "eval_size": len(dataset["eval_ex"]),
        "metrics": metrics,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(results)
    successful_rows = [row for row in results if row["metrics"]["equivalent"]]
    successful = len(successful_rows)

    def mean(metric: str, rows: list[dict[str, Any]] = results) -> float:
        if not rows:
            return 0.0
        return sum(float(row["metrics"][metric]) for row in rows) / len(rows)

    wall_times = [row["metrics"]["wall_time_seconds"] for row in results]
    successes_by_round = Counter(
        row["metrics"]["successful_round"] for row in successful_rows
    )

    return {
        "completed": completed,
        "successful": successful,
        "failed": completed - successful,
        "success_rate": successful / completed if completed else 0.0,
        "successes_by_round": {
            str(round_number): count
            for round_number, count in sorted(successes_by_round.items())
        },
        "mean_successful_round": mean("learning_rounds", successful_rows),
        "total_counterexamples": sum(
            row["metrics"]["num_counterexamples"] for row in results
        ),
        "mean_counterexamples": mean("num_counterexamples"),
        "total_membership_queries": sum(
            row["metrics"]["membership_queries"] for row in results
        ),
        "mean_membership_queries": mean("membership_queries"),
        "total_equivalence_queries": sum(
            row["metrics"]["equivalence_queries"] for row in results
        ),
        "mean_equivalence_queries": mean("equivalence_queries"),
        "mean_wall_time_seconds": mean("wall_time_seconds"),
        "median_wall_time_seconds": statistics.median(wall_times) if wall_times else 0.0,
        "max_wall_time_seconds": max(wall_times, default=0.0),
        "total_wall_time_seconds": sum(wall_times),
        "total_benchmark_time_seconds": sum(
            row["metrics"]["benchmark_time_seconds"] for row in results
        ),
    }


def write_checkpoint(
    path: Path,
    regexes: list[str],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "algorithm": "AALpy L*",
        "task_type": "simplyrx",
        "benchmark": "single" if args.regex else "ablation",
        "num_datasets": len(regexes),
        "config": {
            "max_length": args.max_length,
            "max_learning_rounds": args.max_learning_rounds,
            "closing_strategy": args.closing_strategy,
            "cex_processing": args.cex_processing,
            "equivalence_oracle": "exact_product_dfa",
        },
        "summary": summarize(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    regexes = [args.regex] if args.regex else regexes_from_script(args.script)
    if not regexes:
        raise SystemExit("No SimplyRx regexes selected")

    results = []
    for index, regex in enumerate(regexes, start=1):
        print(f"[{index}/{len(regexes)}] {regex}", flush=True)
        result = run_one(regex, args)
        results.append(result)
        write_checkpoint(args.out, regexes, results, args)
        metrics = result["metrics"]
        print(
            f"  equivalent={metrics['equivalent']} "
            f"rounds={metrics['learning_rounds']} "
            f"cex={metrics['num_counterexamples']} "
            f"mq={metrics['membership_queries']} "
            f"time={metrics['wall_time_seconds']:.4f}s",
            flush=True,
        )
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
