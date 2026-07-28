#!/usr/bin/env python3
"""Run CVC5 DFA synthesis on one SimplyRx regex or its ablation benchmark."""

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

from baselines.cvc5_dfa import learn_cvc5_dfa
from baselines.run_simplyrx_ablation import regexes_from_script
from baselines.run_state_merging import (
    cap_training_examples,
    dataset_path_for_regex,
    load_dataset,
)
from baselines.state_merging import accuracy, equivalent_to_target
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
    parser.add_argument("--max_train_examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_states", type=int, default=1)
    parser.add_argument("--max_states", type=int, default=12)
    parser.add_argument("--timeout_ms_per_round", type=int, default=None)
    parser.add_argument("--max_cegis_rounds", type=int, default=30)
    parser.add_argument(
        "--no_cegis",
        action="store_true",
        help="Disable counterexample-guided refinement and fit only the initial examples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed regexes in --out.",
    )
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/summary/cvc5_simplyrx_ablation_cegis.json"),
    )
    return parser.parse_args()


def run_one(regex: str, args: argparse.Namespace) -> dict[str, Any]:
    benchmark_started_at = time.perf_counter()
    dataset_path = dataset_path_for_regex(args.dataset_dir, regex)
    full_dataset = load_dataset(dataset_path)
    dataset = cap_training_examples(
        full_dataset, args.max_train_examples, args.seed
    )
    task = SimplyRegularLanguage(regex, max_length=args.max_length)
    alphabet = sorted(str(symbol.value) for symbol in task.sigma)

    synthesis_examples = list(dataset["train_ex"])
    synthesis_labels = list(dataset["train_labels"])
    counterexamples = []
    synthesis_rounds = []
    learned = None
    equivalent = False
    witness = None
    max_cegis_rounds = 1 if args.no_cegis else args.max_cegis_rounds

    for cegis_round in range(1, max_cegis_rounds + 1):
        learned, round_metrics = learn_cvc5_dfa(
            synthesis_examples,
            synthesis_labels,
            alphabet,
            min_states=args.min_states,
            max_states=args.max_states,
            timeout_ms_per_round=args.timeout_ms_per_round,
        )
        synthesis_rounds.append(round_metrics)
        if learned is None:
            break
        equivalent, witness = equivalent_to_target(
            learned, task.dfa, alphabet
        )
        if equivalent or args.no_cegis:
            break
        counterexamples.append(witness)
        synthesis_examples.append(witness)
        synthesis_labels.append(int(task.accepts(witness)))

    metrics = {
        "satisfiable": learned is not None,
        "equivalent": equivalent,
        "witness": witness,
        "cegis_rounds": len(synthesis_rounds),
        "successful_round": len(synthesis_rounds) if equivalent else None,
        "num_counterexamples": len(counterexamples),
        "counterexamples": counterexamples,
        "counterexample_lengths": [len(item) for item in counterexamples],
        "initial_num_examples": len(dataset["train_ex"]),
        "final_num_examples": len(synthesis_examples),
        "solver_rounds": sum(
            round_metrics["solver_rounds"]
            for round_metrics in synthesis_rounds
        ),
        "learned_num_states": (
            len(learned.states) if learned is not None else None
        ),
        "synthesis_rounds": synthesis_rounds,
        "total_solver_time_seconds": sum(
            round_metrics["total_solver_time_seconds"]
            for round_metrics in synthesis_rounds
        ),
        "wall_time_seconds": sum(
            round_metrics["wall_time_seconds"]
            for round_metrics in synthesis_rounds
        ),
    }
    if learned is None:
        metrics.update(
            {
                "train_accuracy": None,
                "eval_accuracy": None,
            }
        )
    else:
        metrics.update(
            {
                "train_accuracy": accuracy(
                    learned, dataset["train_ex"], dataset["train_labels"]
                ),
                "eval_accuracy": accuracy(
                    learned,
                    full_dataset["eval_ex"],
                    full_dataset["eval_labels"],
                ),
            }
        )
    metrics["benchmark_time_seconds"] = (
        time.perf_counter() - benchmark_started_at
    )
    return {
        "regex": regex,
        "dataset_path": str(dataset_path),
        "target_num_states": len(task.dfa.states),
        "alphabet": alphabet,
        "train_size": len(dataset["train_ex"]),
        "original_train_size": len(full_dataset["train_ex"]),
        "eval_size": len(full_dataset["eval_ex"]),
        "metrics": metrics,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(results)
    sat_rows = [row for row in results if row["metrics"]["satisfiable"]]
    successful_rows = [row for row in results if row["metrics"]["equivalent"]]
    times = [row["metrics"]["wall_time_seconds"] for row in results]
    successes_by_round = Counter(
        row["metrics"]["successful_round"] for row in successful_rows
    )
    return {
        "completed": completed,
        "satisfiable": len(sat_rows),
        "successful": len(successful_rows),
        "failed": completed - len(successful_rows),
        "success_rate": len(successful_rows) / completed if completed else 0.0,
        "successes_by_solver_round": {
            str(round_number): count
            for round_number, count in sorted(successes_by_round.items())
        },
        "mean_solver_rounds": (
            sum(row["metrics"]["solver_rounds"] for row in results) / completed
            if completed
            else 0.0
        ),
        "total_counterexamples": sum(
            row["metrics"]["num_counterexamples"] for row in results
        ),
        "mean_counterexamples": (
            sum(row["metrics"]["num_counterexamples"] for row in results)
            / completed
            if completed
            else 0.0
        ),
        "mean_eval_accuracy": (
            sum(row["metrics"]["eval_accuracy"] for row in sat_rows)
            / len(sat_rows)
            if sat_rows
            else 0.0
        ),
        "mean_wall_time_seconds": statistics.mean(times) if times else 0.0,
        "median_wall_time_seconds": statistics.median(times) if times else 0.0,
        "max_wall_time_seconds": max(times, default=0.0),
        "total_solver_wall_time_seconds": sum(times),
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
        "algorithm": "CVC5 bounded DFA synthesis",
        "task_type": "simplyrx",
        "benchmark": "single" if args.regex else "ablation",
        "num_datasets": len(regexes),
        "config": {
            "max_train_examples": args.max_train_examples,
            "seed": args.seed,
            "min_states": args.min_states,
            "max_states": args.max_states,
            "timeout_ms_per_round": args.timeout_ms_per_round,
            "cegis": not args.no_cegis,
            "max_cegis_rounds": args.max_cegis_rounds,
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
    if args.resume and args.out.exists():
        checkpoint = json.loads(args.out.read_text(encoding="utf-8"))
        results = checkpoint.get("results", [])
        print(f"Resuming with {len(results)} completed result(s).", flush=True)
    completed_regexes = {result["regex"] for result in results}

    for index, regex in enumerate(regexes, start=1):
        if regex in completed_regexes:
            continue
        print(f"[{index}/{len(regexes)}] {regex}", flush=True)
        result = run_one(regex, args)
        results.append(result)
        write_checkpoint(args.out, regexes, results, args)
        metrics = result["metrics"]
        print(
            f"  sat={metrics['satisfiable']} "
            f"equivalent={metrics['equivalent']} "
            f"states={metrics['learned_num_states']} "
            f"cegis_rounds={metrics['cegis_rounds']} "
            f"cex={metrics['num_counterexamples']} "
            f"solver_rounds={metrics['solver_rounds']} "
            f"time={metrics['wall_time_seconds']:.4f}s",
            flush=True,
        )
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
