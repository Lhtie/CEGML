#!/usr/bin/env python3
"""Run the Regexulator-style LLM tree-search baseline on SimplyRX ablation."""

from __future__ import annotations

import argparse
import json
import random
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.regexulator import accuracy, run_regexulator_search
from baselines.run_simplyrx_ablation import regexes_from_script
from baselines.run_state_merging import (
    cap_training_examples,
    dataset_path_for_regex,
    load_dataset,
)
from keysecrets import api_key
from modeling.llm import load_model_and_tokenizer, run_model
from tasks.rl import SimplyRegularLanguage
from train_icl_gen import extract_ans, extract_reasoning


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
    parser.add_argument("--mkey", default="gpt-oss")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_train_examples", type=int, default=1500)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--initial_splits", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--branching_factor", type=int, default=2)
    parser.add_argument("--max_generation_calls", type=int, default=16)
    parser.add_argument(
        "--time_limit_seconds",
        type=float,
        default=180.0,
        help="Overall wall-clock limit for each regex; <=0 disables it.",
    )
    parser.add_argument("--start_examples", type=int, default=10)
    parser.add_argument("--improve_examples", type=int, default=5)
    parser.add_argument("--max_compile_repairs", type=int, default=1)
    parser.add_argument("--depth_base", type=float, default=0.9)
    parser.add_argument("--sibling_base", type=float, default=0.95)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/summary/regexulator_simplyrx_ablation.json"),
    )
    return parser.parse_args()


def stratified_split(
    examples: list[str],
    labels: list[int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be between 0 and 1")
    rng = random.Random(seed)
    indices_by_label = {0: [], 1: []}
    for index, label in enumerate(labels):
        indices_by_label[int(label)].append(index)

    validation_indices = set()
    for indices in indices_by_label.values():
        rng.shuffle(indices)
        count = (
            min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
            if len(indices) > 1
            else 0
        )
        validation_indices.update(indices[:count])
    train_indices = [
        index for index in range(len(examples)) if index not in validation_indices
    ]
    validation_indices_sorted = sorted(validation_indices)
    return (
        [examples[index] for index in train_indices],
        [int(labels[index]) for index in train_indices],
        [examples[index] for index in validation_indices_sorted],
        [int(labels[index]) for index in validation_indices_sorted],
    )


@contextmanager
def generation_deadline(seconds: float | None):
    """Interrupt a blocking local generation when its regex budget expires."""
    if seconds is None or seconds <= 0:
        yield
        return

    def handle_timeout(signum, frame):
        raise TimeoutError("Regexulator per-regex time limit reached")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_one(
    regex: str,
    args: argparse.Namespace,
    model,
    tokenizer,
) -> dict[str, Any]:
    dataset_path = dataset_path_for_regex(args.dataset_dir, regex)
    full_dataset = load_dataset(dataset_path)
    capped = cap_training_examples(
        full_dataset, args.max_train_examples, args.seed
    )
    train_ex, train_labels, validation_ex, validation_labels = stratified_split(
        capped["train_ex"],
        capped["train_labels"],
        args.validation_fraction,
        args.seed,
    )
    task = SimplyRegularLanguage(regex, max_length=args.max_length)
    search_started_at = time.perf_counter()
    deadline = (
        search_started_at + args.time_limit_seconds
        if args.time_limit_seconds > 0
        else None
    )

    def generate(prompt: str, temperature: float) -> dict[str, Any]:
        remaining = (
            max(0.0, deadline - time.perf_counter())
            if deadline is not None
            else None
        )
        if remaining is not None and remaining <= 0:
            response = None
        else:
            with generation_deadline(remaining):
                response = run_model(
                    args.mkey, model, tokenizer, prompt, temp=temperature
                )
        return {
            "Response": response,
            "Prediction": extract_ans(response),
            "Reasoning": extract_reasoning(response),
        }

    metrics = run_regexulator_search(
        task=task,
        train_examples=train_ex,
        train_labels=train_labels,
        validation_examples=validation_ex,
        validation_labels=validation_labels,
        generate=generate,
        tokenizer=tokenizer,
        initial_splits=args.initial_splits,
        max_depth=args.max_depth,
        branching_factor=args.branching_factor,
        max_generation_calls=args.max_generation_calls,
        time_limit_seconds=args.time_limit_seconds,
        start_examples=args.start_examples,
        improve_examples=args.improve_examples,
        max_compile_repairs=args.max_compile_repairs,
        depth_base=args.depth_base,
        sibling_base=args.sibling_base,
        temperature=args.temp,
        seed=args.seed,
    )
    if metrics["selected_regex"] is None:
        metrics["eval_accuracy"] = 0.0
    else:
        selected_dfa = task.regex_to_dfa(metrics["selected_regex"])
        metrics["eval_accuracy"] = accuracy(
            selected_dfa,
            full_dataset["eval_ex"],
            full_dataset["eval_labels"],
        )
    return {
        "regex": regex,
        "dataset_path": str(dataset_path),
        "target_num_states": len(task.dfa.states),
        "original_train_size": len(full_dataset["train_ex"]),
        "search_train_size": len(train_ex),
        "validation_size": len(validation_ex),
        "eval_size": len(full_dataset["eval_ex"]),
        "metrics": metrics,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(results)
    successes = [row for row in results if row["metrics"]["equivalent"]]
    calls = [row["metrics"]["generation_calls"] for row in results]
    times = [row["metrics"]["wall_time_seconds"] for row in results]
    successes_by_call = Counter(
        row["metrics"]["successful_call"] for row in successes
    )
    return {
        "completed": completed,
        "successful": len(successes),
        "failed": completed - len(successes),
        "success_rate": len(successes) / completed if completed else 0.0,
        "successes_by_generation_call": {
            str(call): count for call, count in sorted(successes_by_call.items())
        },
        "mean_generation_calls": statistics.mean(calls) if calls else 0.0,
        "total_generation_calls": sum(calls),
        "mean_eval_accuracy": (
            statistics.mean(row["metrics"]["eval_accuracy"] for row in results)
            if results
            else 0.0
        ),
        "total_prompt_tokens": sum(
            row["metrics"]["prompt_tokens"] for row in results
        ),
        "total_response_tokens": sum(
            row["metrics"]["response_tokens"] for row in results
        ),
        "mean_wall_time_seconds": statistics.mean(times) if times else 0.0,
        "median_wall_time_seconds": statistics.median(times) if times else 0.0,
        "total_wall_time_seconds": sum(times),
    }


def write_checkpoint(
    path: Path,
    regexes: list[str],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "algorithm": "Regexulator validation-guided LLM tree search",
        "task_type": "simplyrx",
        "benchmark": "single" if args.regex else "ablation",
        "num_datasets": len(regexes),
        "config": {
            "mkey": args.mkey,
            "max_train_examples": args.max_train_examples,
            "validation_fraction": args.validation_fraction,
            "initial_splits": args.initial_splits,
            "max_depth": args.max_depth,
            "branching_factor": args.branching_factor,
            "max_generation_calls": args.max_generation_calls,
            "time_limit_seconds": args.time_limit_seconds,
            "start_examples": args.start_examples,
            "improve_examples": args.improve_examples,
            "max_compile_repairs": args.max_compile_repairs,
            "depth_base": args.depth_base,
            "sibling_base": args.sibling_base,
            "temperature": args.temp,
            "seed": args.seed,
            "selection_metric": "held-out string classification accuracy",
            "success_metric": "exact product-DFA equivalence",
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
        raise SystemExit("No SimplyRX regexes selected")

    results = []
    if args.resume and args.out.exists():
        checkpoint = json.loads(args.out.read_text(encoding="utf-8"))
        results = checkpoint.get("results", [])
        if args.retry_failed:
            failed = {
                row["regex"] for row in results if not row["metrics"]["equivalent"]
            }
            results = [row for row in results if row["regex"] not in failed]
            print(f"Retrying {len(failed)} failed result(s).", flush=True)
        print(f"Resuming with {len(results)} completed result(s).", flush=True)
    completed_regexes = {row["regex"] for row in results}

    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
    for index, regex in enumerate(regexes, start=1):
        if regex in completed_regexes:
            continue
        print(f"[{index}/{len(regexes)}] {regex}", flush=True)
        result = run_one(regex, args, model, tokenizer)
        results.append(result)
        write_checkpoint(args.out, regexes, results, args)
        metrics = result["metrics"]
        print(
            f"  equivalent={metrics['equivalent']} "
            f"stop={metrics['stop_reason']} "
            f"calls={metrics['generation_calls']} "
            f"valid={metrics['valid_candidates']} "
            f"val_acc={metrics['selected_validation_accuracy']:.4f} "
            f"eval_acc={metrics['eval_accuracy']:.4f} "
            f"tokens={metrics['total_tokens']} "
            f"time={metrics['wall_time_seconds']:.1f}s",
            flush=True,
        )
        print(f"  checkpoint: {args.out}", flush=True)
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
