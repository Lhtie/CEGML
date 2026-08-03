#!/usr/bin/env python3
"""Run the type-erased Smore-style baseline on SimplyRx datasets."""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.run_simplyrx_ablation import regexes_from_script
from baselines.run_state_merging import cap_training_examples, dataset_path_for_regex, load_dataset
from baselines.smore import extract_sketch, run_smore_search
from keysecrets import api_key
from modeling.llm import load_model_and_tokenizer, run_model
from tasks.rl import SimplyRegularLanguage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regex", default=None, help="Run one target regex.")
    parser.add_argument("--script", type=Path, default=Path("datasets/scaleup/run_scripts/simplyrx_ablation.sh"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("datasets/scaleup/regex_datasets"))
    parser.add_argument("--mkey", default="gpt-oss")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_train_examples", type=int, default=1500)
    parser.add_argument("--max_prompt_examples", type=int, default=80)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--max_hole_candidates", type=int, default=192)
    parser.add_argument("--max_combinations", type=int, default=50000)
    parser.add_argument(
        "--time_limit_seconds",
        type=float,
        default=200.0,
        help="Total wall-clock budget for each target regex; <=0 disables it.",
    )
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("logs/summary/smore_type_erased_simplyrx.json"))
    return parser.parse_args()


@contextmanager
def generation_deadline(seconds: float | None):
    """Interrupt a blocking model generation when the regex budget expires."""
    if seconds is None or seconds <= 0:
        yield
        return

    def handle_timeout(signum, frame):
        raise TimeoutError("Smore per-regex time limit reached")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r["metrics"] for r in results]
    solved = sum(bool(r["equivalent"]) for r in rows)
    return {
        "completed": len(rows),
        "successful": solved,
        "failed": len(rows) - solved,
        "success_rate": solved / len(rows) if rows else 0.0,
        "consistent_count": sum(bool(r["consistent"]) for r in rows),
        "mean_train_accuracy": statistics.mean(r["train_accuracy"] for r in rows) if rows else 0.0,
        "mean_eval_accuracy": statistics.mean(r["eval_accuracy"] for r in rows) if rows else 0.0,
        "total_model_calls": sum(r["iterations"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "mean_wall_time_seconds": statistics.mean(r["wall_time_seconds"] for r in rows) if rows else 0.0,
    }


def checkpoint(path: Path, regexes: list[str], results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "algorithm": "Smore-style type-erased sketch synthesis (adaptation)",
        "task_type": "simplyrx",
        "benchmark": "single" if args.regex else "ablation",
        "num_datasets": len(regexes),
        "config": {
            "mkey": args.mkey,
            "max_train_examples": args.max_train_examples,
            "max_prompt_examples": args.max_prompt_examples,
            "max_iterations": args.max_iterations,
            "max_hole_candidates": args.max_hole_candidates,
            "max_combinations": args.max_combinations,
            "time_limit_seconds": args.time_limit_seconds,
            "temperature": args.temp,
            "seed": args.seed,
            "hole_type": "Default (ignored)",
            "repair_feedback": "training examples only",
            "success_metric": "exact product-DFA equivalence",
        },
        "summary": summarize(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_one(regex: str, args: argparse.Namespace, model, tokenizer) -> dict[str, Any]:
    path = dataset_path_for_regex(args.dataset_dir, regex)
    data = load_dataset(path)
    capped = cap_training_examples(data, args.max_train_examples, args.seed)
    task = SimplyRegularLanguage(regex, max_length=args.max_length)
    deadline = (
        time.perf_counter() + args.time_limit_seconds
        if args.time_limit_seconds > 0
        else None
    )

    def generate(prompt: str, temperature: float) -> dict[str, Any]:
        remaining = max(0.0, deadline - time.perf_counter()) if deadline is not None else None
        if remaining is not None and remaining <= 0:
            return {"Response": None, "Sketch": None}
        try:
            with generation_deadline(remaining):
                response = run_model(args.mkey, model, tokenizer, prompt, temp=temperature)
        except TimeoutError:
            response = None
        return {"Response": response, "Sketch": extract_sketch(response)}

    metrics = run_smore_search(
        task=task,
        train_examples=capped["train_ex"],
        train_labels=capped["train_labels"],
        eval_examples=data["eval_ex"],
        eval_labels=data["eval_labels"],
        generate=generate,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        max_prompt_examples=args.max_prompt_examples,
        max_hole_candidates=args.max_hole_candidates,
        max_combinations=args.max_combinations,
        temperature=args.temp,
        time_limit_seconds=args.time_limit_seconds,
    )
    return {
        "regex": regex,
        "dataset_path": str(path),
        "target_num_states": len(task.dfa.states),
        "train_size": len(capped["train_ex"]),
        "original_train_size": len(data["train_ex"]),
        "eval_size": len(data["eval_ex"]),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    regexes = [args.regex] if args.regex else regexes_from_script(args.script)
    if args.limit is not None:
        regexes = regexes[: args.limit]
    if not regexes:
        raise SystemExit("No SimplyRx regexes selected")

    results: list[dict[str, Any]] = []
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text(encoding="utf-8")).get("results", [])
        if args.retry_failed:
            results = [r for r in results if r["metrics"]["equivalent"]]
    done = {r["regex"] for r in results}
    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
    for index, regex in enumerate(regexes, 1):
        if regex in done:
            continue
        print(f"[{index}/{len(regexes)}] {regex}", flush=True)
        result = run_one(regex, args, model, tokenizer)
        results.append(result)
        checkpoint(args.out, regexes, results, args)
        m = result["metrics"]
        print(
            f"  equivalent={m['equivalent']} consistent={m['consistent']} "
            f"stop={m['stop_reason']} iterations={m['iterations']} train={m['train_accuracy']:.4f} "
            f"eval={m['eval_accuracy']:.4f} regex={m['selected_regex']}",
            flush=True,
        )
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
