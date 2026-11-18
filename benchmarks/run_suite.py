"""Plan or run all locally available benchmark settings without model imports."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("cryptography", "list_functions", "instruction_induction", "scan", "ilp")
SMOKE = {"cryptography": ["caesar:0"], "list_functions": ["c102"],
         "instruction_induction": ["antonyms"], "scan": ["simple"], "ilp": ["even_succ"]}


def list_complexity_tiers():
    """Rank reference syntax complexity; tiers are proxies, not measured difficulty."""
    path = ROOT / "datasets/list_functions/raw/list_functions.jsonl"
    ranked = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        nodes = list(ast.walk(ast.parse(item["function"])))
        score = len(nodes)
        score += 8 * sum(isinstance(n, (ast.For, ast.While)) for n in nodes)
        score += 4 * sum(isinstance(n, (ast.If, ast.IfExp)) for n in nodes)
        score += 3 * sum(isinstance(n, ast.Call) for n in nodes)
        score += 2 * sum(isinstance(n, ast.Subscript) for n in nodes)
        score += 5 * sum(isinstance(n, ast.comprehension) for n in nodes)
        ranked.append((score, item["id"]))
    ranked.sort()
    count = len(ranked)
    return [ranked[i * count // 3:(i + 1) * count // 3] for i in range(3)]


def balanced_settings(selected, list_per_tier, crypto_per_family, sample_seed):
    if list_per_tier < 1 or crypto_per_family < 1:
        raise ValueError("Balanced sample counts must be positive")
    available = settings("all", selected)
    chosen = set()
    # Independent random streams keep each task's selection stable under --task filters.
    for family in ("caesar", "atbash", "keyboard"):
        ids = [pair for pair in available if pair[0] == "cryptography"
               and pair[1].startswith(family + ":")]
        if ids:
            if crypto_per_family > len(ids):
                raise ValueError(f"Requested more samples than available for {family}")
            chosen.update(random.Random(f"{sample_seed}:crypto:{family}").sample(ids, crypto_per_family))
    if any(task == "list_functions" for task, _ in available):
        for i, tier in enumerate(list_complexity_tiers()):
            if list_per_tier > len(tier):
                raise ValueError("Requested more list samples than available in a tier")
            picked = random.Random(f"{sample_seed}:list:{i}").sample(tier, list_per_tier)
            chosen.update(("list_functions", ident) for _, ident in picked)
    chosen.update(pair for pair in available if pair[0] in ("instruction_induction", "scan"))
    if any(task == "ilp" for task, _ in available):
        from benchmarks.ilp_settings import BALANCED_IDS
        wanted = {("ilp", ident) for ident in BALANCED_IDS}
        missing = wanted - set(available)
        if missing:
            raise ValueError(f"Missing balanced ILP data; run datasets/ilp/prepare.py: {sorted(missing)}")
        chosen.update(wanted)
    return [pair for pair in available if pair in chosen]


def settings(suite="all", selected=None, *, list_per_tier=3, crypto_per_family=2, sample_seed=43):
    """Discover IDs from the data, checking paired train/test files exist."""
    selected = selected or TASKS
    if suite == "balanced":
        return balanced_settings(selected, list_per_tier, crypto_per_family, sample_seed)
    if suite == "smoke":
        return [(task, ident) for task in TASKS if task in selected for ident in SMOKE[task]]
    data = ROOT / "datasets"
    found = []
    for task in TASKS:
        if task not in selected:
            continue
        if task == "cryptography":
            for family in ("caesar", "atbash", "keyboard"):
                path = data / "cryptography/raw" / f"{family}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                found.extend((task, f"{family}:{i}") for i in range(len(rows)))
        elif task == "list_functions":
            path = data / "list_functions/raw/list_functions.jsonl"
            ids = sorted({json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()})
            found.extend((task, ident) for ident in ids)
        elif task == "ilp":
            folder = data / "ilp"
            for path in sorted(folder.glob("*.json")):
                row = json.loads(path.read_text())
                if "id" in row and "train" in row and "eval" in row:
                    found.append((task, row["id"]))
        elif task == "instruction_induction":
            folder = data / "instruction_induction/raw"
            for path in sorted((folder / "induce").glob("*.json")):
                if not (folder / "execute" / path.name).is_file():
                    raise ValueError(f"Missing execution data for {path.stem}")
                found.append((task, path.stem))
        else:
            folder = data / "scan/raw"
            for path in sorted(folder.glob("tasks_train_*.txt")):
                split = path.stem.removeprefix("tasks_train_")
                if not (folder / f"tasks_test_{split}.txt").is_file():
                    raise ValueError(f"Missing SCAN test split {split}")
                found.append((task, split))
        if not any(name == task for name, _ in found):
            raise ValueError(f"No data found for {task}")
    return found


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        index = argv.index("--")
        argv, extra = argv[:index], argv[index + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_positional", nargs="?", choices=("icl", "icl_gen", "both"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--runner", choices=("icl", "icl_gen", "both"),
                        help="Experiment type (default: icl)")
    parser.add_argument("--suite", choices=("smoke", "all", "balanced"), default="smoke",
                        help="smoke: one default per task; all: every setting; balanced: stratified smaller suite")
    parser.add_argument("--list-per-tier", type=int, default=3, help="Balanced suite: list functions per complexity tier")
    parser.add_argument("--crypto-per-family", type=int, default=2, help="Balanced suite: instances per cipher family")
    parser.add_argument("--sample-seed", type=int, default=43, help="Balanced selection seed; independent of model/run seed")
    parser.add_argument("--task", action="extend", nargs="+", choices=TASKS,
                        help="Restrict to one or more tasks; also repeatable")
    parser.add_argument("--mode", choices=("std", "ce", "both"), default="both")
    parser.add_argument("--mkey", default="gpt-oss")
    parser.add_argument("--outdir", default=os.environ.get("OUTDIR", "logs/benchmarks"))
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or writing logs")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed job; exit nonzero if any failed")
    args = parser.parse_args(argv)
    if args.runner and args.runner_positional and args.runner != args.runner_positional:
        parser.error("Conflicting positional runner and --runner")
    args.runner = args.runner or args.runner_positional or "icl"
    reserved = {"--task_type", "--benchmark_id", "--mkey", "--outdir", "--use_ce", "--use_reg"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        parser.error("Task, benchmark ID, model, output directory, CE mode and regularization are controlled by the suite")
    try:
        instances = settings(args.suite, args.task, list_per_tier=args.list_per_tier,
                             crypto_per_family=args.crypto_per_family, sample_seed=args.sample_seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    runners = ("icl", "icl_gen") if args.runner == "both" else (args.runner,)
    modes = ("std", "ce") if args.mode == "both" else (args.mode,)
    total = len(instances) * len(runners) * len(modes)
    counts = {task: sum(name == task for name, _ in instances) for task in TASKS}
    print(f"Suite={args.suite}; settings={len(instances)} {counts}; jobs={total}", flush=True)
    if args.suite == "balanced":
        print(f"Selection seed={args.sample_seed}; list tiers={args.list_per_tier} each; "
              f"cipher families={args.crypto_per_family} each; all instruction tasks and SCAN splits retained; six ILP tasks (two per structural difficulty tier). "
              "Tiers reflect reference syntax complexity, not measured model difficulty.", flush=True)
    failed = []
    done = 0
    for runner in runners:
        for task, ident in instances:
            for mode in modes:
                command = [args.python, f"train_{runner}.py", "--task_type", task,
                           "--benchmark_id", ident, "--mkey", args.mkey, "--use_reg",
                           "--outdir", args.outdir]
                if runner == "icl_gen":
                    command += ["--rerun", "1"]
                if mode == "ce":
                    command += ["--use_ce", "--ce_batch_size", "8"]
                command += extra
                done += 1
                print(f"[{done}/{total}] {runner} / {task} / {ident} / {mode}", flush=True)
                print(shlex.join(command), flush=True)
                if args.dry_run:
                    continue
                result = subprocess.run(command, cwd=ROOT, check=False)
                if result.returncode:
                    failed.append((runner, task, ident, mode, result.returncode))
                    print(f"FAILED: {failed[-1]}", file=sys.stderr, flush=True)
                    if not args.keep_going:
                        return 1
    print(f"{'Planned' if args.dry_run else 'Finished'} {done} jobs; failures={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
