#!/usr/bin/env python3
"""Compute two-stage bootstrap CIs for the main full-dataset success table."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "datasets/scaleup/regex_list.json"
N_BOOTSTRAP = 20_000
SEED = 20260728

METHOD_DIRS = {
    "Standard": {
        "simplyrx": ROOT / "logs/scaleup/icl_gen_simplyrx/model=gpt-oss/std/reg",
        "extrx": ROOT / "logs/scaleup/icl_gen_extrx/model=gpt-oss/std/reg",
    },
    "Single Inference": {
        "simplyrx": ROOT
        / "logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/single_inference",
        "extrx": ROOT
        / "logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/single_inference",
    },
    "Agentic Reflection": {
        "simplyrx": ROOT
        / "logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs",
        "extrx": ROOT
        / "logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/agentic_reflection",
    },
}


def target_regexes(dataset: dict, domain: str) -> dict[int, list[str]]:
    by_depth: dict[int, list[str]] = {}
    for state_group in dataset[domain]:
        for depth_group in state_group["regex_list"]:
            depth = int(depth_group["Stardepth"])
            by_depth.setdefault(depth, []).extend(depth_group["regex_list"][:3])
    return by_depth


def index_logs(log_dir: Path, regexes: set[str]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in log_dir.glob("*.json"):
        name = path.name
        for regex in regexes:
            if name.startswith(f"msgdict_regex={regex}_"):
                if regex in indexed:
                    raise RuntimeError(f"Duplicate log for {regex!r} in {log_dir}")
                indexed[regex] = path
                break
    return indexed


def outcomes(path: Path) -> list[int]:
    summary = json.loads(path.read_text())["summary"]
    values = []
    for run_index in range(3):
        run = summary.get(f"run-{run_index}") or summary.get(f"rerun-{run_index}")
        if not isinstance(run, dict) or "final_accuracy" not in run:
            raise RuntimeError(f"Missing run-{run_index} in {path}")
        values.append(int(float(run["final_accuracy"]) >= 1.0))
    return values


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def hierarchical_ci(
    target_runs: list[list[int]], rng: random.Random
) -> tuple[float, float, float]:
    """Resample targets, then three runs within each selected target."""
    n_targets = len(target_runs)
    observed = sum(map(sum, target_runs)) / (3 * n_targets)
    bootstrap = []
    for _ in range(N_BOOTSTRAP):
        successes = 0
        for _ in range(n_targets):
            runs = target_runs[rng.randrange(n_targets)]
            successes += sum(runs[rng.randrange(3)] for _ in range(3))
        bootstrap.append(successes / (3 * n_targets))
    bootstrap.sort()
    return observed, percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    rng = random.Random(SEED)
    rows = []

    for domain in ("simplyrx", "extrx"):
        regexes_by_depth = target_regexes(dataset, domain)
        all_regexes = {regex for group in regexes_by_depth.values() for regex in group}
        for method, directories in METHOD_DIRS.items():
            log_index = index_logs(directories[domain], all_regexes)
            for depth, regexes in sorted(regexes_by_depth.items()):
                missing = [regex for regex in regexes if regex not in log_index]
                if missing:
                    raise RuntimeError(
                        f"{domain}/{method}/SD={depth}: {len(missing)} missing logs"
                    )
                target_runs = [outcomes(log_index[regex]) for regex in regexes]
                mean, lower, upper = hierarchical_ci(target_runs, rng)
                rows.append(
                    {
                        "dataset": domain,
                        "method": method,
                        "stardepth": depth,
                        "n_regexes": len(regexes),
                        "n_runs": 3 * len(regexes),
                        "successes": sum(map(sum, target_runs)),
                        "mean": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )

    output = Path(__file__).with_name("main_result_hierarchical_bootstrap_ci.csv")
    header = (
        "dataset,method,stardepth,n_regexes,n_runs,successes,"
        "mean,ci_lower,ci_upper\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['dataset']},{row['method']},{row['stardepth']},"
            f"{row['n_regexes']},{row['n_runs']},{row['successes']},"
            f"{100 * row['mean']:.1f}%,"
            f"{100 * row['ci_lower']:.1f}%,{100 * row['ci_upper']:.1f}%\n"
        )
    output.write_text("".join(lines))

    for domain in ("simplyrx", "extrx"):
        print(f"\n{domain}")
        print("Method | SD | mean [95% CI] | successes/runs | regexes")
        for row in rows:
            if row["dataset"] != domain:
                continue
            print(
                f"{row['method']} | {row['stardepth']} | "
                f"{100 * row['mean']:.1f}% "
                f"[{100 * row['ci_lower']:.1f}%, "
                f"{100 * row['ci_upper']:.1f}%] | "
                f"{row['successes']}/{row['n_runs']} | {row['n_regexes']}"
            )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
