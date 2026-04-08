import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from typing import Any, Callable, Dict, List, Optional, Tuple


def _load_token_counter(mkey: str):
    from modeling.llm import resolve_model_path

    model_path = resolve_model_path(mkey)

    if mkey.startswith("gpt-oss"):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Counting tokens for gpt-oss requires transformers to be installed."
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def count_fn(prompt: str) -> int:
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(token_ids)

        return count_fn

    if mkey.startswith(("gpt3", "gpt4")):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "Counting tokens for OpenAI chat models requires tiktoken to be installed."
            ) from e

        encoding = tiktoken.encoding_for_model(model_path)

        def count_fn(prompt: str) -> int:
            return len(encoding.encode(prompt))

        return count_fn

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"Counting tokens for model `{mkey}` requires transformers to be installed."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def count_fn(prompt: str) -> int:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    return count_fn


def _extract_prompt_from_log(log_item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(log_item, dict):
        return None

    prompt = log_item.get("Prompt")
    if isinstance(prompt, str):
        return prompt

    return None


def count_rerun_input_tokens(
    msgdict: Dict[str, Any],
    mkey: str = "gpt-oss",
    token_counter: Optional[Callable[[str], int]] = None,
) -> Dict[str, Any]:
    """
    Read a logged msgdict and count the total input tokens fed to the LLM for each rerun.

    Args:
        msgdict: The loaded JSON log dictionary.
        mkey: Model key used to choose the tokenizer, e.g. "gpt-oss".

    Returns:
        A dict with per-run totals and per-epoch breakdowns.
    """
    count_tokens = token_counter if token_counter is not None else _load_token_counter(mkey)
    results: Dict[str, Any] = {"model": mkey, "runs": {}, "grand_total_tokens": 0}

    for run_key, run_value in msgdict.items():
        if not str(run_key).startswith("run-") or not isinstance(run_value, dict):
            continue

        run_total = 0
        epoch_breakdown: Dict[str, Any] = {}

        for epoch_key, epoch_value in run_value.items():
            if not isinstance(epoch_value, dict):
                continue

            logs = epoch_value.get("Logs", [])
            epoch_total = 0
            prompt_count = 0

            if isinstance(logs, list):
                for log_item in logs:
                    prompt = _extract_prompt_from_log(log_item)
                    if prompt is None:
                        continue
                    epoch_total += count_tokens(prompt)
                    prompt_count += 1

            epoch_breakdown[epoch_key] = {
                "prompt_count": prompt_count,
                "total_tokens": epoch_total,
            }
            run_total += epoch_total

        results["runs"][run_key] = {
            "total_tokens": run_total,
            "epochs": epoch_breakdown,
        }
        results["grand_total_tokens"] += run_total

    return results


def _extract_regex_from_log_path(path: str) -> str:
    basename = os.path.basename(path)
    prefix = "msgdict_regex="
    if prefix not in basename:
        raise ValueError(f"Cannot parse regex from path: {path}")

    regex_part = basename.split(prefix, 1)[1]
    for suffix in ("_totTrain=", "_ceEpochs="):
        if suffix in regex_part:
            return regex_part.split(suffix, 1)[0]
    raise ValueError(f"Cannot parse regex suffix from path: {path}")


def _load_scaleup_regex_metadata(regex_list_path: str, task_type: str) -> Dict[str, Any]:
    with open(regex_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if task_type not in data:
        raise KeyError(f"Task type `{task_type}` not found in {regex_list_path}")

    depth_map: Dict[str, int] = {}
    state_map: Dict[str, int] = {}
    depths = set()
    states = set()
    for state_group in data[task_type]:
        state = int(state_group["#States"])
        states.add(state)
        for depth_group in state_group["regex_list"]:
            depth = int(depth_group["Stardepth"])
            depths.add(depth)
            for regex in depth_group["regex_list"]:
                depth_map[regex] = depth
                state_map[regex] = state
    return {
        "depth_map": depth_map,
        "state_map": state_map,
        "depths": sorted(depths),
        "states": sorted(states),
    }


def _compute_success_rate(summary: Dict[str, Any]) -> float:
    run_items = [
        run_info
        for run_key, run_info in summary.items()
        if str(run_key).startswith("run-") and isinstance(run_info, dict)
    ]
    if not run_items:
        return 0.0

    success_count = sum(
        1 for run_info in run_items if float(run_info.get("final_accuracy", 0)) >= 1.0
    )
    return success_count / len(run_items)


def summarize_scaleup_log_for_pareto(
    log_path: str,
    mkey: str = "gpt-oss",
    token_counter: Optional[Callable[[str], int]] = None,
) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        msgdict = json.load(f)

    token_stats = count_rerun_input_tokens(msgdict, mkey=mkey, token_counter=token_counter)
    run_totals = [
        run_stats["total_tokens"]
        for run_stats in token_stats["runs"].values()
        if isinstance(run_stats, dict)
    ]
    avg_tokens = float(np.mean(run_totals)) if run_totals else 0.0
    summary = msgdict.get("summary", {})
    success_rate = _compute_success_rate(summary)
    total_runs = sum(
        1 for run_key, run_info in summary.items()
        if str(run_key).startswith("run-") and isinstance(run_info, dict)
    )
    success_count = int(round(success_rate * total_runs))

    return {
        "path": log_path,
        "regex": _extract_regex_from_log_path(log_path),
        "avg_total_tokens": avg_tokens,
        "success_rate": success_rate,
        "run_token_totals": run_totals,
        "success_count": success_count,
        "total_runs": total_runs,
    }


def summarize_scaleup_log_for_solve_rate(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        msgdict = json.load(f)

    summary = msgdict.get("summary", {})
    success_rate = _compute_success_rate(summary)
    total_runs = sum(
        1 for run_key, run_info in summary.items()
        if str(run_key).startswith("run-") and isinstance(run_info, dict)
    )
    success_count = int(round(success_rate * total_runs))

    return {
        "path": log_path,
        "regex": _extract_regex_from_log_path(log_path),
        "success_rate": success_rate,
        "success_count": success_count,
        "total_runs": total_runs,
    }


def summarize_scaleup_log_for_sample_efficiency(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        msgdict = json.load(f)

    solved_samples: List[int] = []
    summary = msgdict.get("summary", {})
    run_keys = sorted(
        [
            run_key
            for run_key, run_info in summary.items()
            if str(run_key).startswith("run-") and isinstance(run_info, dict)
        ],
        key=lambda x: int(str(x).split("-")[1]),
    )

    for run_key in run_keys:
        run_epochs = msgdict.get(run_key, {})
        epoch_keys = sorted(
            [
                epoch_key
                for epoch_key, epoch_value in run_epochs.items()
                if str(epoch_key).startswith("epoch-") and isinstance(epoch_value, dict)
            ],
            key=lambda x: int(str(x).split("-")[1]),
        )
        for epoch_key in epoch_keys:
            epoch = run_epochs[epoch_key]
            if float(epoch.get("Accuracy", 0.0)) >= 1.0:
                solved_samples.append(int(epoch["NumTrainingSamples"]))
                break

    mean_solved_samples = float(np.mean(solved_samples)) if solved_samples else None
    median_solved_samples = float(np.median(solved_samples)) if solved_samples else None
    return {
        "path": log_path,
        "regex": _extract_regex_from_log_path(log_path),
        "mean_solved_samples": mean_solved_samples,
        "median_solved_samples": median_solved_samples,
        "solved_samples": solved_samples,
        "num_solved_runs": len(solved_samples),
    }


def _pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not points:
        return []

    front: List[Dict[str, Any]] = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            no_worse_x = other["avg_total_tokens"] <= point["avg_total_tokens"]
            no_worse_y = other["success_rate"] >= point["success_rate"]
            strictly_better = (
                other["avg_total_tokens"] < point["avg_total_tokens"]
                or other["success_rate"] > point["success_rate"]
            )
            if no_worse_x and no_worse_y and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(point)

    return sorted(front, key=lambda p: (p["avg_total_tokens"], -p["success_rate"]))


def _aggregate_method_points(points: List[Dict[str, Any]], method_label: str, color: str) -> Optional[Dict[str, Any]]:
    if not points:
        return None

    pooled_run_tokens: List[int] = []
    total_success_count = 0
    total_runs = 0
    for point in points:
        pooled_run_tokens.extend(point.get("run_token_totals", []))
        total_success_count += point.get("success_count", 0)
        total_runs += point.get("total_runs", 0)

    avg_total_tokens = float(np.mean(pooled_run_tokens)) if pooled_run_tokens else 0.0
    success_rate = (total_success_count / total_runs) if total_runs else 0.0
    return {
        "label": method_label,
        "color": color,
        "avg_total_tokens": avg_total_tokens,
        "success_rate": success_rate,
        "num_msgdicts": len(points),
        "total_runs": total_runs,
    }


def _plot_aggregated_method_points(ax, aggregated_points: List[Dict[str, Any]]):
    if not aggregated_points:
        return

    for point in aggregated_points:
        ax.scatter(
            point["avg_total_tokens"],
            point["success_rate"],
            s=70,
            alpha=0.9,
            color=point["color"],
            label=point["label"],
        )

    front = _pareto_front(aggregated_points)
    if front:
        ax.plot(
            [point["avg_total_tokens"] for point in front],
            [point["success_rate"] for point in front],
            color="black",
            linewidth=1.8,
            linestyle="--",
        )


def _format_token_tick(x: float, _: int) -> str:
    if abs(x) >= 1000:
        return f"{x/1000:.1f}k"
    return f"{int(x)}"


def _set_readable_token_axis(ax, x_values: List[float]):
    if not x_values:
        return

    xmin = min(x_values)
    xmax = max(x_values)
    if xmax <= 0:
        ax.set_xlim(left=0)
        return

    # Start from 0 when it does not compress the interesting region too much.
    # Otherwise zoom into the populated range and explicitly annotate that choice.
    should_start_at_zero = xmin <= 0.18 * xmax
    if should_start_at_zero:
        ax.set_xlim(left=0, right=xmax * 1.08)
    else:
        span = max(xmax - xmin, xmax * 0.08)
        ax.set_xlim(left=max(0, xmin - 0.12 * span), right=xmax + 0.12 * span)
        ax.text(
            0.02,
            0.02,
            "x-axis trimmed for readability",
            transform=ax.transAxes,
            fontsize=9,
            color="dimgray",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    ax.xaxis.set_major_formatter(FuncFormatter(_format_token_tick))


def _style_pareto_axis(ax, x_values: List[float]):
    ax.set_xlabel("Average Total Input Tokens Across Reruns")
    ax.set_ylabel("Success Run Ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    _set_readable_token_axis(ax, x_values)


def _plot_depth_overlay(
    ax,
    per_depth_points_by_method: Dict[str, List[Dict[str, Any]]],
    method_specs: List[Tuple[str, str, str, str]],
):
    depth_markers = {
        0: "o",
        1: "s",
        2: "^",
        3: "D",
        4: "P",
    }
    plotted_points: List[Dict[str, Any]] = []
    plotted_depths = set()

    for method_key, _, label, color in method_specs:
        method_points = sorted(
            per_depth_points_by_method.get(method_key, []),
            key=lambda p: p["depth"],
        )
        if not method_points:
            continue

        ax.plot(
            [point["avg_total_tokens"] for point in method_points],
            [point["success_rate"] for point in method_points],
            color=color,
            linewidth=1.3,
            alpha=0.7,
        )

        for point in method_points:
            depth = point["depth"]
            plotted_depths.add(depth)
            marker = depth_markers.get(depth, "o")
            ax.scatter(
                point["avg_total_tokens"],
                point["success_rate"],
                s=90,
                alpha=0.95,
                color=color,
                marker=marker,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.annotate(
                f"d={depth}",
                (point["avg_total_tokens"], point["success_rate"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8.5,
                color=color,
            )
            plotted_points.append(point)

    pareto_points = _pareto_front(plotted_points)
    if pareto_points:
        ax.plot(
            [point["avg_total_tokens"] for point in pareto_points],
            [point["success_rate"] for point in pareto_points],
            color="black",
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
        )

    setting_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=1.3,
            label=label,
        )
        for _, _, label, color in method_specs
    ]
    depth_handles = [
        Line2D(
            [0], [0],
            marker=depth_markers.get(depth, "o"),
            color="gray",
            markerfacecolor="white",
            markeredgecolor="gray",
            linewidth=0,
            label=f"depth={depth}",
        )
        for depth in sorted(plotted_depths)
    ]

    legend1 = ax.legend(handles=setting_handles, loc="lower right", title="Setting")
    ax.add_artist(legend1)
    ax.legend(handles=depth_handles, loc="center right", title="Depth")


def _build_method_specs(logs_root: str) -> List[Tuple[str, str, str, str]]:
    return [
        ("std_reg", os.path.join(logs_root, "std/reg"), "std/reg", "#1f77b4"),
        (
            "ce_agentic",
            os.path.join(logs_root, "ce/reg/agentic_reflection"),
            "ce/reg/agentic_reflection",
            "#d62728",
        ),
        (
            "ce_non_agentic",
            os.path.join(logs_root, "ce/reg/single_inference"),
            "ce/reg/single_inference",
            "#2ca02c",
        ),
    ]


def _task_label(task_type: str) -> str:
    return "SimplyRx" if task_type == "simplyrx" else "ExtRx"


def _method_depth_offsets() -> Dict[str, float]:
    return {
        "std_reg": -0.12,
        "ce_agentic": 0.0,
        "ce_non_agentic": 0.12,
    }


def plot_solve_rate_by_stardepth(
    logs_root: str, regex_list_path: str, outdir: str, task_type: str = "simplyrx",
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    metadata = _load_scaleup_regex_metadata(regex_list_path, task_type)
    depth_map = metadata["depth_map"]
    available_depths = metadata["depths"]
    task_label = _task_label(task_type)
    method_specs = _build_method_specs(logs_root)
    depth_markers = {
        "std_reg": "o",
        "ce_agentic": "s",
        "ce_non_agentic": "^",
    }
    depth_offsets = _method_depth_offsets()

    all_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, method_dir, _, _ in method_specs:
        points: List[Dict[str, Any]] = []
        for log_path in sorted(glob.glob(os.path.join(method_dir, "*.json"))):
            point = summarize_scaleup_log_for_solve_rate(log_path)
            point["depth"] = depth_map.get(point["regex"])
            points.append(point)
        all_points_by_method[method_key] = points

    plt.figure(figsize=(8.8, 5.8))
    ax = plt.gca()

    for method_key, _, label, color in method_specs:
        method_points = all_points_by_method[method_key]
        depth_values: List[int] = []
        mean_rates: List[float] = []

        for depth in available_depths:
            depth_points = [point for point in method_points if point.get("depth") == depth]
            if not depth_points:
                continue
            depth_values.append(depth)
            mean_rates.append(float(np.mean([point["success_rate"] for point in depth_points])))

        if not depth_values:
            continue

        ax.plot(
            depth_values,
            mean_rates,
            marker=depth_markers.get(method_key, "o"),
            color=color,
            linewidth=2.0,
            markersize=7,
            label=label,
        )
        for depth, rate in zip(depth_values, mean_rates):
            ax.annotate(
                f"{rate:.2f}",
                (depth, rate),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8.5,
                color=color,
            )

    ax.set_title(f"Solve Rate by Stardepth ({task_label})")
    ax.set_xlabel("StarDepth")
    ax.set_ylabel("Mean Solve Rate")
    ax.set_xticks(available_depths)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solve_rate_by_stardepth_compare.png"), dpi=300)
    plt.close()

    return all_points_by_method


def plot_solve_rate_by_states(
    logs_root: str, regex_list_path: str, outdir: str, task_type: str = "simplyrx",
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    metadata = _load_scaleup_regex_metadata(regex_list_path, task_type)
    depth_map = metadata["depth_map"]
    state_map = metadata["state_map"]
    available_states = metadata["states"]
    task_label = _task_label(task_type)
    method_specs = _build_method_specs(logs_root)
    state_markers = {
        "std_reg": "o",
        "ce_agentic": "s",
        "ce_non_agentic": "^",
    }

    all_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, method_dir, _, _ in method_specs:
        points: List[Dict[str, Any]] = []
        for log_path in sorted(glob.glob(os.path.join(method_dir, "*.json"))):
            point = summarize_scaleup_log_for_solve_rate(log_path)
            point["depth"] = depth_map.get(point["regex"])
            point["state"] = state_map.get(point["regex"])
            points.append(point)
        all_points_by_method[method_key] = points

    plt.figure(figsize=(8.8, 5.8))
    ax = plt.gca()

    for method_key, _, label, color in method_specs:
        method_points = all_points_by_method[method_key]
        state_values: List[int] = []
        mean_rates: List[float] = []

        for state in available_states:
            state_points = [point for point in method_points if point.get("state") == state]
            if not state_points:
                continue
            state_values.append(state)
            mean_rates.append(float(np.mean([point["success_rate"] for point in state_points])))

        if not state_values:
            continue

        ax.plot(
            state_values,
            mean_rates,
            marker=state_markers.get(method_key, "o"),
            color=color,
            linewidth=2.0,
            markersize=7,
            label=label,
        )
        for state, rate in zip(state_values, mean_rates):
            ax.annotate(
                f"{rate:.2f}",
                (state, rate),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8.5,
                color=color,
            )

    ax.set_title(f"Solve Rate by #States ({task_label})")
    ax.set_xlabel("#States")
    ax.set_ylabel("Mean Solve Rate")
    ax.set_xticks(available_states)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solve_rate_by_states_compare.png"), dpi=300)
    plt.close()

    return all_points_by_method


def plot_mean_samples_by_stardepth(
    logs_root: str, regex_list_path: str, outdir: str, task_type: str = "simplyrx",
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    metadata = _load_scaleup_regex_metadata(regex_list_path, task_type)
    depth_map = metadata["depth_map"]
    available_depths = metadata["depths"]
    task_label = _task_label(task_type)
    method_specs = _build_method_specs(logs_root)
    depth_markers = {
        "std_reg": "o",
        "ce_agentic": "s",
        "ce_non_agentic": "^",
    }
    depth_offsets = _method_depth_offsets()

    all_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, method_dir, _, _ in method_specs:
        points: List[Dict[str, Any]] = []
        for log_path in sorted(glob.glob(os.path.join(method_dir, "*.json"))):
            point = summarize_scaleup_log_for_sample_efficiency(log_path)
            point["depth"] = depth_map.get(point["regex"])
            points.append(point)
        all_points_by_method[method_key] = points

    plt.figure(figsize=(8.8, 5.8))
    ax = plt.gca()

    for method_key, _, label, color in method_specs:
        method_points = all_points_by_method[method_key]
        depth_values: List[int] = []
        depth_level_mean_samples: List[float] = []
        depth_level_std_samples: List[float] = []

        for depth in available_depths:
            depth_sample_pool: List[int] = []
            for point in method_points:
                if point.get("depth") != depth:
                    continue
                depth_sample_pool.extend(point.get("solved_samples", []))

            if not depth_sample_pool:
                continue
            depth_values.append(depth)
            depth_level_mean_samples.append(float(np.mean(depth_sample_pool)))
            depth_level_std_samples.append(float(np.std(depth_sample_pool)))

        if not depth_values:
            continue

        shifted_depth_values = [
            depth + depth_offsets.get(method_key, 0.0) for depth in depth_values
        ]

        ax.errorbar(
            shifted_depth_values,
            depth_level_mean_samples,
            yerr=depth_level_std_samples,
            marker=depth_markers.get(method_key, "o"),
            color=color,
            linewidth=2.0,
            markersize=7,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            alpha=0.9,
            label=label,
        )
        for shifted_depth, mean_samples in zip(
            shifted_depth_values, depth_level_mean_samples
        ):
            ax.annotate(
                f"{mean_samples:.0f}",
                (shifted_depth, mean_samples),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color=color,
            )

    ax.set_title(f"Average #Solved Samples by Stardepth ({task_label})")
    ax.set_xlabel("StarDepth")
    ax.set_ylabel("Mean #Solved Samples")
    ax.set_xticks(available_depths)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(_format_token_tick))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_samples_by_stardepth_compare.png"), dpi=300)
    plt.close()

    return all_points_by_method


def _aggregate_pooled_median_by_cell(
    points: List[Dict[str, Any]],
    states: List[int],
    depths: List[int],
) -> np.ndarray:
    z = np.full((len(depths), len(states)), np.inf, dtype=float)
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    depth_to_idx = {depth: idx for idx, depth in enumerate(depths)}

    pooled_samples: Dict[Tuple[int, int], List[int]] = {}
    for point in points:
        state = point.get("state")
        depth = point.get("depth")
        if state is None or depth is None:
            continue
        pooled_samples.setdefault((state, depth), []).extend(point.get("solved_samples", []))

    for (state, depth), samples in pooled_samples.items():
        if not samples:
            continue
        z[depth_to_idx[depth], state_to_idx[state]] = float(np.median(samples))

    return z


def plot_3d_median_samples(
    logs_root: str, regex_list_path: str, outdir: str, task_type: str = "simplyrx",
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    metadata = _load_scaleup_regex_metadata(regex_list_path, task_type)
    depth_map = metadata["depth_map"]
    state_map = metadata["state_map"]
    available_depths = metadata["depths"]
    available_states = metadata["states"]
    task_label = _task_label(task_type)
    method_specs = _build_method_specs(logs_root)

    all_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    z_values_by_method: Dict[str, np.ndarray] = {}

    for method_key, method_dir, _, _ in method_specs:
        points: List[Dict[str, Any]] = []
        for log_path in sorted(glob.glob(os.path.join(method_dir, "*.json"))):
            point = summarize_scaleup_log_for_sample_efficiency(log_path)
            point["depth"] = depth_map.get(point["regex"])
            point["state"] = state_map.get(point["regex"])
            points.append(point)
        all_points_by_method[method_key] = points
        z_values_by_method[method_key] = _aggregate_pooled_median_by_cell(
            points, available_states, available_depths
        )

    finite_values = [
        value
        for z in z_values_by_method.values()
        for value in z[np.isfinite(z)]
    ]
    vmin = float(min(finite_values)) if finite_values else 0.0
    vmax = float(max(finite_values)) if finite_values else 1.0

    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(16, 5.6), constrained_layout=True)
    cmap = plt.colormaps.get_cmap("YlOrRd_r")
    norm = Normalize(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(np.array(available_states), np.array(available_depths))
    axes = []

    for subplot_idx, (method_key, _, label, _) in enumerate(method_specs, start=1):
        ax = fig.add_subplot(1, 3, subplot_idx, projection="3d")
        axes.append(ax)
        Z = z_values_by_method[method_key]
        Z_masked = np.ma.masked_invalid(Z)

        ax.plot_surface(
            X,
            Y,
            Z_masked,
            facecolors=cmap(norm(Z_masked.filled(vmin))),
            rstride=1,
            cstride=1,
            linewidth=0.35,
            antialiased=True,
            shade=False,
            alpha=0.95,
        )

        for depth_idx, depth in enumerate(available_depths):
            for state_idx, state in enumerate(available_states):
                value = Z[depth_idx, state_idx]
                if not np.isfinite(value):
                    continue
                ax.text(
                    state,
                    depth,
                    value,
                    f"{value:.0f}",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )

        ax.set_title(label)
        ax.set_xlabel("#States")
        ax.set_ylabel("StarDepth")
        ax.set_zlabel("Median #Solved Samples")
        ax.set_xticks(available_states)
        ax.set_yticks(available_depths)
        ax.set_zlim(vmin, vmax * 1.03 if vmax > 0 else 1.0)
        ax.view_init(elev=24, azim=-55)
        ax.set_box_aspect((1.2, 1.0, 0.8))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    fig.suptitle(f"Median #Solved Samples Surface ({task_label})", fontsize=13)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.array(finite_values) if finite_values else np.array([0.0]))
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.72, pad=0.04)
    cbar.set_label("Median #Solved Samples")

    plt.savefig(os.path.join(outdir, "median_samples_surface_compare.png"), bbox_inches="tight")
    plt.close()

    return all_points_by_method


def plot_median_samples_heatmap(
    logs_root: str, regex_list_path: str, outdir: str, task_type: str = "simplyrx",
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    metadata = _load_scaleup_regex_metadata(regex_list_path, task_type)
    depth_map = metadata["depth_map"]
    state_map = metadata["state_map"]
    available_depths = metadata["depths"]
    available_states = metadata["states"]
    task_label = _task_label(task_type)
    method_specs = _build_method_specs(logs_root)

    all_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    z_values_by_method: Dict[str, np.ndarray] = {}

    for method_key, method_dir, _, _ in method_specs:
        points: List[Dict[str, Any]] = []
        for log_path in sorted(glob.glob(os.path.join(method_dir, "*.json"))):
            point = summarize_scaleup_log_for_sample_efficiency(log_path)
            point["depth"] = depth_map.get(point["regex"])
            point["state"] = state_map.get(point["regex"])
            points.append(point)
        all_points_by_method[method_key] = points
        z_values_by_method[method_key] = _aggregate_pooled_median_by_cell(
            points, available_states, available_depths
        )

    finite_values = [
        value
        for z in z_values_by_method.values()
        for value in z[np.isfinite(z)]
    ]
    vmax = float(max(finite_values)) if finite_values else 1.0
    vmin = float(min(finite_values)) if finite_values else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    cmap = plt.colormaps.get_cmap("YlOrRd_r")
    images = []

    for ax, (method_key, _, label, _) in zip(axes, method_specs):
        z = z_values_by_method[method_key]
        z_plot = np.where(np.isfinite(z), z, vmax)
        image = ax.imshow(
            z_plot,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)

        ax.set_title(label)
        ax.set_xlabel("#States")
        ax.set_ylabel("StarDepth")
        ax.set_xticks(range(len(available_states)))
        ax.set_xticklabels(available_states)
        ax.set_yticks(range(len(available_depths)))
        ax.set_yticklabels(available_depths)

        for row_idx, depth in enumerate(available_depths):
            for col_idx, state in enumerate(available_states):
                value = z[row_idx, col_idx]
                text = "inf" if not np.isfinite(value) else f"{value:.0f}"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    fig.suptitle(f"Median #Solved Samples Heatmap ({task_label})", fontsize=13)
    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Median #Solved Samples")

    plt.savefig(os.path.join(outdir, "median_samples_heatmap_compare.png"), bbox_inches="tight")
    plt.close()

    return all_points_by_method


def plot_pareto(
    logs_root: str, regex_list_path: str, outdir: str, mkey: str, task_type: str = "simplyrx",
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    metadata = _load_scaleup_regex_metadata(regex_list_path, task_type)
    depth_map = metadata["depth_map"]
    available_depths = metadata["depths"]
    token_counter = _load_token_counter(mkey)
    task_label = _task_label(task_type)
    method_specs = _build_method_specs(logs_root)

    all_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, method_dir, _, _ in method_specs:
        points: List[Dict[str, Any]] = []
        for log_path in sorted(glob.glob(os.path.join(method_dir, "*.json"))):
            point = summarize_scaleup_log_for_pareto(
                log_path,
                mkey=mkey,
                token_counter=token_counter,
            )
            point["depth"] = depth_map.get(point["regex"])
            points.append(point)
        all_points_by_method[method_key] = points

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    aggregated_points: List[Dict[str, Any]] = []
    for method_key, _, label, color in method_specs:
        aggregated_point = _aggregate_method_points(all_points_by_method[method_key], label, color)
        if aggregated_point is not None:
            aggregated_points.append(aggregated_point)

    _plot_aggregated_method_points(ax, aggregated_points)
    _style_pareto_axis(ax, [point["avg_total_tokens"] for point in aggregated_points])
    ax.set_title(f"Pareto Front for {task_label} Scaleup (task average)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_task_average.png"), dpi=300)
    plt.close()

    per_depth_points_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, _, label, color in method_specs:
        method_depth_points: List[Dict[str, Any]] = []
        for depth in available_depths:
            depth_points = [
                point
                for point in all_points_by_method[method_key]
                if point.get("depth") == depth
            ]
            aggregated_point = _aggregate_method_points(depth_points, label, color)
            if aggregated_point is not None:
                aggregated_point["depth"] = depth
                method_depth_points.append(aggregated_point)
        per_depth_points_by_method[method_key] = method_depth_points

    plt.figure(figsize=(9.5, 6.5))
    ax = plt.gca()
    _plot_depth_overlay(ax, per_depth_points_by_method, method_specs)
    overlay_x_values = [
        point["avg_total_tokens"]
        for method_points in per_depth_points_by_method.values()
        for point in method_points
    ]
    _style_pareto_axis(ax, overlay_x_values)
    ax.set_title(f"Pareto Front for {task_label} Scaleup (all depths together)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_depth_overlay.png"), dpi=300)
    plt.close()

    return all_points_by_method

def plot_loss_curve(losses, outdir, name):
    epochs = len(losses)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), losses, label='Loss', color='blue', linewidth=2)
    plt.title('Gradient Descent Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(epochs-1, color='red', linestyle='--', label="Final Epoch", linewidth=1)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=500)

def plot_accuracy_curve(num_samples, accs, outdir, name):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump({"num_samples": num_samples, "accs": accs}, f)

    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, accs, label='Accuracy', color='blue', linewidth=2)
    plt.title('Accuracy against #TrainingRound Curve', fontsize=16)
    plt.xlabel('#TrainingRounds', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(num_samples[-1], color='red', linestyle='--', label="Total Train Round", linewidth=1)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=500)

def smooth(y, window_size=5):
    if window_size < 2:
        return y.copy()
    pad = window_size // 2
    y_padded = np.pad(y, pad_width=pad, mode='reflect')
    window = np.ones(window_size) / window_size
    return np.convolve(y_padded, window, mode='valid')

def plot_3dSurface(Z):
    dfa_sizes = np.arange(3, 11)      # 3..10 (8)
    star_depths = np.arange(0, 5)     # 0..4  (5)

    Y, X = np.meshgrid(dfa_sizes, star_depths)  # X: DFA size, Y: star depth

    Z_center = np.median(Z, axis=2)                 # or Z.mean(axis=2)
    Z_min = Z.min(axis=2)
    Z_max = Z.max(axis=2)

    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "figure.dpi": 120,       # interactive
        "savefig.dpi": 300,      # export
    })

    fig = plt.figure(figsize=(7.8, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Colormap: low=green, high=red
    cmap = plt.colormaps.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=Z_center.min(), vmax=Z_center.max())

    ax.plot_surface(
        X, Y, Z_center,
        facecolors=cmap(norm(Z_center)),
        rstride=1, cstride=1,
        linewidth=0.25,
        antialiased=True,
        shade=False
    )

    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):    
            ax.plot(
                [X[i, j], X[i, j]],
                [Y[i, j], Y[i, j]],
                [Z_min[i, j], Z_max[i, j]],
                linewidth=1.0,
                alpha=0.75
            )

    # Colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z_center)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("Samples needed (lower = easier)")

    ax.set_xlabel("Star depth")
    ax.set_ylabel("Minimal DFA size (states)")
    ax.set_zlabel("Sample complexity")

    # Ticks: keep sparse & readable
    ax.set_xticks(star_depths)
    ax.set_yticks(dfa_sizes)

    # View: choose a stable, readable angle
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1.25, 1.0, 0.7))

    # Optional: remove pane fills for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.savefig("accuracy_curves/surface_paper.pdf", bbox_inches="tight")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot_type",
        type=str,
        default="pareto",
        choices=[
            "pareto",
            "solve_rate_by_stardepth",
            "solve_rate_by_states",
            "mean_samples_by_stardepth",
            "median_samples_surface",
            "median_samples_heatmap",
        ],
    )
    parser.add_argument("--task_type", type=str, default="simplyrx", choices=["simplyrx", "extrx"])
    parser.add_argument("--logs_root", type=str, default=None)
    parser.add_argument("--regex_list_path", type=str, default="datasets/scaleup/regex_list.json")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--mkey", type=str, default="gpt-oss")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.logs_root is None:
        args.logs_root = f"logs/scaleup/icl_gen_{args.task_type}/model={args.mkey}"
    if args.outdir is None:
        args.outdir = f"accuracy_curves/scaleup/icl_gen_{args.task_type}/model={args.mkey}"

    if args.plot_type == "pareto":
        plot_pareto(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            mkey=args.mkey,
            task_type=args.task_type,
        )
    elif args.plot_type == "solve_rate_by_stardepth":
        plot_solve_rate_by_stardepth(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            task_type=args.task_type,
        )
    elif args.plot_type == "solve_rate_by_states":
        plot_solve_rate_by_states(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            task_type=args.task_type,
        )
    elif args.plot_type == "mean_samples_by_stardepth":
        plot_mean_samples_by_stardepth(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            task_type=args.task_type,
        )
    elif args.plot_type == "median_samples_surface":
        plot_3d_median_samples(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            task_type=args.task_type,
        )
    elif args.plot_type == "median_samples_heatmap":
        plot_median_samples_heatmap(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            task_type=args.task_type,
        )
