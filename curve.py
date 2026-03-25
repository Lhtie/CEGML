import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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


def _build_scaleup_simplyrx_depth_map(regex_list_path: str) -> Dict[str, int]:
    with open(regex_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    depth_map: Dict[str, int] = {}
    for state_group in data["simplyrx"]:
        for depth_group in state_group["regex_list"]:
            depth = depth_group["Stardepth"]
            for regex in depth_group["regex_list"]:
                depth_map[regex] = depth
    return depth_map


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


def plot_pareto(
    logs_root: str, regex_list_path: str, outdir: str, mkey: str,
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(outdir, exist_ok=True)

    depth_map = _build_scaleup_simplyrx_depth_map(regex_list_path)
    token_counter = _load_token_counter(mkey)

    method_specs: List[Tuple[str, str, str, str]] = [
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

    for depth in range(5):
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        aggregated_points: List[Dict[str, Any]] = []
        for method_key, _, label, color in method_specs:
            depth_points = [
                point
                for point in all_points_by_method[method_key]
                if point.get("depth") == depth
            ]
            aggregated_point = _aggregate_method_points(depth_points, label, color)
            if aggregated_point is not None:
                aggregated_points.append(aggregated_point)

        _plot_aggregated_method_points(ax, aggregated_points)

        ax.set_title(f"Pareto Front for SimplyRx Scaleup (gpt-oss, depth={depth})")
        ax.set_xlabel("Average Total Input Tokens Across Reruns")
        ax.set_ylabel("Success Run Ratio")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"pareto_depth={depth}.png"), dpi=300)
        plt.close()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    aggregated_points = []
    for method_key, _, label, color in method_specs:
        aggregated_point = _aggregate_method_points(all_points_by_method[method_key], label, color)
        if aggregated_point is not None:
            aggregated_points.append(aggregated_point)

    _plot_aggregated_method_points(ax, aggregated_points)

    ax.set_title("Pareto Front for SimplyRx Scaleup (gpt-oss, all depths)")
    ax.set_xlabel("Average Total Input Tokens Across Reruns")
    ax.set_ylabel("Success Run Ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_all_depths.png"), dpi=300)
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
    parser.add_argument("--plot_type", type=str, default="pareto", choices=["pareto"])
    parser.add_argument("--logs_root", type=str, default="logs/scaleup/icl_gen_simplyrx/model=gpt-oss")
    parser.add_argument("--regex_list_path", type=str, default="datasets/scaleup/regex_list.json")
    parser.add_argument("--outdir", type=str, default="accuracy_curves/scaleup/icl_gen_simplyrx/model=gpt-oss")
    parser.add_argument("--mkey", type=str, default="gpt-oss")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.plot_type == "pareto":
        plot_pareto(
            logs_root=args.logs_root,
            regex_list_path=args.regex_list_path,
            outdir=args.outdir,
            mkey=args.mkey,
        )
