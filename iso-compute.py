#!/usr/bin/env python3
"""Iso-compute inference for agentic_no_repair_loop.

This script runs the same CE/reflection rounds as `agentic_no_repair_loop`, but
after the first retry in each round it lets the model "continue thinking" until
either:

1. the produced regex is equivalent to the target; or
2. the cumulative input+output tokens spent by continue-thinking calls in this
   round exceeds T / r.

Here r is the number of rounds used by a previous agentic_no_repair_loop log for
the same setting, and T is the extra token budget spent by retries after the
first retry in the matching full agentic_reflection log.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataset import generate_dataset
from keysecrets import api_key
from learner import LearnerForICLGen
from modeling.llm import is_vllm_model, load_model_and_tokenizer
from prompting.rl import (
    EXTRX_CLUSTRED_CE_INSTR,
    EXTRX_REGULARIZATION,
    EXTRX_SIGMA,
    SIMPLYRX_CLUSTRED_CE_INSTR,
    SIMPLYRX_REGULARIZATION,
)
from tasks.rl import ExtRegularLanguage, SimplyRegularLanguage
from teacher import Teacher
from train_icl_gen import (
    build_prompt_template,
    build_reflection_prompt,
    extract_ans,
    extract_reasoning,
    savejson,
    summarize_label_counts,
    train_data_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="simplyrx", choices=["simplyrx", "extrx"])
    parser.add_argument("--regex", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--mkey", type=str, default="gpt-oss")
    parser.add_argument("--tot_train_size", type=int, default=384)
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--start_size", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rerun", type=int, default=1)
    parser.add_argument("--use_reg", default=False, action="store_true")
    parser.add_argument("--use_ce", default=True, action="store_true")
    parser.add_argument("--ce_epochs", type=int, default=12)
    parser.add_argument("--ce_start_size", type=int, default=8)
    parser.add_argument("--ce_batch_size", type=int, default=250)
    parser.add_argument("--ce_clustered", default=False, action="store_true")
    parser.add_argument(
        "--ce_generation_mode",
        type=str,
        default="dfs",
        choices=["dfs", "bfs", "shortest", "random"],
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "naive_prompt",
            "only_input_instr",
            "only_output_instr",
            "input_output_instr",
            "zero_prompt",
        ],
    )
    parser.add_argument("--indir", type=str, default="datasets/scaleup/regex_datasets")
    parser.add_argument("--outdir", type=str, default="logs/scaleup")
    parser.add_argument(
        "--reference_logdir",
        type=str,
        default=None,
        help="Root containing old logs. Defaults to --outdir.",
    )
    parser.add_argument(
        "--iso_outdir",
        type=str,
        default="logs/iso_compute",
        help="Root for iso-compute logs.",
    )
    parser.add_argument(
        "--continue_max_calls",
        type=int,
        default=16,
        help="Safety cap on continue-thinking calls per round.",
    )
    return parser.parse_args()


def token_count(tokenizer, text) -> int:
    if text is None:
        return 0
    text = str(text)
    if tokenizer is None:
        # Fallback for API models without a tokenizer. It is intentionally a
        # rough, monotone proxy, used only when exact tokenization is unavailable.
        return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def log_total_tokens(msg, tokenizer) -> int:
    return token_count(tokenizer, msg.get("Prompt", "")) + token_count(
        tokenizer, msg.get("Response", "")
    )


def ce_log_filename(args, mode: str, include_generation_mode: bool = True) -> str:
    ce_mode = f"_{args.ce_generation_mode}" if include_generation_mode else ""
    ce_clustered = "_clustered" if args.ce_clustered else ""
    return (
        f"msgdict_regex={args.regex}_ceEpochs={args.ce_epochs}"
        f"_ceBatch={args.ce_batch_size}{ce_mode}{ce_clustered}.json"
    )


def candidate_reference_paths(args, reasoning_mode: str) -> list[Path]:
    root = Path(args.reference_logdir or args.outdir) / f"icl_gen_{args.task_type}"
    reg_dir = "reg" if args.use_reg else "noreg"
    base = root / f"model={args.mkey}" / "ce" / reg_dir / reasoning_mode

    filenames = [ce_log_filename(args, reasoning_mode, include_generation_mode=True)]
    # Historical dfs logs often omitted the explicit "_dfs" suffix.
    if args.ce_generation_mode == "dfs":
        filenames.append(ce_log_filename(args, reasoning_mode, include_generation_mode=False))

    paths = []
    for filename in filenames:
        paths.append(base / filename)
        paths.append(base / args.ce_generation_mode / filename)
    return paths


def find_reference_log(args, reasoning_mode: str) -> Path:
    for path in candidate_reference_paths(args, reasoning_mode):
        if path.exists():
            return path
    candidates = "\n".join(str(p) for p in candidate_reference_paths(args, reasoning_mode))
    raise FileNotFoundError(
        f"Cannot find reference log for {reasoning_mode}. Tried:\n{candidates}"
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_run_summary(log_data, runid: int):
    summary = log_data.get("summary") or {}
    return summary.get(f"run-{runid}") or summary.get("run-0") or {}


def get_round_count(no_repair_log, runid: int, ce_epochs: int) -> int:
    run_summary = get_run_summary(no_repair_log, runid)
    rounds = run_summary.get("epochs")
    if rounds is None:
        return ce_epochs
    return max(1, int(rounds))


def extra_retry_tokens(full_log, runid: int, tokenizer) -> int:
    run = full_log.get(f"run-{runid}") or full_log.get("run-0") or {}
    total = 0
    for epoch_key in sorted(run, key=lambda k: int(k.split("-")[-1]) if k.startswith("epoch-") else 10**9):
        if not epoch_key.startswith("epoch-"):
            continue
        logs = run.get(epoch_key, {}).get("Logs", [])
        for msg in logs[1:]:
            total += log_total_tokens(msg, tokenizer)
    return total


def build_iso_config_name(args) -> str:
    outdir = Path(args.iso_outdir) / f"icl_gen_{args.task_type}"
    config_name = outdir / f"model={args.mkey}" / "ce"
    config_name = config_name / ("reg" if args.use_reg else "noreg")
    config_name = config_name / "agentic_no_repair_loop_iso_compute"
    if args.prompt_mode != "full":
        config_name = config_name / f"prompt={args.prompt_mode}"
    ce_mode = f"_{args.ce_generation_mode}"
    ce_clustered = "_clustered" if args.ce_clustered else ""
    filename = (
        f"msgdict_regex={args.regex}_ceEpochs={args.ce_epochs}"
        f"_ceBatch={args.ce_batch_size}{ce_mode}{ce_clustered}.json"
    )
    return str(config_name / filename)


def make_continue_prompt(previous_msg, token_budget, used_tokens):
    previous_response = previous_msg.get("Response") or ""
    previous_prediction = previous_msg.get("Prediction") or "(none)"
    previous_reasoning = previous_msg.get("Reasoning") or "(none)"
    return (
        previous_msg.get("Prompt", "")
        + "\n\nCONTINUE THINKING INSTRUCTION\n"
        + "- Continue thinking from your previous attempt. You are not given any new counterexamples or repair examples.\n"
        + "- Re-check the structure, syntax, and edge cases using only the prompt above and your previous answer.\n"
        + "- You may keep the same regex or revise it, but you must provide a complete answer again.\n"
        + "- Output one concise <reasoning>...</reasoning> block and one final <ans>...</ans> block.\n"
        + f"- Extra continue-thinking token budget for this round: {token_budget:.1f}; already used: {used_tokens}.\n\n"
        + "Previous response:\n"
        + previous_response
        + "\n\nPrevious extracted reasoning:\n"
        + previous_reasoning
        + "\nPrevious extracted regex:\n"
        + previous_prediction
        + "\n\nContinue now.\n"
    )


def run_iso_episode(
    *,
    args,
    task,
    data,
    teacher,
    learner,
    prompt_template,
    prompt_kwargs,
    per_round_budget,
    msgdict,
    config_name,
    runid,
):
    regex = task.regex_str
    _, fst_gt, _ = task.regex_to_pynini_via_pyformlang(regex)
    agg_train_ex, agg_train_labels = [], []
    current_guess = None
    current_guess_reasoning = None
    epoch_results = []
    accs = []

    for epoch in range(args.ce_epochs):
        reflection_prompt = ""
        try:
            train_ex, train_labels = teacher.generate_counterexamples(
                bs=args.ce_batch_size,
                regex_gt=regex,
                regex_gen=current_guess,
                clustered=args.ce_clustered,
                generation_mode=args.ce_generation_mode,
            )
            reflection_prompt = build_reflection_prompt(
                current_guess_reasoning,
                current_guess,
                train_ex,
                train_labels,
            )
        except Exception as e:
            print(
                f"Cannot generate counterexamples at epoch {epoch}: {e}, "
                f"use {args.ce_start_size} random examples instead."
            )
            train_ex = data["train_ex"][epoch * args.ce_start_size : (epoch + 1) * args.ce_start_size]
            train_labels = data["train_labels"][epoch * args.ce_start_size : (epoch + 1) * args.ce_start_size]

        agg_train_ex += train_ex
        agg_train_labels += train_labels
        train_p = "\n".join(
            [
                train_data_template.format(ex, label)
                for ex, label in zip(agg_train_ex, agg_train_labels)
            ]
        )

        iter_prompt_kwargs = dict(prompt_kwargs)
        iter_prompt_kwargs["agentic_reflection_instr"] = reflection_prompt
        msg = learner.generate(
            prompt_template=prompt_template,
            train_prompt=train_p,
            prompt_format_kwargs=iter_prompt_kwargs,
            temp=args.temp,
            answer_extractor=extract_ans,
            reasoning_extractor=extract_reasoning,
        )
        print(
            f"Epoch {epoch}, initial retry\nPrediction: {msg['Prediction']}\nReasoning: {msg['Reasoning']}",
            flush=True,
        )
        msg = teacher.judge_regex(
            msg=msg,
            fst_gt=fst_gt,
            train_ex=agg_train_ex,
            train_labels=agg_train_labels,
            eval_ex=data["eval_ex"],
            eval_labels=data["eval_labels"],
        )
        msg["RetryIndex"] = 0
        msg["IsoComputeContinue"] = False
        logs = [msg]
        acc = 1 if msg.get("Equivalent") else 0

        continue_used_tokens = 0
        continue_calls = 0
        previous_msg = msg
        while (
            not previous_msg.get("Equivalent")
            and per_round_budget > 0
            and continue_calls < args.continue_max_calls
        ):
            if continue_used_tokens > per_round_budget:
                break
            continue_prompt = build_continue_prompt(
                previous_msg,
                token_budget=per_round_budget,
                used_tokens=continue_used_tokens,
            )
            cont_msg = learner.generate(
                prompt_template="{0}",
                train_prompt=continue_prompt,
                prompt_format_kwargs={},
                temp=args.temp,
                answer_extractor=extract_ans,
                reasoning_extractor=extract_reasoning,
            )
            cont_msg = teacher.judge_regex(
                msg=cont_msg,
                fst_gt=fst_gt,
                train_ex=agg_train_ex,
                train_labels=agg_train_labels,
                eval_ex=data["eval_ex"],
                eval_labels=data["eval_labels"],
            )
            continue_calls += 1
            cont_msg["RetryIndex"] = continue_calls
            cont_msg["IsoComputeContinue"] = True
            cont_msg["IsoComputeTokenCost"] = log_total_tokens(cont_msg, learner.tokenizer)
            continue_used_tokens += cont_msg["IsoComputeTokenCost"]
            cont_msg["IsoComputeCumulativeTokens"] = continue_used_tokens
            cont_msg["IsoComputeRoundBudget"] = per_round_budget
            logs.append(cont_msg)
            previous_msg = cont_msg
            print(
                f"Epoch {epoch}, continue {continue_calls}, "
                f"tokens {continue_used_tokens}/{per_round_budget:.1f}, "
                f"equiv={bool(cont_msg.get('Equivalent'))}\n"
                f"Prediction: {cont_msg.get('Prediction')}",
                flush=True,
            )
            if cont_msg.get("Equivalent"):
                acc = 1
                break

        best_msg = next((m for m in reversed(logs) if m.get("Prediction") is not None), logs[-1])
        current_guess = best_msg.get("Prediction")
        current_guess_reasoning = best_msg.get("Reasoning")
        epoch_result = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "CurrentGuess": current_guess,
            "CurrentGuessReasoning": current_guess_reasoning,
            "IsoComputeRoundBudget": per_round_budget,
            "IsoComputeContinueTokens": continue_used_tokens,
            "IsoComputeContinueCalls": continue_calls,
            "Logs": logs,
        }
        msgdict[f"run-{runid}"][f"epoch-{epoch}"] = epoch_result
        savejson(msgdict, config_name)
        epoch_results.append(epoch_result)
        accs.append(acc)
        print(f"Accuracy at epoch {epoch}: {acc}", flush=True)
        if acc == 1:
            break

    return {
        "epoch_results": epoch_results,
        "epochs_completed": len(epoch_results),
        "final_num_samples": summarize_label_counts(agg_train_labels),
        "final_accuracy": accs[-1] if accs else 0.0,
        "current_guess": current_guess,
        "current_guess_reasoning": current_guess_reasoning,
    }


def main() -> None:
    args = parse_args()
    args.reasoning_mode = "agentic_no_repair_loop"

    use_vllm = is_vllm_model(args.mkey)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not use_vllm and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.task_type == "extrx":
        task = ExtRegularLanguage(args.regex, args.max_length, alphabet=EXTRX_SIGMA)
        prompt_kwargs = {
            "sigma": EXTRX_SIGMA,
            "regularization_instr": EXTRX_REGULARIZATION if args.use_reg else "",
            "agentic_reflection_instr": "",
            "clustered_ce_instr": EXTRX_CLUSTRED_CE_INSTR if args.ce_clustered else "",
        }
    else:
        task = SimplyRegularLanguage(args.regex, args.max_length)
        prompt_kwargs = {
            "regularization_instr": SIMPLYRX_REGULARIZATION if args.use_reg else "",
            "agentic_reflection_instr": "",
            "clustered_ce_instr": SIMPLYRX_CLUSTRED_CE_INSTR if args.ce_clustered else "",
        }
    prompt_template = build_prompt_template(args.task_type, args.prompt_mode)

    no_repair_path = find_reference_log(args, "agentic_no_repair_loop")
    full_path = find_reference_log(args, "agentic_reflection")
    print(f"Using no-repair reference: {no_repair_path}")
    print(f"Using full-agentic reference: {full_path}")
    no_repair_log = load_json(no_repair_path)
    full_log = load_json(full_path)

    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
    teacher = Teacher(task)
    learner = LearnerForICLGen(args.mkey, model, tokenizer, task)

    generate_dataset(args, task_type=args.task_type, outdir=args.indir)
    dataset_path = Path(args.indir) / (
        f"regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json"
    )
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    config_name = build_iso_config_name(args)
    msgdict = {
        "summary": None,
        "iso_compute": {
            "no_repair_reference": str(no_repair_path),
            "full_agentic_reference": str(full_path),
            "budget_rule": "per_round_budget = extra_retry_tokens_from_full_agentic / rounds_from_no_repair",
        },
    }
    finish_states = {}
    print(f"Starting iso-compute training for regex: {args.regex} with model {args.mkey}")

    for runid in range(args.rerun):
        rounds = get_round_count(no_repair_log, runid, args.ce_epochs)
        extra_tokens = extra_retry_tokens(full_log, runid, tokenizer)
        per_round_budget = extra_tokens / rounds if rounds > 0 else 0.0
        print(
            f"=== Rerun {runid}: r={rounds}, T={extra_tokens}, "
            f"T/r={per_round_budget:.1f} ===",
            flush=True,
        )
        msgdict[f"run-{runid}"] = {
            "IsoComputeReferenceRounds": rounds,
            "IsoComputeFullExtraTokens": extra_tokens,
            "IsoComputePerRoundBudget": per_round_budget,
        }
        savejson(msgdict, config_name)

        episode_result = run_iso_episode(
            args=args,
            task=task,
            data=data,
            teacher=teacher,
            learner=learner,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            per_round_budget=per_round_budget,
            msgdict=msgdict,
            config_name=config_name,
            runid=runid,
        )

        finish_states[f"run-{runid}"] = {
            "epochs": episode_result["epochs_completed"],
            "final_num_samples": episode_result["final_num_samples"],
            "final_accuracy": episode_result["final_accuracy"],
            "IsoComputeReferenceRounds": rounds,
            "IsoComputeFullExtraTokens": extra_tokens,
            "IsoComputePerRoundBudget": per_round_budget,
        }
        msgdict["summary"] = finish_states
        savejson(msgdict, config_name)


if __name__ == "__main__":
    main()
