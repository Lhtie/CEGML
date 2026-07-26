"""LLM-based LTL inference.

This entrypoint intentionally mirrors ``train_icl_gen.py`` so the shared
episode, learner, logging, and retry machinery can be extracted later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Mapping

import numpy as np
import torch

from keysecrets import api_key
from learner import LearnerForICLGen
from modeling.llm import is_vllm_model, load_model_and_tokenizer
from prompting.ltl import (
    AGENTIC_REFLECTION_INSTR,
    AGENTIC_REPAIR_INSTR,
    DIRECT_OUTPUT_FORMAT_INSTR,
    INFERENCE_STRATEGY,
    INPUT_INSTR,
    OUTPUT_FORMAT_INSTR,
    PROMPT_TEMPLATE,
    REGULARIZATION,
    SYNTAX_INSTR,
    TASK_INSTR,
    TRAINING_DATA_INSTR,
)
from tasks.ltl import BlackSolver, LTLNode, LTLTask
from train_icl_gen import (
    extract_all_ans,
    extract_ans,
    extract_reasoning,
    log_scaling,
    savejson,
    summarize_label_counts,
    uses_candidate_repair,
    uses_reflection_prompt,
    uses_repair_loop,
)


def format_state(state: Mapping[str, bool]) -> str:
    values = ", ".join(
        f"{variable}={int(bool(value))}"
        for variable, value in state.items()
    )
    return "{" + values + "}"


def format_trace(trace: Mapping) -> str:
    states = ", ".join(format_state(state) for state in trace["states"])
    return f"states=[{states}], loop={int(trace['loop'])}"


def format_training_row(trace: Mapping, label: int) -> str:
    return f"{format_trace(trace)}, {int(label)}"


def formula_key(formula: str) -> str:
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16]


def build_prompt_template(task_type, prompt_mode):
    if task_type != "ltl":
        raise ValueError(f"Unsupported task_type: {task_type}")
    if prompt_mode == "full":
        return PROMPT_TEMPLATE
    if prompt_mode == "naive_prompt":
        return TASK_INSTR + DIRECT_OUTPUT_FORMAT_INSTR + TRAINING_DATA_INSTR
    if prompt_mode == "only_input_instr":
        return (
            TASK_INSTR
            + INPUT_INSTR
            + DIRECT_OUTPUT_FORMAT_INSTR
            + TRAINING_DATA_INSTR
        )
    if prompt_mode == "only_output_instr":
        return (
            TASK_INSTR
            + SYNTAX_INSTR
            + DIRECT_OUTPUT_FORMAT_INSTR
            + TRAINING_DATA_INSTR
        )
    if prompt_mode == "input_output_instr":
        return (
            TASK_INSTR
            + INPUT_INSTR
            + SYNTAX_INSTR
            + DIRECT_OUTPUT_FORMAT_INSTR
            + TRAINING_DATA_INSTR
        )
    if prompt_mode == "zero_prompt":
        return (
            TASK_INSTR
            + INPUT_INSTR
            + SYNTAX_INSTR
            + OUTPUT_FORMAT_INSTR
            + TRAINING_DATA_INSTR
        )
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def build_reflection_prompt(
    previous_reasoning,
    previous_formula,
    train_ex,
    train_labels,
    feedback_note=None,
):
    ce_lines = "\n".join(
        format_training_row(assignment, label)
        for assignment, label in zip(train_ex, train_labels)
    )
    prompt = (
        AGENTIC_REFLECTION_INSTR
        + "Reasoning of previous epoch:\n"
        + (previous_reasoning or "(none)")
        + "\nFormula of previous epoch:\n"
        + (previous_formula or "(none)")
    )
    if feedback_note:
        prompt += "\nWhat failed in the previous epoch:\n" + feedback_note + "\n"
    return prompt + "\nNew counterexamples:\n" + (ce_lines or "(none)") + "\n\n"


def build_repair_prompt(
    previous_reasoning,
    previous_formula,
    train_ex,
    train_labels,
    feedback_note=None,
):
    repair_lines = "\n".join(
        format_training_row(assignment, label)
        for assignment, label in zip(train_ex, train_labels)
    )
    prompt = (
        AGENTIC_REPAIR_INSTR
        + "Reasoning of previous attempt:\n"
        + (previous_reasoning or "(none)")
        + "\nFormula of previous attempt:\n"
        + (previous_formula or "(none)")
    )
    if feedback_note:
        prompt += "\nRepair feedback:\n" + feedback_note + "\n"
    return prompt + "\nRepair examples:\n" + (repair_lines or "(none)") + "\n\n"


def build_candidate_repair_prompt(candidate_count=10):
    return (
        "\nCANDIDATE FORMULA GENERATION INSTRUCTION\n"
        f"- Generate exactly {candidate_count} diverse candidate formulas.\n"
        "- Wrap every candidate in its own <ans> and </ans> block.\n"
        "- You may include one shared <reasoning> block before the candidates.\n"
        "- Do not include text after the final </ans> block.\n\n"
    )


def score_formula(
    task: LTLTask,
    formula_tree: LTLNode,
    traces,
    labels,
):
    if not traces:
        return None
    correct = sum(
        int(int(task.evaluate(trace, formula_tree)) == int(label))
        for trace, label in zip(traces, labels)
    )
    return correct / len(traces)


class LTLTeacher:
    """LTL counterpart of ``teacher.Teacher`` with matching public methods."""

    def __init__(self, task: LTLTask, solver: BlackSolver):
        self.task = task
        self.solver = solver

    def generate_counterexamples(
        self,
        bs,
        formula_gt,
        formula_gen,
        generation_mode="search",
        timeout_seconds=10,
    ):
        del timeout_seconds
        if formula_gt != self.task.formula:
            raise ValueError("formula_gt does not match the teacher task")
        if formula_gen is None:
            raise ValueError("Cannot generate counterexamples without a hypothesis")
        if generation_mode not in {"search", "random"}:
            raise ValueError(
                f"Unknown counterexample generation mode: {generation_mode}"
            )

        traces, labels = self.solver.counterexamples(
            ground_truth=formula_gt,
            hypothesis=formula_gen,
            variables=self.task.variables,
            count=bs,
        )
        if generation_mode == "random":
            paired = list(zip(traces, labels))
            random.shuffle(paired)
            if paired:
                traces, labels = map(list, zip(*paired))
        return traces[:bs], labels[:bs]

    def judge_formula(
        self,
        msg,
        formula_gt,
        train_ex,
        train_labels,
        eval_ex,
        eval_labels,
        timeout_seconds=10,
    ):
        del timeout_seconds
        result = dict(msg)
        prediction = result.get("Prediction")
        try:
            if formula_gt != self.task.formula:
                raise ValueError("formula_gt does not match the teacher task")
            if prediction is None:
                raise ValueError("Unable to extract a formula from the response")
            prediction_tree = self.task.validate_formula(prediction)
            equivalent, witness = self.solver.equivalent(
                formula_gt, prediction, self.task.variables
            )
            result["Equivalent"] = equivalent
            result["Witness"] = witness
            result["scoreTrainSet"] = score_formula(
                self.task, prediction_tree, train_ex, train_labels
            )
            result["scoreEvalSet"] = score_formula(
                self.task, prediction_tree, eval_ex, eval_labels
            )
        except Exception as error:
            result["Equivalent"] = False
            result["Witness"] = None
            result["scoreTrainSet"] = None
            result["scoreEvalSet"] = None
            result["Error"] = str(error)
        return result


def build_retry_prompt(
    task,
    msg,
    train_ex,
    train_labels,
    max_examples=16,
    timeout_seconds=10,
):
    del timeout_seconds
    previous_formula = msg.get("Prediction")
    previous_reasoning = msg.get("Reasoning")
    feedback = []
    if previous_formula is None:
        feedback.append("Unable to extract a formula from the previous response.")
    if msg.get("Error"):
        feedback.append(msg["Error"])
    if msg.get("scoreTrainSet") is not None:
        feedback.append(
            f"Training accuracy of previous formula: {msg['scoreTrainSet']:.3f}"
        )
    if msg.get("scoreEvalSet") is not None:
        feedback.append(
            f"Eval accuracy of previous formula: {msg['scoreEvalSet']:.3f}"
        )

    formula_tree = None
    repair_ex, repair_labels = [], []
    if previous_formula is not None and msg.get("Error") is None:
        try:
            formula_tree = task.validate_formula(previous_formula)
        except Exception:
            formula_tree = None
    if formula_tree is not None:
        for trace, label in zip(train_ex, train_labels):
            if int(task.evaluate(trace, formula_tree)) != int(label):
                repair_ex.append(trace)
                repair_labels.append(label)

    compile_score = 1.0 if formula_tree is not None else 0.0
    fit_score = (
        1.0 - len(repair_ex) / len(train_ex)
        if formula_tree is not None and train_ex
        else 0.0
    )
    retry_done_score = 0.5 * compile_score + 0.5 * fit_score
    if retry_done_score >= 1.0:
        return None, retry_done_score

    return build_repair_prompt(
        previous_reasoning=previous_reasoning,
        previous_formula=previous_formula,
        train_ex=repair_ex[:max_examples],
        train_labels=repair_labels[:max_examples],
        feedback_note="; ".join(feedback) if feedback else None,
    ), retry_done_score


def run_episode(
    *,
    config,
    task,
    data,
    teacher,
    prompt_template,
    prompt_kwargs,
    generate_fn,
    on_retry=None,
    on_epoch_end=None,
):
    formula = task.formula
    accs = []
    agg_train_ex, agg_train_labels = [], []
    num_samples = log_scaling(
        config.tot_train_size, config.start_size, config.scale_factor
    )
    epochs = config.ce_epochs if config.use_ce else len(num_samples)
    current_guess = None
    current_guess_reasoning = None
    epoch_results = []

    for epoch in range(epochs):
        reflection_prompt = ""
        if config.use_ce:
            try:
                train_ex, train_labels = teacher.generate_counterexamples(
                    bs=config.ce_batch_size,
                    formula_gt=formula,
                    formula_gen=current_guess,
                    generation_mode=config.ce_generation_mode,
                )
                if uses_reflection_prompt(config.reasoning_mode):
                    reflection_prompt = build_reflection_prompt(
                        current_guess_reasoning,
                        current_guess,
                        train_ex,
                        train_labels,
                    )
            except Exception as error:
                print(
                    f"Cannot generate counterexamples at epoch {epoch}: {error}; "
                    f"use {config.ce_start_size} initial examples instead."
                )
                left = epoch * config.ce_start_size
                right = (epoch + 1) * config.ce_start_size
                train_ex = data["train_ex"][left:right]
                train_labels = data["train_labels"][left:right]
        else:
            left, right = len(agg_train_ex), num_samples[epoch]
            train_ex = data["train_ex"][left:right]
            train_labels = data["train_labels"][left:right]

        agg_train_ex += train_ex
        agg_train_labels += train_labels
        train_prompt = "\n".join(
            format_training_row(assignment, label)
            for assignment, label in zip(agg_train_ex, agg_train_labels)
        )

        msgs, acc = [], 0
        retry_prompt, retry_done_score = "", 0.0
        best_retry_score = -1.0
        best_retry_msg = None
        msg = None

        for retry_idx in range(config.retries):
            iter_prompt_kwargs = dict(prompt_kwargs)
            if uses_candidate_repair(config.reasoning_mode):
                iter_prompt_kwargs["agentic_reflection_instr"] = (
                    reflection_prompt
                    + build_candidate_repair_prompt(config.candidate_repair_count)
                )
            elif uses_reflection_prompt(
                config.reasoning_mode
            ) or uses_repair_loop(config.reasoning_mode):
                iter_prompt_kwargs["agentic_reflection_instr"] = (
                    reflection_prompt + retry_prompt
                )

            msg = generate_fn(
                prompt_template=prompt_template,
                train_prompt=train_prompt,
                prompt_format_kwargs=iter_prompt_kwargs,
                answer_extractor=(
                    extract_all_ans
                    if uses_candidate_repair(config.reasoning_mode)
                    else extract_ans
                ),
                reasoning_extractor=extract_reasoning,
            )

            if uses_candidate_repair(config.reasoning_mode):
                candidates = msg.get("Prediction") or []
                msg["IsCandidateRepairBatch"] = True
                msg["CandidateCount"] = len(candidates)
                msgs.append(msg)
                if on_retry is not None:
                    on_retry(epoch, msgs)
                if not candidates:
                    msg["Error"] = "No candidate formulas were extracted."
                    break

                for candidate_idx, candidate_formula in enumerate(
                    candidates[: config.candidate_repair_count]
                ):
                    candidate_msg = {
                        **msg,
                        "Prediction": candidate_formula,
                        "CandidateIndex": candidate_idx,
                        "IsCandidateRepair": True,
                    }
                    candidate_msg = teacher.judge_formula(
                        msg=candidate_msg,
                        formula_gt=formula,
                        train_ex=agg_train_ex,
                        train_labels=agg_train_labels,
                        eval_ex=data["eval_ex"],
                        eval_labels=data["eval_labels"],
                    )
                    _, candidate_score = build_retry_prompt(
                        task, candidate_msg, agg_train_ex, agg_train_labels
                    )
                    candidate_msg["RepairScore"] = candidate_score
                    msgs.append(candidate_msg)
                    if on_retry is not None:
                        on_retry(epoch, msgs)
                    if candidate_msg.get("Equivalent"):
                        acc = 1
                    if (
                        best_retry_msg is None
                        or candidate_score >= best_retry_score
                    ):
                        best_retry_score = candidate_score
                        best_retry_msg = candidate_msg
                    if candidate_score >= 1.0:
                        break
                break

            print(
                f"Epoch {epoch}, Retry {retry_idx}\n"
                f"Prediction: {msg['Prediction']}\n"
                f"Reasoning: {msg['Reasoning']}",
                flush=True,
            )
            msg = teacher.judge_formula(
                msg=msg,
                formula_gt=formula,
                train_ex=agg_train_ex,
                train_labels=agg_train_labels,
                eval_ex=data["eval_ex"],
                eval_labels=data["eval_labels"],
            )
            if msg.get("Equivalent"):
                acc = 1
            msgs.append(msg)
            if on_retry is not None:
                on_retry(epoch, msgs)

            if uses_repair_loop(config.reasoning_mode):
                retry_prompt, retry_done_score = build_retry_prompt(
                    task, msg, agg_train_ex, agg_train_labels
                )
                if best_retry_msg is None or retry_done_score >= best_retry_score:
                    best_retry_score = retry_done_score
                    best_retry_msg = msg
                if retry_done_score >= 1.0:
                    break
            else:
                best_retry_msg = None
                break
            if msg.get("Equivalent"):
                break

        if best_retry_msg is not None:
            current_guess = best_retry_msg.get("Prediction")
            current_guess_reasoning = best_retry_msg.get("Reasoning")
        elif msg is not None and not msg.get("IsCandidateRepairBatch"):
            current_guess = msg.get("Prediction")
            current_guess_reasoning = msg.get("Reasoning")
        else:
            current_guess = None
            current_guess_reasoning = None

        epoch_result = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "CurrentGuess": current_guess,
            "CurrentGuessReasoning": current_guess_reasoning,
            "Logs": msgs,
        }
        epoch_results.append(epoch_result)
        accs.append(acc)
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_result)
        if acc == 1.0:
            break

    return {
        "epoch_results": epoch_results,
        "epochs_completed": len(epoch_results),
        "final_num_samples": summarize_label_counts(agg_train_labels),
        "final_accuracy": accs[-1] if accs else 0.0,
        "current_guess": current_guess,
        "current_guess_reasoning": current_guess_reasoning,
    }


def generate_dataset(config, task, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"formula={formula_key(task.formula)}.json")
    train_ex, train_labels = task.generate_random_data(
        config.tot_train_size, balanced=False
    )
    eval_ex, eval_labels = task.generate_random_data(
        config.eval_size, balanced=False
    )
    data = {
        "formula": task.formula,
        "variables": list(task.variables),
        "train_ex": train_ex,
        "train_labels": train_labels,
        "eval_ex": eval_ex,
        "eval_labels": eval_labels,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    return path, data


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", default="ltl", choices=["ltl"])
    parser.add_argument(
        "--formula",
        "--regex",
        dest="formula",
        default="G(p -> F q)",
        help="Ground-truth formula. --regex is accepted as a compatibility alias.",
    )
    parser.add_argument(
        "--variables", nargs="+", default=["p", "q", "r", "s", "t"]
    )
    # Compatibility arguments retained for future shared-runner extraction.
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--mkey", type=str, default="gpt5")
    parser.add_argument("--tot_train_size", type=int, default=8)
    parser.add_argument("--eval_size", type=int, default=8)
    parser.add_argument("--start_size", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rerun", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--use_reg", action="store_true")
    parser.add_argument("--use_ce", action="store_true")
    parser.add_argument("--ce_epochs", type=int, default=8)
    parser.add_argument("--ce_start_size", type=int, default=3)
    parser.add_argument("--ce_batch_size", type=int, default=8)
    parser.add_argument(
        "--ce_generation_mode",
        default="search",
        choices=["search", "random"],
    )
    parser.add_argument(
        "--prompt_mode",
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
    parser.add_argument(
        "--reasoning_mode",
        default="agentic_reflection",
        choices=[
            "single_inference",
            "agentic_reflection",
            "agentic_no_reflection",
            "agentic_no_repair_loop",
            "agentic_candidate_repair",
        ],
    )
    parser.add_argument("--candidate_repair_count", type=int, default=10)
    parser.add_argument("--min_trace_length", type=int, default=1)
    parser.add_argument("--max_trace_length", type=int, default=8)
    parser.add_argument("--black_binary", default="black")
    parser.add_argument("--black_timeout", type=float, default=30.0)
    parser.add_argument("--black_bound", type=int)
    parser.add_argument(
        "--indir", default="datasets/ltl/traces"
    )
    parser.add_argument("--outdir", default="logs/ltl")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.outdir = os.path.join(args.outdir, f"icl_gen_{args.task_type}")
    use_vllm = is_vllm_model(args.mkey)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not use_vllm and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    task = LTLTask(
        args.formula,
        variables=args.variables,
        seed=args.seed,
        min_trace_length=args.min_trace_length,
        max_trace_length=args.max_trace_length,
    )
    prompt_template = build_prompt_template(args.task_type, args.prompt_mode)
    prompt_kwargs = {
        "variables": ", ".join(task.variables),
        "regularization_instr": REGULARIZATION if args.use_reg else "",
        "agentic_reflection_instr": "",
    }
    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
    solver = BlackSolver(
        binary=args.black_binary,
        timeout_seconds=args.black_timeout,
        bound=args.black_bound,
    )
    solver.ensure_available()
    teacher = LTLTeacher(task, solver)
    learner = LearnerForICLGen(args.mkey, model, tokenizer, task)
    _, data = generate_dataset(args, task, args.indir)

    mode = "ce" if args.use_ce else "std"
    reg = "reg" if args.use_reg else "noreg"
    config_dir = os.path.join(
        args.outdir, f"model={args.mkey}", mode, reg
    )
    if args.use_ce:
        config_dir = os.path.join(config_dir, args.reasoning_mode)
    if args.prompt_mode != "full":
        config_dir = os.path.join(config_dir, f"prompt={args.prompt_mode}")
    config_name = os.path.join(
        config_dir,
        (
            f"msgdict_formula={formula_key(args.formula)}"
            f"_ceEpochs={args.ce_epochs}_ceBatch={args.ce_batch_size}"
            f"_{args.ce_generation_mode}.json"
            if args.use_ce
            else (
                f"msgdict_formula={formula_key(args.formula)}"
                f"_totTrain={args.tot_train_size}"
                f"_startSize={args.start_size}"
                f"_scaleFactor={args.scale_factor}.json"
            )
        ),
    )

    msgdict = {
        "formula": args.formula,
        "variables": list(task.variables),
        "summary": None,
    }
    finish_states = {}
    print(
        f"Starting training for formula: {args.formula} with model {args.mkey}"
    )
    for runid in range(args.rerun):
        print(f"=== Rerun {runid} ===")
        msgdict[f"run-{runid}"] = {}

        def generate_fn(
            prompt_template,
            train_prompt,
            prompt_format_kwargs,
            answer_extractor=extract_ans,
            reasoning_extractor=extract_reasoning,
        ):
            return learner.generate(
                prompt_template=prompt_template,
                train_prompt=train_prompt,
                prompt_format_kwargs=prompt_format_kwargs,
                temp=args.temp,
                answer_extractor=answer_extractor,
                reasoning_extractor=reasoning_extractor,
            )

        def on_retry(epoch, msgs):
            msgdict[f"run-{runid}"][f"epoch-{epoch}"] = {"Logs": msgs}
            savejson(msgdict, config_name)

        def on_epoch_end(epoch, epoch_result):
            msgdict[f"run-{runid}"][f"epoch-{epoch}"] = epoch_result
            savejson(msgdict, config_name)
            print(f"Accuracy at epoch {epoch}: {epoch_result['Accuracy']}")

        episode_result = run_episode(
            config=args,
            task=task,
            data=data,
            teacher=teacher,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            generate_fn=generate_fn,
            on_retry=on_retry,
            on_epoch_end=on_epoch_end,
        )
        finish_states[f"run-{runid}"] = {
            "epochs": episode_result["epochs_completed"],
            "final_num_samples": episode_result["final_num_samples"],
            "final_accuracy": episode_result["final_accuracy"],
        }
        msgdict["summary"] = finish_states
        savejson(msgdict, config_name)


if __name__ == "__main__":
    main()
