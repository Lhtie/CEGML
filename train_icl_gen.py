import torch
import argparse
import random
import os
import json
import re
import signal
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from modeling.llm import is_vllm_model, load_model_and_tokenizer
from tasks.rl import SimplyRegularLanguage, ExtRegularLanguage
from learner import LearnerForICLGen
from teacher import Teacher
from dataset import generate_dataset
from keysecrets import api_key
from tasks.utils import dfa_accepts_ex
from prompting.rl import (
    AGENTIC_REFLECTION_INSTR,
    AGENTIC_REPAIR_INSTR,
    EXTRX_CLUSTRED_CE_INSTR,
    EXTRX_INPUT_INSTR,
    EXTRX_OUTPUT_INSTR,
    EXTRX_PROMPT_TEMPLATE,
    EXTRX_REGULARIZATION,
    EXTRX_SIGMA,
    EXTRX_TASK_INSTR,
    SIMPLYRX_CLUSTRED_CE_INSTR,
    SIMPLYRX_INPUT_INSTR,
    SIMPLYRX_OUTPUT_INSTR,
    SIMPLYRX_PROMPT_TEMPLATE,
    SIMPLYRX_REGULARIZATION,
    SIMPLYRX_TASK_INSTR,
    TRAINING_DATA_INSTR,
)

train_data_template = "{0}, {1}"

def extract_ans(res):
    if res is None: return None
    matches = re.search(r"(?:.*)<ans>\s*(.*?)\s*</ans>", res, re.DOTALL)
    if matches:
        return matches.group(1)
    return None

def extract_all_ans(res):
    if res is None:
        return []
    return [
        match.strip()
        for match in re.findall(r"<ans>\s*(.*?)\s*</ans>", res, re.DOTALL)
        if match.strip()
    ]

def extract_reasoning(res):
    if res is None: return None
    matches = re.search(r"(?:.*)<reasoning>\s*(.*?)\s*</reasoning>", res, re.DOTALL)
    if matches:
        return matches.group(1)
    return None

def log_scaling(total, start, scale_factor):
    sizes = []
    current = start
    while current < total:
        sizes.append(current)
        current = int(current * scale_factor)
    sizes.append(total)
    return sizes

def savejson(ctx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ctx, f, indent=4)


def summarize_label_counts(labels):
    pos = sum(int(label == 1) for label in labels)
    neg = sum(int(label == 0) for label in labels)
    return {
        "positive": pos,
        "negative": neg,
        "total": len(labels),
    }


def build_prompt_template(task_type, prompt_mode):
    if task_type == "extrx":
        task_instr = EXTRX_TASK_INSTR
        input_instr = EXTRX_INPUT_INSTR
        syntax_instr = EXTRX_OUTPUT_INSTR["syntax"]
        output_format_instr = EXTRX_OUTPUT_INSTR["output_format"]
        direct_output_format_instr = EXTRX_OUTPUT_INSTR["direct_output_format"]
        full_prompt_template = EXTRX_PROMPT_TEMPLATE
    else:
        task_instr = SIMPLYRX_TASK_INSTR
        input_instr = SIMPLYRX_INPUT_INSTR
        syntax_instr = SIMPLYRX_OUTPUT_INSTR["syntax"]
        output_format_instr = SIMPLYRX_OUTPUT_INSTR["output_format"]
        direct_output_format_instr = SIMPLYRX_OUTPUT_INSTR["direct_output_format"]
        full_prompt_template = SIMPLYRX_PROMPT_TEMPLATE

    if prompt_mode == "full":
        return full_prompt_template
    if prompt_mode == "naive_prompt":
        return task_instr + direct_output_format_instr + TRAINING_DATA_INSTR
    if prompt_mode == "only_input_instr":
        return task_instr + input_instr + direct_output_format_instr + TRAINING_DATA_INSTR
    if prompt_mode == "only_output_instr":
        return task_instr + syntax_instr + direct_output_format_instr + TRAINING_DATA_INSTR
    if prompt_mode == "input_output_instr":
        return (
            task_instr
            + input_instr
            + syntax_instr
            + direct_output_format_instr
            + TRAINING_DATA_INSTR
        )
    if prompt_mode == "zero_prompt":
        return (
            task_instr
            + syntax_instr
            + input_instr
            + output_format_instr
            + TRAINING_DATA_INSTR
        )
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def build_reflection_prompt(previous_reasoning, previous_regex, train_ex, train_labels, feedback_note=None):
    ce_lines = "\n".join(
        [train_data_template.format(ex, label) for ex, label in zip(train_ex, train_labels)]
    )
    prompt = (
        AGENTIC_REFLECTION_INSTR
        + "Reasoning of previous epoch:\n"
        + (previous_reasoning or "(none)")
        + "\nRegex of previous epoch:\n"
        + (previous_regex or "(none)")
    )
    if feedback_note:
        prompt += "\nWhat failed in the previous epoch:\n" + feedback_note + "\n"
    prompt += (
        "\nNew counterexamples (string, label):\n"
        + ce_lines
        + "\n\n"
    )
    return prompt


def build_repair_prompt(previous_reasoning, previous_regex, train_ex, train_labels, feedback_note=None):
    ce_lines = "\n".join(
        [train_data_template.format(ex, label) for ex, label in zip(train_ex, train_labels)]
    )
    prompt = (
        AGENTIC_REPAIR_INSTR
        + "Reasoning of previous attempt:\n"
        + (previous_reasoning or "(none)")
        + "\nRegex of previous attempt:\n"
        + (previous_regex or "(none)")
    )
    if feedback_note:
        prompt += "\nRepair feedback:\n" + feedback_note + "\n"
    if ce_lines:
        prompt += (
            "\nRepair examples (string, label):\n"
            + ce_lines
            + "\n\n"
        )
    else:
        prompt += (
            "\nRepair examples (string, label):\n"
            + "(none available; the previous regex either failed to compile or no repair examples were found)\n\n"
        )
    return prompt


def build_candidate_repair_prompt(candidate_count=10):
    return (
        "\nCANDIDATE REGEX GENERATION INSTRUCTION\n"
        + f"- Do not return only one repaired regex. Generate exactly {candidate_count} diverse candidate regexes.\n"
        + "- Wrap every candidate regex in its own <ans> and </ans> block.\n"
        + "- You may include one concise shared <reasoning>...</reasoning> block before the candidates.\n"
        + "- Do not include any text after the final </ans> block.\n\n"
    )


def build_retry_prompt(task, msg, train_ex, train_labels, max_examples=16, timeout_seconds=10):
    previous_regex = msg.get("Prediction")
    previous_reasoning = msg.get("Reasoning")

    feedback = []
    if previous_regex is None:
        feedback.append("Unable to extract a regex from previous response.")
    if msg.get("Error"):
        feedback.append(msg["Error"])
    if msg.get("scoreTrainSet") is not None:
        feedback.append(f"Training accuracy of previous regex: {msg['scoreTrainSet']:.3f}")
    if msg.get("scoreEvalSet") is not None:
        feedback.append(f"Eval accuracy of previous regex: {msg['scoreEvalSet']:.3f}")

    repair_ex, repair_labels = [], []
    pred = previous_regex
    dfa_pred = None
    if pred is not None and msg.get("Error") is None:
        try:
            dfa_pred, _, _ = task.regex_to_pynini_via_pyformlang(pred)
        except Exception:
            dfa_pred = None
        if dfa_pred is not None:
            for ex, label in zip(train_ex, train_labels):
                if int(dfa_accepts_ex(dfa_pred, ex)) != label:
                    repair_ex.append(ex)
                    repair_labels.append(label)

    # two objectives: 1. produce a regex that compiles, 2. fix mistakes on training data
    compile_score = 1.0 if dfa_pred is not None else 0.0
    total_train = len(train_ex)
    if dfa_pred is not None and total_train > 0:
        repair_fit_score = 1.0 - (len(repair_ex) / total_train)
    else:
        repair_fit_score = 0.0
    retry_done_score = 0.5 * compile_score + 0.5 * repair_fit_score

    if retry_done_score >= 1.0:
        return None, retry_done_score

    if repair_ex:
        repair_ex = repair_ex[:max_examples]
        repair_labels = repair_labels[:max_examples]

    return build_repair_prompt(
        previous_reasoning=previous_reasoning,
        previous_regex=previous_regex,
        train_ex=repair_ex,
        train_labels=repair_labels,
        feedback_note="; ".join(feedback) if feedback else None,
    ), retry_done_score


def uses_reflection_prompt(reasoning_mode):
    return reasoning_mode in {
        "agentic_reflection",
        "agentic_no_repair_loop",
        "agentic_candidate_repair",
    }

def uses_repair_loop(reasoning_mode):
    return reasoning_mode in {"agentic_reflection", "agentic_no_reflection"}

def uses_candidate_repair(reasoning_mode):
    return reasoning_mode == "agentic_candidate_repair"

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
    regex = task.regex_str
    _, fst_gt, _ = task.regex_to_pynini_via_pyformlang(regex)
    accs = []
    agg_train_ex, agg_train_labels = [], []
    num_samples = log_scaling(config.tot_train_size, config.start_size, config.scale_factor)
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
                    regex_gt=regex,
                    regex_gen=current_guess,
                    clustered=config.ce_clustered,
                    generation_mode=config.ce_generation_mode,
                )
                if uses_reflection_prompt(config.reasoning_mode):
                    reflection_prompt = build_reflection_prompt(
                        current_guess_reasoning,
                        current_guess,
                        train_ex,
                        train_labels,
                    )
            except Exception as e:
                print(f"Cannot generate counterexamples at epoch {epoch}: {e}, use {config.ce_start_size} random examples instead.")
                train_ex = data["train_ex"][epoch * config.ce_start_size:(epoch + 1) * config.ce_start_size]
                train_labels = data["train_labels"][epoch * config.ce_start_size:(epoch + 1) * config.ce_start_size]
                
        else:
            l, r = len(agg_train_ex), num_samples[epoch]
            train_ex = data["train_ex"][l:r]
            train_labels = data["train_labels"][l:r]

        agg_train_ex += train_ex
        agg_train_labels += train_labels
        train_p = "\n".join(
            [train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)]
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
                    + build_candidate_repair_prompt(
                        candidate_count=config.candidate_repair_count,
                    )
                )
            elif uses_reflection_prompt(config.reasoning_mode) or uses_repair_loop(config.reasoning_mode):
                iter_prompt_kwargs["agentic_reflection_instr"] = (
                    reflection_prompt + retry_prompt
                )

            msg = generate_fn(
                prompt_template=prompt_template,
                train_prompt=train_p,
                prompt_format_kwargs=iter_prompt_kwargs,
                answer_extractor=extract_all_ans if uses_candidate_repair(config.reasoning_mode) else extract_ans,
                reasoning_extractor=extract_reasoning,
            )

            if uses_candidate_repair(config.reasoning_mode):
                candidates = msg.get("Prediction") or []
                msg["IsCandidateRepairBatch"] = True
                msg["CandidateCount"] = len(candidates)
                msgs.append(msg)
                if on_retry is not None:
                    on_retry(epoch, msgs)

                if len(candidates) == 0:
                    msg["Error"] = "Not enough candidate regexes extracted from candidate repair response."
                    break

                for candidate_idx, candidate_regex in enumerate(candidates[:min(len(candidates), config.candidate_repair_count)]):
                    candidate_msg = {
                        **msg,
                        "Prediction": candidate_regex,
                        "Reasoning": msg.get("Reasoning"),
                        "CandidateIndex": candidate_idx,
                        "IsCandidateRepair": True,
                    }
                    print(f"Epoch {epoch}, Candidate {candidate_idx}\nPrediction: {candidate_regex}", flush=True)
                    candidate_msg = teacher.judge_regex(
                        msg=candidate_msg,
                        fst_gt=fst_gt,
                        train_ex=agg_train_ex,
                        train_labels=agg_train_labels,
                        eval_ex=data["eval_ex"],
                        eval_labels=data["eval_labels"]
                    )
                    _, candidate_retry_score = build_retry_prompt(
                        task=task,
                        msg=candidate_msg,
                        train_ex=agg_train_ex,
                        train_labels=agg_train_labels,
                    )
                    candidate_msg["RepairScore"] = candidate_retry_score
                    msgs.append(candidate_msg)
                    if on_retry is not None:
                        on_retry(epoch, msgs)

                    if candidate_msg.get("Equivalent"):
                        acc = max(acc, 1)
                    if best_retry_msg is None or candidate_retry_score >= best_retry_score:
                        best_retry_score = candidate_retry_score
                        best_retry_msg = candidate_msg

                    if candidate_retry_score >= 1.0:
                        break
                break

            print(f"Epoch {epoch}, Retry {retry_idx}\nPrediction: {msg['Prediction']}\nReasoning: {msg['Reasoning']}", flush=True)
            msg = teacher.judge_regex(
                msg=msg,
                fst_gt=fst_gt,
                train_ex=agg_train_ex,
                train_labels=agg_train_labels,
                eval_ex=data["eval_ex"],
                eval_labels=data["eval_labels"]
            )
            if msg.get("Equivalent"):
                acc = max(acc, 1)
            msgs.append(msg)
            if on_retry is not None:
                on_retry(epoch, msgs)
            
            if uses_repair_loop(config.reasoning_mode):
                retry_prompt, retry_done_score = build_retry_prompt(
                    task=task,
                    msg=msg,
                    train_ex=agg_train_ex,
                    train_labels=agg_train_labels,
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
        "current_guess_reasoning": current_guess_reasoning
    }

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="extrx", choices=["simplyrx", "extrx"])
    parser.add_argument("--regex", type=str, default="[A-Za-z0-9#]*z[A-Za-z]*[A-Za-z0-9#]*")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--mkey", type=str, default="gpt5")
    parser.add_argument("--tot_train_size", type=int, default=384)
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--start_size", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rerun", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--use_reg", default=False, action="store_true")
    parser.add_argument("--use_ce", default=False, action="store_true")
    parser.add_argument("--ce_epochs", type=int, default=8)
    parser.add_argument("--ce_start_size", type=int, default=8)
    parser.add_argument("--ce_batch_size", type=int, default=128)
    parser.add_argument("--ce_clustered", default=False, action="store_true")
    parser.add_argument(
        "--ce_generation_mode",
        type=str,
        default="dfs",
        choices=["dfs", "bfs", "random"],
        help="Strategy used to generate counterexamples from the difference DFA.",
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
    parser.add_argument(
        "--reasoning_mode",
        type=str,
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
    parser.add_argument("--indir", type=str, default="datasets")
    parser.add_argument("--outdir", type=str, default="logs/opt_prompt")
    args = parser.parse_args(argv)
    
    args.outdir = os.path.join(args.outdir, f"icl_gen_{args.task_type}")
    use_vllm = is_vllm_model(args.mkey)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not use_vllm and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.task_type == "extrx":
        task = ExtRegularLanguage(args.regex, args.max_length, alphabet=EXTRX_SIGMA)
        prompt_template = build_prompt_template(args.task_type, args.prompt_mode)
        prompt_kwargs = {
            "sigma": EXTRX_SIGMA,
            "regularization_instr": EXTRX_REGULARIZATION if args.use_reg else "",
            "agentic_reflection_instr": "",
            "clustered_ce_instr": EXTRX_CLUSTRED_CE_INSTR if args.ce_clustered else ""
        }
    else:
        task = SimplyRegularLanguage(args.regex, args.max_length)
        prompt_template = build_prompt_template(args.task_type, args.prompt_mode)
        prompt_kwargs = {
            "regularization_instr": SIMPLYRX_REGULARIZATION if args.use_reg else "",
            "agentic_reflection_instr": "",
            "clustered_ce_instr": SIMPLYRX_CLUSTRED_CE_INSTR if args.ce_clustered else ""
        }

    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
    teacher = Teacher(task)
    learner = LearnerForICLGen(args.mkey, model, tokenizer, task)

    if not args.use_ce:
        config_name = os.path.join(args.outdir, f"model={args.mkey}/std/")
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"prompt={args.prompt_mode}/" if args.prompt_mode != "full" else ""
        config_name += f"msgdict_regex={args.regex}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}.json"
    else:
        config_name = os.path.join(args.outdir, f"model={args.mkey}/ce/")
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"{args.reasoning_mode}/"
        config_name += f"prompt={args.prompt_mode}/" if args.prompt_mode != "full" else ""
        ce_mode = f"_{args.ce_generation_mode}"
        ce_clustered = "_clustered" if args.ce_clustered else ""
        config_name += f"msgdict_regex={args.regex}_ceEpochs={args.ce_epochs}_ceBatch={args.ce_batch_size}{ce_mode}{ce_clustered}.json"

    generate_dataset(args, task_type=args.task_type, outdir=args.indir)
    dataset = os.path.join(
        args.indir,
        f"regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json",
    )
    with open(dataset, "r") as f:
        data = json.load(f)

    msgdict, finish_states = {}, {}
    msgdict["summary"] = None
    print(f"Starting training for regex: {args.regex} with model {args.mkey}")
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
