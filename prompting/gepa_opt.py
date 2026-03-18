import gepa
import argparse
import json
import os
import sys
import random
import numpy as np
import torch
from statistics import mean

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import keysecrets
from gepa_adapter import DFAMatchAdapter
from local_vllm_callable import build_chat_callable
from train_icl_gen import (
    AGENTIC_REFLECTION_INSTR,
    EXTRX_CLUSTRED_CE_INSTR,
    EXTRX_PROMPT_TEMPLATE,
    EXTRX_REGULARIZATION,
    EXTRX_SIGMA,
    SIMPLYRX_CLUSTRED_CE_INSTR,
    SIMPLYRX_PROMPT_TEMPLATE,
    SIMPLYRX_REGULARIZATION,
)

os.environ["OPENAI_API_KEY"] = keysecrets.api_key


def build_seed_prompt(
    task_type: str,
    use_reg: bool = False,
    ce_clustered: bool = False,
    use_agentic_reflection: bool = False,
) -> str:
    if task_type == "extrx":
        prompt = EXTRX_PROMPT_TEMPLATE.format(
            "",
            regularization_instr=EXTRX_REGULARIZATION if use_reg else "",
            agentic_reflection_instr=AGENTIC_REFLECTION_INSTR if use_agentic_reflection else "",
            sigma=EXTRX_SIGMA,
            clustered_ce_instr=EXTRX_CLUSTRED_CE_INSTR if ce_clustered else "",
        )
    else:
        prompt = SIMPLYRX_PROMPT_TEMPLATE.format(
            "",
            regularization_instr=SIMPLYRX_REGULARIZATION if use_reg else "",
            agentic_reflection_instr=AGENTIC_REFLECTION_INSTR if use_agentic_reflection else "",
            clustered_ce_instr=SIMPLYRX_CLUSTRED_CE_INSTR if ce_clustered else "",
        )

    training_data_marker = "\nTraining Data (Each line has one input-output pair separated by comma):\n"
    if training_data_marker in prompt:
        prompt = prompt.split(training_data_marker, 1)[0].rstrip()
    return prompt


def build_task_splits(data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    task_splits = {"train": [], "val": [], "test": []}
    seen_regexes: set[str] = set()
    for split in ["train", "val", "test"]:
        for item in data.get(split, []):
            regex = item["answer"]
            if regex in seen_regexes:
                continue
            seen_regexes.add(regex)
            task_splits[split].append({"answer": regex})
    return task_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_metric_calls", type=int, default=150)
    parser.add_argument("--task_lm", type=str, default="openai/gpt-5.1")
    parser.add_argument("--reflection_lm", type=str, default="openai/gpt-5.1")
    parser.add_argument("--task_type", type=str, default="simplyrx", choices=["simplyrx", "extrx"])
    parser.add_argument("--use_reg", default=False, action="store_true")
    parser.add_argument("--use_agentic_reflection", default=False, action="store_true")
    
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--total_train_size", type=int, default=384)
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--start_size", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--use_ce", default=False, action="store_true")
    parser.add_argument("--ce_epochs", type=int, default=8)
    parser.add_argument("--ce_start_size", type=int, default=8)
    parser.add_argument("--ce_batch_size", type=int, default=128)
    parser.add_argument("--ce_clustered", default=False, action="store_true")
    parser.add_argument("--indir", type=str, default="datasets")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(f"prompting/gepa_icl_gen_{args.task_type}.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = build_task_splits(raw_data)
    print("Task count:", {split: len(data.get(split, [])) for split in ["train", "val", "test"]})
    
    adapter = DFAMatchAdapter(
        model=build_chat_callable(args.task_lm),
        task_type=args.task_type,
        max_length=args.max_length,
        eval_max_length=args.eval_max_length,
        indir=args.indir,
        use_ce=args.use_ce,
        total_train_size=args.total_train_size,
        eval_size=args.eval_size,
        start_size=args.start_size,
        scale_factor=args.scale_factor,
        retries=args.retries,
        ce_epochs=args.ce_epochs,
        ce_start_size=args.ce_start_size,
        ce_batch_size=args.ce_batch_size,
        ce_clustered=args.ce_clustered,
    )
    
    gepa_result = gepa.optimize(
        seed_candidate={
            "system_prompt": build_seed_prompt(
                args.task_type,
                args.use_reg,
                args.ce_clustered,
                args.use_agentic_reflection,
            )
        },
        trainset=data["train"],
        valset=data["val"],
        adapter=adapter,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=build_chat_callable(args.reflection_lm)
    )

    print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])

    if data.get("test"):
        test_result = adapter.evaluate(data["test"], gepa_result.best_candidate, capture_traces=False)
        test_score = mean(test_result.scores) if test_result.scores else 0.0
        print("Test score:", test_score)
