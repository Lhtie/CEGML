import gepa
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import keysecrets
from gepa_adapter import DFAMatchAdapter
from local_vllm_callable import build_chat_callable
from train_icl_gen import (
    EXTRX_CLUSTRED_CE_INSTR,
    EXTRX_PROMPT_TEMPLATE,
    EXTRX_REGULARIZATION,
    EXTRX_SIGMA,
    SIMPLYRX_CLUSTRED_CE_INSTR,
    SIMPLYRX_PROMPT_TEMPLATE,
    SIMPLYRX_REGULARIZATION,
)

os.environ["OPENAI_API_KEY"] = keysecrets.api_key


def build_seed_prompt(task_type: str, ce_clustered: bool = False) -> str:
    if task_type == "extrx":
        prompt = EXTRX_PROMPT_TEMPLATE.format(
            EXTRX_REGULARIZATION,
            "",
            sigma=EXTRX_SIGMA,
            clustered_ce_instr=EXTRX_CLUSTRED_CE_INSTR if ce_clustered else "",
        )
    else:
        prompt = SIMPLYRX_PROMPT_TEMPLATE.format(
            SIMPLYRX_REGULARIZATION,
            "",
            clustered_ce_instr=SIMPLYRX_CLUSTRED_CE_INSTR if ce_clustered else "",
        )

    training_data_marker = "\nTraining Data (Each line has one input-output pair separated by comma):\n"
    if training_data_marker in prompt:
        prompt = prompt.split(training_data_marker, 1)[0].rstrip()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_metric_calls", type=int, default=150)
    parser.add_argument("--task_lm", type=str, default="openai/gpt-5.1")
    parser.add_argument("--reflection_lm", type=str, default="openai/gpt-5.1")
    parser.add_argument("--task_type", type=str, default="simplyrx", choices=["simplyrx", "extrx"])
    parser.add_argument("--ce_clustered", default=False, action="store_true")
    args = parser.parse_args()

    with open(f"prompting/gepa_icl_gen_{args.task_type}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Data size:", len(data["train"]))
    
    adapter = DFAMatchAdapter(
        model=build_chat_callable(args.task_lm),
        task_type=args.task_type,
        str_max_length=args.max_length
    )
    
    gepa_result = gepa.optimize(
        seed_candidate={
            "system_prompt": build_seed_prompt(args.task_type, args.ce_clustered)
        },
        trainset=data["train"],
        adapter=adapter,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=build_chat_callable(args.reflection_lm)
    )

    print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
