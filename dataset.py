import torch
import argparse
import random
import os
import json
import numpy as np
from tqdm import tqdm

from tasks.rl import SimplyRegularLanguage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", type=str, default="(a(b+c)(a+b)c(a+c)b(a+b+c)(a+b+c))*")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--tot_train_size", type=int, default=1280)
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    task = SimplyRegularLanguage(args.regex, args.max_length)

    train_ex = task.generate_random_strings_balanced(
        m=args.tot_train_size, 
        n=args.max_length
    )
    train_labels = [1 if task.accepts(x) else 0 for x in train_ex]

    eval_ex = task.generate_random_strings_balanced(
        m=args.eval_size, 
        n=args.eval_max_length
    )
    eval_labels = [1 if task.accepts(x) else 0 for x in eval_ex]

    os.makedirs(".cache", exist_ok=True)
    with open(f".cache/dataset_regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json", "w") as f:
        json.dump({
            "train_ex": train_ex,
            "train_labels": train_labels,
            "eval_ex": eval_ex,
            "eval_labels": eval_labels
        }, f, indent=4)