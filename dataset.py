import torch
import argparse
import random
import os
import json
import numpy as np
from tqdm import tqdm

from tasks.rl import SimplyRegularLanguage

regex_list_train = [
    "(a(a)*b)*",                                # 2 states
    "(c(a+c)(a+b+c)a(b+c))*c",                  # 5 states
    "((b+c)(a+b+c)a b c(a+c))*b",               # 7 states
    "((a b)*+(a c)*+(b c)*)",                   # 8 states
    "(a(b+c)(a+b)c(a+c)b(a+b+c)(a+b+c))*"       # 8 states
]

regex_list_test = [
    "((a+b)c(a+b))*",                           # 3 states
    "((a* b)* c)*",                             # 3 states
    "((a a)*+(b b)*+(c c)*)*",                  # 4 states
    "(b(b+c))* a* (cc)*",                       # 4 states
    "((a+b)(b+c)(a+c)a(b+c)(a+b+c))*",          # 6 states
    "(c b(b+a c)a b)* (a+(b+c)*a)*",            # 8 states
    "((a*(b+c))*c + c((a+c)*b)*)* a"            # 8 states
]

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

    for r in regex_list_test:
        t = SimplyRegularLanguage(r, args.max_length)
        print(f"{r}: {len(t.dfa.states)}")

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

    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json", "w") as f:
        json.dump({
            "train_ex": train_ex,
            "train_labels": train_labels,
            "eval_ex": eval_ex,
            "eval_labels": eval_labels
        }, f, indent=4)