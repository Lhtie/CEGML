import torch
import argparse
import random
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Iterable, Set, Tuple

from tasks.rl import SimplyRegularLanguage, PythonRegularLanguage

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

def generate_dataset(args):
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

def generate_dataset_pyrx(args):
    task = PythonRegularLanguage(args.regex, args.max_length)

    # for r in regex_list_test:
    #     t = PythonRegularLanguage(r, args.max_length)
    #     print(f"{r}: {len(t.dfa.states)}")

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

def nl_rx_turk():
    from pyformlang.regular_expression import PythonRegex
    from tqdm import tqdm

    bins = {}
    with open("datasets/NL-RX-Turk.txt", "r") as f:
        lines = f.readlines()
        for rx in tqdm(lines[100:200]):
            pyrx = PythonRegex(rx.strip())
            dfa = pyrx.to_epsilon_nfa().to_deterministic().minimize()
            bins[len(dfa.states)] = bins.get(len(dfa.states), []) + [rx.strip()]

    bins = dict(sorted(bins.items(), key=lambda x: x[0]))

    print("Available number of states: ", list(bins.keys()))
    for state_num in bins:
        print(f"Regexes with {state_num} states:")
        for rx in bins[state_num]:
            print(f"  {rx}")

def enum_regexes(max_n: int, max_k: int, sigma: Tuple[str, ...]):
    from dataclasses import dataclass
    from functools import lru_cache
    from itertools import product

    @dataclass(frozen=True)
    class Sym:
        s: str

    @dataclass(frozen=True)
    class Concat:
        a: object
        b: object

    @dataclass(frozen=True)
    class Union:
        a: object
        b: object

    @dataclass(frozen=True)
    class Star:
        a: object
    
    def star_depth(r) -> int:
        if isinstance(r, Sym):
            return 0
        if isinstance(r, Concat) or isinstance(r, Union):
            return max(star_depth(r.a), star_depth(r.b))
        if isinstance(r, Star):
            return 1 + star_depth(r.a)
        raise TypeError(r)

    def size(r) -> int:
        if isinstance(r, Sym):
            return 1
        if isinstance(r, Concat) or isinstance(r, Union):
            return 1 + size(r.a) + size(r.b)
        if isinstance(r, Star):
            return 1 + size(r.a)
        raise TypeError(r)

    def states_count(s) -> int:
        from pyformlang.regular_expression import Regex

        regex = Regex(s)
        dfa = regex.to_epsilon_nfa().to_deterministic().minimize()
        return len(dfa.states)
    
    def to_str(r) -> str:
        if isinstance(r, Sym):
            return r.s
        if isinstance(r, Star):
            inner = to_str(r.a)
            return f"({inner})*"
        if isinstance(r, Concat):
            return f"({to_str(r.a)} {to_str(r.b)})"
        if isinstance(r, Union):
            x, y = to_str(r.a), to_str(r.b)
            if x > y:
                x, y = y, x
            return f"({x}+{y})"
        raise TypeError(r)
    
    def _split_budget(total: int) -> Tuple[int, int]:
        # split total into two positive integers
        if total == 2:
            return 1, 1
        a = random.randint(1, total - 1)
        return a, total - a
    
    def sample(
            k: int, n: int, sigma: Tuple[str, ...],
            p_union: float=0.35, p_concat: float=0.45, p_star: float=0.20
        ):
        if n == 1:
            return Sym(random.choice(sigma))
        
        constructor = None
        if n <= 2:
            constructor = "star"
        elif  k <= 0:
            constructor = random.choices(
                ["union", "concat"],
                weights=[p_union, p_concat],
                k=1
            )[0]
        else:
            constructor = random.choices(
                ["union", "concat", "star"],
                weights=[p_union, p_concat, p_star],
                k=1
            )[0]

        if constructor == "star":
            inner = sample(k - 1, n - 1, sigma, p_union, p_concat, p_star)
            return Star(inner)
        else:
            left_size, right_size = _split_budget(n - 1)
            left = sample(k, left_size, sigma, p_union, p_concat, p_star)
            right = sample(k, right_size, sigma, p_union, p_concat, p_star)
            if constructor == "concat":
                return Concat(left, right)
            else:
                return Union(left, right)

    cans = {}
    for n in tqdm(range(1, max_n + 1)):
        print(f"Generating regexes with size={n}...")
        ss = []
        for _ in range(2048):
            r = sample(max_k, n, sigma)
            s = to_str(r)
            d = star_depth(r)
            ss.append(s)
            cans[d] = cans.get(d, set()).union(set([s]))
        print(f"  Generated {len(ss)} regexes of size {n}, first 5 regexes: {ss[:5]}")
    cans = dict(sorted(cans.items(), key=lambda x: x[0]))

    regex_list = {}
    for d in range(0, 5):
        regex_list[d] = {}
        if cans[d] is not None:
            s = [(r, states_count(r)) for r in cans[d]]
            s = sorted(s, key=lambda x: x[1])
            for m in range(3, 11):
                regex_list[d][m] = regex_list[d].get(m, []) + [r for r, state_num in s if state_num == m]

    # for d in regex_list:
    #     print(f"Star depth: {d}")
    #     for m in regex_list[d]:
    #         print(f"  Number of states: {m}")
    #         for rx in regex_list[d][m][:5]:
    #             print(f"    {rx}")

    for d in regex_list:
        print(f"{d}\t{'\t'.join([('\\n').join(regex_list[d][m][:5]) for m in regex_list[d]])}")

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

    # nl_rx_turk()

    # generate_dataset(args)
    generate_dataset_pyrx(args)

    # enum_regexes(max_n=25, max_k=4, sigma=('a', 'b', 'c'))
    