"""Direct input/output ICL runner for the four added benchmarks."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from prompting.benchmarks import ICL_PROMPT, ICL_TASKS
from tasks.benchmarks import BenchmarkTask


DEFAULT_IDS = {
    "cryptography": "caesar:0",
    "list_functions": "c102",
    "instruction_induction": "antonyms",
    "scan": "simple",
}


def _schedule(total: int, start: int, factor: float) -> list[int]:
    if total < 1 or start < 1 or factor <= 1:
        raise ValueError("tot_train_size and start_size must be positive, and scale_factor must exceed 1")
    sizes = []
    current = start
    while current < total:
        sizes.append(current)
        current = max(current + 1, int(current * factor))
    sizes.append(total)
    return sizes


def _parse_answers(response: str | None, count: int):
    if not response:
        return None
    blocks = re.findall(r"<ans>\s*(.*?)\s*</ans>", response, flags=re.DOTALL)
    if not blocks:
        return None
    try:
        answers = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None
    return answers if isinstance(answers, list) and len(answers) == count else None


def _equal(kind: str, predicted, expected) -> bool:
    if kind == "list_functions":
        return isinstance(predicted, list) and predicted == expected
    if not isinstance(predicted, str):
        return False
    if kind == "scan":
        return predicted.split() == expected.split()
    return predicted.strip() == expected.strip()


def _prompt(task: BenchmarkTask, seen, inputs, regularize=False):
    examples = "\n".join(task.format_pair(x, y) for x, y in seen) or "(none)"
    queries = "\n".join(json.dumps(x, ensure_ascii=False) for x in inputs)
    return (ICL_PROMPT.replace("{task}", ICL_TASKS[task.kind])
            .replace("{regularization}", "Prefer the simplest rule consistent with the examples." if regularize else "")
            .replace("{examples}", examples)
            .replace("{count}", str(len(inputs)))
            .replace("{inputs}", queries))


def _query(task, seen, batch, call_model, regularize=False):
    prompt = _prompt(task, seen, [x for x, _ in batch], regularize)
    response = call_model(prompt)
    answers = _parse_answers(response, len(batch))
    correct = None if answers is None else sum(
        _equal(task.kind, prediction, expected)
        for prediction, (_, expected) in zip(answers, batch)
    )
    return {"Prompt": prompt, "Response": response, "Prediction": answers,
            "GroundTruth": [y for _, y in batch], "Correct": correct}


def run_icl(args, response_fn=None):
    """Run baseline or mistake-driven CE ICL; response_fn enables offline checks."""
    kind = args.task_type
    benchmark_id = args.benchmark_id or DEFAULT_IDS[kind]
    task = BenchmarkTask(kind, benchmark_id, seed=args.seed, eval_size=args.eval_size)
    if response_fn is None:
        from keysecrets import api_key
        from modeling.llm import load_model_and_tokenizer, run_model
        model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
        response_fn = lambda prompt: run_model(args.mkey, model, tokenizer, prompt, temp=args.temp)

    rng = random.Random(args.seed)
    pool = list(task.train)
    rng.shuffle(pool)
    total = min(args.tot_train_size, len(pool))
    sizes = _schedule(total, args.start_size, args.scale_factor)
    ce_limit = getattr(args, "ce_batch_size", 8)
    if ce_limit < 1:
        raise ValueError("ce_batch_size must be positive")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", benchmark_id)
    mode = "ce" if args.use_ce else "std"
    reg = "reg" if args.use_reg else "noreg"
    outpath = Path(args.outdir) / f"icl_{kind}" / f"model={args.mkey}" / mode / reg / f"benchmark={safe_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    result = {"task_type": kind, "benchmark_id": benchmark_id, "mode": mode, "epochs": []}
    seen = []
    cursor = 0
    for epoch, target_size in enumerate(sizes):
        candidates = pool[cursor:target_size]
        cursor = target_size
        if args.use_ce:
            additions = []
            ce_logs = []
            for start in range(0, len(candidates), args.eval_batch_size):
                if len(additions) >= ce_limit:
                    break
                batch = candidates[start:start + args.eval_batch_size]
                log = _query(task, seen, batch, response_fn, args.use_reg)
                ce_logs.append(log)
                if log["Prediction"] is not None:
                    additions.extend((x, y) for (x, y), prediction in zip(batch, log["Prediction"])
                                     if not _equal(kind, prediction, y))
            additions = additions[:ce_limit]
            seen.extend(additions)
        else:
            additions = candidates
            ce_logs = []
            seen.extend(candidates)

        correct = 0.0
        eval_logs = []
        for start in range(0, len(task.eval), args.eval_batch_size):
            batch = task.eval[start:start + args.eval_batch_size]
            retry_logs = [_query(task, seen, batch, response_fn, args.use_reg)
                          for _ in range(args.retries)]
            eval_logs.extend(retry_logs)
            valid = [log["Correct"] for log in retry_logs if log["Correct"] is not None]
            if valid:
                correct += sum(valid) / len(valid)
        accuracy = correct / len(task.eval) if task.eval else None
        result["epochs"].append({"epoch": epoch, "NumTrainingSamples": len(seen),
                                 "AddedExamples": len(additions), "Accuracy": accuracy,
                                 "CounterexampleLogs": ce_logs, "EvalLogs": eval_logs})
        with outpath.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(f"Epoch {epoch}: accuracy={accuracy}, training samples={len(seen)}")
    return result
