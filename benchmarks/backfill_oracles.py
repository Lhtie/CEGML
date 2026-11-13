"""Backfill std verification in existing logs, preserving generation and eval data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from tasks.benchmarks import BenchmarkTask, _validate_function, _time_limit

ROOT = Path(__file__).resolve().parents[1]


def _task(kind, ident, seed, crypto_setting="legacy"):
    task = BenchmarkTask(kind, ident, seed=seed, crypto_setting=crypto_setting)
    if kind == "scan":
        # Same validator and per-input timeout as predict(), but compile once per
        # candidate rather than once per command in the finite enumeration.
        compile_candidate = lru_cache(maxsize=4)(_validate_function)
        def predict(candidate, x):
            with _time_limit():
                return compile_candidate(candidate)(x)  # SCAN inputs are immutable strings.
        task.predict = predict
    return task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, default=ROOT / "logs/benchmarks_balanced")
    parser.add_argument("--instruction-model", help="Explicit LLM judge model, e.g. gpt-oss; otherwise defer language tasks")
    parser.add_argument("--seed", type=int, default=43, help="Post-hoc oracle search seed")
    parser.add_argument("--force", action="store_true", help="Repeat already completed checks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = sorted(args.logdir.glob("icl_gen_*/model=*/std/reg/*.json"))
    if not paths:
        parser.error("No std/reg benchmark logs found")
    for path in paths:
        json.loads(path.read_text())  # Preflight readability before changing any file.
    if args.dry_run:
        print(f"Found {len(paths)} std logs; language judge={args.instruction_model or 'deferred'}")
        return 0
    call_model = None
    if args.instruction_model:
        # Load at most once, and only when explicitly requested. No silent change
        # from the original local model to a hosted model/provider.
        if args.instruction_model == "gpt-oss":
            import torch
            if torch.cuda.device_count() < 2:
                parser.error("Original gpt-oss configuration requires two GPUs; use its original compute host")
        from keysecrets import api_key
        from modeling.llm import load_model_and_tokenizer, run_model
        model, tokenizer = load_model_and_tokenizer(args.instruction_model, api_key)
        call_model = lambda prompt: run_model(args.instruction_model, model, tokenizer, prompt, temp=0.0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = args.logdir / "analysis" / ("std_oracle_backup_" + stamp)
    backup.mkdir(parents=True)
    for path in paths:
        target = backup / path.relative_to(args.logdir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"Backup: {backup}", flush=True)
    records = []
    counts = Counter()
    cache = {}
    for index, path in enumerate(paths, 1):
        data = json.loads(path.read_text())
        kind, ident = data["task_type"], data["benchmark_id"]
        crypto_setting = data.get("TaskSettings", {}).get("crypto_setting", "legacy")
        task = _task(kind, ident, args.seed, crypto_setting)
        for runid, run in enumerate(data["runs"]):
            for epoch in run:
                prior = epoch.get("OracleBackfill", {})
                if not args.force and prior.get("status") in ("checked", "invalid_candidate"):
                    counts[(kind, "already_done")] += 1
                    continue
                candidate = epoch.get("Prediction")
                attempts = epoch.get("Attempts", [])
                historical_error = attempts[-1].get("Error") if attempts else None
                meta = {"timestamp_utc": stamp, "seed": args.seed,
                        "candidate_sha256": hashlib.sha256((candidate or "").encode()).hexdigest(),
                        "scope": "logged_epoch_candidate", "original_eval_preserved": True}
                start = time.monotonic()
                if candidate is None or historical_error:
                    meta.update(status="invalid_candidate", reason=historical_error or "No candidate extracted")
                    epoch["Equivalent"] = None
                elif kind == "instruction_induction" and call_model is None:
                    meta.update(status="deferred_llm_judge", reason="Original LLM judge not available; rerun with --instruction-model gpt-oss on original host")
                    # Keep any original verdict untouched; no new verdict claimed.
                else:
                    key = (kind, ident, candidate)
                    try:
                        if key not in cache:
                            # Independent deterministic search per distinct candidate.
                            task = _task(kind, ident, args.seed, crypto_setting)
                            cache[key] = task.verify(candidate, 1, llm_call=call_model)
                            cached = False
                        else:
                            cached = True
                        verdict = cache[key]
                        if verdict.error:
                            raise RuntimeError(verdict.error)
                        epoch.update(verdict.log_fields())
                        if kind == "instruction_induction":
                            meta["judge_model"] = args.instruction_model
                        epoch.pop("OracleError", None)
                        meta.update(status="checked", cached=cached)
                    except Exception as exc:
                        epoch["Equivalent"] = None
                        epoch["OracleError"] = str(exc)
                        meta.update(status="oracle_error", reason=str(exc))
                meta["elapsed_seconds"] = round(time.monotonic() - start, 3)
                epoch["OracleBackfill"] = meta
                counts[(kind, meta["status"])] += 1
                records.append({"file": str(path.relative_to(args.logdir)), "run": runid,
                                "epoch": epoch["epoch"], "equivalent": epoch.get("Equivalent"), **meta})
        temporary = path.with_suffix(".json.backfill_tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n")
        temporary.replace(path)
        print(f"[{index}/{len(paths)}] {kind}/{ident}", flush=True)
    report = {"backup": str(backup), "instruction_model": args.instruction_model,
              "counts": {f"{task}:{status}": n for (task, status), n in sorted(counts.items())},
              "records": records}
    report_path = args.logdir / "analysis" / ("std_oracle_backfill_" + stamp + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False), flush=True)
    print(f"Report: {report_path}", flush=True)
    return 1 if any(status == "oracle_error" for _, status in counts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
