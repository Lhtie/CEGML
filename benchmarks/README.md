# Program-induction benchmarks

`python -m benchmarks <name> [options]` runs one of the four added datasets through `train_icl_gen.py`.

| Name | Oracle |
|---|---|
| `cryptography` | cvc5 substitution check |
| `list_functions` | cvc5 sequence query and bounded search |
| `instruction_induction` | LLM judge |
| `scan` | Exhaustive enumeration |

```bash
python -m benchmarks --list
python -m benchmarks cryptography --benchmark_id caesar:0 --mkey gpt5 --use_ce
python -m benchmarks list_functions --benchmark_id c170 --mkey gpt5 --use_ce
python -m benchmarks instruction_induction --benchmark_id antonyms --mkey gpt5 --use_ce
python -m benchmarks scan --benchmark_id simple --mkey gpt5 --use_ce
```

The existing regex task continues to use `train_icl_gen.py` directly. Dataset sources, IDs, and oracle limits are documented in [`datasets/BENCHMARKS.md`](../datasets/BENCHMARKS.md).

To run all four default benchmark IDs with `gpt-oss-120b`, regularization, and both standard and CE settings, use [`datasets/benchmarks/run_gpt_oss_120b.sh`](../datasets/benchmarks/run_gpt_oss_120b.sh). It runs direct ICL by default; pass `--runner icl_gen` to use rule generation, or `--runner both` for both runners. The default smoke suite executes eight jobs per runner. Add `--suite all` to discover and run every locally available task ID and SCAN split. Logs go under `logs/benchmarks/`.

## Direct ICL prediction

Use `train_icl.py` for the same direct input/output ICL experiment as the regex task. Omit `--use_ce` to add each scheduled training example. Add `--use_ce` to predict each scheduled candidate with the current examples and add only mistakes as counterexamples. This CE mode uses the benchmark's labeled training pool; the symbolic and LLM equivalence oracles belong to `train_icl_gen.py`.

```bash
# Baseline
python train_icl.py --task_type cryptography --benchmark_id caesar:0 --mkey gpt5
# Counterexample ICL
python train_icl.py --task_type cryptography --benchmark_id caesar:0 --mkey gpt5 --use_ce
```

Replace `cryptography --benchmark_id caesar:0` with `list_functions --benchmark_id c170`, `instruction_induction --benchmark_id antonyms`, or `scan --benchmark_id simple`. The regular `--tot_train_size`, `--start_size`, `--scale_factor`, `--eval_size`, `--eval_batch_size`, `--retries`, and `--seed` flags apply. Logs go to `logs/icl_<task_type>/model=<mkey>/std|ce/...`.

## Full dataset suites

The full suite currently contains 582 settings: 300 cryptography instances (100 each of Caesar, Atbash, and keyboard), 250 list functions, 24 instruction-induction tasks, and 8 SCAN splits. Every setting runs both standard and CE by default: 1,164 jobs per runner, or 2,328 with `both`. IDs are discovered from local data files rather than inferred from a numeric difficulty order.

```bash
# Preview the complete rule-generation run without calling any model.
python -m benchmarks.run_suite --runner icl_gen --suite all --dry-run

# Run all rule-generation settings (std and CE, with regularization).
python -m benchmarks.run_suite --runner icl_gen --suite all

# Run all settings for direct input/output ICL.
python -m benchmarks.run_suite --runner icl --suite all

# Run both experiment types.
python -m benchmarks.run_suite --runner both --suite all

# Restrict to a benchmark; repeat --task to select more than one.
python -m benchmarks.run_suite --runner icl_gen --suite all --task list_functions

# CE only, continue after failed jobs, and pass additional runner arguments.
python -m benchmarks.run_suite --runner icl_gen --suite all --mode ce --keep-going -- --retries 3 --eval_size 64
```

The existing invocation without `--suite all` still runs the four-instance smoke suite. `--mode std|ce|both` controls methods; `--mkey` controls the model. `PYTHON_BIN` and `OUTDIR` remain supported (also available as `--python` and `--outdir`). Use `--outdir logs/benchmarks_full` to keep a full experiment separate from existing smoke logs: the underlying runners overwrite the same task's output path when rerun. Generation defaults to one run per job, as before; override with `-- --rerun 3` for `icl_gen` only. Options after `--` apply to every selected runner and must be supported by each of them.

“All” means all task IDs/splits, not an automatic expansion of model, regularization, seed, training-size or evaluation-size settings. The existing runner defaults still apply, including the evaluation cap. Jobs run sequentially; a failed job stops the suite unless `--keep-going` is supplied, and any failed job yields a nonzero final exit status. Dry-run only discovers data and prints commands; it does not create result logs or call models. The launcher can also be used directly with `python -m benchmarks.run_suite`.

## Smaller suite with stratified coverage

Use `--suite balanced` to reduce the number of settings while retaining broader complexity coverage than the smoke suite:

- Cryptography: 4 reproducibly sampled instances per cipher family (12 total). This balances families, not difficulty: instances from the same family need not differ substantially in difficulty.
- List Functions: sort the 250 reference programs by a structural score, divide into three equal-size rank bands, and sample 8 from each (24 total).
- Instruction Induction: keep all 24 distinct task types.
- SCAN: keep all 8 splits.

This gives 68 settings / 136 std-and-CE jobs per runner, down from 1,164 jobs. Both methods receive exactly the same selected IDs. The structural score is AST node count + 8 per for/while + 4 per if/conditional expression + 3 per call + 2 per subscript + 5 per comprehension clause. It ignores comments/docstring text length; equal scores are ordered by ID. The three bands approximate reference-program complexity, **not measured model difficulty**; sampling is uniform within each band, not uniform across observed accuracy. All instruction task types and SCAN splits are retained rather than assigned unvalidated difficulty labels. This is a stratified diagnostic subset, not an unbiased estimate of the full dataset average.

```bash
# Preview the default 136-job subset.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --dry-run

# Run it in a separate output directory.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --outdir logs/benchmarks_balanced

# Smaller variant: 4 list tasks per band and 2 instances per cipher family.
# 50 settings / 100 std-and-CE jobs; all instruction tasks and SCAN splits remain.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --list-per-tier 4 --crypto-per-family 2
```

`--sample-seed` defaults to 43 and controls selection only; it is distinct from the runner's `--seed` passed after `--`. Selection for a task stays the same when using `--task` filters. Dry-run prints the exact selected IDs and commands for recording the experiment. If 1,494 seconds was the previous eight-job smoke total, linear extrapolation gives about 7.1 hours for 136 jobs; complex tasks may change this substantially. Empirical difficulty strata should be recalibrated after actual pilot results become available.
