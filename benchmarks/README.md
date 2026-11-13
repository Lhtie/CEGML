# Program-induction benchmarks

`python -m benchmarks <name> [options]` runs one of the four added datasets through `train_icl_gen.py`.

| Name | Oracle |
|---|---|
| `cryptography` | cvc5 symbolic execution of a generated Python string program |
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

- Cryptography: 2 reproducibly sampled instances per cipher family (6 total). This balances families, not difficulty: instances from the same family need not differ substantially in difficulty.
- List Functions: sort the 250 reference programs by a structural score, divide into three equal-size rank bands, and sample 3 from each (9 total).
- Instruction Induction: keep all 24 distinct task types.
- SCAN: keep all 8 splits.

This gives 47 settings / 94 std-and-CE jobs per runner, down from 1,164 jobs. Both methods receive exactly the same selected IDs. The structural score is AST node count + 8 per for/while + 4 per if/conditional expression + 3 per call + 2 per subscript + 5 per comprehension clause. It ignores comments/docstring text length; equal scores are ordered by ID. The three bands approximate reference-program complexity, **not measured model difficulty**; sampling is uniform within each band, not uniform across observed accuracy. All instruction task types and SCAN splits are retained rather than assigned unvalidated difficulty labels. This is a stratified diagnostic subset, not an unbiased estimate of the full dataset average.

```bash
# Preview the default 94-job subset.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --dry-run

# Run it in a separate output directory.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --outdir logs/benchmarks_balanced

# Slightly larger variant: 4 list tasks per band and 2 instances per cipher family.
# 50 settings / 100 std-and-CE jobs; all instruction tasks and SCAN splits remain.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --list-per-tier 4 --crypto-per-family 2
```

`--sample-seed` defaults to 43 and controls selection only; it is distinct from the runner's `--seed` passed after `--`. Selection for a task stays the same when using `--task` filters. Dry-run prints the exact selected IDs and commands for recording the experiment. Using the observed 6.5 hours for the former 68-setting balanced suite, proportional scaling gives approximately 4.5 hours for the new 47-setting suite (std and CE combined, one runner). This is an estimate: per-task cost, oracle changes and early stopping affect runtime. Restore the former sample size with `--list-per-tier 8 --crypto-per-family 4`.

## Equivalence logging for rule generation

`train_icl_gen.py` checks every valid standard **and** CE candidate with the same task oracle. Epoch logs contain `Equivalent` and `OracleCounterexamples`; the latter are verification evidence, not standard training examples. Instruction Induction also records `OracleResponse` (an additional LLM judge call per valid candidate). Both standard and CE stop the current run as soon as a candidate receives `Equivalent: true`, after saving the epoch log. Until that point, standard follows its scheduled samples and never receives oracle witnesses in prompts. `false` or `null` does not trigger equivalence-based early stopping.

Oracle failures are recorded as `OracleError` (and the legacy `Error`) with `Equivalent: null`; test-set scoring is attempted independently. List Functions now returns `true` when a completed nonempty bounded search finds no witness, as a noisy operational oracle, while Instruction Induction uses a heuristic judge. Neither is a global equivalence proof. `OracleDetails` records list search scope and policy; solver unknown/errors and empty searches remain `null` when no counterexample is found. Mutable list inputs are copied before candidate/reference execution so verification cannot change training data, test data, or recorded witnesses. Existing logs are not backfilled: these fields and protections apply to future runs.

## Backfill verification in existing standard logs

```bash
# Offline cryptography/list/SCAN; language candidates explicitly marked deferred.
python -m benchmarks.backfill_oracles --logdir logs/benchmarks_balanced

# Include the original language judge on its original two-GPU compute host.
python -m benchmarks.backfill_oracles --logdir logs/benchmarks_balanced --instruction-model gpt-oss
```

Each invocation backs up all matching std/reg files under the log directory's analysis folder before atomic in-place updates. Existing prompts, candidates, scores and training entries are preserved. Invalid original attempts remain unverified with an explicit status. Previously checked candidates are skipped unless `--force` is supplied. Instruction tasks require an explicit judge model and may incur additional model calls; there is no automatic provider substitution. Post-hoc searches use `--seed 43` by default and are distinct from the original run's random state. The report records per-epoch provenance and errors.

## Shared regex episode pipeline

All four benchmarks now call the **same `train_icl_gen.py::run_episode` used by the existing regex experiments**. There is no separate benchmark epoch/retry/selection loop. `run_benchmark` only loads data, constructs the domain adapter, calls `run_episode`, and converts checkpoint callbacks to the existing benchmark log format.

The regex entry point still calls `run_episode` without an adapter. Its data scheduling, initial CE failure fallback, reflection modes, training-fit repair loop, candidate-batch mode, best-retry selection, callback order, and acceptance-based stopping behavior are retained.

`BenchmarkEpisodeAdapter` supplies only domain hooks:

- JSON input/output example formatting and task-specific prompt text.
- A teacher implementation: `task.verify` returns oracle verdicts and witnesses; executable training examples provide repair feedback; supported test execution supplies diagnostic scores.
- Reflection/repair prompts that describe programs or instructions rather than regex syntax.
- Serialization of the shared epoch result. Flat benchmark fields remain available, with detailed attempts in `Logs` and the selected `CurrentGuess`/`CurrentGuessReasoning`.

The shared controller decides when to sample examples, reflect, retry, stop repairing a training-consistent candidate, select the best retry, emit checkpoints, and finish the run after oracle acceptance. Default benchmark configuration follows the same `agentic_reflection` semantics as regex. Increase `--retries` to allow within-epoch repair, or choose another `--reasoning_mode` using the existing runner options. Candidate-batch repair is supported through the same shared controller. Natural-language instructions do not have executable training-fit scores; their semantic failures come from the LLM oracle at the next CE epoch.

Oracle backends differ in evidence, not the controller: bounded cvc5 symbolic execution (Cryptography), bounded noisy search (List Functions), LLM judgment (Instruction Induction), and finite command enumeration (SCAN). A `true` verdict ends the run for either std or CE; `false` and `null` do not. Standard never receives oracle witnesses in its training prompts.

CE can reuse witnesses from verification of the selected candidate. When an oracle is inconclusive, labeled supplemental examples are obtained by the task adapter; oracle errors use the **original regex controller's epoch-slice fallback**. `TrainingExampleSource` records initial/counterexample/supplemental/fallback/scheduled, while the legacy `Counterexamples` field is retained for compatibility. Samples accumulate with the original regex controller's semantics, without the old benchmark-specific deduplication or no-new-example early exit. Only oracle acceptance or the configured budget ends a normal run.

`BenchmarkHandle` and the post-hoc backfill tool use the same `task.verify` contract. Existing logs and analysis snapshots have not been rewritten. Future benchmark runs may differ from old runs because they now use the original regex repair/selection flow, preserve previous reasoning, request a short reasoning summary, and follow its fallback/data-accumulation behavior.

## Migrate historical benchmark logs to the shared episode schema

```bash
# Preview the default scope: both logs/benchmarks and logs/benchmarks_balanced.
python -m benchmarks.migrate_logs --dry-run

# Back up and migrate those historical files in place.
python -m benchmarks.migrate_logs

# Limit the scope to a single experiment directory.
python -m benchmarks.migrate_logs --logdir logs/benchmarks_balanced
```

Migration reconstructs `Logs`, `Attempts`, `CurrentGuess`, `CurrentGuessReasoning`, `Accuracy`, and source/status fields from recorded evidence. It preserves prompts, responses, scores and candidate selection; it does not rerun the model, oracle or new repair algorithm. Only an existing `Equivalent: true` triggers truncation after that epoch. Older `null` values are not reinterpreted under the newer bounded-acceptance oracle policy.

`LogMigration` explicitly identifies historical execution with a new log schema. Unrecorded reasoning summaries, training scores, repair examples and earlier retry prompts remain null; the complete original Response remains available. Ambiguous historical CE/fallback entries use `TrainingExampleSource=unknown_legacy`. Generation-free terminal events are preserved in migration metadata rather than turned into fictitious LLM attempts. An accepted recheck of the same preceding candidate is attached as `OracleAuditEvents` to that candidate. All source files are archived and byte-verified before the first update, under `logs/benchmark_log_migrations/`. The migration report records original/migrated hashes and removed epoch indices. Re-running migration skips already migrated files.

## Cryptography: original paper task protocol

Cryptography rule generation now defaults to the task interface from *Patterns Over Principles* ([author prompt](https://github.com/HKUST-KnowComp/Robust-Rule-Induction/blob/main/prompts/crypto_prompt.py), [author data mixing](https://github.com/HKUST-KnowComp/Robust-Rule-Induction/blob/main/tasks/base.py)):

- The model receives 10 published input/output examples and generates a complete **Python function `fn(x)`**, taking a string and returning a string. Bare functions, Python fences, and explicit ans wrappers are parsed. JSON cipher names are not accepted in this setting.
- The prompt does **not** reveal Caesar/Atbash/Keyboard, enumerate candidate rules, give a shift parameter example, or provide an implemented encryption function.
- Examples use the author's `Input: ...` / `Output: ...` format and clean base prompt. Only clean labels are supported: the default uses 10 normal examples; optional correctly labeled OOD examples can replace normal examples while keeping the total at 10. There is no noise option; published noise examples are never loaded into the observed training set.
- The task defaults override the incremental benchmark sample schedule: std starts with all 10 examples, and CE starts with the same 10 examples. Up to 10 published test examples are executed; the sample accuracy is `scoreEvalSet`, and a task is fully test-correct iff that score is 1.0 over all 10. Explicitly capping `--eval_size` below 10 no longer matches the full paper test protocol.
- The prompt requests a concise summary in `<reasoning>...</reasoning>` and the complete Python function in `<ans>...</ans>`, using the shared benchmark output instruction. This output format extends the original paper prompt. No extra regularization preamble is appended, even if the suite passes `--use_reg`. Within-epoch repair, CE refinement and acceptance stopping continue to use the shared regex episode controller.

```bash
# Clean paper task interface, std and CE, balanced Cryptography IDs.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task cryptography --outdir logs/benchmarks_crypto_paper

# Add published OOD examples if desired (still ten examples total).
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task cryptography --outdir logs/benchmarks_crypto_paper -- --crypto-ood-ratio 0.1
```

These are rule-generation (`icl_gen`) options. The original paper's SRR/SC/SR methods, model choice and noise-consistency aggregation are **not reproduced by our shared CE method**. This change aligns the task, data mixtures and program hypothesis space; it does not label our method as a reproduction of every paper experiment. Direct IO ICL remains a separate experiment type.

The Cryptography oracle now uses the **cvc5 Python API** to symbolically execute the candidate `fn(x)` and assert that it disagrees with the known reference transformation. Every position is an arbitrary ASCII letter (`A–Z`, `a–z`); lengths 0 through 10 are checked, without enumerating mappings or sampling words. Neither training words nor held-out test words are used in this query. A SAT model is replayed through the Python executor before being returned as an input/output counterexample. Runtime failures also count as disagreements.

If all lengths are UNSAT, `Equivalent: true` means equivalence **only on this bounded domain**, and the shared std/CE controller stops. Unsupported Python constructs and solver unknown/timeout yield `null`, with no random-search fallback. Each solver check has a 10-second limit. `OracleDetails` records the backend, domain, policy, searched/proved lengths, query count and status. Supported constructs include bounded `for` loops, branches, arithmetic and modulo, character indexing, `ord`/`chr`, ASCII case operations, string concatenation, `join` comprehensions and translation tables. Arbitrary Python (for example `while`, nested helper functions or symbolic comprehension filters) is intentionally inconclusive. The closed-JSON SMT oracle remains only for explicit compatibility with historical logs; the backfill tool selects that mode for old files without paper-task metadata. Existing logs are not rewritten by this change.

## List Functions: original clean task protocol

List Functions uses the clean task interface from *Patterns Over Principles* ([author prompt](https://github.com/HKUST-KnowComp/Robust-Rule-Induction/blob/main/prompts/list_function_prompt.py), [author generator](https://github.com/HKUST-KnowComp/Robust-Rule-Induction/blob/main/tasks/list_function.py)). The 250 published reference functions and their IO data are unchanged. Both std and CE start with all 10 published normal examples, without observation noise; the default evaluation executes all 10 published test examples. An explicit `--eval_size` below 10 caps evaluation and does not match the full test protocol.

The model generates `fn(x)`, taking and returning a list of integers. The prompt does not disclose the reference function or restrict the Python language to arithmetic/list operations. Normal Python constructs, helper functions and imports are executed in a fresh namespace with a 10-second timeout per call, like the author execution approach. This executor is not a security sandbox. Historical `transform(x)` candidates remain executable. Fenced code and ans-wrapped code are extracted before execution.

As requested for our experiments, the shared prompt still requests concise `<reasoning>` and `<ans>` blocks; these tags extend the original prompt. Extra regularization is disabled even when the suite passes `--use_reg`. TaskSettings records the clean paper task protocol. Shared retry/CE/acceptance control and the existing bounded noisy list oracle remain our method, not a reproduction of the original paper's SRR algorithm or a global equivalence proof. The list oracle's symbolic fragment is unchanged in expressive scope: an unsupported SMT translation uses the existing concrete search, rather than rejecting valid Python. Existing logs are not rewritten.

## SCAN: official data and exhaustive oracle

SCAN retains the official command/action pairs and eight locally available splits (`simple`, `length`, two `addprim` splits and four `template` splits), without reducing grammar or rewriting outputs. The source repository also provides other experimental splits not included in this local suite; these eight should not be described as every setting in the original paper. The original task predicts action sequences with a sequence model ([author repository](https://github.com/brendenlake/SCAN)); generating a Python interpreter and using CE is our method extension, not an original-paper prompt or training protocol.

Our prompt requests `transform(x)`, returning a space-separated action string, with concise reasoning/ans tags. The executor now accepts standard Python, including helper functions, loops and imports, with a fresh namespace and a 10-second per-input timeout. It is not a security sandbox. Official train data remain available in full; the configured example budget and epoch schedule determine which training examples reach the LLM. Full selected-split test accuracy is exact sequence matching, regardless of the generic evaluation sample limit.

The oracle is brute-force enumeration over all 20,910 entries of `tasks.txt`, comparing candidate outputs against published outputs. Exceptions/timeouts count as failures. It returns at most the requested number of counterexamples; `Equivalent: true` requires no failures on the entire finite command set. This does not cover arbitrary strings outside SCAN. `OracleDetails` records coverage and disagreement count. Because the domain includes the held-out split, CE can reveal test commands/labels; this is the intended oracle-assisted method extension and is not the original train-only generalization evaluation. Historical logs are unchanged.
