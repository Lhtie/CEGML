# Program-induction benchmarks

`python -m benchmarks <name> [options]` runs one of the registered datasets through `train_icl_gen.py`.

| Name | Oracle |
|---|---|
| `cryptography` | cvc5 symbolic execution of a generated Python string program |
| `list_functions` | cvc5 sequence query and bounded search |
| `instruction_induction` | LLM judge |
| `scan` | Exhaustive enumeration |
| `ilp` | cvc5 finite Horn least-model encoding and replayed counterexamples |

```bash
python -m benchmarks --list
python -m benchmarks cryptography --benchmark_id caesar:0 --mkey gpt5 --use_ce
python -m benchmarks list_functions --benchmark_id c170 --mkey gpt5 --use_ce
python -m benchmarks instruction_induction --benchmark_id antonyms --mkey gpt5 --use_ce
python -m benchmarks scan --benchmark_id simple --mkey gpt5 --use_ce
```

The existing regex task continues to use `train_icl_gen.py` directly. Dataset sources, IDs, and oracle limits are documented in [`datasets/BENCHMARKS.md`](../datasets/BENCHMARKS.md).

To run all registered default benchmark IDs with `gpt-oss-120b`, regularization, and both standard and CE settings, use [`datasets/benchmarks/run_gpt_oss_120b.sh`](../datasets/benchmarks/run_gpt_oss_120b.sh). It runs direct ICL by default; pass `--runner icl_gen` to use rule generation, or `--runner both` for both runners. The default smoke suite executes ten jobs per runner. Add `--suite all` to discover and run every locally available task ID and SCAN split. Logs go under `logs/benchmarks/`.

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

The full suite currently contains 697 settings: 300 cryptography instances (100 each of Caesar, Atbash, and keyboard), 250 list functions, 24 instruction-induction tasks, 104 SCAN splits, and nineteen ILP configurations. Every setting runs both standard and CE by default: 1,394 jobs per runner, or 2,788 with `both`. IDs are discovered from local data files rather than inferred from a numeric difficulty order.

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

The existing invocation without `--suite all` still runs the five-instance smoke suite. `--mode std|ce|both` controls methods; `--mkey` controls the model. `PYTHON_BIN` and `OUTDIR` remain supported (also available as `--python` and `--outdir`). Use `--outdir logs/benchmarks_full` to keep a full experiment separate from existing smoke logs: the underlying runners overwrite the same task's output path when rerun. Generation defaults to one run per job, as before; override with `-- --rerun 3` for `icl_gen` only. Options after `--` apply to every selected runner and must be supported by each of them.

“All” means all task IDs/splits, not an automatic expansion of model, regularization, seed, training-size or evaluation-size settings. The existing runner defaults still apply, including the evaluation cap. Jobs run sequentially; a failed job stops the suite unless `--keep-going` is supplied, and any failed job yields a nonzero final exit status. Dry-run only discovers data and prints commands; it does not create result logs or call models. The launcher can also be used directly with `python -m benchmarks.run_suite`.

## Smaller suite with stratified coverage

All 697 locally runnable setting IDs now have versioned, task-specific difficulty annotations; the full pool is annotated within each task; native configurations are all retained, not just the smaller suite. The annotations are a priori proxies, not calibrated LLM difficulty. Training/test IO and oracle domains are unchanged.

| Task | Full-pool levels × IDs per level | Default balanced levels × IDs per level |
|---|---|---|
| Cryptography | 3 × 100 | 3 × 2 |
| List Functions | 5 × 50 | 5 × 2 |
| Instruction Induction | 3 × 8 | 3 × 8 |
| SCAN | 4 × 26 | 4 × 2 |
| ILP | 3 tiers: 4/8/7 | 3 × 2 |

Balanced has 54 settings / 108 std-and-CE jobs per runner; all retains 697 settings / 1,394 jobs. List Functions now uses five AST rank bands and `--list-per-tier` defaults to 2 (ten selected IDs). Crypto keeps `--crypto-per-family=2`, using cyclic quotas to balance both family and coverage-tier marginals. These are not equal counts in every family×tier cell. Instruction retains all 24 IDs; SCAN samples two of 26 IDs per tier; ILP retains the same six selected IDs. The all suite keeps every native configuration, including the naturally unequal ILP tier sizes.

Crypto tiers measure training-letter coverage, List Functions uses reference AST complexity, Instruction Induction uses explicit manual task-contract strata, SCAN uses a heuristic generalization-protocol ordering, and ILP uses relational/recursive construction. Levels are not comparable across tasks. Rank-band ties and manual annotation limits are documented. See the [complete 697-ID inventory and rationale](../analysis/benchmark_difficulty/setting_difficulty_catalog_zh.md), [CSV](../analysis/benchmark_difficulty/all_setting_difficulty.csv), and [machine-readable catalog](../datasets/benchmarks/difficulty_catalog.json).

```bash
# Preview the default 108-job subset.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --dry-run

# Run it in a separate output directory.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --outdir logs/benchmarks_balanced

# Slightly larger variant: 4 list tasks per band and 2 instances per cipher family.
# 64 settings / 128 std-and-CE jobs; all instruction tasks remain and SCAN samples 8 of 104 splits.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --list-per-tier 4 --crypto-per-family 2
```

`--sample-seed` defaults to 43 and controls selection only; it is distinct from the runner's `--seed` passed after `--`. Selection for a task stays the same when using `--task` filters. Dry-run prints the exact selected IDs and commands for recording the experiment. The current selection changes the number of list-function bands and IDs, so older runtime estimates do not transfer directly; oracle cost and early stopping also affect runtime.

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

All registered benchmarks call the **same `train_icl_gen.py::run_episode` used by the existing regex experiments**. There is no separate benchmark epoch/retry/selection loop. `run_benchmark` only loads data, constructs the domain adapter, calls `run_episode`, and converts checkpoint callbacks to the existing benchmark log format.

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

SCAN now retains all 104 paired train/test splits from the pinned official repository revision `c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f`, including basic, size variations, additional jump examples, filler and few-shot replications. All original lines are preserved. These are native split conditions, not 104 independent language targets. `datasets/scan/source_manifest.json` records source paths, hashes and row counts; `python datasets/scan/prepare.py` restores the pinned files. Generating a Python interpreter and using CE remain our method adaptation, not the original sequence-learning protocol.

Our prompt requests `transform(x)`, returning a space-separated action string, with concise reasoning/ans tags. The executor now accepts standard Python, including helper functions, loops and imports, with a fresh namespace and a 10-second per-input timeout. It is not a security sandbox. Official train data remain available in full; the configured example budget and epoch schedule determine which training examples reach the LLM. Full selected-split test accuracy is exact sequence matching, regardless of the generic evaluation sample limit.

The oracle is brute-force enumeration over all 20,910 entries of `tasks.txt`, comparing candidate outputs against published outputs. Exceptions/timeouts count as failures. It stops once the requested number of counterexamples is found; `Equivalent: true` requires no failures on the entire finite command set. SCAN reuses candidate/input execution results across oracle, training and evaluation. Each of these stages checks a 30-second elapsed-time budget between inputs (a current input may add up to its 10-second timeout). Incomplete oracle search without a witness returns null; incomplete training fit or evaluation returns null rather than presenting a partial score as full accuracy. Coverage, stop reason and progress are recorded/printed. This does not cover arbitrary strings outside SCAN. `OracleDetails` records coverage and disagreement count. Because the domain includes the held-out split, CE can reveal test commands/labels; this is the intended oracle-assisted method extension and is not the original train-only generalization evaluation. Historical logs are unchanged.

## Instruction Induction: grounded semantic teacher (v3)

The teacher now uses the official human annotations under `datasets/instruction_induction/annotations`, induction-only reference examples spread across the whole induction pool, and task-specific acceptance rules. The learner still sees IO examples only. `cause_and_effect` is corrected to selecting the cause from two sentences in alternating order, matching the official prepared induction prompts; historical cause-to-effect logs describe a different task and are not rewritten.

Prompt construction and CE validation live in `benchmarks/instruction_oracle.py`. Calibration examples address arithmetic shortcuts, one-word animal shortcuts, legal answer alternatives, set ordering, and task domains. `Equivalent` remains a noisy semantic task-alignment verdict, not execution accuracy or a formal proof. Actual output execution scores for Instruction Induction remain unavailable. CE schema, known answer aliases, deterministic labels for supported tasks and simple-past passive domain are checked before feedback. Invalid witnesses cannot cause rejection, and an unsubstantiated `false` or contradictory JSON yields `null`, not automatic acceptance. Novel open-language CE semantics still depend on the same LLM; there is no independent executor certifying its proposed candidate output.

Future logs include policy, actual oracle prompt, raw verdict, rejection reasons and GT source under `OracleDetails`. No execution/test examples are used to build or validate teacher context. The shared regex episode controller is unchanged. Historical audit, reviewed cases, prompt snapshots and gpt-oss-api replays are in `analysis/instruction_oracle_review/`; original benchmark logs are preserved. The replay results are selected diagnostic cases, not an unbiased estimate of oracle accuracy or updated experiment success rates.

## ILP: HRI-inspired scaleup tasks

`ilp/even_succ` adds positive Horn rule induction from [Glanois et al. (ICML 2022)](https://proceedings.mlr.press/v162/glanois22a.html). It is registered beside List Functions rather than flattened into a Python list classifier. The vendored author generator/configuration is pinned and licensed under `datasets/ilp/source`; regenerate the deterministic data with `python datasets/ilp/prepare.py`.

The learner receives task-specific relational background knowledge and Boolean target atoms. It returns recursive, function-free unary/binary Horn clauses inside ans tags, with the shared concise reasoning summary. Runtime scaleup pools contain all 384 unary or 400 binary training atoms, with a default scheduling budget of 384, with finite least-fixed-point inference. Unary targets use N=384/512 for training/evaluation; binary targets use N=20/24 with all 400 training and 576 evaluation atoms. The original author 11/15 contexts and six/eight-round budgets remain in the dataset as provenance, not runtime semantics. Standard generation follows the unchanged regex geometric schedule: 3, 6, 12, 24, 48, 96, 192, 384 accumulated examples by default. CE starts with the same first three examples, adds up to eight oracle counterexamples per round and allows ten epochs by default. Both methods stop early on equivalence; ten epochs are a budget, not a guaranteed number of generations. `--start_size`, `--tot_train_size`, `--scale_factor` and `--ce_epochs` remain configurable through the shared runner. Auxiliary predicate invention is supported; Python/modulo shortcuts and numeric constants in rules are excluded. The oracle checks every evaluation atom in the declared context, returns real ground-atom CE and preserves the shared std/CE stopping contract. Accuracy and hard-label MSE/1e-4 success are logged.

This is an LLM+CE task adaptation, not reproduction of HRI's fuzzy optimizer, proto-rule hierarchy, Gumbel training noise or ten-run experiment. Runtime syntax/resource limits and finite inference semantics are explicit in TaskSettings. Finite acceptance does not prove rule equivalence over arbitrary domains. See [dataset provenance](../datasets/ilp/README.md), [Chinese setting/usage report](../analysis/ilp_integration/setting_and_usage_zh.md) and [undecidable SMT task design notes](../analysis/ilp_integration/undecidable_smt_task_notes_zh.md).

```bash
# Preview both methods without a model call.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --dry-run

# Run std and CE with the same shared generation controller.
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --mkey gpt-oss --outdir logs/benchmarks
```

`--runner icl` is also supported as the existing direct-IO baseline, using Boolean predictions rather than generated rules; its mistake-driven CE is distinct from the generation oracle. Direct ICL now honors task-declared sample/shuffle defaults, including the previously declared Crypto/List Functions paper defaults.

`--task` accepts multiple names together (and remains repeatable), e.g. `--task ilp instruction_induction`.

Balanced selects six ILP tasks from an explicit three-tier inventory of all 19 author synthetic configurations (two per tier): easy `predecessor`, `undirected_edge`; medium `grandparent`, `graph_coloring`; hard `fizz`, `length`. Current catalog tiers use direct relations/ordered-chain closure, relational joins/general-graph paths, and recursive numerical/list construction; these are not measured LLM difficulty or a uniform sample of task identities. Selection is fixed and independent of model results and `--sample-seed`. All nineteen author synthetic configurations are now locally available under `--suite all`. EvenOdd/EvenSucc and Grandparent/Grandparent_NLM overlap in target semantics and are recorded as native configurations, not independent concepts. The seven newly added configurations use the pinned author generators; the prior backgrounds and labels are preserved, with binary training pools completed to all 400 atoms. Every task has 384 training atoms; unary tasks evaluate 512 atoms, binary tasks evaluate all 24×24=576 atoms. Graph/family tasks use seeded sparse backgrounds rather than the author's sampler. Historical logs remain unchanged. See [selection rationale](../analysis/ilp_integration/balanced_six_selection_zh.md).


## Token accounting (regex and benchmarks)

New round logs include `InputTokens` and `OutputTokens`. `TokenUsage` contains the same totals, `TotalTokens`, `ModelCalls`, `UsageComplete`, known partial totals, and a `Calls` array. Round totals include every LLM call within that round: candidate generation, retries/repairs, CE generation and LLM oracle verification where applicable. Judging multiple candidates from one response does not multiply its token count. Each generation attempt/query also retains its own `TokenUsage`.

API calls use provider `prompt_tokens`/`completion_tokens`; output tokens therefore include reasoning tokens when the provider includes them. Provider-reported reasoning counts are preserved per call when available. vLLM uses actual prompt/completion token IDs; Transformers uses actual input/generated sequence lengths. Counts include model input formatting and generated special tokens, not just visible response text. No extra generation, tokenization pass or model call is made to collect usage.

When exact usage is unavailable or a call fails without usage, the corresponding total is `null`, `UsageComplete` is false, and known partial totals remain available. Missing usage is never treated as zero. Round scopes are restored after early stopping or exceptions. Existing model callers continue receiving string-compatible responses. Historical logs are not backfilled.

`NumTrainingSamples` already records accumulated training entries used by the round, including added CE, and is not a unique-example count or number of tokens. Generation logs and benchmark direct-IO logs store an integer; the original regex direct-IO runner retains its existing `{positive, negative, total}` schema. Validation queries, oracle inputs and repeated presentations across retries are not additional training entries.


### ILP SMT oracle

ILP now uses the cvc5 Python API (no `cvc5` executable). `finite_Horn_ranked_smt_v1` checks the full training label pool and every evaluation target atom. A resource-bounded relevance grounder builds ground rule supports; query-directed slicing then removes auxiliary atoms irrelevant to these targets. The grounder does positive support closure, so SMT does not eliminate finite grounding cost. The oracle does not compare Python prediction labels and merely wrap that verdict in SMT: candidate atoms are Boolean variables; Horn closure and well-founded integer derivation ranks encode their exact least-model truth. A mismatch query against the GT labels drives SAT counterexample extraction or UNSAT acceptance.

SAT witnesses are replayed through the independent Horn executor and stored with their background `num_constants`. `true` requires UNSAT for the complete combined domain; `false` requires a replayed witness; SMT unknown, time/grounding limits or a failed replay do not turn into acceptance. `OracleDetails` records context sizes, support/encoded-atom counts, status, solver query count, elapsed time and proof scope. Defaults: 10 seconds per SMT query, 30 seconds for the whole search, 10 million grounding/join work items, 300,000 facts/supports and 200,000 intermediate bindings. Runtime scoring has its own bounds; an oracle resource-limit result skips repeated expensive scoring. Empty background relations retain their declared arity.

This remains finite-context equivalence, not a proof over arbitrary graphs or arbitrary domain sizes. Training/graph samplers and GraphColoring's existing unary target are unchanged. Previously generated logs retain their original oracle policy. See [implementation and validation](../analysis/ilp_integration/smt_oracle_implementation_zh.md).


## Full difficulty inventory and recorded selections

`python -m benchmarks.difficulty` regenerates the catalog, including source SHA-256 checksums. The catalog covers every `--suite all` ID. New benchmark logs record `TaskSettings.Difficulty`; non-default Crypto legacy/OOD settings are flagged as outside the classification basis. Classification metadata is not inserted into learner prompts.

Each normal suite run writes `suite_manifest_<suite>_<mode>_<plan-hash>.json` to its output directory. It records all selected IDs, difficulty metadata, model/mode/runner and extra arguments. Override the path with `--manifest PATH`; explicit manifests can also be exported with `--dry-run`. No model calls are made by catalog generation or dry runs. Historical logs remain unchanged.

```bash
python -m benchmarks.run_suite --runner icl_gen --suite all --dry-run --manifest analysis/benchmark_difficulty/all_settings.json
python -m benchmarks.run_suite --runner icl_gen --suite balanced --dry-run --manifest analysis/benchmark_difficulty/balanced_settings.json
```


## Native dataset coverage

Instruction Induction has 24 native task types, containing 67,662 induce IO and 2,156 execute IO in this checkout. Full execute IO is now loaded by default; a positive `--eval_size` explicitly limits supported tasks. Regex keeps its existing default of 32 evaluation examples. Direct-IO ICL evaluates all loaded IO; instruction generation still uses its semantic oracle, not a new hidden executor over the execute split.

Every SCAN split loads its complete original train/test rows; all 104 native conditions are available in `all`. The oracle still checks the 20,910-command universe. ILP covers all 19 author synthetic configurations; new configurations are generated from pinned author code at our declared scaleup sizes, not copied from a nonexistent fixed ILP sample file. WN remains a separate unsupported knowledge-graph task.

The per-run benchmark sample budget now defaults to the complete training pool; `-- --tot_train_size 0` explicitly selects this same default (fixed paper protocols retain their task-specific defaults). Equivalence early stopping remains active, and large training pools can exceed model context limits if reached. Logs distinguish `TrainingPoolSize`, `ScheduledTrainingBudget` and `EvaluationPoolSize`. This change does not concatenate every original IO into the first prompt.

See [native coverage and per-task counts](../analysis/benchmark_difficulty/native_dataset_coverage_zh.md) and the updated [full difficulty inventory](../analysis/benchmark_difficulty/setting_difficulty_catalog_zh.md). Existing logs are unchanged.


## Default budgets: full training pool, ten rounds

Benchmarks default to the full available training IO pool (`--tot_train_size 0` has the same meaning). Std geometrically redistributes cumulative sample counts from start_size to that full budget over ten rounds, reaching the full pool in round ten unless equivalence stops earlier. Small pools may repeat sizes. This replaces truncation or padding of the previous doubling schedule. `--scale_factor` applies to the natural schedule restored by `--rounds 0`; a fixed positive round budget determines the effective geometric growth ratio.

CE has ten rounds by default and adds oracle counterexamples as before; it does not forcibly consume the whole training pool. Explicit `--ce_epochs` overrides CE rounds. Explicit positive `--tot_train_size` sets a sample cap for tasks without fixed paper-protocol overrides. Equivalence early stopping remains active. Large full pools can exceed model context limits if reached; no silent truncation is added. New logs record TrainingPoolSize, ScheduledTrainingBudget, RoundBudget and StandardSampleSchedule.

Original regex has no fixed finite native IO pool, so its omitted sample-budget default remains 384; its ten-round schedule also fits the chosen budget. Direct-IO benchmark runners use the same full-pool/ten-round defaults.
