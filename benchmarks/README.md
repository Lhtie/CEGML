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

To run all four default benchmark IDs with `gpt-oss-120b`, regularization, and both standard and CE settings, use [`datasets/benchmarks/run_gpt_oss_120b.sh`](../datasets/benchmarks/run_gpt_oss_120b.sh). It runs direct ICL by default; pass `icl_gen` to use the rule-generation runner. Both invocations execute eight jobs and write logs under `logs/benchmarks/`.

## Direct ICL prediction

Use `train_icl.py` for the same direct input/output ICL experiment as the regex task. Omit `--use_ce` to add each scheduled training example. Add `--use_ce` to predict each scheduled candidate with the current examples and add only mistakes as counterexamples. This CE mode uses the benchmark's labeled training pool; the symbolic and LLM equivalence oracles belong to `train_icl_gen.py`.

```bash
# Baseline
python train_icl.py --task_type cryptography --benchmark_id caesar:0 --mkey gpt5
# Counterexample ICL
python train_icl.py --task_type cryptography --benchmark_id caesar:0 --mkey gpt5 --use_ce
```

Replace `cryptography --benchmark_id caesar:0` with `list_functions --benchmark_id c170`, `instruction_induction --benchmark_id antonyms`, or `scan --benchmark_id simple`. The regular `--tot_train_size`, `--start_size`, `--scale_factor`, `--eval_size`, `--eval_batch_size`, `--retries`, and `--seed` flags apply. Logs go to `logs/icl_<task_type>/model=<mkey>/std|ce/...`.
