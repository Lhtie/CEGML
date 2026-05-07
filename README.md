# CEGML

This repository contains code for learning regular expressions from labeled
examples with LLM learners, counterexample-guided feedback, and several ablation
settings. The main experiments use two regex domains:

- `simplyrx`: simple regular expressions over a small alphabet.
- `extrx`: extended regexes with character classes, intersection, complement,
  and bounded repetition.

## Repository Layout

```text
.
|-- train_icl_gen.py                  # Main LLM regex-learning experiment runner
|-- curve.py                          # Result aggregation and plotting
|-- dataset.py                        # Dataset generation utilities
|-- teacher.py                        # Counterexample and verifier feedback
|-- learner.py                        # LLM learner wrapper
|-- tasks/
|   |-- rl.py                         # Regex task definitions and DFA/FST utilities
|   `-- utils.py                      # DFA helpers, parsing, acceptance checks
|-- prompting/
|   `-- rl.py                         # Prompt templates and task instructions
|-- modeling/
|   `-- llm.py                        # Model/API/vLLM loading and inference helpers
|-- datasets/
|   `-- scaleup/
|       |-- regex_list.json           # Regex benchmark split by dataset/#states/StarDepth
|       |-- regex_datasets/           # Per-regex train/eval JSON datasets
|       |-- run_simplyrx.sh           # Generated large-scale Simple Regex commands
|       |-- run_extrx.sh              # Generated large-scale Extended Regex commands
|       `-- run_scripts/              # Main experiment and ablation shell scripts
|-- logs/
|   |-- scaleup/                      # Main experiment logs
|   |-- opt_prompt/                   # Prompt-optimization experiment logs
|   `-- scaleup/summary/              # Generated LaTeX/Markdown summary tables
|-- accuracy_curves/                  # Generated plots
`-- GEPA_opt/                         # Prompt optimization utilities
```

## Setup

Install the Python dependencies used by the codebase, including:

- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `pyformlang`
- `pynini`
- `transformers`
- `openai`
- `tiktoken`
- `vllm` if running local `gpt-oss`
- `gepa` if using `GEPA_opt/`

API keys are read from `keysecrets.py`:

```python
api_key = {
    "openai": "...",
    "together": "...",
    "google": "...",
    "deepseek": "...",
    "claude": "...",
}
```

Model aliases are defined in `modeling/llm.py`, e.g. `gpt-oss`, `gpt5`,
`gpt5.4-medium`, `gpt5.4-xhigh`, `qw3-235b`, and `gm2.5`.

## Data

The scale-up benchmark is defined by:

```text
datasets/scaleup/regex_list.json
```

It stores regexes grouped by task type, number of DFA states, and `StarDepth`.
The actual train/eval examples live in:

```text
datasets/scaleup/regex_datasets/
```

Each dataset file is named:

```text
regex=<REGEX>_trainMaxLen=32_evalMaxLen=32.json
```

and contains:

```json
{
  "train_ex": [...],
  "train_labels": [...],
  "eval_ex": [...],
  "eval_labels": [...]
}
```

`train_icl_gen.py` calls `dataset.generate_dataset(...)` automatically before
running an experiment, so missing or undersized datasets are created/extended
when the requested `--tot_train_size` or `--eval_size` is larger than what is
already on disk.

Example single-dataset generation:

```bash
python -c "
from types import SimpleNamespace
from dataset import generate_dataset
args = SimpleNamespace(
    regex='(((b)*+a)+c)',
    max_length=32,
    eval_max_length=32,
    tot_train_size=3000,
    eval_size=3000,
)
generate_dataset(args, task_type='simplyrx', outdir='datasets/scaleup/regex_datasets')
"
```

## Running Experiments

The main entrypoint is:

```bash
python train_icl_gen.py [options]
```

Important arguments:

- `--task_type {simplyrx,extrx}`: dataset/domain.
- `--regex`: target regex.
- `--mkey`: model alias from `modeling/llm.py`.
- `--use_reg`: include regex regularization instructions.
- `--use_ce`: use counterexample-guided learning.
- `--reasoning_mode`: one of `single_inference`, `agentic_reflection`,
  `agentic_no_reflection`, `agentic_no_repair_loop`, or
  `agentic_candidate_repair`.
- `--ce_epochs`: maximum CE epochs.
- `--ce_batch_size`: number of counterexamples per CE epoch.
- `--ce_clustered`: use clustered counterexamples.
- `--rerun`: number of independent runs per regex.
- `--prompt_mode`: prompt ablation setting.
- `--indir`: dataset directory.
- `--outdir`: log root.

### Standard ICL

```bash
python train_icl_gen.py \
  --task_type simplyrx \
  --regex '(((b)*+a)+c)' \
  --mkey gpt-oss \
  --use_reg \
  --tot_train_size 3000 \
  --start_size 3 \
  --scale_factor 2.0 \
  --indir datasets/scaleup/regex_datasets \
  --outdir logs/scaleup
```

### Counterexample-Guided Agentic Learning

```bash
python train_icl_gen.py \
  --task_type simplyrx \
  --regex '(((b)*+a)+c)' \
  --mkey gpt-oss \
  --use_reg \
  --use_ce \
  --ce_epochs 12 \
  --ce_batch_size 250 \
  --ce_clustered \
  --reasoning_mode agentic_reflection \
  --indir datasets/scaleup/regex_datasets \
  --outdir logs/scaleup
```

### Single Inference

```bash
python train_icl_gen.py \
  --task_type simplyrx \
  --regex '(((b)*+a)+c)' \
  --mkey gpt-oss \
  --use_ce \
  --ce_epochs 12 \
  --ce_batch_size 250 \
  --ce_clustered \
  --reasoning_mode single_inference \
  --indir datasets/scaleup/regex_datasets \
  --outdir logs/scaleup
```

### Candidate Repair Ablation

```bash
python train_icl_gen.py \
  --task_type simplyrx \
  --regex '(((b)*+a)+c)' \
  --mkey gpt-oss \
  --use_reg \
  --use_ce \
  --ce_epochs 12 \
  --ce_batch_size 250 \
  --ce_clustered \
  --reasoning_mode agentic_candidate_repair \
  --candidate_repair_count 10 \
  --indir datasets/scaleup/regex_datasets \
  --outdir logs/scaleup
```

## Batch Scripts

Pre-generated batch scripts are in:

```text
datasets/scaleup/run_scripts/
```

Common scripts:

- `simplyrx_main.sh`: main Simple Regex runs.
- `extrx_main.sh`: main Extended Regex runs.
- `simplyrx_ablation.sh`: Simple Regex algorithm ablations.
- `extrx_ablation.sh`: Extended Regex algorithm ablations.
- `simplyrx_ablation_prompt.sh`: prompt ablations.
- `simplyrx_ablation_hard.sh`: harder Simple Regex settings.
- `extrx_ablation_no_clustered.sh`: non-clustered CE ablations.

Run one with:

```bash
bash datasets/scaleup/run_scripts/simplyrx_main.sh
```

The larger generated scripts are also available:

```bash
bash datasets/scaleup/run_simplyrx.sh
bash datasets/scaleup/run_extrx.sh
```

## Logs

Experiment logs are written under:

```text
logs/scaleup/icl_gen_<task_type>/model=<mkey>/
```

The main layout is:

```text
std/{reg,noreg}/...
ce/{reg,noreg}/<reasoning_mode>/...
```

Example:

```text
logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/
```

Each `msgdict_*.json` file contains:

- `summary`: final run-level metadata.
- `run-0`, `run-1`, ...: per-run trajectories.
- `epoch-*`: per-epoch prompt, response, extracted regex, training/eval scores,
  counterexample counts, and current hypothesis.

Tables used in the paper are stored in:

```text
logs/scaleup/summary/
```

## Plotting and Summaries

Use `curve.py` for aggregations and figures:

```bash
python curve.py --plot_type <PLOT_TYPE> --task_type simplyrx --mkey gpt-oss
```

Available plot types include:

- `pareto`
- `ce_composition_heatmaps`
- `solve_rate_by_stardepth`
- `solve_rate_by_states`
- `solve_rate_heatmap`
- `sample_budget_guess_eval_accuracy`
- `sample_budget_composite_rerun0`
- `sample_budget_diff_ratio_composite_rerun0`
- `mean_samples_by_stardepth`
- `median_samples_surface`
- `median_samples_heatmap`

### Pareto Frontier

```bash
MPLCONFIGDIR=/tmp python -B curve.py \
  --plot_type pareto \
  --task_type simplyrx \
  --mkey gpt-oss
```

Output:

```text
accuracy_curves/scaleup/icl_gen_simplyrx/model=gpt-oss/
```

### Sample-Budget Accuracy Composite

```bash
MPLCONFIGDIR=/tmp python -B curve.py \
  --plot_type sample_budget_composite_rerun0 \
  --outdir accuracy_curves/scaleup \
  --mkey gpt-oss
```

Output:

```text
accuracy_curves/scaleup/current_guess_eval_accuracy_sample_budget_composite_rerun0.png
```

### Sample-Budget Regex Distance Composite

```bash
MPLCONFIGDIR=/tmp python -B curve.py \
  --plot_type sample_budget_diff_ratio_composite_rerun0 \
  --outdir accuracy_curves/scaleup \
  --mkey gpt-oss
```

Output:

```text
accuracy_curves/scaleup/current_guess_diff_ratio_sample_budget_composite_rerun0.png
```

This plot uses `tasks.rl.RegularLanguage.diff_ratio`, which measures the
normalized symmetric difference between the current hypothesis regex and the
target regex over strings up to length `k`.

### Heatmaps

```bash
MPLCONFIGDIR=/tmp python -B curve.py \
  --plot_type solve_rate_heatmap \
  --task_type simplyrx \
  --mkey gpt-oss
```

## Core Code Map

- `tasks/rl.py`: builds `SimplyRegularLanguage` and `ExtRegularLanguage`
  objects, converts regexes to DFA/FST, checks acceptance, samples balanced
  strings, and computes `diff_ratio`.
- `teacher.py`: judges candidate regexes, computes correctness, and generates
  counterexamples.
- `learner.py`: formats prompts and extracts model predictions/reasoning.
- `prompting/rl.py`: task prompts, output format instructions, regularization
  text, clustered-CE instructions, and agentic repair/reflection instructions.
- `modeling/llm.py`: selects API or local vLLM backends and runs generation.
- `curve.py`: reads logs, counts tokens, computes success rates, produces plots,
  and writes summary artifacts.

## Prompt Optimization Utilities

`GEPA_opt/` contains utilities for prompt optimization:

```bash
python GEPA_opt/collect_data.py \
  --input_dir logs/opt_prompt/icl_gen_extrx \
  --output_path prompting/gepa_icl_gen_extrx.json
```

```bash
python GEPA_opt/gepa_opt.py \
  --task_type simplyrx \
  --use_reg \
  --use_ce \
  --ce_clustered \
  --max_metric_calls 150
```

## Legacy / Auxiliary Scripts

- `train_icl.py`: earlier ICL learner.
- `train_rnn.py`: RNN baseline with counterexample training modes.
- `baseline.py`: older neural baseline.
- `datasets/scaleup/synthesize.py`: benchmark synthesis helpers.
- `datasets/scaleup/convert.py`: conversion utilities for scale-up data.

## Reproducibility Notes

- Most experiment runners expose `--seed`; defaults are usually `43`.
- `train_icl_gen.py` saves logs incrementally after each epoch/retry, so partial
  runs can still be inspected.
- If using local `gpt-oss`, `modeling/llm.py` currently loads it through vLLM
  with tensor parallelism set to 2.
- Matplotlib can write cache files; on restricted systems use
  `MPLCONFIGDIR=/tmp` before plotting.
