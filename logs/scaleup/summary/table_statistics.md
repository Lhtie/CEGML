# Summary 表格统计说明

本文档说明 `logs/scaleup/summary/` 中每个 summary 表格的数据来源和统计方式。所有表格的核心成功指标都来自实验 log JSON 顶层的 `summary` 字段。

## 通用规则

### Regex 元信息来源

每个 regex 的 domain、`#States`、`StarDepth` 来自：

- `datasets/scaleup/regex_list.json`

结构是：

- `simplyrx` / `extrx`
- 每个 `#States`
- 每个 `Stardepth`
- 对应的 `regex_list`

大多数表格只取每个 `(#States, StarDepth)` cell 里的前 3 个 regex，即：

```text
datasets/scaleup/regex_list.json
  -> <domain>
  -> #States group
  -> Stardepth group
  -> regex_list[:3]
```

部分 ablation 表只取 `StarDepth=1,2,3`，因此 SimplyRx 下通常是：

```text
7 个 #States * 3 个 regex = 21 个 regex per StarDepth
```

ExtRx 的 `StarDepth=2` 没有 `#States=9` 的 regex，因此 full-coverage 分母是：

```text
6 个 #States * 3 个 regex = 18 个 regex
```

### Log 文件如何匹配 regex

每个实验 log 是一个 JSON 文件，文件名形如：

```text
msgdict_regex=<REGEX>_ceEpochs=12_ceBatch=250.json
msgdict_regex=<REGEX>_ceEpochs=12_ceBatch=250_clustered.json
```

统计时从文件名中取出 `<REGEX>`，再去 `regex_list.json` 查它属于哪个 domain、`#States` 和 `StarDepth`。

### 单个 run 是否成功

每个 log 的核心字段是：

```json
{
  "summary": {
    "run-0": {
      "final_accuracy": 1,
      "final_num_samples": {
        "positive": 5,
        "negative": 4,
        "total": 9
      }
    },
    "run-1": {...},
    "run-2": {...}
  }
}
```

一个 run 成功当且仅当：

```text
summary["run-i"]["final_accuracy"] >= 1.0
```

表格中的 `successful runs / observed runs` 就是统计满足上面条件的 run 数量，以及实际在 log 中观察到的 run 数量。

### Per-regex success rate

如果一个 regex 跑了 3 次，则：

```text
per-regex success rate = 成功 run 数 / 3
```

例如 2 个 run 成功、1 个失败，则该 regex 的 success rate 是 `66.7%`。

### Run-level success rate

有些表格不是先算每个 regex 的平均，而是直接把所有 regex 的 run 池化：

```text
run-level success rate = 所有成功 run 数 / 所有 observed run 数
```

例如 21 个 regex，每个只统计 `run-0`，则分母是 `21`；如果每个 regex 统计 3 个 run，则分母是 `63`。

### `run-0 only`

有些 ablation 表为了和“一 regex 一次独立尝试”的设定对齐，只统计每个 log 的：

```text
summary["run-0"]
```

这时每个 regex 最多贡献 1 个 observed run。

## 表格逐项说明

## `main_result_success_by_stardepth`

文件：

- `main_result_success_by_stardepth.md`
- `main_result_success_by_stardepth.csv`
- `main_result_success_by_stardepth.tex`

用途：主结果表，比较两个 domain 上三种方法随 `StarDepth` 变化的 run-level success rate。

Domain：

- `SimpleRx`
- `ExtendedRx`

方法和 log 目录：

- `Standard`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/std/reg/`
  - `logs/scaleup/icl_gen_extrx/model=gpt-oss/std/reg/`
- `Single Inference`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/single_inference/`
  - `logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/single_inference/`
- `Agentic Reflection`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/`
  - `logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/agentic_reflection/`

统计方式：

1. 从 `regex_list.json` 中取每个 `(#States, StarDepth)` cell 的前 3 个 regex。
2. 按 `StarDepth` 聚合，不再区分 `#States`。
3. 对匹配到的 log，读取 `summary["run-*"]["final_accuracy"]`。
4. 成功条件是 `final_accuracy >= 1.0`。
5. 每个 cell 显示：

```text
100 * 成功 run 数 / observed run 数
```

显示格式：

```text
success rate (successful runs / observed runs)
```

当前表格 note 中记录的覆盖率：

- SimpleRx 的 `Standard` `Single Inference` 和 `Agentic Reflection` 是 full coverage：每个 StarDepth 分母为 `7 states * 3 regexes * 3 reruns = 63`。
- ExtendedRx 的 `StarDepth=0,1` 分母为 `63`。
- ExtendedRx 的 `StarDepth=2` 分母为 `54`，因为 `#States=9, StarDepth=2` 没有定义 regex。

## `full_matrix_mean_success_by_state_stardepth`

文件：

- `full_matrix_mean_success_by_state_stardepth.md`
- `full_matrix_mean_success_by_state_stardepth.csv`
- `full_matrix_mean_success_by_state_stardepth.tex`

用途：主结果的完整矩阵版本，保留 `#States x StarDepth` 的 cell。

Domain：

- `simplyrx`
- `extrx`

方法和 log 目录同 `main_result_success_by_stardepth`：

- `Standard`: `std/reg`
- `Single Inference`: `ce/reg/single_inference`
- `Agentic Reflection`: `ce/reg/agentic_reflection`

统计方式：

1. 对每个 domain、每个方法、每个 `(#States, StarDepth)` cell，取 `regex_list.json` 中前 3 个 regex。
2. 对每个 regex 读取对应 log 的 `summary["run-0"]`、`summary["run-1"]`、`summary["run-2"]`。
3. 单个 regex 的 success rate：

```text
成功 run 数 / observed run 数
```

通常 observed run 数为 3。

4. cell 的 mean success rate：

```text
3 个 regex 的 per-regex success rate 的平均值
```

5. 表格中方括号里的 `[r1, r2, r3]` 是该 cell 前 3 个 regex 各自的 success rate。

显示格式：

```text
mean success rate% [regex1_rate, regex2_rate, regex3_rate]
```

CSV 字段：

- `dataset`: `simplyrx` 或 `extrx`
- `method`: 方法名
- `states`: `#States`
- `stardepth`: `StarDepth`
- `mean_success_rate`: cell 平均值
- `regex1_rate`, `regex2_rate`, `regex3_rate`: 前 3 个 regex 的 per-regex success rate
- `logged_regexes`: 当前找到 log 的 regex 数
- `total_regexes`: 该 cell 期望统计的 regex 数，通常是 3

## `simplyrx_model_method_stardepth_1_2_3`

文件：

- `simplyrx_model_method_stardepth_1_2_3.md`
- `simplyrx_model_method_stardepth_1_2_3.csv`
- `simplyrx_model_method_stardepth_1_2_3.tex`

用途：模型消融表，比较不同模型在 SimplyRx 的 `StarDepth=1,2,3` 上，`Standard` 和 `Agentic Reflection` 的成功率。

Domain：

- `simplyrx`

StarDepth：

- `1`
- `2`
- `3`

当前表中模型：

- `gpt-oss`
- `gpt5`
- `qw3-235b`

方法和 log 目录模式：

```text
logs/scaleup/icl_gen_simplyrx/model=<MODEL>/std/reg/
logs/scaleup/icl_gen_simplyrx/model=<MODEL>/ce/reg/agentic_reflection/
```

统计方式：

1. 从 `regex_list.json` 取 SimplyRx 中 `StarDepth=1,2,3`、所有 `#States`、每个 cell 前 3 个 regex。
2. 每个 `StarDepth` 共 `7 states * 3 regexes = 21` 个 regex。
3. 对当前表格，每个 regex 只统计 `summary["run-0"]`。
4. 成功条件是 `final_accuracy >= 1.0`。
5. 每个 cell 显示：

```text
成功 run-0 数 / observed run-0 数
```

所以 full coverage 时每个 StarDepth 的分母是 `21`。

特殊说明：

- 当前表格对所有 model 都只统计每个 regex 的 first rerun，也就是 `run-0`。
- `gpt5` 当前是 `NA`，因为 `model=gpt5` 还没有跑完，暂时不从未完成 logs 统计数值。
- `qw3-235b` 当前已有 full coverage，因此直接从 `logs/scaleup/icl_gen_simplyrx/model=qw3-235b` 统计。

## `simplyrx_algorithm_components_ablation`

文件：

- `simplyrx_algorithm_components_ablation.md`
- `simplyrx_algorithm_components_ablation.csv`
- `simplyrx_algorithm_components_ablation.tex`

用途：算法组件消融，比较 regularization、reflection、repair loop 三个组件对 SimplyRx 的影响。

Domain：

- `simplyrx`

StarDepth：

- `1`
- `2`
- `3`

当前表中的 ablation 和 log 目录：

- `standard`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/std/reg/`
- `single_inference`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/single_inference/`
- `noreg/agentic`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/noreg/agentic_reflection/`
- `agentic_no_reflection`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_no_reflection/`
- `agentic_no_repair_loop`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_no_repair_loop/`
- `agentic_reflection`
  - `logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/`

组件列含义：

- `Regularization`: 是否使用 `reg`
- `Reflection`: 是否有 agentic reflection
- `Repair Loop`: 是否有 repair loop

统计方式：

1. 从 `regex_list.json` 取 SimplyRx 中 `StarDepth=1,2,3`、所有 `#States`、每个 cell 前 3 个 regex。
2. 每个 StarDepth full coverage 是 21 个 regex。
3. 每个 regex 只统计 `summary["run-0"]`。
4. 成功条件是 `final_accuracy >= 1.0`。
5. 每个 cell 显示：

```text
成功 run-0 数 / observed run-0 数
```

显示格式：

```text
success rate (successful run-0s / observed run-0s)
```

## `simplyrx_prompt_ablation`

文件：

- `simplyrx_prompt_ablation.md`
- `simplyrx_prompt_ablation.csv`
- `simplyrx_prompt_ablation.tex`

用途：Prompt 消融，比较不同 prompt information components 对 SimplyRx 推理效果的影响。

Domain：

- `simplyrx`

StarDepth：

- `1`
- `2`
- `3`

Dataset slice：

- 所有 `#States`
- 每个 `(#States, StarDepth)` cell 取前 3 个 regex
- 每个 StarDepth full coverage 是：

```text
7 个 #States * 3 个 regex = 21 个 regex
```

Run setting：

- 每个 regex 运行 1 次，即只统计 `summary["run-0"]`
- 比较 `standard` 和 `single_inference`
- 不使用 regularization，即 `noreg`
- 不使用 clustered counterexamples，即没有 `_clustered` 后缀

方法和 log 目录模式：

- `standard`
  - `logs/scaleup/icl_gen_simplyrx/model=<MODEL>/std/noreg/prompt=<PROMPT_MODE>/`
- `single_inference`
  - `logs/scaleup/icl_gen_simplyrx/model=<MODEL>/ce/noreg/single_inference/prompt=<PROMPT_MODE>/`

特殊目录规则：

- 当 `prompt_mode != "full"` 时，log 会放进 `prompt=<PROMPT_MODE>/` 子目录。
- 当 `prompt_mode == "full"` 时，`train_icl_gen.py` 不会创建 `prompt=full/` 子目录，而是直接放在 method 目录外层：
  - `standard`: `logs/scaleup/icl_gen_simplyrx/model=<MODEL>/std/noreg/`
  - `single_inference`: `logs/scaleup/icl_gen_simplyrx/model=<MODEL>/ce/noreg/single_inference/`

Prompt variants：

| Paper name | `prompt_mode` in `train_icl_gen.py` | Prompt contents |
|---|---|---|
| naive prompt | `naive_prompt` | task instruction only, plus direct answer format and training-data block |
| naive + input info | `only_input_instr` | task instruction + input-format instruction |
| naive + output info | `only_output_instr` | task instruction + regex syntax/output instruction |
| naive + input + output info | `input_output_instr` | task instruction + input-format instruction + regex syntax/output instruction |
| zero prompt | `zero_prompt` | naive + input + output info, with reasoning/COT-style output format |
| current / GEPA-optimized prompt | `full` | full prompt template, i.e. zero-style prompt plus the optimized inference strategy |

统计方式：

1. 从 `regex_list.json` 取 SimplyRx 中 `StarDepth=1,2,3`、所有 `#States`、每个 cell 前 3 个 regex。
2. 对每个 prompt variant 和 method，找到对应 log 目录中的 JSON。
3. 每个 regex 只统计 `summary["run-0"]`。
4. 成功条件是 `summary["run-0"]["final_accuracy"] >= 1.0`。
5. 每个 cell 显示：

```text
成功 run-0 数 / observed run-0 数
```

显示格式：

```text
success rate (successful run-0s / observed run-0s)
```

Full coverage 时，每个 `StarDepth` 的分母是 `21`。如果某个 prompt variant / method 的 log 缺失，则 run-level 表中应报告 `NA` 或按 observed runs 显示实际分母；不要把缺失 log 误当成失败 run，除非表格明确声明使用保守估计。

## `extrx_clustered_vs_nonclustered`

文件：

- `extrx_clustered_vs_nonclustered.md`
- `extrx_clustered_vs_nonclustered.csv`
- `extrx_clustered_vs_nonclustered.tex`

用途：ExtRx 上 clustered counterexample 和 non-clustered counterexample 的比较。

Domain：

- `extrx`

Model：

- `gpt-oss`

方法和 log 目录：

- `agentic_no_clustered`
  - `logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/agentic_no_clustered/`
- `agentic_reflection`
  - `logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/agentic_reflection/`

统计方式：

1. 统计 ExtRx 的 `StarDepth=0,1,2`。
2. 从 `regex_list.json` 取所有 `#States`、每个 cell 前 3 个 regex。
3. 每个 regex 只统计 `summary["run-0"]`。
4. 成功条件是 `final_accuracy >= 1.0`。
5. 每个 cell 显示：

```text
成功 run-0 数 / observed run-0 数
```

当前 coverage：

- `StarDepth=0`: 21 个 observed run-0
- `StarDepth=1`: 21 个 observed run-0
- `StarDepth=2`: 18 个 observed run-0，因为 ExtRx 没有 `#States=9, StarDepth=2`

## 和实验列表的对应关系

你的实验列表可以对应到当前 summary 表如下：

- Main results
  - `main_result_success_by_stardepth`
  - `full_matrix_mean_success_by_state_stardepth`
- Model ablation
  - `simplyrx_model_method_stardepth_1_2_3`
- Algorithm ablation
  - `simplyrx_algorithm_components_ablation`
- Prompt ablation
  - `simplyrx_prompt_ablation`
- Clustered cex
  - `extrx_clustered_vs_nonclustered`
