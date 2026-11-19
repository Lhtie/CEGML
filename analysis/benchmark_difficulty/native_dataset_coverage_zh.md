# 原数据集覆盖：Instruction Induction、SCAN、ILP

本次针对“后面三个为什么只有这么几个测试点”补齐覆盖。setting 是一种任务/官方划分，不是其中每条 IO；不能把几千条 IO 复制成几千个独立的规则归纳任务。

| 任务 | 之前可运行的 setting | 现在 all 中的 setting | 原始样本覆盖 |
|---|---:|---:|---|
| Instruction Induction | 24 | 24 | 67,662 条 induce、2,156 条 execute；全部加载 |
| SCAN | 8 | 104 | 官方全部配对划分和完整 20,910 命令全集 |
| ILP | 12 | 19 | 全部作者合成配置；新增七个直接调用作者 generator |

全 benchmark 从 594 个 setting 增加为 697 个，std/CE 共 1394 个 job（一个 runner）；balanced 保留 54 个 setting、108 个 job。

## Instruction Induction

作者公开数据只有二十四种任务类型，不是只含二十四条样本。当前已经接入所有类型，新增更多类型需要新数据集或人为变体，不能说是原数据集。

之前默认 eval_size=32 会截断 execute IO，现在 benchmark 默认加载完整 execute。显式正数 --eval_size N 仍可限制加载。原 regex 未显式指定 eval_size 时仍保持三十二，不受此修改影响。

生成式 instruction pipeline 依旧用语义 oracle，scoreEvalSet 仍可为 null；“加载了完整测试 IO”不等于新实现了逐条 instruction executor。没有将 execute 数据泄漏到 teacher oracle；直接 IO runner 会遍历完整加载的测试集。

| ID | induce IO | execute IO |
|---|---:|---:|
| active_to_passive | 900 | 100 |
| antonyms | 2592 | 100 |
| cause_and_effect | 26 | 25 |
| common_concept | 17 | 16 |
| diff | 4950 | 100 |
| first_word_letter | 3306 | 100 |
| informal_to_formal | 15 | 15 |
| larger_animal | 3028 | 100 |
| letters_list | 3306 | 100 |
| negation | 904 | 100 |
| num_to_verbal | 9900 | 100 |
| orthography_starts_with | 3000 | 100 |
| rhymes | 3000 | 100 |
| second_word_letter | 3306 | 100 |
| sentence_similarity | 3716 | 100 |
| sentiment | 1167 | 100 |
| singular_to_plural | 1943 | 100 |
| sum | 4950 | 100 |
| synonyms | 2222 | 100 |
| taxonomy_animal | 3000 | 100 |
| translation_en-de | 2808 | 100 |
| translation_en-es | 2711 | 100 |
| translation_en-fr | 2811 | 100 |
| word_in_context | 4084 | 100 |

## SCAN

固定官方仓库 revision c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f，补齐此前没有下载的原始文件；没有重新合成或改写任何命令/答案。原来八个文件对与官方内容相同，保持原样。

| 官方来源目录 | 划分数 |
|---|---:|
| add_prim_split | 32 |
| few_shot_split | 55 |
| filler_split | 4 |
| length_split | 1 |
| simple_split | 8 |
| template_split | 4 |

累计 train 行数 1,516,059、test 行数 451,506，是跨划分的总行数，含大量重叠和官方原生重复；唯一命令全集仍为 20,910。少样本复制编号、额外 primitive 示例和数据比例是原生实验条件，不是不同语言规律。

所有 209 个文本文件（104 对加 tasks.txt）的 SHA-256 已逐一核对；来源、样本数在 datasets/scan/source_manifest.json。可用 python datasets/scan/prepare.py 恢复数据，或 --archive 指定本地官方 zip。下载需要网络；没有在线模型/API 费用。

SCAN loader 改为从真实配对文件发现 ID，去掉原先八个 ID 的白名单。原始完整 train/test 行全部保留，oracle 的 20,910 命令范围没有扩大或缩小。评分仍有原来的时间预算，未完成时应读 completeness/null 状态，不能当作已评全。

四个难度 rank band 现在各二十六个；使用输出长度外推、compositional-jump 覆盖缺口和训练命令覆盖指标，详见 setting_difficulty_catalog_zh.md。balanced 每档选两个，而不是把 104 个划分都塞进 balanced。

## ILP

作者 Task.py 有十九个合成配置，另有 WN 知识图谱任务。此次补齐七个：adj_to_red、two_children、relatedness、member、son、even_odd、grandparent_nlm。作者提供的是生成器，不是一个可以耗尽的固定有限样本表；每个配置的当前背景、IO 和规模仍明确记录在数据文件中。

新增七个使用固定版本作者 Dataset.py/Graph.py/Family.py 生成背景和目标。保留旧背景和已有标注，二元训练池只追加末尾十六个缺失原子，不改旧标签；因此这不是十九个配置都重新统一为原论文规模或采样分布。Family.py 的 NumPy 选择异质序列兼容性问题通过均匀下标抽样解决，vendored 文件本身未修改。

EvenOdd/EvenSucc 的目标重叠，Grandparent_NLM/Grandparent 的目标相同而背景不同，记录为原生配置，不声称十九个独立概念。既有 GraphColoring 保持作者代码的一元口径，未切换成论文文字的二元口径。

| ID | 当前训练池 | 验证 IO | 数据来源 |
|---|---:|---:|---|
| adj_to_red | 384 | 512 | pinned_author_generator_seeded_scaleup |
| buzz | 384 | 512 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| connectedness | 400 | 576 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| cyclic | 384 | 512 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| even_odd | 384 | 512 | pinned_author_generator_seeded_scaleup |
| even_succ | 384 | 512 | deterministic_author_numeric_extension |
| fizz | 384 | 512 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| grandparent | 400 | 576 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| grandparent_nlm | 400 | 576 | pinned_author_generator_seeded_scaleup |
| graph_coloring | 384 | 512 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| has_father | 384 | 512 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| length | 400 | 576 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| less_than | 400 | 576 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| member | 400 | 576 | pinned_author_generator_seeded_scaleup |
| predecessor | 400 | 576 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |
| relatedness | 400 | 576 | pinned_author_generator_seeded_scaleup |
| son | 400 | 576 | pinned_author_generator_seeded_scaleup |
| two_children | 384 | 512 | pinned_author_generator_seeded_scaleup |
| undirected_edge | 400 | 576 | seeded_sparse_scaleup_adaptation_not_author_random_sampler |

为保留全部原配置，all 的 ILP 难度档数量为 4/8/7；不能继续声称每档四个。balanced 仍每档两个，原来六个 ID 不变。

对包含 neq/color 的非递归规则，SMT grounder 和独立回放执行器采用目标导向展开，避免无意义地展开全部辅助关系对；递归程序保留原来的正向支持闭包和 rank 最小模型编码。TwoChildren 的二次方 neq 背景在 prompt 中用定义表示，实际数据关系不变。十九个参考程序均通过 SMT，且完整训练池与独立执行一致。

Graph.py 的 Apache-2.0 许可已随 source/LICENSE-APACHE-2.0 保存，源文件头保留。

## 默认预算：完整训练池，十轮

benchmark 默认样本预算等于该 setting 的完整训练 IO 数，不再默认限制为 384。std 在十轮中按几何增长重新分配累计样本数，第十轮达到完整预算；equivalent 后仍提前结束。小数据集可能出现相同样本数，但不伪造新增条目。

CE 最多十轮，仍按 oracle 反例增量训练，不强制把整个原始池加入 prompt。--tot_train_size 0 与默认全池相同，显式正数可限制支持任务的样本预算。--rounds 0 恢复自然的 scale_factor 调度；默认十轮模式由起始规模、完整预算和轮数共同决定实际增长比例。

日志记录 TrainingPoolSize、ScheduledTrainingBudget、EvaluationPoolSize、RoundBudget、StandardSampleSchedule。使用全部训练池作为预算不等于第一轮就输入全部 IO；如果达到很大的最终池，仍可能超过模型上下文，pipeline 不会悄悄截断。

```bash
# 默认完整训练池、最多十轮；std/CE 共 294 job。
python -m benchmarks.run_suite --runner icl_gen --suite all --task instruction_induction scan ilp --mode both --mkey gpt-oss --outdir logs/benchmarks_native

# 如需手动限制样本数：
python -m benchmarks.run_suite --runner icl_gen --suite all --task instruction_induction scan ilp --mode both --mkey gpt-oss --outdir logs/benchmarks_native_capped -- --tot_train_size 384
```

原 regex 没有一个固定有限的“原始最大 IO 池”，省略预算时仍生成/使用默认 384 条；显式指定预算后也按十轮分配。

## 验证

- 24 个 instruction task 的原始 train/test 数量全部与 loader 一致，显式 eval_size=32 仍按请求限制。
- 209 个 SCAN 官方文本文件哈希全部一致；新增各类划分均可加载。
- 作者十九个合成注册名全部有对应本地配置，没有遗漏或虚构新目标。
- 十九个参考规则均获得 SMT true，所有训练标注与独立执行一致。
- 三十六个独立小背景的 SMT/枚举解释器对照通过；原 regex std/CE 和 token 记录回归通过。
- 完整训练池开关和早停行为通过离线假响应验证。

详见 native_coverage_validation.json。此次没有运行正式 LLM benchmark，没有修改历史实验日志。




最新默认预算验证：full_default_budget_validation.json；取代此前“384 条后重复”的十轮方案。
