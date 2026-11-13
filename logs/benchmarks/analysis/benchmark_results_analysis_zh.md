# logs/benchmarks 结果分析

> 日志迁移更新：原日志现已转换为共享 pipeline 的结构，并按首次已有 Equivalent=true 截断。本文及原指标文件仍是迁移前快照；迁移与备份见 [说明](../../benchmark_log_migrations/migration_summary_zh.md)。

## 结论

当前目录只有 **4 个任务实例 × std/CE 两种方法 × 各 1 个 run，共 8 份日志**，模型均为 gpt-oss，均带 regularization。它们来自规则/程序生成路径 icl_gen，而不是直接预测每条输入输出的 ICL。这批结果可用于检查 pipeline 和观察个案，尚不足以得出跨 benchmark 的方法优劣结论。

最重要的发现是：**目前没有一个清楚展示“新增的真实 counterexample 促成错误规则被修正”的成功轨迹。** Caesar 和列表任务第一轮已经学对；SCAN 的成功发生在相同数据、完全相同 prompt 的再次生成；反义词的指令会变化，但其反馈来自启发式 LLM judge，且没有实际执行准确率。

## 结果汇总

Epoch 从 0 开始；“调用数”仅统计日志中的规则生成调用，不包括 LLM oracle；N 是累计去重后的训练条目数。null 表示未判定或没有评分，不能记成 0。

| 任务实例 | 方法 | Epoch 数 / 规则生成调用数 | N 轨迹 | 日志 eval accuracy | 日志 Equivalent | 主要结论 |
|---|---|---|---|---|---|---|
| Cryptography / caesar:0 | std | 3 / 3 | 3 → 6 → 10 | 每轮 100%（10 条测试样本） | 每轮 null | 第一轮已输出正确 Caesar +3 |
| Cryptography / caesar:0 | CE | 8 / 8 | 始终 8 | 每轮 null | 每轮 null | 每轮因缺少 cvc5 失败；候选实际正确 |
| List Functions / c102 | std | 3 / 3 | 3 → 6 → 10 | 每轮 100%（10 条测试样本） | 每轮 null | 第一轮已学到 identity |
| List Functions / c102 | CE | 8 / 8 | 8 → 16 → … → 64 | 每轮 100%（10 条测试样本） | 每轮 null | 第一轮已正确，之后追加普通样本 |
| Instruction Induction / antonyms | std | 8 / 8 | 3 → 6 → … → 384 | 每轮 null | 每轮 null | 已识别“反义词”，没有执行评测 |
| Instruction Induction / antonyms | CE | 8 / 8 | 8 → 10 → 10 → 11 → 11 → 12 → 12 → 16 | 每轮 null | 每轮 false（LLM judge） | 指令变宽泛，但 judge 持续不接受 |
| SCAN / simple | std | 8 / 8 | 3 → 6 → … → 384 | 每轮 null | 每轮 null | 6 次 Python 子集校验拒绝、2 次无 ans |
| SCAN / simple | CE | 2 / 2 | 8 → 8 | epoch 0 无评分；epoch 1 为 100% | epoch 0 null；epoch 1 true | 第二次生成通过全 20,910 条枚举验证 |

所有 48 个 epoch 的结构化统计见 [epoch_metrics.csv](epoch_metrics.csv)。

## 1. SCAN：最终程序正确，但不能将改善归因于新增 CE

这是当前最值得看的一对日志。

### std 的 trajectory 与失败模式

- Epoch 0，N=3：生成了 Python 程序，但包含 while 和 nonlocal，报 Unsupported Python syntax。
- Epoch 1，N=6：包含 import 和 while，被拒绝。
- Epoch 2，N=12：使用 pop 和 replace，报 Unsupported method。
- Epoch 3，N=24：包含 while，被拒绝。
- Epoch 4，N=48：包含 while，被拒绝。
- Epoch 5，N=96：没有提取到 ans；Response 长 106,666 字符，末尾持续重复 I_TURN_RIGHT。
- Epoch 6，N=192：包含 while，被拒绝。
- Epoch 7，N=384：没有提取到 ans；Response 长 105,190 字符，末尾同样重复动作 token。

因此日志没有提供任何可评分的 std 程序。**这不能直接解释成 std 的 SCAN 语义准确率为 0。** 需要先区分：程序被受限执行器拒绝、输出无法解析，以及算法本身的语义错误。目前前两类已足以阻止评分。重复输出的形态说明存在生成退化，但日志未保存 finish_reason 或 token usage，不能断言具体截断原因。

### CE 的 trajectory

- Epoch 0，N=8：首个程序包含 import re，还使用了不被允许的 pop，因此没有通过子集校验。程序中还存在语义问题：将 around 带次数的重复解释为 (count + 1) × 3，以及未正确处理 turn 作为动作的特例。
- Epoch 1，N=8：生成了更短、无 import/pop 的程序；将 around 正确拆成 4 次 turn+verb 的基础动作，再整体乘 twice/thrice；单独处理 turn；通过 and/after 组合动作。日志记录 Equivalent=true、eval=1.0。
- 该任务的 oracle 枚举 tasks.txt 中的全部 20,910 个命令。等价成立的范围是这个有限公开命令域，不能外推到任意自然语言或域外命令。

### 最关键的证据：两轮 prompt 完全相同

直接比较日志发现，CE epoch 0 和 epoch 1 的 Prompt **逐字符一致**，Counterexamples 字段中的 8 条数据也完全一致，累计 N 没有增长。

原因是第一个候选未通过校验，previous 没有被更新。下一轮再次取得初始 8 条数据，没有获得基于候选差异的新反例；上一轮失败反馈也没有跨 epoch 保留下来。因此，这次成功是**同一 prompt 下第二次生成的程序改善**，同时修正了语法兼容性和语义，而非新增 CE 导致的修复。

Std 与 CE 的初始样本也不同：std 是随机打乱后的 3 条，CE 是原训练集前 8 条。不能把两者的差异单独归因于 CE 机制。

## 2. Cryptography：两边第一轮都学对，但 CE 的 solver 故障掩盖结果

目标是保持大小写的 Caesar +3。Std 和 CE 每轮都输出 {"cipher":"caesar","shift":3}。

- std：N=3 的第一轮 eval 已达 100%；之后 N=6、10 仍是 100%。没有执行 CE oracle，因此 Equivalent 一直是 null。
- CE：每轮都记录 [Errno 2] No such file or directory: 'cvc5'；Equivalent 与 scoreEvalSet 均为 null。
- CE 的 8 个 epoch 使用相同 prompt、相同 8 条训练数据。Oracle 异常发生在评分和 previous 更新之前，因此没有进入有效的反例迭代。

本次分析对保存的 CE 候选做了**离线补充评测**：使用现有 BenchmarkTask.score，在原 10 条测试样本上得到 100%。这是本次补充验证，不是原日志中已有的成绩；没有改动原日志，也没有调用 LLM 或 SMT solver。

结论：应记作“候选正确、CE 验证流程异常”，不能记作“CE 没学会 Caesar”。该例本身太容易，不适合展示 CE 的学习收益。

## 3. List Functions：c102 是 identity，后续所谓 CE 实际为补充样本

数据中的参考函数是：

```python
def c102(x):
    # the input
    return x
```

两种方法从第一轮起就返回 x[:]，是该任务的正确映射。Std 在 N=3、6、10 时 eval 均为 100%；CE 在 N=8 到 64 的八轮也始终为 100%。

### 为什么 CE 一直运行，还不断添加数据？

当前实现刻意不为任意 Python 的有限无反例搜索返回 Equivalent=true：无反例时仍返回 null。Runner 在“无反例、不能证明等价”时追加带标签的普通样本。

本次离线逐条检查了 epoch 1–7 的新增条目：**每轮 8 条，共 56 条，全部已被上一轮候选正确预测，实际 disagreement 数为 0。** 因此这些是 fallback 补充数据，而不是真实反例；Counterexamples 字段名没有反映它们的来源。

还有一个实现细节：当前 SMT 翻译器要求函数体只包含一个 return。CE 候选都带 docstring，因此不进入该直接翻译子集。至少 docstring 已足以使这里退回有限输入搜索。由于 identity 在池上没有错误，也不需要调用用于选择 disagreement 的 cvc5。

结论：此例只能说明两者都轻易识别 identity；不能用持续运行八轮、N=64 来表示 CE 学得更慢或更差，也不能把普通补充样本分析成 CE 的纠错轨迹。

## 4. Instruction Induction：指令已抓住主旨，但 judge 与精确映射存在冲突

Std 第一轮就提出“返回输入词的反义词”，后续逐渐补充否定前缀等细节。CE 的轨迹大致是：

- Epoch 0–1：standard opposite / antonym。
- Epoch 2：扩展为 opposite or contrasting counterpart，并包含增删否定前缀。
- Epoch 3–6：进一步强调词形变换、prefix/suffix、同类语义对立。
- Epoch 7：允许 single-word contrast / any natural opposite。

LLM oracle 在 epoch 0–6 的判定里反复用 linear → cubic 作为不等价证据，猜测该指令可能产生 nonlinear。Epoch 7 又用 powerful → powerless 对比可能输出 weak。

### 如何解释反例与 reasoning 的变化？

- 反馈确实推动指令从“标准反义词”扩展到“语义对比”，训练集也加入 credit → cash、gaseous → solid 等条目。
- 但“更宽泛的对比指令”并没有明确规定 linear 必须输出 cubic，因此 judge 仍能质疑其精确映射。
- OracleResponse 中的 candidate_output 是 judge 推测的结果，**并未真正调用候选指令执行输入并验证输出**。故 false 表示 judge 不接受语义等价，不等于测得执行准确率为 0。
- BenchmarkTask.score 对这个任务直接返回 None；std 也没有等价 oracle 调用。因此 std 的 null 与 CE 的 false 不能作为两者的胜负比较。
- 有重复反馈而无新增信息的停滞：epoch 2 和 4 的累计 N 都没有增长，虽然 Counterexamples 字段仍非空。说明 oracle 会重复给已有条目，不能简单把该字段长度当作新增样本数。

此任务需要将“识别任务类型”“对给定词输出参考答案”“自然语言指令被 judge 判为等价”三种目标分开。目前日志主要记录第三种目标，缺少第二种的实测结果。

## 5. 推理与计算开销

以下仅是规则生成的 Response 字符数，包含 analysis 和最终答案，**不是 reasoning token 数或总 API 成本**。Instruction Induction 的额外 judge 调用不包含在这里。

| 任务 | std 生成调用 / Response 字符 | CE 生成调用 / Response 字符 |
|---|---:|---:|
| Cryptography | 3 / 2,973 | 8 / 7,756 |
| List Functions | 3 / 2,336 | 8 / 19,211 |
| Instruction Induction | 8 / 26,561 | 8 / 16,839 |
| SCAN | 8 / 383,726 | 2 / 36,301 |

SCAN std 的开销很大一部分来自两次重复退化的输出。Crypto CE 的重复调用则来自 oracle 故障后未推进 previous。List CE 的额外调用来自 incomplete oracle 下持续补样本。它们是不同原因，不能统一解释为方法需要更多推理。

Instruction Induction 每轮候选生成之后还调用一次 judge；从 epoch 1 起，下一轮之前又调用 judge 获取反例。按当前代码路径，CE 八轮对应 8 次规则生成和 15 次 judge 调用。原日志仅保留每轮生成后的 OracleResponse，无法从 Response 字符总数恢复全部成本。

## 6. 对后续实验的具体建议

1. **先让运行环境与 prompt 契约一致。** 安装并预检 cvc5；SCAN prompt 明确允许的 Python 子集，说明 import/while/nonlocal/pop/replace 等限制，或按安全策略适度扩展 validator。目前“Use Python standard syntax only”不足以告知这些限制。
2. **区分失败状态。** 将生成/解析失败、validator 拒绝、oracle 异常、无反例但未证明、真实反例、严格验证通过分别记录；把 Attempts.Error 提升到易汇总的 epoch 状态，避免只检查顶层 Error 时漏掉 SCAN 的失败。
3. **区分反例与 fallback 样本。** 为新增条目标注来源、实际新增长度，以及是否真的被前一候选判错。当前 list_functions 的字段容易误导分析。
4. **避免无进展的重复调用。** 同一 prompt 遇到 oracle 异常时应优先暴露运行故障；记录是否复用了初始数据、是否有新反馈。将错误反馈跨必要的重试保留，并单独报告相同 prompt 下的再次生成结果。
5. **给 incomplete oracle 单独的停止与评分标准。** List 可以报告有限 eval/pool 的准确率和验证域，不能将 null 当失败；不应仅因不能证明任意 Python 等价而把已正确的候选一直补样本到上限。
6. **对指令任务补 execution evaluation。** 固定执行模型、prompt 与评测集，真正执行归纳指令，计算其输出与目标的匹配情况；judge 的语义意见另列。对于多义反义词或数据中的特殊对比关系，应明确要精确参考答案还是允许合理同义答案。
7. **扩展难度与重复次数。** 当前每个 benchmark 只有一个实例、一个 run，且 caesar:0、c102 太简单。固定相同初始样本，并给 std/CE 匹配生成预算，再扩展到更多 ID、SCAN split 和随机种子，才能讨论 CE 的总体收益。

以上为分析建议，本次没有改动 runner、validator、数据集或原始日志，没有安装依赖、调用模型或重新运行 benchmark。

## 源文件

- [运行逻辑](../../../train_icl_gen.py)：run_benchmark。
- [任务、执行器与 oracle](../../../tasks/benchmarks.py)：BenchmarkTask、_validate_function、_list_sequence_smt。
- [Prompt](../../../prompting/benchmarks.py)。
- 原日志：logs/benchmarks/icl_gen_<task>/model=gpt-oss/std|ce/reg/<id>.json。
