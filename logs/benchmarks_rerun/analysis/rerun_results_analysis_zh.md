# Rerun 结果分析：ILP 与 Instruction Induction

统计目录：`logs/benchmarks_rerun`，模型目录标签为 `gpt-oss`。本次共 25 个 setting、std/CE 各一次运行，总计 50 份日志，均无 InProgress 记录；日志中未见 OracleError 或 ValidationError。本报告不改写实验日志，也不修改 oracle/推理代码。

成功定义：任一 epoch 出现 `Equivalent: true`。这两个 task 的 oracle 性质不同，因此不将接受率直接混成同一种真实正确率。

| Task | std 接受率 | CE 接受率 | 如何解释 |
|---|---:|---:|---|
| ILP / EvenSucc | 0/1 | 1/1 | 指定有限域、固定推理轮数下全部 ground atoms 是否正确 |
| Instruction Induction | 22/24（91.7%） | 24/24（100%） | v3 LLM oracle 对 instruction 的接受；不是执行测试集 accuracy |

## 1. ILP：CE 成功，但主要修正的是推理深度

| 方法/阶段 | 累积训练条目 | 训练准确率 | 评估准确率 | Equivalent |
|---|---:|---:|---:|---|
| std，epoch 0 | 11 | 8/11 = 72.7% | 11/15 = 73.3% | false |
| CE，epoch 0 | 11 | 8/11 = 72.7% | 11/15 = 73.3% | false |
| CE，epoch 1 | 15 | 100% | 15/15 = 100% | true |

两种方法都从相同的 11 个标签开始。std 只有一次 candidate generation，CE 有两次。CE 最后 MSE 为 0，满足当前 1e-4 阈值；std MSE 为 4/15。这里只有一个 ILP setting、每种一次 run，不能据此估计广泛 ILP 任务上的成功概率。

### std 的 hypothesis 其实定义了正确的偶数关系

```prolog
target(X) :- zero(X).
target(X) :- succ(Y,X), odd(Y).
odd(X) :- succ(Y,X), target(Y).
```

数学上，这是正确的偶/奇互递归规则。但当前训练仅推理 6 轮、评估仅推理 8 轮，且采用同步 Boolean 更新。由 target(0) 得到 odd(1)，再得到 target(2)，每前进两个整数需要两轮。于是 8 轮只推出 target(0)、target(2)、target(4)、target(6)，漏掉 8、10、12、14。

CE 首轮用了另一个互递归写法，先定义 even，再令 target(X) :- even(X)，还增加了一个转发步骤，但 8 轮的 target 结果相同。它并非没有发现 parity；模型的 reasoning 已明确识别出偶数模式。

### Oracle 给出的 4 个 CE

```text
N=15, target(8)  => true
N=15, target(10) => true
N=15, target(12) => true
N=15, target(14) => true
```

它们是当前八轮执行语义下的有效失败证据：GT 为 true，候选在截断时尚未推出。注意输入包含 N：其中 8、10 虽已在 N=11 训练域出现，N=15 的查询上下文仍不同；12、14 则是更大域的新常量。

### CE 后的 reasoning 与规则改动

模型改为先构造一次跳过两个 successor 的二元辅助关系：

```prolog
step2(X,Y) :- succ(X,Z), succ(Z,Y).
target(X) :- zero(X).
target(Y) :- target(X), step2(X,Y).
```

新 reasoning 明确提到要在有限 forward-chaining rounds 内高效传播 evenness。step2 第一轮就可从背景推导，之后 target 每轮向前推进两个数；target(14) 在第八轮得到。它改善了 proof/inference efficiency，而不是把错误的奇偶定义改成正确的奇偶定义。

### 独立复核：跑到有限域不动点后三个程序都正确

本轮只为分析执行了更长的有限域推理，没有修改原 oracle 或日志。结果：

| 程序 | target(14) 首次推出的轮数 | 不动点处的 target 关系 |
|---|---:|---|
| std epoch 0 | 15 | {0,2,4,6,8,10,12,14} |
| CE epoch 0 | 16 | {0,2,4,6,8,10,12,14} |
| CE epoch 1 | 8 | {0,2,4,6,8,10,12,14} |

证据：[有限域不动点复核](ilp_fixpoint_recheck.json)。

这暴露了当前 ILP 适配的一个重要限制：将 6/8 步预算应用到任意 LLM Horn 程序，会对逻辑等价但分解方式不同的程序产生不同判定。当前 `Equivalent` 是 bounded-inference equivalence；如果研究目标是 Horn 规则的逻辑正确性，建议将有限域 oracle 改为推理至不动点，并把“6/8 轮能否完成”另记为效率指标。这里是后续建议，本次没有替你改动 setting。

另外，CE 使用了评估域中的标签，而且比 std 多一次生成机会，100% 也是经过 oracle 辅助修正后的有限域结果，不能当作完全未接触测试信息的独立外推成绩。

## 2. Instruction Induction：成功率增加，仍需区分接受与真实正确

两种方法都使用 grounded_instruction_oracle_v3。std 最终 22 个 accepted、1 个 rejected、1 个 inconclusive；CE 24 个 accepted。相较 std，CE 多通过 singular_to_plural 和 synonyms，没有出现 std 成功而 CE 最终不成功的 task。

### 逐任务结果

表中“第几轮”从 1 开始，等于成功 epoch+1；条目数为最终 NumTrainingSamples，可能含重复/补充条目，不等同于去重后的有效 CE 数。

| Task | std | CE |
|---|---|---|
| active_to_passive | 成功，第1轮，3条 | 成功，第1轮，8条 |
| antonyms | 成功，第2轮，6条 | 成功，第5轮，12条 |
| cause_and_effect | 成功，第3轮，12条 | 成功，第1轮，8条 |
| common_concept | 成功，第1轮，3条 | 成功，第1轮，8条 |
| diff | 成功，第1轮，3条 | 成功，第2轮，9条 |
| first_word_letter | 成功，第2轮，6条 | 成功，第1轮，8条 |
| informal_to_formal | 成功，第1轮，3条 | 成功，第1轮，8条 |
| larger_animal | 成功，第2轮，6条 | 成功，第3轮，10条 |
| letters_list | 成功，第1轮，3条 | 成功，第1轮，8条 |
| negation | 成功，第2轮，6条 | 成功，第3轮，11条 |
| num_to_verbal | 成功，第1轮，3条 | 成功，第2轮，9条 |
| orthography_starts_with | 成功，第1轮，3条 | 成功，第2轮，9条 |
| rhymes | 成功，第1轮，3条 | 成功，第6轮，15条 |
| second_word_letter | 成功，第1轮，3条 | 成功，第1轮，8条 |
| sentence_similarity | 成功，第3轮，12条 | 成功，第1轮，8条 |
| sentiment | 成功，第1轮，3条 | 成功，第1轮，8条 |
| singular_to_plural | 未接受，8轮，384条 | 成功，第2轮，9条 |
| sum | 成功，第1轮，3条 | 成功，第2轮，9条 |
| synonyms | 未判定，8轮，384条 | 成功，第1轮，8条 |
| taxonomy_animal | 成功，第1轮，3条 | 成功，第1轮，8条 |
| translation_en-de | 成功，第4轮，24条 | 成功，第2轮，16条 |
| translation_en-es | 成功，第1轮，3条 | 成功，第1轮，8条 |
| translation_en-fr | 成功，第1轮，3条 | 成功，第3轮，17条 |
| word_in_context | 成功，第1轮，3条 | 成功，第1轮，8条 |

### CE 的明确收益：反例排除了真实捷径

- **sum**：CE 首轮返回第二个 token，被新版 oracle 拒绝；加入 `5 7 → 12` 后，第二轮改为真正求和。
- **diff**：首轮返回第一个数；`87 5 → 82` 后改为第一数减第二数。
- **larger_animal**：先用“选非 snail”，再用“默认选第一个”；`elephant, donkey → elephant` 和 `mouse, elephant → elephant` 逐步排除了这两种捷径，第三轮得到典型体型比较规则。
- **orthography_starts_with**：首轮只取连续匹配块；`Bob bought a blue balloon. [b] → bob bought blue balloon` 表明 a 之后仍有匹配词，第二轮改为提取全句所有匹配项。
- **singular_to_plural**：首轮统一加 s；一个 `mouse → mice` 就使第二轮改成“常规名词加 s，有不规则形式则使用不规则复数”。std 则在 384 条训练条目后仍停留在主要规则型复数表，遗漏 mouse/child，最终失败。

这些例子支持“CE 通过提供区分候选规则的证据而改善归纳”的解释。sum/diff/snail 捷径也不再像旧版 oracle 那样直接被接受。

### 不能全部归功于 CE 的差异

- **synonyms**：CE 首轮已经输出正确的通用 synonym 指令，因此这项新增成功没有经历 CE 修正。std 的初始样本不同，逐步学成了 hypernym、dictionary-definition term 等更宽泛的规则。原数据还包含 `pillow → rest` 这样的弱对应，容易与 human GT 的“真正同义词”目标发生张力。
- **cause_and_effect**：CE 首轮正确，std 三轮后正确。首轮输入分别有 8 条和 3 条，不能将差距单独解释成反例反馈。新版任务已是两句中选原因，旧日志是原因生成结果，两版不能直接当同一任务比较。

### 样本和调用成本：看总体与共同成功子集，结论不同

| 指标 | std | CE |
|---|---:|---:|
| 接受任务数 | 22/24 | 24/24 |
| 首轮接受任务数 | 15/24 | 13/24 |
| candidate generation 调用总数 | 49 | 45 |
| 日志记录的 LLM oracle 响应数 | 49 | 45 |
| 平均轮数，含失败任务 | 2.04 | 1.88 |
| 平均最终训练条目，含失败任务 | 36.88 | 9.58 |
| 最终训练条目中位数 | 3 | 8 |
| 共同成功的 22 项：candidate 调用总数 | 33 | 42 |
| 共同成功的 22 项：平均轮数 | 1.50 | 1.91 |
| 共同成功的 22 项：平均最终训练条目 | 5.32 | 9.68 |

CE 在总体平均轮数、样本条目上更少，主要受 std 的两个 8 轮/384 条失败项影响。对共同成功的 22 项，CE 反而用了更多调用和条目。因此不能概括为 CE 对所有任务都更快。尤其当前 std 默认从 3 条开始，CE 从 8 条开始，而且 std 使用随机打乱后的普通样本、CE 初始样本来自原始前缀，初始信息量和分布并不匹配。要更严格分析 CE 的因果收益，需要匹配初始样本及调用/标签预算。

日志没有完整的 token usage 或每次调用耗时，不能据此精确比较总 reasoning tokens、费用或时间。

## 3. 新版 oracle 有改善，但这次仍有伪 CE

本次 std 中 7 条、CE 中 3 条 CE 提议被过滤：其中 4 条因为输出在任务评分下已经等价，6 条因为 candidate_output 是官方可接受答案。CE 有两轮因此转为 inconclusive，下一轮追加的是 supplemental 样本，而不是真 CE；不能把这两次新增样本称为反例反馈。

仍然能确认或明显质疑的问题包括：

1. **rhymes / CE / epoch 3**：oracle 说 `blue` 的候选输出 `true` 不押韵，遂返回 false。标准发音中两者共享 /uː/ 韵部，声母差异不妨碍押韵；[Cambridge 的 true-blue 发音](https://dictionary.cambridge.org/us/pronunciation/english/true-blue)列出 /ˌtruːˈbluː/。这条提议通过了代码过滤并进入后续训练，是明确的残余语义错误。
2. **rhymes / CE / epoch 0**：把 `silver → sliver` 当参考押韵答案，并宣称候选无定义。这里涉及拼写/语音混淆和未经独立词典验证的标签，不宜直接把六轮最终接受解释成六轮有效修正。
3. **antonyms / CE / epoch 1–3**：hypothesis 明说“如果前缀操作能产生反义词就做，否则用通常词汇反义词”，oracle 却断言它强制 inflammable → flammable。该输出并非候选条件必然允许的行为：前提恰恰要求产生 opposite。模型把条件性规则误读成无条件字符剥离。
4. **translation_en-fr / CE / epoch 0**：以 run“是动词，所以 noun-only 指令无定义”为由给 CE，没有考虑 run 的名词用法；这是需要审查的词义/词性输入域假设。
5. **synonyms / std**：最终的 raw 拒绝又产生了 `pillow → cushion`、candidate_output 为 rest 的提议；rest 恰好是原数据所存答案，因此被校验器拦截，最终为 null。过滤避免了直接传播某些矛盾，但也显示人类语义目标与词典派生数据并不完全一致。

综上，Instruction Induction 的 CE 最终 24/24，只能解释为这个 noisy teacher 的 operational acceptance rate。它不是 24 条指令被独立执行、在测试集上 100% 正确，也不能说明全部 CE 已可靠。当前 scoreEvalSet 在全部 48 份 Instruction Induction 日志中仍为 null。

## 4. 与旧日志的描述性对照

| Instruction Induction | 旧 logs/benchmarks | 本次 rerun |
|---|---:|---:|
| std 接受率 | 14/24 = 58.3% | 22/24 = 91.7% |
| CE 接受率 | 16/24 = 66.7% | 24/24 = 100% |

这不能解释成同一个模型“准确率提高了 33.3 个百分点”：teacher prompt、GT 参考、CE 校验、部分任务适配都已变化，hypothesis 也是新运行生成的。更合理的描述是：新版 pipeline 少了明显捷径的误接受，也少了许多合法释义/排序/翻译的误拒绝，但仍存在语言语义层面的伪 CE。

## 5. 可支持的结论与后续优先级

- **ILP**：当前实验显示 CE 能诱导更短的有限轮数推导。若要研究语义归纳，优先区分不动点正确性与 6/8 轮效率，再增加不同 ILP 任务/多次 run。
- **Instruction Induction**：CE 帮助修复了复数等真实规则缺陷，也纠正了算术、snail、连续匹配等早期捷径；但整体接受率仍受 teacher 噪声和初始样本不匹配影响。
- 后续 oracle 审查优先处理“条件前提被忽略”“凭空宣称 candidate 无定义”“押韵语音判断”“词性/输入域歧义”；可以对 accepted CE 再进行独立语义核验，而不是只继续增加同一模型的自我说明。
- 对成功率和成本结论，分别报告全部任务与共同成功子集；不要把失败项带来的长尾误读成所有 task 上的效率优势。

逐 setting 统计：[CSV](results_by_setting.csv)；机器可读统计：[audit_data.json](audit_data.json)；ILP 不动点复核：[ilp_fixpoint_recheck.json](ilp_fixpoint_recheck.json)。

本轮只做分析与单独的离线 ILP 语义复核，没有调用新 LLM、修改原实验日志或改变 pipeline。
