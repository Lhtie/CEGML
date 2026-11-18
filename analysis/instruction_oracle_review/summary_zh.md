# Instruction Induction oracle 审查与 prompt 迭代

本轮结论：现有 oracle 的判断不够可靠，问题不仅是“LLM 偶尔判断错误”。它原先不知道真实任务，只根据原始训练数据的前 20 个 IO 猜规则，又把一个记录答案误当成唯一合法答案。这样既会接受明显错误的捷径，也会否定正确的翻译、同义词或集合筛选指令，还会把假想的执行错误当成指令本身的错误。

本轮保留所有原始 benchmark 日志。完成了 48 份 std/CE 日志、214 个 epoch 的结构及 CE 一致性扫描，对照了全部 24 个任务的官方人类指令注释，并重点复核明显错误的 hypothesis 和 transition。小规模 API 复核使用 gpt-oss-api，结果单独保存，未重跑整套 learner 实验，也未修改旧日志的 Equivalent。

## 1. 先明确 ground truth 和“成功”的含义

GT 以仓库内 `datasets/instruction_induction/annotations/*.json` 的人类指令为依据，并结合官方原始 IO、输入域和评分规则。人类注释是任务语义的依据，不是某个唯一程序，也不是每个输入的唯一合法字符串答案。

新版 teacher 可以看 GT 注释；learner 仍然只接收 IO/CE，不接收这些注释。这与我们的 oracle-assisted 方法一致，但不能称为原论文直接提供的 oracle。原论文执行生成的指令，在测试输入上评估输出；我们的 `Equivalent` 仍是 LLM 对任务指令是否对齐的判断，不能替代 execution accuracy。

[官方仓库](https://github.com/orhonovich/instruction-induction)提供人类注释和任务数据。[官方评分代码](https://github.com/orhonovich/instruction-induction/blob/main/evaluate.py)中，taxonomy_animal、orthography_starts_with 使用集合匹配，翻译、rhymes 等使用可接受答案列表，formalization/common_concept 使用 F1。因而“和一条显示答案不同”并不自动构成语义反例；反过来，评分规则也不意味着所有相关词都算正确答案。

## 2. 最严重的误接受：从退化样本猜出错误 GT

旧 prompt 只提供 `self.train[:20]`，既没有官方 GT，也没有跨数据池覆盖。

- **sum / CE / epoch 0**：hypothesis 是 `Return the second integer from the input string.`，oracle 返回 true。真实 GT 是加法。前 20 个排序样本的第一个数恰好都是 0，导致“加法”和“返回第二个数”无法区分。`7 4 → 11` 就能推翻该 hypothesis，其输出应为 4。
- **diff / CE / epoch 0**：hypothesis 是返回第一个数，oracle 返回 true。真实 GT 是第一数减第二数。排序前缀中第二个数都是 0，掩盖了错误；`7 4 → 3` 与候选输出 7 不同。
- **larger_animal / CE / epoch 0**：hypothesis 是返回不是 snail 的词，oracle 返回 true。真实 GT 是比较动物体型。前缀里 snail 密集出现，因此错误地学成了固定词排除规则。对 `whale, rabbit`，GT 选 whale，而“不是 snail”无法唯一选出正确结果。
- **orthography_starts_with / CE / epoch 0**：只取首个匹配词及随后连续匹配词，被接受。GT 要取整句话中所有匹配词；相隔的匹配词会被漏掉。
- **singular_to_plural / CE / 最后一轮**：仅 add-s 和 consonant-y → ies 的规则被接受，但原始输入域包含 child、person、mouse、tooth 等不规则名词。

这些不是“equivalence 有一点 noise”的小偏差，而是 oracle 缺少 GT 和覆盖面，容易系统性地奖励过拟合捷径。

## 3. 最严重的误拒绝：把合法答案差异和想象的错误当成 CE

- **antonyms / CE / epoch 0**：`Return the standard English antonym of the input word.` 被拒绝。oracle 把 `powerful → weak` 与记录答案 `powerless` 的差异当成 CE；两者均可为合法反义词。还出现了候选已经明确写“移除负前缀”，oracle 却假设它对 unpopular 输出 unknown 的情况：这不是执行该指令得出的必然结果。
- **translation_en-es**：普通翻译指令被反复拒绝，理由是可能输出 `tiempo` 而不是显示的 `hora`。原始记录的 `possible_translations` 明确列出了时间一词的这些合法译法。不能让 learner 为满足 oracle 而去记忆一条特定词典答案。
- **taxonomy_animal**：动物集合相同，仅顺序不同，仍被当成 CE。官方是集合评分；“按字母排序”或“按字母位置和排序”本身不会改变筛选出的动物集合。
- **translation_en-fr / CE / epoch 2**：`team` 的 expected_output 和 candidate_output 都为 `équipe`，却放入了反例数组。旧代码没有检查二者是否真的不同。
- **sentence_similarity**：出现两个输出都为 `3 - probably` 的 CE，也出现把原本的字符串输入改成 `{sentence1, sentence2}` 对象的提议。
- **word_in_context**：真实输入是 `Sentence 1: ... Sentence 2: ... Word: ...` 字符串，旧 oracle 多次生成字典输入并给出可争议的词义标签；部分提议被代码直接作为新训练 IO 接受。

扫描结果如下。各项存在重叠，不能相加得到“错误 CE 总数”。

| 自动检查项 | 数量 | 能说明什么 |
|---|---:|---|
| epoch 记录 | 214 | 覆盖两种方法的现有记录 |
| 原始 CE 提议 | 1,109 | 包括被旧代码过滤的提议 |
| 候选输出属于官方已知可接受答案 | 396 | 强证据：这些提议不能直接支持“候选输出错误” |
| 两个输出按任务评分规则等价 | 94 | 包括完全相同、大小写归一化、集合顺序差异 |
| 输入结构错误 | 79 | 本应是字符串的任务出现对象等结构 |
| expected_output 与所存答案/别名不一致 | 63 | 仅是待审查差异；开放语言任务的答案列表不保证穷尽，不能全部判作标签错误 |
| 无可解析 oracle response | 2 | 不能推断为接受或拒绝 |
| 历史 cause_and_effect 适配改变了任务 | 4 个 epoch | 与官方任务不是同一个问题 |

完整逐 epoch hypothesis 和证据见 [CSV 清单](hypothesis_inventory.csv) 与 [全量扫描 JSON](historical_audit.json)。这是自动一致性扫描和重点人工语义审查，不是给全部 1,109 条 CE 都人工判了最终真伪。

## 4. 需要修复数据适配的问题：cause_and_effect

官方 prepared induction prompt 的样例是：

- Input: `Sentence 1: The ship capsized. Sentence 2: The wind blew strong.`
- Output: `The wind blew strong.`

GT 是从两句话中选出原因。旧 `_instruction_pairs` 却把原始的 cause/effect 字段转换成单个 cause → effect 的 IO。于是日志中的 hypothesis 都在“生成后果”，oracle 接受它们并不能说明它们匹配官方 GT。

本轮修复为两句话输入、输出原始 cause，并交替 cause/effect 在句子 1、句子 2 中的位置，避免重新引入固定位置捷径。历史 cause_and_effect 日志保持原样；后续该任务的运行结果与旧适配结果不能混为同一个 setting。

## 5. Prompt 怎样迭代

**原 prompt**：只有前 20 个 IO、candidate 和要求输出 true/false/CE 的一段话。没有 GT、任务域、答案容许规则，也不要求证明 candidate_output 是候选真正要求的行为。

**V1：grounded context + ICL。** 加入官方人类注释；从整个 induction pool 分布取参考例，而非只取排序前缀；保留可接受答案别名；说明 task alignment 不等于一次执行结果的 exact match。加入算术退化样例、动物捷径、合法翻译、集合排序校准例。

- 六个开发案例在最终审查标签下为 4/6 一致。
- 初始人工审查曾把带“专有名词不翻译”条件的 hypothesis 误当成普通翻译正例，首次显示的 5/6 因而不能直接作为最终指标。所有初始结果保留，重新按修正标签比较的结果见下方 JSON。

**V2：区分合法选择与真正错误。** V1 仍把 standard antonym 当成“比 GT 允许的答案更少”，从而误拒绝。加入“选择一个常见合法答案已足够，不要求枚举所有正确答案”的规则，以及普通反义词正例、机械去掉 -less 的反例。

- 同一开发子集达到 6/6；加上首批预留案例共 11/12。
- 发现新的越界问题：用 perfect/future tense 反驳简单过去时 passive 指令。

**V3：明确输入域并加强 CE 校验。** 把简单过去时的有效输入模板正面写清楚，增加域内/域外的被动语态校准例，并拦截此类越界 CE。明确不能为了保护“普通翻译正例”，就忽略 hypothesis 真正加入的错误条件。

- 原有及新增控制案例 13 条中，12 条判定一致、1 条 null。
- 冻结 prompt 后另取 4 条未用于继续调 prompt 的历史 hypothesis：正确加法、正确减法、只改变动物顺序、选择所有匹配词，全部判定正确。
- 最终为 **16/17 与审查结论一致，1/17 null，API 错误 0**。这是人工精选的诊断集，不是对全部任务/所有 epoch 的无偏 oracle accuracy 估计。

最初的“预留”案例后来部分用于 V3 调整，因此不能再把它们称为完全独立测试集；只有最后四条是在最终 prompt 冻结后追加的检查。同一任务的题材在校准和检查中可能重复，也不能据此宣称跨任务泛化已证明。

## 6. 不是所有与人工初判不一致的判断都该“优化掉”

一条 std 西语 hypothesis 实际写的是：

`Translate the given English word into its usual Spanish equivalent (keeping the word unchanged if it is already the same in Spanish or is a proper noun).`

我最初将其标成正确的普通翻译指令，模型给出 `London → Londres` 后，重新检查发现“专有名词一律不翻译”确实可能强制错误输出 London。这是有意义的真实限制，与 `time → hora/tiempo` 的合法答案差异不同。

因此修正的是人工预期标签，而不是强迫 oracle 接受这条 hypothesis。初始人工预期、修订理由、原始模型响应均保留在 [review_cases.json](review_cases.json)、各版本结果和 [重新计分对照](comparison_regraded.json) 中。还额外加入了不带这个例外条件的真实翻译 hypothesis 作为正对照。

## 7. 最终仍然存在的失败与保守处理

普通西语正对照 `Translate the given English noun into its usual Spanish equivalent.` 在 V3 下仍被 LLM 提议拒绝：它把 `time` 的 `hora` 与 `tiempo` 当成冲突。这仍是错误推理，不能因为加了 ICL 就假设彻底消失。

代码根据官方 `possible_translations` 发现 candidate_output 为合法答案，丢弃这条伪 CE。由于原始 verdict 是 false，但没有有效 CE，最终返回 **null**，不返回 false，也不自动返回 true。对应记录：[V3 普通翻译正对照](eval_v3/plain_translation_positive.json)。

这类 null 会让共享 pipeline 继续使用补充样本，而不会把错误 CE 注入 learner 或错误早停。代价是可能增加轮次。本轮没有再专门针对这一条反复调参直到命中，避免用诊断集掩盖残余问题。

## 8. 代码中的保护，不只依赖 prompt

- input 保持任务原始 schema；词义、相似度等字符串输入不能擅自改成字典。
- CE 必须包含 expected_output、candidate_output 和 conflict，后者要求说明哪条候选约束导致输出错误。它是简短证据说明，不是让模型预测任意弱执行器的失误。
- expected_output/candidate_output 已相同或在集合规则下等价时，直接丢弃。
- candidate_output 是已知可接受别名时，直接丢弃。
- 加减法、字母位置、字母拆分、前缀单词筛选、支持的被动语态模板可确定性检查标签；无法验证的该类 CE 保守拒收。
- 开放语言任务的答案列表不视作穷尽；未知但可能合法的 expected_output 不仅因未列出就判错。新语言案例的语义仍依赖 LLM，不是独立执行证明。
- false 没有有效 CE、JSON 自相矛盾、响应格式错误，都返回 null；不会因“CE 被过滤为空”就自动接受。
- 返回 true 的语义是指令与任务对齐的 noisy 判断，不是程序全域 equivalence 或测试集满分。
- OracleDetails 记录 policy、完整 oracle prompt、raw verdict、被过滤的 CE 及理由、GT 来源；便于后续重新审查。

实现：[instruction_oracle.py](../../benchmarks/instruction_oracle.py)。入口仍经 `BenchmarkTask.verify → counterexamples` 接入共享 episode；没有改 regex 的控制流程。learner 的生成 prompt 和重试/早停控制不因此变成另一套 pipeline。

## 9. 24 个任务逐项审查重点

| 任务 | 真实任务 | 本轮主要判断 |
|---|---|---|
| active_to_passive | 简单过去时句子的被动改写 | 正确 was/were 指令可接受；不能用 perfect/future 域外例误拒绝 |
| antonyms | 给出反义词 | 普通 antonym 指令被错拒；强制机械剥前后缀则可能真错 |
| cause_and_effect | 从两句话选原因 | 旧适配改了任务，已修复未来输入 |
| common_concept | 找全部对象共有性质 | 允许合法共有概念和释义；不是固定显示句子的逐字记忆 |
| diff | 第一数减第二数 | 返回第一数是被误接受的捷径；正确减法也曾被拒 |
| first_word_letter | 首字母 | 指令总体对齐；大小写/短词 CE 应服从实际输入域和评分 |
| informal_to_formal | 保持意义的正式改写 | 不应惩罚合法释义；后期无条件短语替换需要检查语义，不能一律接受 |
| larger_animal | 比较典型动物体型 | 非 snail 捷径明显错误；争议体型比较不要当高置信 CE |
| letters_list | 按顺序拼出字母列表 | 无必要的大小写和分隔符差异不应制造 CE；按实际域检查 |
| negation | 否定原陈述 | 允许正确语法改写；必须区分真的错否定与想象的执行错误/域外句型 |
| num_to_verbal | 英语数字词 | 参考应覆盖超过 99；简短正确指令省略细节不等于明确限制范围 |
| orthography_starts_with | 全句全部匹配单词 | 首个连续块捷径错误；顺序差异不构成集合 CE |
| rhymes | 语音押韵词 | 多答案、按音非拼写；不要求固定显示词，也不把争议音系判断当确定 CE |
| second_word_letter | 第二字母 | 正确位置指令应接受；不存在第二字母的域外输入不宜乱造标签 |
| sentence_similarity | 0–5 语义相似度 | 保留字符串和标签格式；相同输出、边界主观分数都不足以构成可靠 CE |
| sentiment | 正负情绪 | 服从二分类语义；不要仅用可能产生的弱执行错误否定正确指令 |
| singular_to_plural | 单数变复数 | add-s/y 捷径漏掉明确存在于域内的不规则词 |
| sum | 两数相加 | 返回第二数被错误接受；需非零第一数样例 |
| synonyms | 同义词 | 合法变体不能错拒；但“任意相关词/上位词”也不自动是同义词 |
| taxonomy_animal | 选全部动物 | 官方集合评分，排序策略本身不是错误 |
| translation_en-de | 英译德 | 保留合法多译法；错误的强制词性/复数限制应另作具体审查 |
| translation_en-es | 英译西 | 普通翻译与专有名词不翻译例外必须区分；仍有一个 null 残余案例 |
| translation_en-fr | 英译法 | 固定某个词典词条、合法同义译法和候选真的强加“罕见词优先”不能混为一谈 |
| word_in_context | 两句中目标词是否同义 | 正确的词义判别指令应接受；不改变输入结构、不凭模糊义项差别编 CE |

## 10. 复核文件与实际限制

- [逐 epoch 清单](hypothesis_inventory.csv)、[214 个 epoch 的扫描证据](historical_audit.json)。
- [人工审查案例和标签修订](review_cases.json)。
- [跨版本重新计分对照](comparison_regraded.json)。
- `eval_v1/`、`eval_v2/`、`eval_v3/`：每次 prompt、response、usage（包含可用的 reasoning_tokens）、过滤理由和耗时。
- [完整最终 prompt 示例](oracle_prompt_v3.md)。
- [复核脚本](evaluate_prompt.py)；[历史扫描脚本](review_historical.py)。从仓库根目录设置 `PYTHONPATH=.` 可复现，API 调用会产生费用；缓存结果默认复用。

这次优化提高了严重捷径、伪 CE 和域外 CE 的处理质量，但它依然是 noisy LLM oracle。新生成的语言 CE 没有独立执行器/第二模型逐条证实，“conflict”也仍可能被模型编错。精选案例上的 16/17 不能用于替换原 benchmark 成功率，更不能据此宣称旧的接受/拒绝全部已得到修复。
