# SimplyRx 案例分析：std 与 full pipeline 的推理轨迹

## Two simpler examples

- [Case A — 3 states / star depth 1: b-star plus singleton a/c](simplyrx_3state_sd1_union_case_study.md). Positive CEs reveal the missing empty string and singleton branches; full run-0 succeeds after 2 epochs / 10 entries.
- [Case B — 4 states / star depth 2: arbitrary a/c prefix, then bc or bac](simplyrx_4state_sd2_suffix_case_study.md). Positive CEs relax the prefix while negative CEs restrict the suffix; full run-0 succeeds after 2 epochs / 12 entries.

Each linked Markdown is self-contained: ground-truth interpretation, separate std/full epoch summary tables, and analysis of every full transition with the actual counterexample evidence.

## Ground truth: natural-language interpretation

**Accept the single string `a`, or any string formed by concatenating zero or more blocks, each of which is either `a` or `cc`, followed by exactly one final symbol, either `b` or `c`.**

Compact ground-truth regex: `(a + c c)* (b + c) + a`. Here `+` means union, spaces mean concatenation, and `*` means zero or more repetitions.

Equivalently, for a string ending in `b`, every maximal run of `c` has even length (and `b` occurs only at the end). For a string ending in `c`, every internal run of `c` has even length and the final run has odd length. The single string `a` is an additional accepted case; the empty string is rejected. Thus the constraint concerns **each run of c, not just the total number of c symbols**.

Examples: `a`, `b`, `c`, `aab`, `aac`, `accb`, and `ccac` are accepted; `aa`, `cc`, `acb`, and `acacac` are rejected.

## Epoch-by-epoch comparison: summary reasoning and regex

The table uses **run-0** for both methods. Each row reproduces the logged **epoch-selected `CurrentGuessReasoning` and `CurrentGuess`**, rather than the full analysis or necessarily the last retry. In full pipeline, epochs 1, 2, and 3 select retry 1; epoch 0 selects retry 2; epoch 4 selects retry 1 (retries are zero-based). The methods use different training sets and sample counts, so matching epoch numbers do not mean matching evidence or compute.

**The reasoning below is the model’s stated hypothesis at that epoch, and may be incorrect or inconsistent with its regex.** Regex strings are preserved as logged, including invalid syntax in std epochs 4 and 5. Full pipeline succeeds at epoch 4 and has no later epochs.

### Standard (std)

| Epoch | Training entries | Summary reasoning (verbatim logged output) | Selected regex |
|---:|---:|---|---|
| 0 | 3 | All positive strings are either exactly “a” or consist solely of one or more “c” characters; no other letters appear in any positive. | `a + c c*` |
| 1 | 6 | All positives are either a single “a”, a non‑empty string of only “c”, or start with “a”, end with “b”, and have only “a” or “c” in between. Negatives contain “b” inside the middle part or start with other letters, so they are excluded. | `a (a + c)* b + a + c c*` |
| 2 | 12 | All positive strings never contain the substrings “aa” or “bb”.  Negatives all have at least one of these double letters, while positives may have any number of c’s and isolated a or b (never two a’s or two b’s in a row). | `(c + a (c + b) + b (c + a))* (a + b + c)` |
| 3 | 24 | All positives consist of only c’s, or a single a, or a sequence of optional leading c’s followed by one or more blocks “a⁺ c⁺”, optionally ending with a single b. No b appears except possibly as the last character. | `a + c c* + (c* (a a* c c*) (a a* c c*)* (b + epsilon))` |
| 4 | 48 | All positives either contain no b, end with c, or have a single trailing b.  If a b appears it must be the last character.  Strings without b never end with a (except the single‑letter string “a”), and any string that has at least two a’s may appear anywhere.  The language can be expressed as a union of four simple patterns covering these cases. | `(a c* + c+ + c* a (a + c)* a (a + c)* c+ + (a + c)+ b)` |
| 5 | 96 | All accepted strings consist of a’s and c’s (at least one character) and may optionally end with a single b.  If a b appears it must be the only b and be the final character.  The empty string is not allowed. | `(a + c)+ b? + a + b` |
| 6 | 192 | All positive strings either consist of an odd number of c’s, or contain at least one a and never end with a (except the single‑letter string “a”).  Any b that appears can only be the last symbol and must be preceded by c. | `c (c c)* + (a + c)* a (a + c)* (c + c b)` |
| 7 | 384 | All accepted strings either end with a single b (and contain no other b), or consist only of a and c and end with c while having an odd number of c’s. The only other accepted string is the single a. No string containing “bb” or ending with a (except the solitary a) is allowed. | `((a + c)* b + (a* c a* c)* a* c + a)` |
| 8 | 768 | All strings of length 1 (a, b, c) are accepted. For longer strings only the letters a and c may appear anywhere, and the final character must be either c or b. In particular, b can appear only as the last symbol, and a cannot be the last symbol of a string longer than one. | `(a + b + c) + ((a + c) (a + c)* (c + b))` |
| 9 | 1536 | All positive strings are non‑empty and consist only of the letters a, b and c.  A simple language that includes every positive example while excluding the empty string and the clearly negative short examples (like “aa”, “bb”, “cc”, “ca”, “bc”, …) is the set of all non‑empty strings over {a,b,c} that contain at least one occurrence of the letter c. | `(a + b + c)* c (a + b + c)*` |
| 10 | 3000 | All positive strings are either a single letter, or end with a single b (with no other b’s), or end with c and contain an odd number of c’s.  Any string ending with a longer than one character is negative, and strings with a b not at the very end are negative (the only exception “acb” is excluded by requiring at least two c’s when a b‑ending string contains any c). | `( a + a * b + ( (a + c) * c c (a + c) * b ) + a * c a * (c a * c a *) * )` |

### Full pipeline

| Epoch | Training entries | Summary reasoning (verbatim logged output) | Selected regex |
|---:|---:|---|---|
| 0 | 8 | All positives contain only a and c. The language consists of (i) only c’s, (ii) the single string a, or (iii) strings with at least three alternating a‑runs and c‑runs, always ending with a c. Thus strings with exactly two a‑runs (e.g., “ccccacca”, “ccccccaaaccccccccacccccc”) are rejected. | `( a + c c* + a a* (c c* a a*) (c c* a a*) (c c* a a*)* c c* + c c* a a* (c c* a a*) (c c* a a*) (c c* a a*)* c c* )` |
| 1 | 26 | All positives are either a single a, b, ab, ac; a run of at least three c’s; or strings that may start with c’s, contain one or more “acc” blocks and end with b, ab, ac or with at least two a’s followed by a c (allowing extra c’s between later “ac” pieces). Negatives violate these mandatory pieces. | `( a + b + a b + a c + c c c* + c c* a c c (a c c)* ( b + a b + a c + a a a* c c* (c* a c)* ) + c c* a a a* c c* (c* a c)* )` |
| 2 | 61 | All positives consist of runs of c’s (except the isolated “cc”), and any ‘a’ or ‘b’ that follows a single c must be preceded by another c (i.e., only “cca” and “ccb” are allowed).  The language is therefore built from the tokens a, cca, ccb, cc and optional trailing c‑runs (single c, ≥3 c) or a ≥2‑c run ending in b. | `( c + c c c c* ) + ( (a + c c a + c c b + c c)* b ) + ( (a + c c a + c c b + c c)* (a + c c a + c c b) (epsilon + c + c c c c* + c c c* b) )` |
| 3 | 186 | Positives are either an odd‑length run of only c, or contain at least one a (or just b) where every a after the start is preceded by “cc”.  Between a‑clusters there must be at least two c’s, and a final b may appear only at the end. | `( c (c c)* ) + ( ( epsilon + c c c* ) ( a a* (c c (c c)*) )* a a* c* (b + epsilon) )` |
| 4 | 197 | All positives consist of an even number (possibly zero) of leading c’s, then any number of “a” separated by an even number of c’s; the string may end with a single b, or with a single c (giving an overall odd count of c’s). The lone string “a” is also allowed. | `( (c c)* ( a (c c)* )* b ) + ( (c c)* ( a (c c)* )* c ) + a` |

## Full pipeline: what changed in the reasoning, and did counterexamples drive it?

This section follows **run-0 only**. “CE-linked” means that the actual prompt contains examples that directly challenge the previous hypothesis and the next summary changes in a corresponding direction. It is evidence of a plausible mechanism, **not proof of isolated causality**: there is no matched run here that removes only those examples while holding all other conditions fixed. Positive counterexamples are target-accepted strings rejected by the candidate; negative counterexamples are candidate-accepted strings rejected by the target.

### Changes between epochs (selected candidates)

| Transition | Change in the selected summary reasoning | Relevant counterexamples supplied in the new epoch | Is the change CE-linked? |
|---|---|---|---|
| Initialization → epoch 0 | Forms an initial a/c-only explanation: pure c strings, the singleton a, or strings with at least three a-runs ending in c. | The initial data contain positives using only a/c. Within-epoch repair highlights `ccccccaaaccccccccacccccc, 0`. | **Initialization is not a correction of a previous candidate.** The later “at least three a-runs” restriction is strongly linked to the highlighted negative, but is an incorrect generalization. |
| Epoch 0 → epoch 1 | Abandons the a/c-only and minimum-a-run explanation. Adds b-ending strings and short cases; reorganizes longer strings around acc blocks. | Positives `b`, `ab`, `ac`, `accb`, `accab`, `accac`, `accaccb`, `accaccab`; negatives include `cc`, `accaccacc`, and `acacac`. | **Strong correspondence.** Positive examples directly refute the old alphabet/block-count restrictions. Negative `cc` refutes unrestricted nonempty c-runs. The specific acc-based replacement is the model’s guess, not a structure uniquely implied by these examples. |
| Epoch 1 → epoch 2 | Moves from acc-centered templates toward reusable a/cca/ccb/cc tokens, with special trailing c-runs. | Positives include `c`, `ccb`, `ccab`, `aab`, `aac`; negatives include `caac`, `caccab`, `ccaacc`, and `ccaacac`. | **Strong correspondence for broadening and position constraints.** `c` refutes the ≥3-c restriction; `aab`/`aac` refute the mandatory acc template; contrasting cc-leading positives with c-leading negatives encourages paired-c structure. The resulting token system remains incorrect. |
| Epoch 2 → epoch 3 | Makes b explicitly terminal, recognizes odd-length pure-c strings, and describes a-clusters separated by c-runs rather than allowing cca/ccb blocks freely. | 125 new negatives include `cccc`, `ccbb`, `ccbc`, `ccba`, `cca`, `ccacccc`, and many related or grouped strings. | **Strong correspondence for terminal-b and pure-c restrictions.** Internal b blocks are contradicted by the new negatives; `cccc` challenges the old ≥3-c rule. The exact “at least two c’s” separation condition remains an imperfect hypothesis. |
| Epoch 3 → epoch 4 | Replaces loose c-run restrictions with **even-length leading and internal c-runs**, followed by a final b or c. Keeps singleton a. | New examples include `b, 1` and negatives `ccca`, `cccab`, `cccac`, `cccacb`, `cca`, `ccacb`, `acb`, `acc`, `aa`, `aacb`. Final repair separately highlights four old negatives: `ccacacac`, `acacac`, `caccaacac`, `caacac`. | **The clearest link is the final repair, not merely the new CE batch.** The first attempt only captures global c-count parity. The four highlighted training mistakes distinguish global parity from local even-length c-runs; the next attempt explicitly uses cc blocks and passes equivalence checking. |

### Changes inside each epoch (every LLM call)

Retries are zero-based. The starting point for the next epoch is the **selected** retry, not necessarily the last retry.

| Epoch.retry | Change relative to the preceding attempt or selected epoch | Feedback actually present in the prompt | Assessment of the connection |
|---|---|---|---|
| 0.0 | Initial hypothesis: alternate nonempty a-runs and c-runs, end in c; also accept pure c strings and singleton a. | Initial labeled data; no previous candidate to refute. | Baseline inference, not an established CE-induced change. |
| 0.1 | Adds an “at least three a-runs” restriction to the c-leading branch. | Repair highlights `ccccccaaaccccccccacccccc, 0`. This string has two a-runs and was incorrectly accepted. | Directly addresses the highlighted failure, but invents the wrong structural explanation. |
| 0.2 | Extends the minimum-three-a-runs restriction to both a-leading and c-leading strings. | The same negative is highlighted again because the previous regex still accepts it. | Strong repair linkage. The stated restriction and the actual regex in 0.1 did not align; 0.2 makes the restriction more explicit. This retry is selected. |
| 1.0 | Replaces the previous a/c-only explanation with short a/b cases and acc-based patterns. | New positive and negative CEs listed in the epoch-transition table. | Strong link to the new evidence; output has invalid syntax. |
| 1.1 | Keeps the acc-based explanation but allows leading c’s and adjusts suffixes. | Syntax/error feedback; **no concrete repair examples** are available for this retry. The outer-epoch CEs remain in the prompt. | The immediate retry is primarily error-driven. Do not attribute its specific changes to newly highlighted CEs. This retry is selected. |
| 1.2 | Loosens the template toward “contains acc” plus permitted endings. | Several missed positives are highlighted, together with negative `cc`. | Plausible response to missed positives, but the output is invalid and not selected. |
| 2.0 | Introduces cca blocks and disallows a directly after a single leading c; treats pure c-runs separately. | New CEs include c/ccb/ccab/aab/aac positives and caac/caccab-like negatives. | Strong evidence-to-hypothesis correspondence, but not yet the correct parity rule. |
| 2.1 | Expands the stated token system to a/cca/ccb/cc and revises trailing pieces. | Eight highlighted negatives, including `ccccacca`, `abc`, `cc`, `accaccacc`, and `ccaacc`. | Negative-example-driven restructuring, but still overgeneralizes. This retry is selected. |
| 2.2 | Replaces the token system with cca/acc templates and a longer enumeration of allowed endings. | Two remaining negatives: `ccccacca` and `ccccccaaaccccccccacccccc`. | Addresses the two highlighted failures but damages other behavior: training accuracy falls from about 96.7% to 50.8%. Not selected. |
| 3.0 | Makes b terminal and proposes separation constraints between a’s; pure-c rule is still broad. | New batch of 125 negatives exposing internal b and invalid c patterns. | Strong link for restricting b placement. Output has invalid syntax. |
| 3.1 | Explicitly switches pure-c strings to odd length; describes a-clusters separated by at least two c’s. | Syntax/error feedback; **no concrete repair examples** in this retry. The outer batch remains available. | Plausibly informed by the accumulated negatives, but the immediate retry is syntax-driven; no isolated CE attribution. This retry is selected. |
| 3.2 | Loosens the internal structure to “start with a or cc, then a/c, end in c or b.” | Sixteen highlighted training errors, including positive `b`/`ccb` and several negative c-run variants. | Feedback-linked revision, but it regresses in training accuracy and is not selected. |
| 4.0 | Reorganizes the rule around odd total c-count for c-ending strings and restrictions immediately before terminal b. | New 11-example batch, including acb/acc/aa/aacb negatives and positive b. | Strong link to end-position and parity evidence, but confuses global parity with local run structure. Eval=100% does not imply equivalence. |
| 4.1 | Replaces global parity with **cc as a reusable block**, enforcing even leading/internal c-runs. | Four highlighted old training negatives: `ccacacac`, `acacac`, `caccaacac`, `caacac`. | **Strongest mechanistic evidence in the trace.** All four satisfy the prior global-parity approximation but violate local run parity. The next regex excludes the entire error family and is formally equivalent. Selected and successful. |

### Interpretation

The observed pattern is **hypothesis → targeted failure evidence → revised hypothesis**, but the revision can be wrong or harmful. Counterexamples do not supply a correct explanation directly. They reveal which distinction the current explanation misses; the model must still invent and encode a replacement.

The most convincing step is **4.0 → 4.1**: the highlighted negatives all have odd total c-count and end in c, yet contain forbidden odd-length internal c-runs. The model then changes the representation to cc blocks. Earlier transitions also respond to examples, but often through ad hoc restrictions or invalid regexes. Thus “CE causes improvement every round” is not supported; “targeted feedback can trigger a useful structural revision” is well supported by this trace.

The final improvement specifically follows **inner-loop repair on old training errors**, after several outer-loop CE batches have built the evidence set. It should not be attributed solely to the last batch of newly generated counterexamples.

**Key transition:** Standard identifies global c-count parity but never reaches an equivalent regex. In full pipeline, the final selected regex explicitly uses `cc` blocks. At epoch 4, repair highlights four negative examples—`ccacacac`, `acacac`, `caccaacac`, and `caacac`—that have odd total c counts but forbidden odd-length internal c-runs. The final candidate rejects these by enforcing local block structure and passes formal equivalence checking.

---
## 1. 案例、配置与主要结果

本案例选择 scaleup 中的 **5 states、star depth=2** 的 SimplyRx 任务，模型为 **gpt-oss**。该复杂度位于 scaleup 设置的中间。本文重点分析 `run-0`，并核对另外两次运行的最终结果。这是一个用于理解机制的案例，不代表所有任务的平均表现。

目标 regex：

```text
(((((a)*+(c c)))* (b+c))+(a+c))
```

等价的紧凑形式：

```text
(a + c c)* (b + c) + a
```

这里 `+` 表示并集，空格表示连接，`*` 表示 Kleene star。直观地说：**主体由单个 `a` 和成对的 `cc` 拼接，最后接一个 `b` 或 `c`；另外接受单独的 `a`。**

难点是发现 **连续 c 段的局部奇偶约束**，而不只是统计整条字符串中 c 的总数：在以 b 结尾的分支中，所有 c 段都为偶数长度；在以 c 结尾的分支中，内部 c 段为偶数长度，最后一段为奇数长度。

两种方法的配置：

- **std**：`std/reg`，累计训练样本量从 3 增加到 3000。
- **full pipeline**：`ce/reg/agentic_reflection/dfs`，使用 clustered CE，`ceBatch=250`，并启用 reflection 和 epoch 内的 repair loop。250 是配置的 batch 上限，并非每轮实际新增 250 条。

| 指标 | std | Full pipeline |
|---|---:|---:|
| run-0 累计训练条目 | 3000 | 197 |
| run-0 LLM 调用数 | 11 | 14 |
| run-0 是否与目标语言形式等价 | 否 | 是 |
| 三次运行成功数 | 0/3 | 3/3 |
| 三次运行结束时的累计训练条目 | 3000 / 3000 / 3000 | 197 / 56 / 56 |

训练条目包含重复样本；full 还有字符类压缩的样本，不能直接理解为不同具体字符串数。这个案例体现了更高的样本效率，但 **不能说明 full 使用了更少的推理计算**。

原始日志：

- [std](</mnt/c/Users/lhxlh/Documents/UW-Madison/research/CEGML/logs/scaleup/icl_gen_simplyrx/model=gpt-oss/std/reg/msgdict_regex=(((((a)*+(c c)))* (b+c))+(a+c))_totTrain=3000_startSize=3_scaleFactor=2.0.json>)
- [full](</mnt/c/Users/lhxlh/Documents/UW-Madison/research/CEGML/logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs/msgdict_regex=(((((a)*+(c c)))* (b+c))+(a+c))_ceEpochs=12_ceBatch=250_clustered.json>)


## 2. 如何理解日志中的 reasoning 与长度

日志保存了完整 `Response`，其中包含 `analysis` 与最终输出；另外还有一个单独的 `Reasoning` 字段，对应最终输出中简短的 `<reasoning>` 摘要。本文同时检查了完整 analysis 和简短摘要，不能把后者当成完整推理过程。

**日志没有保存 API reasoning-token usage 或 token IDs，因此本文不报告精确 reasoning token 数。** 表中的 `analysis 字符数` 是 `Response` 中 `assistantfinal` 之前的 Unicode 字符数，去掉开头的 `analysis` 标记。它用于展示推理文本的规模，不是 token 数，也不等于推理质量。

在 run-0 中，std 的 analysis 总计 **249,722 字符**；full 总计 **359,874 字符**。full 在这个例子中调用更多，累计输出的 analysis 也更长，因此不能将结果解释为“更少的推理换来了成功”。

## 3. std 的 trajectory：逐渐接近，但规则不断漂移

以下 epoch 从 0 开始，长度单位为千字符。每个 epoch 只有一次 LLM 调用。

| Epoch | 累计样本 | analysis 长度 | 这一轮的主要推断 |
|---|---:|---:|---|
| 0 | 3 | 1.1k | 正例是单独 a 或任意非空 c 串 |
| 1 | 6 | 3.7k | 加入“a 开头、b 结尾”的分支 |
| 2 | 12 | 16.0k | 转而猜测不能出现 aa、bb |
| 3 | 24 | 16.3k | 猜测 a 段、c 段交替，允许末尾 b |
| 4 | 48 | 43.5k | 添加多个分支，输出出现语法错误 |
| 5 | 96 | 49.2k | 放宽成 a/c 串加可选 b，仍有语法错误 |
| 6 | 192 | 36.0k | 开始注意到 c 的奇偶性，但附带错误限制 |
| 7 | 384 | 28.5k | 抓住“无 b 时 c 总数为奇数”；eval=100%，仍不等价 |
| 8 | 768 | 9.5k | 又退回宽泛的首尾字符规则 |
| 9 | 1536 | 15.1k | 退化成“包含至少一个 c” |
| 10 | 3000 | 30.8k | 再次接近，但仍将全局奇偶性作为核心条件 |

**std 并不是完全没有发现关键线索。** 完整 analysis 中，它反复对短串进行比较、数 c、提出奇偶性假设，并检查结尾是否允许 b。epoch 7、10 都达到了 eval accuracy=100%，但形式验证仍给出反例，分别为 `cb` 和 `cccb`。

它花了很多推理篇幅列举样本、计数和尝试不同规则，却没有把这些观察稳定组织成正确的 `a` / `cc` 构造块。epoch 7 已经接近，epoch 8、9 又明显退步，说明其推断并非随着样本数量增长而单调改善。

直观地说，std 的问题不是“完全想不到奇偶性”，而是 **未能可靠地区分全局统计条件与局部结构约束**。更长的推理和更多普通样本，没有自动消除剩余的结构性错误。

## 4. Full pipeline 的 trajectory：错误决定下一轮看到什么

同一 epoch 内的多个长度，依次对应该 epoch 的每次 LLM 调用。

| Epoch | 累计样本 | 每次 analysis 长度 | 推断如何变化 |
|---|---:|---|---|
| 0 | 8 | 46.0k / 17.0k / 18.0k | 起初只允许 a/c，后来靠限制 a 段数量拟合现有数据 |
| 1 | 26 | 21.4k / 15.4k / 26.4k | b、ab、ac 等正反例迫使它推翻旧限制；开始围绕 acc 组织模式 |
| 2 | 61 | 24.7k / 36.0k / 16.7k | 引入 cc、cca、ccb 等块，但仍加入错误分支 |
| 3 | 186 | 37.9k / 27.5k / 27.3k | 大量负反例限制 b 的位置和 c 段；修复仍不稳定 |
| 4 | 197 | 25.2k / 20.4k | 先得到“全局 c 奇偶性”近似，随后改成局部 cc 块，最终等价 |

### Epoch 0：少量数据下形成错误解释

初始 8 条数据没有正例要求使用 b，模型认为正例只包含 a/c。它尝试用 a 段与 c 段交替、以及“至少三个 a 段”等条件解释数据。这些是对当前样本的拟合，并非目标语言的真实结构。

三次尝试的训练准确率依次为 87.5%、87.5%、100%。第三次虽然拟合了已有训练数据，仍与目标不等价，形式验证的 witness 为 `b`。

### Epoch 1：正反例推翻过窄假设

新增 18 条数据中，正例包含 `b、ab、ac、accb、accab、accac、accaccb、accaccab`，负例包含 `cc` 以及多条由 a/c 交替形成的字符串。

这些正例直接推翻了“只能使用 a/c”和“需要很多 a 段”的旧解释；负例 `cc` 推翻了“任意非空 c 串均可接受”。模型开始围绕 `acc` 重复块组织解释，但出现了语法错误，也未能得到正确结构。

### Epoch 2：开始识别可复用块，但仍过度泛化

新增 35 条数据包含 `c、ccb、ccab、aab、aac` 等正例，以及 `caac、caccab` 等负例。这些例子要求模型扩展原来的 acc 模式，同时约束 c 出现的位置。

模型开始使用 `cc、cca、ccb` 等块，但还会允许错误的 b 位置、错误的尾部和额外分支。三次尝试中第二次训练准确率最高，约为 96.72%，第三次反而降到 50.82%。因此不能把该 epoch 最后一次输出简单理解为对前一次的改善。

### Epoch 3：大量负例暴露过度泛化，修复仍不稳定

新增 125 条负例，包括 `cccc`、b 不在末尾的字符串，以及带字符类的反例族。它们暴露了候选对于 c 段和 b 位置的过度泛化。

模型经历一次语法错误和两次修复尝试，仍未收敛。外层下一轮采用的是按 repair score 选择的候选，并不一定是这一轮最后一次输出。

### Epoch 4：从全局奇偶性转向局部构造块

新增 11 条数据，其中 `b` 是正例，`ccca、cccab、cccac、cccacb、cca、ccacb、acb、acc、aa、aacb` 是负例。

第一候选已经达到训练准确率约 97.97%、eval accuracy=100%，但仍不等价。它主要用“c 总数的奇偶性”组织规则，尚未限制每一段内部 c。

最后一次 repair 明确列出了它在已有训练集上判错的四条负例：

```text
ccacacac, 0
acacac, 0
caccaacac, 0
caacac, 0
```

**这四条都以 c 结尾，且 c 的总数为奇数，但中间存在奇数长度的 c 段。** 它们直接区分了“全局 c 奇偶性”与“局部 c 段必须成对”这两个竞争假设。

例如：

```text
acacac = a · c · a · c · a · c
```

“c 总数为奇数”的规则会接受它，但目标语言要求内部 c 成对出现，因此必须拒绝。

最后一次 analysis 逐步转向“a 之间的 c 必须成对”的解释，并构造出：

```text
((c c)* (a (c c)*)* b)
+
((c c)* (a (c c)*)* c)
+
a
```

该 regex 通过了等价检查。关键变化是 **从统计整串的 c 数量，转向使用 cc 作为可重复的构造单元**。它排除了一族错误字符串，而不只是为四个反例分别添加例外。

## 5. 对三个问题的直观回答

### 5.1 Counterexample 在 LLM reasoning 中间起什么作用？

这个案例中，它最重要的作用是 **把“当前解释到底错在哪里”转成具体、可比较的证据**。

- **正反例纠正过窄假设。** 模型早期认为正例只包含 a/c，`b、ab、accb` 等正例直接要求扩展语言。
- **负反例纠正过宽假设。** 最后四条负例说明：“总数奇偶性正确”仍不能保证“每段的位置和奇偶性正确”。

反例不只是增加训练数据，它还把 reasoning 的注意力引向 **当前候选遗漏的那个区别**。最后四条 repair examples 尤其像是在追问：“你现在这套解释为什么把这几条也接受了？”

从实际推理文本看，模型会重新比较字符串、检查重复块和奇偶性、再修改 regex。但这不是反例一出现，模型就必然找到正确解释；中间仍可能计数出错、形成新的错误规则。

### 5.2 LLM 有没有明显的处理 counterexample 的步骤和方法？

**有可观察的处理模式，但没有表现出稳定、严格执行的算法。** 完整 analysis 中反复出现以下行为：

1. 重新整理正负例。
2. 比较短串和相似字符串，检查首尾、重复块、奇偶性。
3. 提出一个结构条件，再用若干样本手动检验。
4. 修改分支、重复块或末尾限制。

最后一次成功修复中，可以清楚看到它转向“a 之间的 c 必须成对”的解释，并据此构造 regex。不过两种方法的推理里都存在错误计数、误读标签、添加临时条件、使用不支持的语法，以及发现正确线索后又退回错误假设等现象。

因此，不能将成功描述为“模型每轮都精准定位反例，再做最小修复”。更准确的描述是：**系统持续提供具有针对性的失败证据，模型反复尝试解释，最终某次重构得到了正确结构。**

还需要区分模型的行为与 pipeline 承担的工作：

- **外层 CE 生成**：系统从目标语言与候选语言的差异中生成新数据。正反例分别对应目标接受而候选拒绝、候选接受而目标拒绝的区域。
- **内层 repair**：系统检查候选在累计训练集上的错误，最多突出显示 16 条错误样本，或提供编译错误反馈。
- **reflection/repair prompt**：提供旧 regex、旧的简短 Reasoning 以及相应反馈。它没有把上一轮完整 analysis 全部回传。
- **候选选择**：repair score 用于选择可编译且更符合训练集的候选，因此外层采用的候选不一定是最后一次调用的结果。

日志中的 `Witness` 字段也不能直接当成“这条字符串一定被传给下一次 LLM”的证据，必须查看实际 prompt。本案例最后四条 repair examples 确实出现在最后一次调用的 prompt 中。

### 5.3 用 counterexample 带来的提升是怎么来的？

本案例最直接支持的解释是：**反例帮助模型区分那些看起来都能解释大部分样本的竞争假设。**

这里两个关键假设分别是：

- 近似规则：无 b 时，c 总数为奇数。
- 正确规则：主体由 a 和 cc 组成，再接终止符。

大量普通样本可能在两种规则下得到相同标签，因此 std 可以达到很高的 eval accuracy，却保留错误的结构。针对当前候选生成、并在 repair 中突出显示的反例，恰好位于两种解释发生分歧的地方，推动模型更换构造单元。

提升来自 **更有辨别力的证据、明确的修复反馈，以及多次验证机会**。在这个案例中，它最终触发了结构层面的重新解释，而不是简单记住几个反例。

这并不意味着推理越来越长或每轮越来越准。Full 在本例中有更多 LLM 调用，也包含 CE、reflection、repair 等多个组件；这个单案例不能将收益单独归因于 CE 的某个组件，更不能据此声称其计算效率更高。可以直接观察到的是：**full 在更少累计训练条目下获得了形式等价解，并且最后一次结构修正与 prompt 中突出显示的错误样本有清楚对应关系。**

## 6. 机制观察汇总（Mechanistic observations）

- **样本效率与计算量需要分开。** Std run-0 使用 11 次调用、3000 条训练数据仍不等价；full 使用 14 次调用、197 条累计训练数据，在 epoch 4 的 retry 1 成功。三次 full 都成功，三次 std 都在 3000 条数据时失败，但 full run-0 的调用次数和 analysis 总字符数都更多。
- **高 eval accuracy 不等于结构正确。** Std epoch 7、10，以及 full epoch 4 第一次尝试，都达到 eval=100%，却仍未形式等价。有限评估集不能替代语言等价检查。
- **正反例负责扩展，负反例负责约束。** Full epoch 1 的 b/ab/ac 正例推翻旧的 a/c 限制；负例 cc 推翻任意 c 串的规则。Epoch 2 的 c/ccb/ccab/aab/aac 正例要求更一般的块，而 caac/caccab 等负例限制 c 的位置。
- **反例可能暴露一整个错误族。** Epoch 3 的 125 条新增负例揭示 b 的位置和 c 段的过度泛化；最后四条 repair examples 则直接击中“全局奇偶性正确、局部结构错误”的盲点。
- **最后的有效变化是表示方式改变。** 成功候选使用 cc 块，表达内部 c 段的局部偶数约束，不再只依赖 c 总数的奇偶性。
- **推理不是单调可靠的。** 两种方法都有反复列样本、计数错误、错误条件、无效语法和很长但失败的 analysis；full 的同一 epoch 内也可能越修越差。
- **外层反例与内层修复应当区分。** 外层通过形式语言差异提供新数据，内层突出已有训练集中的错误；最终成功修复依赖的四条负例是被重新强调的旧训练错误。
- **模型并未获得完整的历史推理。** Reflection 和 repair 回传旧的简短 Reasoning 与 regex，因此延续的是显式假设摘要，而非完整上一轮 analysis。
- **因果结论有边界。** 该案例支持“针对当前错误的反馈有助于识别结构约束”的解释，但不是隔离 CE、reflection、repair 与额外计算的消融实验。

## 7. run-0 每次 LLM 输出的完整统计

`epoch.retry` 从 0 开始。N 为该 epoch 累计训练条目数。Train/Eval 为日志记录的准确率；Equivalent 为形式等价结果；Witness 为形式验证记录的差异字符串。`—` / `None` 表示没有对应评分或结果，不能当作 0；本表中未评分的调用在日志中记录了错误。

| Method | Epoch.retry (zero based) | N | analysis chars | Train | Eval | Equivalent | Witness |
|---|---:|---:|---:|---:|---:|---|---|
| std | 0.0 | 3 | 1129 | 1.0000 | 0.5000 | False | 'cc' |
| std | 1.0 | 6 | 3726 | 1.0000 | 0.5938 | False | 'cc' |
| std | 2.0 | 12 | 15997 | 1.0000 | 0.4375 | False | 'cc' |
| std | 3.0 | 24 | 16321 | 1.0000 | 0.7812 | False | 'cc' |
| std | 4.0 | 48 | 43472 | — | — | None | None |
| std | 5.0 | 96 | 49205 | — | — | None | None |
| std | 6.0 | 192 | 36002 | 0.7865 | 0.8125 | False | 'ccacc' |
| std | 7.0 | 384 | 28465 | 0.9948 | 1.0000 | False | 'cb' |
| std | 8.0 | 768 | 9538 | 0.9089 | 0.8750 | False | 'cc' |
| std | 9.0 | 1536 | 15067 | 0.5046 | 0.5312 | False | 'cc' |
| std | 10.0 | 3000 | 30800 | 0.9940 | 1.0000 | False | 'cccb' |
| full | 0.0 | 8 | 46013 | 0.8750 | 0.7500 | False | 'b' |
| full | 0.1 | 8 | 16951 | 0.8750 | 0.7500 | False | 'b' |
| full | 0.2 | 8 | 17987 | 1.0000 | 0.7500 | False | 'b' |
| full | 1.0 | 26 | 21395 | — | — | None | None |
| full | 1.1 | 26 | 15423 | 0.6538 | 0.6875 | False | 'aab' |
| full | 1.2 | 26 | 26405 | — | — | None | None |
| full | 2.0 | 61 | 24733 | 0.8689 | 0.7500 | False | 'ba' |
| full | 2.1 | 61 | 36016 | 0.9672 | 0.7812 | False | 'aa' |
| full | 2.2 | 61 | 16682 | 0.5082 | 0.5938 | False | 'a' |
| full | 3.0 | 186 | 37854 | — | — | None | None |
| full | 3.1 | 186 | 27471 | 0.9140 | 0.7812 | False | 'b' |
| full | 3.2 | 186 | 27322 | 0.8978 | 0.9375 | False | 'acb' |
| full | 4.0 | 197 | 25247 | 0.9797 | 1.0000 | False | 'cacab' |
| full | 4.1 | 197 | 20375 | 1.0000 | 1.0000 | True | None |


所有三次运行的逐次预测、简短 Reasoning、错误信息与长度统计另存于 [CSV](simplyrx_5state_sd2_trajectory.csv)。本文已包含理解本案例所需的结果、轨迹、关键反例与机制解释，无需依赖聊天记录。

## 8. 实现位置

- `train_icl_gen.py`：`build_reflection_prompt` 组织上一轮摘要、regex 与新反例；`build_retry_prompt` 收集累计训练集上的错误、构建 repair feedback；主循环负责重试、候选选择和外层 CE 迭代。
- `teacher.py`：`_generate_counterexamples_worker` 按目标与候选的语言差异生成正反例，本例采用 DFS 与 clustered 配置。
