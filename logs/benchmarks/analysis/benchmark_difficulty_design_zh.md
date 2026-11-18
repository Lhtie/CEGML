# Benchmark difficulty 如何控制：与 regex star depth 的对应关系

本笔记提出实验设计，不修改现有 task、数据、oracle 或 balanced 采样代码。结论是：可以为各任务寻找类似 star depth 的结构参数，但不宜用一个标量把不同任务强行排成统一难度。应分别记录规则结构、输入/推理规模、样本可辨识性，以及 oracle 验证成本。

## 1. 建议先区分“什么变难了”

| 轴 | 含义 | 例子 |
|---|---|---|
| 规则结构复杂度 | 表达目标规律需要多少种操作、组合或依赖 | filter 后再 map；辅助谓词与递归 |
| 输入/推理规模 | 同一个规律在更大实例上执行需要多少步 | 更长路径、更大整数、更多列表元素 |
| 样本可辨识性 | 给定 IO 是否能排除错误但一致的捷径 | sum 的所有例子都以 0 开头；字母覆盖不足 |
| 泛化距离 | 测试需要多少训练未见的组合/结构 | 学过 jump 和 twice，没见过 jump twice |
| oracle 难度 | 证明/寻找 CE 的代价与不完备程度 | SMT unknown、量词实例化、未覆盖输入 |

只有第一项最接近想用 star depth 衡量的“目标规则复杂度”。其他几项也值得控制，但需要另命名，不能把增加 oracle timeout 或删掉关键示例解释成目标规则更复杂。

Regex 自身也有表示层面的区别：`(a*)*` 与 `a*` 等价，却有不同的语法嵌套深度。建议对生成表达式做基本简化，并联合报告最小 DFA 状态数；最小表达式所需的 star height 与一个给定表达式的 syntactic star depth 也不是同一个量。

## 2. 各 task 的推荐主指标

| Task | 最推荐的可控结构指标 | 应另外记录 | 当前 setting 能否直接形成该梯度 |
|---|---|---|---|
| List Functions | 固定 DSL 下的规范化组合深度 | 分支数、跨元素依赖、输入长度、稀有触发率 | 可先给现有 250 个函数标注；精确控制深度需要独立生成变体 |
| ILP | 规则依赖层级、必要的辅助谓词/递归结构 | 固定参考程序的推导深度、常量数 | 当前仅 EvenSucc，只有一个目标规律，不能据此声称已有三个结构难度层 |
| SCAN | 命令的语义操作组合深度 | 输出长度、训练未见组合、held-out template | 可以按现有命令分桶；全部 split 的完整解释器规则仍相同 |
| Instruction Induction | 对可程序化子任务，用规范化规则组合深度 | 例外率、语义歧义、label noise | 原始 24 个任务不适合共用一个客观深度标尺 |
| Cryptography | 原 setting 内按 cipher family 分组；扩展任务可用最小转导器状态数/位置周期 | 字母覆盖、映射自由度、输入长度 | 原有三种都是逐字符替换，不存在自然的“层数深度”梯度 |
| 拟议 NIA task | 变量交互结构、次数、单项式/原子数的向量 | 训练支持范围、非线性耦合、solver 状态 | 尚未实现；任何单一参数都不保证 solver 运行时间单调 |

这些是设计建议，不是已经由实验验证的 easy/medium/hard 标签。

## 3. List Functions：最适合先做 composition depth

### 比原始 Python AST 大小更有意义的定义

先固定一个有类型的简洁 DSL，例如：

- 元素变换：Map(Add(k))、Map(Affine(a,b))。
- 筛选：Filter(IsEven)、Filter(GreaterThan(k))。
- 顺序/位置：Reverse、Take、Drop、Rotate、IndexSelect。
- 聚合：Length、Sum、Count，按任务要求封装为 list 输出。
- 邻域操作：SelectByNext(predicate)、相邻差值。
- 条件分支：If(condition, branch1, branch2)。

在这个固定语言里，记录规范化后最长依赖链的操作数。例如一个 map 是一层；map 的结果再送入 filter 是两层；再 reverse 是三层。分支深度和嵌套循环/邻域依赖另记，不强塞进同一个深度。

不要直接按 Python 写法定深度。显式 for、list comprehension、内置 sum 可能表达同一个语义操作。`Add(1)` 后 `Add(2)` 可合并为 Add(3)，Reverse 两次可消去。若 DSL 自带 Affine，`3*x+7` 也只是一种元素变换，而不是因为 Python 多两个 AST 节点就算高难度。

“最短 DSL 程序深度”理论上最理想，但未必容易求到。实际可分两层标注：生成/参考程序的简化后深度，以及在限定枚举预算内有没有发现更短等价解。没有证据时不要声称已经求得真正的最小描述。

### 当前 9 个 balanced 函数给出的警示

- c071：逐元素 +2，c162：逐元素 affine 变换，都可以是一层 Map；后者 AST 较大不代表结构上多一层。
- c067：首尾交换，c103：返回长度，c122：取倒数第二项，都是单个常见语义操作，但 LLM 熟悉程度不同。
- c204：保留后一个元素为 0 的项，涉及相邻关系，与纯逐元素 filter 不同；而且若训练没有 0，很容易误学为空列表。
- c181：偶数位置元素的顺序变换，涉及位置条件和跨位置依赖。
- c224：参考代码含多个切片和拼接，需按实际 Python 语义展开，不能只把代码注释里的自然语言步骤当作严格的串行组合深度。

当前 `list_complexity_tiers()` 用 AST 节点、for/while、if、call、subscript、comprehension 的加权和分三层。这适合先做覆盖采样，却不是类似 star depth 的语义复杂度；现有权重没有实验证明，循环/内置函数写法也会影响排名。

### 实验建议

- 原始复现实验：不改 250 个原任务；增加 `operator_family`、`composition_depth`、`branch_depth`、`dependency_span` 等标签，再分层抽样。
- 受控难度实验：新增独立的 `list_composition` 变体，用同一 DSL 生成 d=1/2/3 的目标程序，先简化，再剔除退化/重复语义。
- 在每层固定训练 IO 数、输入长度/数值分布和 oracle 搜索预算；额外确保分支和稀有条件有足够覆盖。
- 保持当前 List Functions oracle 的长度 3–8 不变；如果难度实验同时增加真实列表长度，需要明确承认 oracle 覆盖不足也在改变，不能把接受率提升当成学习更好了。

## 4. ILP：规则依赖和 proof depth 要分开

[HRI 论文](https://proceedings.mlr.press/v162/glanois22a.html)的建模涉及辅助谓词、递归和层级结构。这给 ILP 提供了很自然的结构维度。但当前我们只实现了一个 EvenSucc setting，尚不存在多个不同规则结构的 ILP 层级。

### 规则层面的指标

- 用一个固定 Horn/meta-rule 语言衡量必要的辅助谓词数量，而非任意命名数：把同一条规则机械拆成三个辅助谓词，不应自动获得更高难度。
- 非递归谓词依赖 DAG 的最长路径；有递归时先标出强连通分量，区分单谓词递归、相互递归和不同递归组件间依赖。
- body atoms 数、变量数、关系 arity、是否需要 predicate invention 都可辅助描述。
- HRI 的 max_depth 是模型 bias/容量上限，不等于任意任务真正需要的最小规则深度。我们现有 metadata 的 paper_max_depth 也不能直接当 difficulty 标签。

可后续选择论文中的 Grandparent、Connectedness、Member 等不同关系任务，按规则结构分组；但不要未经验证就断言它们构成一个所有模型都严格递增的排序。

### 同一 EvenSucc 里控制推理规模

扩大 N 会增加较远 target atoms 所需的推导长度，但目标规则还是相同的三个 Horn clauses，不是新添了规则结构。

当前参考程序在同步 Boolean 推理下，target(2k) 需要 k+1 轮：zero 的 base case 在第 1 轮，之后按两步 successor 扩展。因此：

| 常量数 N | 最大目标偶数 | 参考程序完整覆盖所需轮数 |
|---:|---:|---:|
| 11 | 10 | 6 |
| 15 | 14 | 8 |
| 31 | 30 | 16 |
| 63 | 62 | 32 |

这是相对于当前固定参考程序和同步更新语义的推导步数；引入不同的辅助关系或等价规则重构可能改变 proof depth，不能把它当语言无关的最小复杂度。

如果做 N=15/31/63 的泛化梯度，必须同步增加 evaluator 的推理预算，并扩展目前只支持 N≤15、steps≤8 的实现。否则正确递归规则会仅因被截断而被判错。保留训练 N=11，扩大评估 N，可以明确测“递归外推”；作为新诊断变体标记，不能称为原论文默认 setting。

当前一元 EvenSucc 的 query 数是 N；其他二元 target 的完整域有 N² 个 atoms。因此也不要把所有 ILP 任务的“常量数相同”直接等同于验证成本相同。

## 5. SCAN：组合深度主要是样例难度，不是 8 套不同规则

[官方 SCAN 说明](https://github.com/brendenlake/SCAN)区分随机划分、动作长度外推、新 primitive、新 template 等泛化条件。这些是不同的泛化轴，不是按同一个 difficulty 参数递增的 8 级。

一个可操作的样例指标是固定语义算子词表后的 AST depth：把 twice/thrice、around/opposite、and/after 等作为约定的操作节点，primitive 为叶子。比如按“around 是一个原子算子”的约定：

- jump：depth 0。
- jump twice：depth 1。
- jump around left twice：depth 2。
- jump around left twice after walk thrice：depth 3。

这个深度依赖操作粒度；若把 around 展成“转向+动作+重复”，深度会改变，必须固定定义后再比较。原始 SCAN 语法本身也限制组合层级，不支持无限堆叠 after/and；要生成更深命令，需要新 grammar、新 GT generator 和新 oracle domain，并明确为扩展任务。

在现有 setting 中，8 个 split 的 GT 都是同一个完整 SCAN interpreter，我们的 oracle 也统一检查所有 20,910 条命令。只让某个 split 的测试例较浅，并没有把这个 oracle 要求的目标规则变简单。

因此有两种有意义的实验：

1. 保留完整 SCAN，按命令的 composition depth、动作长度、训练未见组合分别报告分桶准确率；它衡量输入与泛化难度。
2. 新增 `scan_depth` 诊断族，让 d=1/2/3 分别定义较小语法子域，并让训练、评估和 oracle 都使用对应子域。否则 d=1 的 candidate 仍须通过完整复杂 grammar，难度标签就名不副实。这个变体与原始 full-SCAN 成功率分开报告。

若只增加 around 的重复次数或输出动作长度，改变的是执行规模；这不一定增加规则描述复杂度。

## 6. Cryptography：别把可折叠的 substitution 层数当结构深度

目前 Caesar、Atbash、Keyboard 都是逐字符替换。串联两个固定逐字符替换仍是一个逐字符替换，因此“做 k 次 substitution”通常可折叠，不能自然对应 k 层必要的结构复杂度。

保留原 setting 时，更合理的是：

- 按 cipher family 分组，不把 family 名称直接排成数学上有保证的 easy/medium/hard。
- 记录观测覆盖了多少字母、多少大小写，以及未见映射的比例。这改变可辨识性，不改变映射规则本身。
- 区分 Caesar 的低自由度参数、预定义 Atbash/Keyboard 规则和任意 substitution 的自由度；但 LLM 对字母表/键盘先验的熟悉程度也影响实际难度，不能只按参数数目排名。

若需要一个真正可调的结构族，可以新增位置周期为 k 的 substitution：第 i 个字符使用第 i mod k 张替换表。k=1 是普通 substitution；k=2 需要区分奇偶位置；k=4 需要跟踪四个位置状态。确保这些映射确实不同、没有更小周期后，最小转导器状态数/必要位置周期就比较接近 regex 的状态复杂度。

这已不再是原论文那三类 cipher 的原始 setting。必须另外命名和生成数据，按每个相位分别检查字母覆盖。更长字符串本身不是更高 k，也不要把这种结构难度描述成实际密码学安全性。

## 7. Instruction Induction：先挑可程序化子集，再谈统一深度

原始 24 个任务包含数学、字母位置、词形、翻译、语义分类、正式改写等，缺少一个共用的组合语法。[官方任务及评分](https://github.com/orhonovich/instruction-induction)也包含不同的可接受输出规则。不能仅凭“翻译看起来比字母提取高级”，就建立可验证的复杂度阶梯。

可分别控制：

- 规则型子任务：已固定的字符串/算术 DSL 中，单步抽取、条件抽取、多个操作组合。
- 词形任务：规则变化与不规则例外比例，但例外比例应独立于规则组合深度。
- 语义任务：上下文长度、关系组合、歧义程度；标注质量和多答案容许规则需要先固定。

例如把 first letter 变成 second letter，并没有增加多少规则结构；二者都是常数位置索引。给 sum 加更多 operands 则改变原始“两个数求和”的任务，须作为新变体。给 formalization 增加长度也可能只是更长输入，不是更复杂的目标 instruction。

更贴近本研究目标的办法是新增 `instruction_composition`，从可执行 DSL 生成 GT 和 NL 描述，比如“选匹配项 → 执行字符串变换 → 聚合”，按 d=1/2/3 控制目标组合。这样能确定 CE 的标签，并避免只靠另一个 LLM 判断规则有多复杂。它与原始 24 任务必须分开报告。

集合任务尤其要小心：taxonomy_animal 的排序要求在当前集合评分下可能完全不增加任务难度。若想研究顺序组合，必须明确改变输出规格和评分，而不是继续用 set metric 却声称模型需要学会排序。

本轮审查已发现的“输入退化导致捷径”“合法翻译被错拒”属于样本与 oracle 的问题，不应重命名为高难度来掩盖。

## 8. 拟议 NIA task：结构与 solver 难度并列记录

对尚未实现的无界整数非线性谓词族，可以记录：

- 变量数、最高次数、非零单项式数、关系原子数、Boolean 组合深度。
- 非线性变量交互图，例如哪些变量出现在同一乘积项中、图的耦合结构。
- 若以后加入 quantifiers，另记 alternation depth；出现一个 forall 不自动构成不可判定或更难。
- 系数大小、训练取值范围、正例密度、CE 到训练区域的距离。

不建议只按多项式次数分层。例如一个明显非负的高次方可能比次数更低、但耦合更复杂的整数关系更易证明。solver timeout/unknown 应作为 oracle 指标独立报告，不能用它直接给 learner 难度打分。

保留 noisy 接受时，这一点更重要：难题可能更容易耗尽搜索预算，从而被“没找到 CE 就接受”的 policy 提前接受，造成表面成功率反而更高。应拆分 operational accepted、proved equivalent、held-out 具体标签正确率，以及 CE 搜索覆盖。

## 9. 建议的第一阶段实施顺序

优先做 List Functions、ILP、SCAN，它们最容易给出可审查的结构定义。原始 setting 保持不动，先加标签/分桶；需要改变生成规律、输入域或评估语义的部分单独建诊断变体。

1. List Functions：现有 250 函数先标操作家族、组合深度、分支/邻域依赖；保留当前 AST rank 作为辅助，不再把它单独称作语义难度。
2. ILP：增加几类不同规则结构的原文任务；EvenSucc 另外做推理规模梯度，调整 inference budget 防止错误截断。
3. SCAN：先对现有 20,910 命令做固定语义 AST 分桶，再按 depth、length、novel composition 报告准确率；没有必要先改 grammar。
4. Crypto 和 Instruction Induction：先承认原 setting 没有共同的连续结构难度参数。若需要严格梯度，再做独立的 periodic-cipher / instruction-composition 族。

每一层保证相同实例数、重复次数、模型参数、oracle 预算；在每个 family 内对齐初始信息量。目前非固定样本 task 的默认 std 从 3 个 IO 开始，CE 从 8 个开始，这会影响难度/收益比较。新的分层对照中应对齐 n_init，再比较追加普通样本和追加 CE 的效果。Crypto/List Functions/ILP 已声明的全量初始样本协议另按各自 setting 保留。

完成结构分层后，再通过固定模型的小规模多次运行校准实际表现，报告成功率、首个有效 CE、达到正确 hypothesis 的调用/样本/token 成本。不应根据本次模型是否答对，再回头把同一批样本重新命名为 easy/hard；若确实用经验成功率校准，需要独立 pilot，并在评估前冻结分层。

**推荐用一个难度向量而非总分：规则组合深度、依赖/递归结构、实例规模、样本可辨识性。Oracle 的覆盖和不完备状态另列。** 先在每个 family 内比较随一个轴变化的曲线，再汇总各 family 的 std/CE 效果；不同 family 的 depth=2 不代表同一种绝对难度。
