# 另一个 oracle 一般不可判定的任务：设计笔记

当前已实现的是有限域 ILP；本笔记讨论下一个候选任务，尚未将它注册为可运行 benchmark。不能为了满足“undecidable”这个名字，把当前有限域的规则验证或有限长度字符串搜索直接改名。

## 1. 区分四个层次

- **问题族不可判定**：不存在对所有输入都正确、总能终止并回答 yes/no 的算法；不意味着每个具体例子都困难。
- **半判定程序**：必须明确对哪一侧保证最终找到证据，例如“有 CE 时，总有一天枚举到它”；对另一侧可能永远运行。不是遇到 unknown 的 solver 都自动具备这个保证。
- **Best effort**：算法可能给出证据、证明或 unknown，但不保证对某一整侧都最终成功。实际 quantified SMT 和非线性整数 SMT 通常属于这类。
- **预算耗尽后的 noisy 接受**：是实验策略，不是数学上的证明或半判定保证。可以保持我们的 pipeline 早停行为，但要与 proof status 分开记录。

cvc5 的官方[理论教程](https://cvc5.github.io/tutorials/beginners/theories.html)明确区分实数上的非线性多项式算术与整数上的 QF_NIA：后者一般不可判定，solver 实现的是不完备启发式。官方[unknown 排查说明](https://cvc5.github.io/blog/2024/04/15/interfaces-for-understanding-cvc5.html)还列出量词实例化、资源限制等原因；unknown 不能解释成没有反例。

## 2. 首选：无界整数非线性谓词归纳

建议名称：`nia_predicates`。

输入是任意精度整数 tuple，GT 是可计算的多项式关系谓词，标签为 Boolean。例如：

```text
GT:        y = x*x
IO:        [0,0] -> true, [1,1] -> true, [1,2] -> false
Candidate: y = x
CE:        [2,4] -> true   (candidate gives false)
```

这里 y=x² 只是最简单的结构示例，不代表这个具体式子就不可判定。任务族需要容纳一般的整系数多项式零集/布尔组合，理论上不固定为有限个变量、极低次数的一小类方程；具体实验可以先从容易的实例开始，但不要把受限可判定子类的结果称为不可判定性证明。

与我们已有任务一样，learner 从 IO 归纳一个可解析的逻辑表达式。输入域为 Z^k，不是 bit-vector，也没有偷偷加上的全局 [-B,B] 范围。

SMT oracle 构造：

```text
寻找整数 tuple x，使 F_gt(x) XOR F_candidate(x) 为真。
```

这些自由变量在 satisfiability 查询中表达“存在某个输入”；无需刻意添加一个 forall 才能进入困难理论。若公式涉及一般整数乘法，其 QF_NIA satisfiability 已一般不可判定。可以用 `P(x)=0` 与恒 false 比较来联系整数多项式解的存在性。

尤其注意：**比较多项式函数的数值是否恒等**通常可通过展开和系数比较解决，不是我们要的困难问题。这里比较的是多项式关系/零集给出的 Boolean 分类，不是单纯判断两个多项式是不是同一个多项式。

### Oracle 工作流

1. 验证输入表达式语法，用精确整数语义编码 reference 和 candidate。
2. cvc5 Python API 尝试无界 QF_NIA 的 mismatch 查询，设置明确时间/资源预算。
3. SAT：取模型里的整数 tuple，用独立的具体表达式求值器重放两边；确实不同才返回 CE。不要把近似模型、未解释项或转换器错误直接当 CE。
4. UNSAT：如果是忠实的无界编码，则可以作为这一候选等价的证明。实际系统仍信任编码和 solver；不能把 bounded UNSAT 混进来。
5. UNKNOWN/timeout：可进行有界枚举、抽象证明尝试，或按实验策略返回 unknown/noisy acceptance。

有界枚举只作为搜索策略，不能替换 GT 定义域。若依次枚举 [-1,1]^k、[-2,2]^k 等整数壳，每一点的多项式关系求值都会终止；若真的存在有限整数 CE，公平枚举最终会找到它。因此“持续枚举 CE”构成这一任务 CE 存在性的半判定程序。把它截断到有限预算后，才变成 best effort。不能把单次 cvc5 启发式求解本身宣称为这个完整半判定保证。

### 与现有 benchmark 接口一致

- task：整数 tuple -> Boolean，与 ILP 一样可使用 Boolean labels。
- hypothesis：逻辑/算术表达式 AST，不允许任意 Python 副作用或求解器命令注入。
- training_errors / score：具体精确整数求值，通常可直接决定，不依赖 SMT 是否成功。
- verify：只在 oracle 内实现 SMT；epoch、repair、候选选择、早停继续走相同 controller。
- 数据：对已知存在正负例的实例分层采样，防止全负例的平凡猜测；单独保留较远整数点作诊断，并记录训练范围，避免把范围偷偷变成任务定义。一般多项式零集也可能为空，随机采不到正例不能证明它为空。小规模人工生成的平衡子集不自动继承整个逻辑的不可判定性。

## 3. Quantified logic 作为后续扩展

可考虑“带量词的规范/不变量归纳”，例如从状态或关系样例归纳 forall/exists 约束，再问 SMT 是否存在违反候选和 GT 对齐的状态。一般 FOL 及适当的理论组合是不完备求解的自然来源。

但需要小心：

- 不是所有带量词的逻辑都不可判定，不能仅凭出现 forall/exists 就下结论。
- 若实例化到固定有限域后穷举，得到的仍是一个有限可判定检查；它只能说明有界范围。
- 量词消去可处理的一些算术片段仍可判定。
- 用普通一阶公理编码递归谓词时，SMT 模型可能不是 Datalog 的最小模型；不能拿任意模型直接当 ILP 程序的运行反例。
- 有 forall-exists 的规范在一个具体 tuple 上也可能需要额外求解，不能自动沿用上面“逐点可计算，所以公平枚举必然找到 CE”的论证。
- 优先用真实执行/可计算的 pointwise evaluator 验证 CE；如果没有独立可检验的 witness，记录 unknown，而不是编一个训练标签。

因此第一版更建议选无量词、但一般不可判定的整数非线性关系族：具体标签可靠，理论边界清楚，再扩展量词会更容易解释实验结果。

## 4. Abstract interpretation 能起到什么作用

[抽象解释的基本作用](https://cs.nyu.edu/~pcousot/absint.html)是 sound abstraction/approximation。可以给某些候选提供快速证明或筛选线索，例如 interval/sign/congruence domains。

例子：若能证明 `x*x+1 >= 1` 对全部整数成立，则 `(x*x+1=0)` 与 false 对齐，这是一个可用的无解证明。普通区间域若丢失 x 与自身的相关性，可能无法自动得到这个结论；这里需要 sound 的平方/sign 传递规则，而非假装任意抽象域都能证明。

对循环安全/不变量任务，sound 的 over-approximation 可以证明 absence of bad states；若抽象结果报告“可能有 bad state”，它可能是 spurious，需要 concretization/SMT 检查。不能直接把抽象反例塞进训练。widening 帮助终止，会损失精度，并不使所有不可判定性质变得可判定。

抽象解释通常既不是“完整解法”，也不是“对有 CE 的一侧总能结束”的保证。它与公平 CE 枚举、SMT 是不同性质的组件，适合组成 portfolio，而非共用一个模糊的 semi-decision 标签。

## 5. 按当前 noisy oracle 偏好的返回策略

你之前希望“预算内没找到 CE 就接受并停止”。下一个任务也可以保留这种 operational 策略，同时记录证据强度：

| 实际结果 | Equivalent | 建议记录 |
|---|---|---|
| SAT 且 CE 具体重放成功 | false | solver_result=sat, witness_replayed=true |
| 无界 mismatch 查询 UNSAT | true | decision_basis=proved_equivalent, proof_found=true |
| sound abstraction 证明无 mismatch | true | decision_basis=abstract_proof, abstract_domain=... |
| UNKNOWN/timeout/有限枚举无 CE，noisy policy | true | decision_basis=accepted_without_refutation, proof_found=false, guarantee=heuristic_not_global_proof |
| 同一情形，strict policy | null | decision_basis=inconclusive |
| parser/编码异常、模型不能重放 | null | error/translation_mismatch；不能拿实现故障充当正常的 no-CE 搜索 |

noisy 策略下返回 true 后仍按统一 pipeline 早停。必须同时保留原始 solver_result、timeout、查询公式/哈希、枚举范围、已查数量和 acceptance policy；分析时分开报告“被接受”和“被证明”。尤其不能改写 solver 的 unknown 为 unsat，也不能把有限枚举没找到 CE 叫全域 equivalence。

本笔记只提出新任务方案；当前 List Functions 的 3–8 长度 noisy oracle、现有 Crypto/SCAN/Instruction Induction 及 regex 的语义未因此改动。
