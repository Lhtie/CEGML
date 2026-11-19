# ILP SMT oracle 实现

已接入 cvc5 Python API，使用现有安装依赖，不需要系统 cvc5 可执行文件。ILP 的 std 与 CE 通过同一个 BenchmarkTask.verify 接口调用。没有修改原 regex 的控制流程，也没有改变 ILP 的训练数据、balanced 六任务选择或 learner prompt。

## 返回契约

- true：SMT 证明，候选在全部约定检查输入上，与 GT 一致。
- false：SMT 找到差异输入，并通过独立 Horn 执行器和 GT 标签回放。
- unknown：求解器返回 unknown、展开/求解超时、关系或连接资源预算不足，或反例回放不一致。没有执行“没找到反例就当 true”的 fallback。
- 语法非法：仍由候选验证阶段报告 invalid/error，不混成已证明不等价的合法程序。

检查域现在包括完整训练标签池和验证域的全部 target 原子。Unary 为 384+512 条，binary 为 384+576 条；binary 的训练池仍是二十常量域四百个可能原子中的 384 个，不宣称覆盖未标注的十六个训练原子。这是有限背景上的等价，不是对任意规模/任意图的程序等价证明。

## SMT 怎样表示规则

1. 用已知背景事实进行正向相关性展开，收集可能出现的 ground atoms 和 ground rule supports。这里只建立约束结构，不以 Python 输出与 GT 的比较代替 SMT verdict。这个 grounder 自身需要有限域的正向展开，SMT 并没有消除这部分成本。
2. 从要检查的 target 原子向依赖反向切片，省掉无关辅助原子。GraphColoring 中不是所有同色节点对都与最终目标有关，切片可明显减少 SMT 公式。
3. 每个保留的候选原子对应一个 Boolean SMT 变量，另有一个整数 rank。
4. 每条规则编码为 body 成立则 head 成立；每个为真的 head 还必须存在一个支持规则，且该支持的所有派生 body 原子 rank 都严格小于 head。
5. 所有 rank 都限制在零到保留原子数量之间。这允许任何有限无环推导证明，排除没有事实基础的循环自我支持，保证编码的是正 Horn 最小模型。
6. 对全部训练/验证检查输入，构造候选 target 输出与 GT 标签不同的析取，交给 cvc5 做 SAT/UNSAT 判定。

例如仅有 target(X) :- target(X). 时，单纯 Horn 蕴含允许 target 任意为真；rank 条件要求 rank(target)>rank(target)，因此没有事实基础的 target 不能为真。离线检查也直接绕开 grounder，对包含互相支持循环的 SMT 编码做了 UNSAT 验证。

SAT 模型可直接提取多个不同的 ground-atom CE。每条 CE 均带原有 num_constants 背景标识，先经过问题的真实执行器和标签回放，再返回 pipeline。若是训练域 CE，则来自训练背景；不再仅从 eval 背景挑选。

## 资源预算与 unknown

默认每次 checkSat 最多十秒，整个搜索墙钟预算三十秒；正向展开与连接工作计数上限一千万，关系事实加支持数上限三十万，中间 binding 上限二十万。反例回放继承搜索的剩余墙钟预算；普通执行器另有二十秒和相应关系/连接预算。

预算不足而没有确认 CE 时返回 Equivalent=null，OracleStatus=inconclusive，具体原因写在 OracleDetails.status/reason。已有确认 CE 则保持 false。训练/验证评分在资源耗尽时记 null；不会对后续每条样本反复启动同一昂贵执行。

ILP 的初始候选验证改为语法/schema 检查，避免在调用 SMT 之前先执行一次完整关系推理。oracle 因 grounding 资源耗尽时跳过重复评分。其他任务和 regex 的判断/早停流程未因此改变。

## 本次同时修正的问题

- 训练/验证不一致：原来合法的“仅在 N=24 正确、N=20 训练标签错误”的 Predecessor witness，现在返回 false 和训练域反例。
- 空关系：schema 独立于第一条事实，空 edge/father 等已知背景关系可以正常编码和执行，不再因 rows[0] 越界。
- 资源不足：新增明确的超时、工作量、事实量、连接 binding 预算，并保留 unknown。

## 新日志

顶层 oracle 为 cvc5_finite_Horn_least_fixed_point。OracleDetails 记录：

- backend=cvc5_python_api，policy=finite_Horn_ranked_smt_v1；
- encoding、grounding 和有限域 guarantee；
- contexts 中训练/验证规模、ground_atoms、encoded_atoms、ground_supports；
- 查询次数、状态、完整性、预算和耗时；
- unbounded_rule_equivalence_proved=false、fallback=null。

原日志保持原有 oracle 版本，不自动重算。新的运行才使用 SMT oracle。旧审计报告记录的是改动前状态。

## 离线验证

- 本地十二个任务的正确参考程序全部返回 true；对应错误程序均返回 false，生成的 CE 均可回放。
- 现有实验中 24 个有效候选用 SMT 重查，八个 true、十六个 false，与原日志一致；没有 unknown。
- 原训练/eval 缺口 witness 被拒绝，并返回 N=20 的真实训练反例。
- 额外三十六个小背景/规则组合，与独立的全变量赋值枚举解释器对照一致。
- 直接测试 SMT 中无根循环不能自我支持。
- 空背景经 SMT 返回正常结果；工作量、墙钟、事实预算耗尽均返回 unknown。
- 模拟 solver unknown 后仍保留 unknown，没有执行 Python verdict fallback。
- std/CE 共享控制器的早停、resource-limit 后跳过重复评分、token/example 日志保持有效。

参考规则实测：balanced 中 Predecessor、UndirectedEdge、Grandparent、Length 的检查约零点零几秒，Fizz 约两秒，GraphColoring 约三秒。时间是本机离线记录，不承诺复杂候选也如此。

验证数据：smt_oracle_validation.json、smt_pipeline_validation.json、smt_random_validation.json。验证仅使用本地规则和假 LLM 响应，没有调用实际 LLM API，也没有生成新的模型性能结果。

## 仍未改变的 setting 差异

图/家庭数据仍使用此前的固定种子、分块模板；GraphColoring 仍是作者代码的 unary 版本。该次修改只实现并加固 oracle，不意味着数据分布或目标口径已经完全复现原论文。增加多个独立同规模背景时，还需要扩展背景标识，不能只依赖 num_constants。

建议用新输出目录比较：

```bash
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --mode both --mkey gpt-oss --outdir logs/benchmarks_ilp_smt
```


原始配置覆盖更新：当前已接入十九个作者合成配置。为支持新增 TwoChildren 等包含密集背景的任务，非递归程序使用目标导向 grounding 和独立 backward interpreter 回放；递归程序仍用正向支持展开。SMT 的 least-model rank 编码和 true/false/unknown 契约不变。二元训练池补齐到全部四百个原子，旧标签不变；详见 analysis/benchmark_difficulty/native_dataset_coverage_zh.md。
