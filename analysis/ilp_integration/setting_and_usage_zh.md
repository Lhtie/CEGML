# 新增 ILP benchmark：HRI EvenSucc

已增加并列的 `task_type=ilp`，第一个 setting 为 `even_succ`。没有将 ILP 包装成普通 list-to-list Python 任务：它保留背景知识、正负 ground atoms、Horn clauses、递归和辅助谓词发明。它与 List Functions 共用 benchmark 基础设施，但输出对象是逻辑程序。

## 论文及源码对照

依据 [原论文](https://proceedings.mlr.press/v162/glanois22a.html)和[作者代码](https://github.com/claireaoi/hierarchical-rule-induction)，本轮采用 EvenSucc。源码固定在 commit `1a3a0465a5a5305ad8fec958901710dc6ead76e4`，原文件与 CC0 许可证保存在 `datasets/ilp/source`。

| 项目 | 本次 setting |
|---|---|
| 背景关系 | zero/1、succ/2 |
| target | 一元谓词；给 learner 的 prompt 不直接说明其数学含义 |
| 训练常量 | 0–10，共 11 个 |
| 评估常量 | 0–14，共 15 个 |
| 训练/评估推理轮数 | 6 / 8 |
| 观测标签 | 干净的正负 Boolean labels，全量 11/15 个 ground atoms |
| hypothesis | 正 Horn clauses；一元/二元辅助谓词；允许递归和相互递归 |
| 输出格式 | 简要 reasoning 标签 + ans 中的 Horn 程序，可包 prolog fence |
| 额外语言/资源限制 | 每条最多两个 body atoms、三个变量；32 条规则、12 个辅助谓词；规则中不用数字常量、Python 和算术 |
| oracle | 有限域 Boolean forward chaining，检查 N=15 的全部 15 个 target atoms |
| 评估指标 | 完整 15 个标签的 accuracy、MSE 和 MSE < 1e-4 的 symbolic success |

这不是 HRI 训练算法的重实现：其 optimizer、embedding、Gumbel 参数噪声、R-star proto-rule 参数化、层级 bias 没有搬过来。`paper_max_depth=4` 仅记录论文配置，不表示本实现已强制相同层级。我们使用同步 Boolean 推理，也不声称与作者 fuzzy/layer update 数值上完全一样。因而适合描述为“采用 HRI EvenSucc 的数据与主要推理预算，使用 LLM+CE 学习方法”。

作者的 EvenOdd 与 EvenSucc 在其通用模板下目标相同，本轮没有将两者重复计作两个独立 benchmark。

## 一条完整 trace 的结构

背景知识包括 `zero(0)`、`succ(0,1)`、`succ(1,2)` 等。训练示例是：

```text
{"num_constants": 11, "atom": "target(0)"} => true
{"num_constants": 11, "atom": "target(1)"} => false
{"num_constants": 11, "atom": "target(2)"} => true
...
```

假设初始 hypothesis 只有：

```prolog
target(X) :- zero(X).
```

有限 oracle 在较大背景 N=15 下发现正例无法推出，例如：

```text
{"num_constants": 15, "atom": "target(2)"} => true
{"num_constants": 15, "atom": "target(12)"} => true
```

CE 作为同样格式的 IO 加入训练，learner 可修正为：

```prolog
target(X) :- zero(X).
aux(X,Y) :- succ(X,Z), succ(Z,Y).
target(X) :- target(Y), aux(Y,X).
```

它在全部 15 个 target atoms 上符合标签时，oracle 返回 true，共享 controller 早停。这只表示指定有限背景和推理轮数内通过，不证明任意常量数下都等价。

## 与现有框架统一的部分

注册、suite 发现、std/CE、reasoning/ans、反思/repair、训练一致性、候选选择、早停、Logs/Attempts/OracleDetails 都复用现有接口。`train_icl_gen.py::run_episode` 未改。额外代码只负责 ILP 数据、解析、Boolean 推理和有限 oracle。

训练默认一次提供全部 11 个标签，与原任务的全量训练 valuation 对齐。CE 模式随后可加入 N=15 的反例；这是我们的方法扩展，会暴露评估域标签，不是原文原样的训练协议。Direct ICL 入口也支持 Boolean IO，并统一遵守 task 声明的样本默认值与 shuffle 设置；因此已经声明固定全量样本的 Crypto/List Functions 的 direct ICL 也会尊重那些配置。

## 运行指令

只预览，不调用模型：

```bash
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --dry-run
```

运行 std、CE，默认 mode=both：

```bash
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --mkey gpt-oss --outdir logs/benchmarks
```

若要显式进行 10 个 LLM rerun：

```bash
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --mkey gpt-oss --outdir logs/benchmarks -- --rerun 10
```

单独跑一项：

```bash
python -m benchmarks ilp --benchmark_id even_succ --mkey gpt-oss --use_ce --outdir logs/benchmarks --rerun 1
```

日志按原框架写入 `icl_gen_ilp/model=.../std|ce/reg|noreg/even_succ.json`。Suite 新增此 setting 后，all 为 583 settings；balanced 为 48 settings / 96 个 std+CE 作业，旧任务选中的 IDs 不变。

## 已完成的验证

- 以作者的原始 `EvenSuccDataset` 在 CPU 实际生成 N=11/N=15 数据，逐项比对 target labels、zero、succ，与本次 JSON 完全一致。
- 用论文描述的递归程序跑 6/8 轮，得到正确训练/评估 valuation。
- 使用实际共享 `run_episode` 和固定返回内容做 std/CE 离线验证：std 一轮成功；CE 先提供错误规则，再提供正确规则，两轮成功，训练条目从 11 增至 18。
- 验证没有 base case 的递归会按有限轮数结束；算术、数字规则常量、改写背景关系等非法 hypothesis 被拒绝。
- 这些是离线实现检查，不是 LLM 的 benchmark 成绩；记录在 `offline_validation.json` 和 `smoke_logs`。本轮没有调用付费模型跑 ILP 实验。
