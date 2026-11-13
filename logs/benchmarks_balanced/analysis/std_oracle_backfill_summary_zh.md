# std 原日志 oracle 补算结果

已在原路径更新 logs/benchmarks_balanced 中 std/reg 的 68 份日志；未重新生成任何候选。补算由 benchmarks.backfill_oracles 执行。

## 已完成的离线验证

- Cryptography：36 个有效 epoch 全部 Equivalent=true，覆盖 12 个实例。
- List Functions：59 个有效 epoch 中，10 个 false、49 个 null。Null 仍表示有限搜索没有证据，不能证明全域等价；另 13 个 epoch 原程序无效。
- SCAN：17 个有效 epoch 中，4 个 true、13 个 false；另 47 个 epoch 原程序无效。True 出现在 simple 的 epoch 2，以及 template_around_right 的 epoch 5/6/7。
- 合计完成 112 个有效候选的 oracle 检查，没有发生 oracle 异常。离线程序中的 60 个无效候选保留 Equivalent=null，OracleBackfill.status=invalid_candidate。

## 待完成的自然语言验证

Instruction Induction 有 180 个有效候选需原 gpt-oss-120b judge，另 1 个原输出无效。当前本地只有一张 8GB 显卡，而项目 gpt-oss 配置要求两张 GPU，且本地未缓存模型，因此未加载模型、未切换 API provider。这 180 条记录保留 Equivalent=null，并明确标为 OracleBackfill.status=deferred_llm_judge。

在原实验计算环境继续执行以下命令，会跳过已完成的记录，只补语言 judge：

```bash
python -m benchmarks.backfill_oracles --logdir logs/benchmarks_balanced --instruction-model gpt-oss
```

这会额外调用 LLM judge，但不会重新生成候选，也不会将 oracle 反例反馈给 std。若再次失败，可重新运行；已完成记录会跳过。

## 公平比较现在能得到的结论

- Cryptography：std 和 CE 都覆盖 12/12 等价实例；CE 第一轮已通过，std 旧实验则继续执行了原计划。
- SCAN：std 曾有 2/8 split 生成完整公开命令域等价的程序，CE 为 4/8；只看最后一轮，std 为 1/8、CE 为 4/8。
- List Functions：两种方法都不提供任意 Python 全域等价证明；false/null 应分别解释，且旧 CE 的 c067 数据污染没有通过本次 std 补算得到修复。
- Instruction Induction：judge 补算尚未完成，不能比较两边等价接受率。

## 日志字段与可复现性

- Equivalent：本次 oracle 的 true/false/null。
- OracleCounterexamples：仅用于事后验证的反例，不添加到 Counterexamples 或 Prompt。
- OracleBackfill：状态、候选 SHA-256、补算 seed=43、时间及是否复用同一候选的缓存结果。
- 检查过的相同任务/相同候选复用结果；List 搜索每个不同候选独立使用 seed=43。它是本次事后搜索，不是重放原实验的随机数状态。
- SCAN 使用同一 Python 子集校验与单输入超时，枚举相同的全部 20,910 条命令；为加速只缓存编译结果。
- 逐项比对 68 份备份后确认：原 Prompt、Response、Attempts、scoreEvalSet、训练条目及 epoch 数均保留。这里只补验证字段，不回填历史测试分数。

原始日志备份：logs/benchmarks_balanced/analysis/std_oracle_backup_20261111T002803771437Z

详细补算记录：[std_oracle_backfill_20261111T002803771437Z.json](std_oracle_backfill_20261111T002803771437Z.json)
