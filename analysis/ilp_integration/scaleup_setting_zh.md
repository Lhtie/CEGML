# ILP scaleup 配置

当前 balanced 选择 6 个 ILP 目标任务：简单档 Predecessor、UndirectedEdge；中等档 Grandparent、GraphColoring；较难档 Fizz、Length。std、CE 各跑一次，共 12 个 job。分档覆盖作者 19 个合成配置，按规则结构选择，不代表实测难度。当前本地已实现 12 个不同目标，`--suite all` 可运行全部 12 个。选择依据见 balanced_six_selection_zh.md。

每个任务训练池都有 384 个真/假标注。Unary 任务训练 N=384，验证 N=512 的全部 512 个标注；binary 任务训练 N=20，从 400 个原子中选择 384 个，验证 N=24 的全部 576 个原子。输入、Horn 规则语法和递归归纳目标保持一致。原论文 N=11/15 数据保存在同一 JSON 的 train/eval 字段，运行扩展数据位于 scaleup 字段。这是实验扩展，不是原论文规模。

std 复用原 regex run_episode：默认累计样本数为 3、6、12、24、48、96、192、384，共最多 8 个 epoch。CE 从相同的前三条开始，默认每轮最多 8 个反例、最多 8 个 epoch。两者 equivalent 后早停，因此正确规则可能第一轮就结束，不能保证生成八轮。没有修改 regex 控制器。

ILP 有限域 oracle 改为正 Horn 最小不动点，消除六/八轮截断对不同规则分解的偏差；该 verdict 仍只证明各任务声明的有限验证域上结果一致，不证明无界等价。关系连接使用已绑定变量索引，以支持更大域。CE 可以获得验证域标注，属于原有 pipeline 的主动反馈。

离线检查使用固定假响应：错误规则在两种方法下都运行八轮，正确参考规则在第一轮早停；两种逻辑正确的递归表达均通过 512 原子验证。未调用 LLM，也未重写历史实验日志。检查摘要见 scaleup_validation.json。

运行：

```bash
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --mode both --mkey gpt-oss --outdir logs/benchmarks_ilp_scaleup
```

总共 12 个 job。全任务 balanced 为 53 个 setting、106 个 std/CE job。使用新目录避免覆盖旧规模日志。预算参数可在命令末尾通过 `-- --start_size 3 --tot_train_size 384 --scale_factor 2 --ce_epochs 8` 传给共享 runner。


新增任务使用相同的正 Horn 引擎，支持 unary/binary target 和每个任务独立的不可重定义背景谓词。数值任务保留作者定义；图/家庭关系使用固定种子的稀疏结构，目标标签在 Horn 引擎之外独立计算，不复现作者随机采样分布。背景事实可见，但不会把参考规则或目标含义写入 learner prompt。训练和验证背景不同，输入 num_constants 用于定位对应背景。

10 个任务的离线验证结果见 ten_tasks_validation.json，包含 std/CE 八轮预算、正确规则早停和背景不可重定义检查。未调用实际 LLM、未修改历史日志。
