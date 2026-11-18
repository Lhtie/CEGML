# ILP balanced：六个任务的选择依据

作者 Task.py 注册 19 个合成任务配置和 WN。当前本地实现 12 个不同目标任务；balanced 仅运行六个，共 12 个 std/CE job；all 保留本地全部十二个，共 24 个 job，不宣称已实现全部 19 个。

这是事先按参考规则结构进行的三档均衡选择，每档两个。没有用模型表现事后挑选，也没有声称这就是实测难度或从 19 个任务中等概率抽样。作者部分配置目标重叠，如 EvenOdd/EvenSucc，未重复计入六个代表任务。

- easy：HasFather, Predecessor, UndirectedEdge。
- medium：AdjToRed, Grandparent, Grandparent_NLM, Son, TwoChildren, GraphColoring。
- hard：LessThan, Connectedness, Member, EvenOdd, EvenSucc, Buzz, Cyclic, Relatedness, Fizz, Length。

简单档是直接关系映射、存在投影或正反向并集。中等档需要组合背景谓词、连接条件或辅助谓词，但不需要递归。较难档包含递归闭包、递归计数或递归加辅助谓词构造。档内并无严格难度次序；背景信息量、可识别性和 LLM 的先验也会影响表现。

选择：

- Predecessor：简单，反转 successor 的参数。
- UndirectedEdge：简单，合并边的两个方向。
- Grandparent：中等，合并父母关系，再组合两代关系。
- GraphColoring：中等，连接颜色关系判断同色，再与邻接关系组合；目标是存在同色邻居，不是求合法图着色方案。
- Fizz：较难，从 successor 构造三步关系，再递归识别三的倍数。
- Length：较难，结合列表后继和数值后继，通过 target/aux 递归累计长度。

每个任务保留 384 条训练标注，std 为 3→6→12→24→48→96→192→384；CE 初始 3 条，最多八轮，每轮最多八个反例。两者 equivalent 后早停。规模、oracle 和原 regex 主流程没有因为 balanced 子集选择而改变。

为补齐代表任务，新增了 GraphColoring 和 Length 的本地适配数据；其余此前已实现任务仍然可用。全 benchmark balanced 为 53 个 setting、106 个 std/CE job。

离线检查：全 19 个配置在分档中恰好出现一次，六个任务每档两个；参考规则对全部训练和验证标注正确，错误规则返回的 CE 与独立标签一致。详见 balanced_six_validation.json。未调用实际 LLM。

```bash
python -m benchmarks.run_suite --runner icl_gen --suite balanced --task ilp --mode both --mkey gpt-oss --outdir logs/benchmarks_ilp_balanced6
```
