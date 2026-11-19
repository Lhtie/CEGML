# ILP setting 与 oracle 审计

本轮只做审计和离线复核，没有修改 pipeline、任务数据、prompt 或历史实验日志。结论：当前实现是 HRI 任务启发的 LLM+CE scaleup 适配，不是原论文实验复现。递归、辅助谓词和关系推理目标没有整体被换成简单分类；但图/家庭背景结构确实被缩窄，GraphColoring 还有论文与作者代码的目标口径差异。oracle 对声明的单个有限验证背景可精确判定，但不能充当所有背景上的程序等价证明，也没有资源耗尽返回 unknown 的完整保障。

## 依据

- [原论文](https://proceedings.mlr.press/v162/glanois22a/glanois22a.pdf)：§6.1、Appendix B.1、Table 14。
- 本地固定版本作者代码：datasets/ilp/source/HRI/utils/Dataset.py、Task.py、UniversalParam.py，revision 1a3a0465a5a5305ad8fec958901710dc6ead76e4。
- 当前实现：benchmarks/ilp.py、benchmarks/ilp_settings.py、datasets/ilp/prepare.py、tasks/benchmarks.py。
- 离线数据：同目录 audit.json。

## 与论文/作者代码一致的部分

输入是背景关系和正负 ground target atoms；输出是无函数、正 Horn 规则，保留 unary/binary 谓词、递归、相互递归和辅助谓词构造。两个 body atoms、最多三个变量与该类规则归纳的语言范围相容，并非把输出换成任意 Python 分类器。禁止规则里直接写数字常量可防止枚举标注，当前六个目标都有无需数值常量的正确规则。

Predecessor 的逆 successor、UndirectedEdge 的对称化、Grandparent 的两代关系、Fizz 的三的倍数、Length 的列表长度，都保留了目标概念。直接抽取并执行作者 Dataset.py 中的六个数值/列表 generator，在 N=11、15 的十二个背景上，当前参考 Horn 规则均与作者标签一致。这是目标/规则语义核对，不是对完整 HRI 训练过程的复现。

论文 Table 14 中 Predecessor 常量规模为 10/14、UndirectedEdge 为 4/6、Grandparent 为 9/11、GraphColoring 为 8/10；当前对应任务分别采用 20/24 或 384/512，不能称原始规模。原论文采用模型层级和 proto-rules、有限推理预算及重复运行；我们采用 LLM 生成、自由 Horn 辅助谓词和最小不动点。后者有利于公平比较不同规则分解的逻辑输出，却改变了原实验协议。论文通常以多次运行的成功比例汇报；当前六个任务各一次，4/6 只是本轮任务通过率。

## 发现的问题

### 1. 图和家庭数据的结构多样性被简化

作者随机图/家庭任务会生成随机背景；我们每个任务只固定一个训练背景和一个验证背景。图又被拆成八节点小块，强制三角环并留下孤立点；家庭只在固定八人、三代模板上随机删除少量边。即使 unary 输入域扩大到 512，也只是拼接更多小块，没有增加图的长程连接深度。

因此，样本数量更大不代表结构泛化更强；某些短程或模板特定规则可能通过当前验证。这属于数据分布上的实际简化，建议优先恢复作者随机背景生成方式，并使用多个独立背景、多个随机种子和节点重编号。对于 binary 任务，384 个训练条目来自同一个二十常量背景的四百个可能原子，并不是 384 个独立结构。

### 2. GraphColoring：论文文字与作者代码不一致

论文 Appendix B.1 文字定义二元的同色邻接关系 target(X,Y)，正例也写作节点对。作者 Dataset.py 的 GraphColoringDataset 实际产生 unary target：节点 X 是否存在同色的出邻居；我们跟的是作者代码。二者不是相同输出，不能笼统说完全与原 paper 一致。

此外，论文列出的示例规则和递归标记也与其文字/代码有不完全一致之处，不能自动视为一套无歧义规范。当前无需递归的 unary 同色邻居规则确实能完整表达作者代码目标；它不是图着色求解任务。若严格以论文文字为准，应单独引入明确命名的 binary 版本，再调整标签、arity 和 oracle；不能悄悄把历史 unary 日志解释成 binary 结果。

### 3. oracle 只查 eval 背景，存在早停一致性缺口

当前 counterexamples 只遍历 self.eval，没有检查训练背景或其他随机背景。程序在这个完整有限域上的 true 是准确的，但它可能同时违反训练标签，pipeline 仍会依据 true 早停。

已构造一个完全合法、变量安全且不含数字常量的 Predecessor 程序：先组合 successor 定义二十步关系；若存在二十步路径，通过 gate 向整个 successor 链传播，再受 gate 控制输出逆 successor。在 N=24 中 gate 可激活，逆 successor 全部正确；在 N=20 中没有二十步路径，输出为空。当前 oracle 返回 true，但它在现有训练池中错了十九条正例。

这不证明历史日志发生了这种错误，也不意味着 oracle 对自己声明的 N=24 域判断错误；它证明了“单一验证域通过”不足以作为更广任务正确性的验收标准。完整 witness 及训练反例在 audit.json 的 train_eval_gap_witness 中。

建议最低限度同时检查训练背景和验证背景，进一步增加固定种子的多个背景。每次 verdict 明确记录验证了哪些背景；true 仅对应全部检查完成且无差异。CE 必须带背景标识，必要时扩展现有 num_constants 标识，因为不同图可能具有同一个 N。

### 4. 没有完整的资源预算/unknown 机制

ILP 不走 SMT，而是自写有限域 forward chaining。对于当前函数自由、正 Horn、有限常量和有限谓词集合，最小不动点可以精确计算，不需要借助 SMT 的 unknown 来近似正常结果。

但当前推理循环没有时间或中间关系/连接规模上限。合法规则可能产生 N² 个关系事实，两个 body atoms 的连接还可能产生 N³ 量级的临时 binding，512 常量下可能非常昂贵。当前“遇到异常转 None”不能代替资源预算，正常推理若一直消耗资源，就不会及时返回 unknown。

建议加入真实的时间、事实量和连接预算。已验证的不同输出应返回 false 和可复现 CE；完整遍历无差异才返回 true；超时或预算不足且没有已确认 CE 才返回 unknown，并标记原因、已检查背景和完整性。语法非法应保留 invalid_candidate/error，不能说它是已证明不等价的合法程序。

### 5. 空背景关系有实际实现缺陷

infer 使用 len(rows[0]) 推断背景谓词 arity。合法的空 edge/father 等关系会触发 IndexError，而不是计算出正常的空关系结果。当前已生成的数据没有触发，所以不能据此否定现有日志；但一旦恢复更广随机背景，这个缺陷就可能出现。应该独立声明背景 schema，不从第一条事实推断 arity。

### 6. 格式约束与学习难度混在一起

现有五十六个 epoch 中，三十二个最后候选无效。数字常量和 ground facts 违反当前明确规定，应保留限制以避免记忆标签；但是模型频繁犯此类错误，意味着结果同时测了格式遵从和规则归纳。匿名变量 `_` 是正常 Prolog 习惯，但当前语法只允许大写变量，不能直接拿来推断任务本身太难。

建议提供不泄漏当前目标解法的合法语法 ICL、明确的背景谓词签名、具体的解析错误反馈；若将 `_` 作为语法糖支持，必须按每次出现创建独立新变量，不能误当同一个共享变量。这里属于表达兼容性，不应放开 ground facts 来“提高成功率”。

### 7. 正负样本不平衡需要与 equivalence 分开汇报

Grandparent 当前验证集只有八个正例、五百六十八个负例；恒 false 就能达到 98.61% accuracy，却完全没有学到祖父母关系。当前要求 Equivalent=true 才成功是正确的，应保持。增加报告中的正例召回、合法候选率或分类混淆统计会有帮助，但不应用高 accuracy 替代等价验收。

## oracle 的实际复核结果

历史日志中 24 个获得布尔判定的 epoch 全部重算一致；回放 128 条 OracleCounterexamples，没有发现 GT 标签错误或候选其实已答对的伪反例。另有 32 个 epoch 无有效布尔判定，不能计为 oracle 已精确判断。

另用独立的“枚举变量赋值、同步加入事实”的朴素解释器，与当前带索引的 join 引擎对照六个选定任务、三十六个小随机背景，结果一致。作者 generator 的十二个数值/列表背景也通过参考规则核对。这些测试支持当前有限域实现的基本正确性，不构成对所有合法候选或未知背景的完整证明。

## 是否应该换成 SMT

当前有限域问题使用精确最小不动点加全量原子比较是合理的，通常比额外编码成 SMT 更直接。不是所有 ILP 都不可判定，更不能因为用了 SMT 就认为一定要有 noise。若将来扩大成跨任意背景的程序语义检查，那是更强的新 oracle 契约，不能把当前一个有限域的 true 沿用为全局证明。

若选择 SMT 实现，也必须编码完整的最小模型/推导语义。只将 Horn 规则写成蕴含约束可能允许任意额外 target 事实，不能自动代表程序的 least-fixed-point 输出。任何 SAT 候选 CE 都应通过独立的程序执行和 GT 重新检查；unknown 必须如实保留。

## 修改优先级

1. 首先补齐 oracle 的训练/验证背景一致性检查、资源预算以及空关系 schema 支持。
2. 恢复作者式随机结构生成，添加多背景验证和节点重编号；保持可复现，并避免八节点模板限制。
3. 明确 GraphColoring 采用 author-code-unary 还是 paper-text-binary；新增版本必须单独记录，不重解释历史结果。
4. 提高合法 Horn 输出率，再重复多种子实验。保留原 regex 共享主流程，仅修改 ILP 的 oracle、数据和领域 prompt。

本轮没有执行这些修改，以免在审计过程中静默改变正在比较的 setting；证据和复现结果已写入本目录。若下一步落实，上述变化需要新 setting/oracle 版本及新日志目录，历史结果保持原义。


后续实现更新：已按用户请求接入 cvc5 SMT oracle，同时修正训练/验证检查缺口、空关系 schema 和资源预算；详见 smt_oracle_implementation_zh.md。本文保留的是修改前审计证据，数据分布与 GraphColoring 口径问题仍然存在，尚未在本次 oracle 改动中调整。
