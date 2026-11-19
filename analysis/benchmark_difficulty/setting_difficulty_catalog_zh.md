# 全量 benchmark setting 难度分档与测试点清单

本次按 setting ID 分档，不修改 setting 内的测试 IO、训练 IO、ground truth 或 oracle。覆盖 run_suite --suite all 当前发现的全部 697 个 ID，没有只覆盖 balanced，也没有复制 IO 充当新任务。SCAN 包括官方的五次随机复制划分，ILP 包括作者语义有重叠的配置；这些原生变体不应解释为互不相同的目标概念。

版本：benchmark_difficulty_v2。分档是任务内、事先定义的难度代理，不是经模型重复实验校准的难度；不同任务的 L1/L2 不可直接横向比较。List Functions 五档，SCAN 四档，其他三档。ILP 保留全部十九个原生配置，三档自然为 4/8/7；不再为了等量而少接任务。balanced 仍保持各档等量选点。

## 数量

| 任务 | 全部 ID 数 | 档数 | all 每档 ID 数 | balanced 每档 ID 数 | balanced 总 ID 数 |
|---|---:|---:|---:|---:|---:|
| Cryptography | 300 | 3 | 100, 100, 100 | 2, 2, 2 | 6 |
| List Functions | 250 | 5 | 50, 50, 50, 50, 50 | 2, 2, 2, 2, 2 | 10 |
| Instruction Induction | 24 | 3 | 8, 8, 8 | 8, 8, 8 | 24 |
| SCAN | 104 | 4 | 26, 26, 26, 26 | 2, 2, 2, 2 | 8 |
| ILP | 19 | 3 | 4, 8, 7 | 2, 2, 2 | 6 |

all 为 697 个 setting、1394 个 std/CE job；balanced 为 54 个 setting、108 个 std/CE job，均指一个 runner。两种方法使用同一份 setting 清单。

## 难度定义

### Cryptography：训练信息覆盖的难度，不是加密强度

保持原有三种逐字符替换，不创造新的加密层数。使用默认 clean paper setting 的十个 train/normal IO，计算缺失的大小写归一字母数量（26 减去覆盖量）；缺失越多，直接观察到的映射越少。次排序指标为缺失的大小写敏感字母数量（52 减去覆盖量），最后用 row ID 确定性打破平局。

每个 cipher family 内按此顺序分成三个 rank band；一百不能被三整除，所以三个 family 分别将多出的一个 ID 放到不同档，合计每档正好一百个。相同主指标可能跨档，这些是覆盖量 rank 分层，不是三个互不重叠的密码学复杂度区间。能识别 cipher family 的模型可从少量字母推断未见字母，因此覆盖少不保证实测更难。

balanced 默认仍每个 family 两个、每个难度档两个，共六个；使用轮换配额同时保持 family 和难度两个边际均衡，不代表 family×tier 的每个交叉单元都等量。--crypto-per-family 仍可配置。非默认 OOD/legacy 配置会改变输入信息量，日志会把 Difficulty.basis_config_matches 记为 false，不能直接套用默认覆盖难度解释。

### List Functions：参考 AST 复杂度的五分位

score = AST 节点数 + 8×for/while + 4×if/条件表达式 + 3×调用 + 2×下标 + 5×comprehension。

按 (score, ID) 排序，L1 到 L5 各五十个。它继承先前的语法复杂度代理，不宣称已得到规范化 DSL 的最小组合深度。等分位边界可能切开同分函数，ID 只用于确定性排序，不能把这些同分 ID 当作有已证实难度差异。

为了同时覆盖全部二百五十个 ID 且各档精确等量，从以前三档改成五档；balanced 默认每档两个，共十个。--list-per-tier 默认由 3 改为 2，显式指定 3 时会选五档共十五个，而不是以前的九个。

### Instruction Induction：任务契约和语义要求的手工代理

- L1：操作明确的抽取、算术、列表化、数词和动物类别筛选。
- L2：词汇关系、例外、局部上下文或判断。
- L3：句子关系/改写、语义评分和跨语言翻译。

每档八个，保留全部二十四个任务。它是覆盖不同推理机制的先验分层；词汇先验、语言熟悉度和标签歧义可能改变实测排序，不能保证某个具体 L1 必然比任意 L3 简单。没有使用本轮 std/CE 结果事后挑任务。

### SCAN：完整官方划分的泛化信息代理

官方固定版本 c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f 的全部 104 对 train/test 已接入：simple/size variations、length、add primitive/with additional examples、template、filler 和 few-shot。源文件路径、SHA-256、样本数记录在 datasets/scan/source_manifest.json。

对每个划分记录：test 与 train 最长输出动作数的正向差、训练中尚未覆盖的 compositional-jump 命令数量、训练未覆盖的完整命令数量；按这个向量和 ID 排序，分四个 rank band，每档二十六个。它仍是泛化条件的代理，档位变化不代表目标 interpreter 本身变了。相同指标的复制划分可跨档，ID 只确定排序。

balanced 从每档固定种子选两个，共八个；all 保留全部 104 个。先前只覆盖八个基础 split 的难度表不再用于当前目录。

### ILP：全部十九个作者合成配置

- L1（4）：HasFather、Predecessor、UndirectedEdge、LessThan。
- L2（8）：Grandparent、Grandparent_NLM、GraphColoring、Connectedness、Cyclic、AdjToRed、TwoChildren、Son。
- L3（7）：EvenSucc、EvenOdd、Buzz、Fizz、Length、Member、Relatedness。

分档依据仍为直接关系/有序链闭包、关系连接/一般图推理、递归数字/列表构造的代理，不是严格最小规则深度。

补齐七个原生配置，新增配置直接运行固定版本作者 generator，保持其背景关系和标签；规模仍采用我们此前的 scaleup 范围。原有背景和已有标注保留；二元训练池补齐到全部四百个原子，原先前三百八十四条逐条不变。作者 Family.py 的 NumPy choice 不能直接处理 [None,(man,woman),...]，适配层按均匀下标抽样修复这一兼容性问题，未改 vendored 源文件。

EvenOdd/EvenSucc 共享奇偶目标；Grandparent_NLM/Grandparent 的目标相同但背景不同。因此十九个配置不是十九个独立语义规律。WN 属于独立知识图谱任务，不混入这十九个合成配置。既有 scaleup 采样和 GraphColoring 的 unary 口径仍保留，不宣称全部等同论文实验。

为使用全部原配置，all 的三个档位自然为 4/8/7，而不是丢掉七个来维持每档四个。balanced 仍每档选两个，原来六个 ID 不变。

## 原始 IO 的使用范围

Instruction Induction 的二十四类共有 67,662 条 induce IO、2,156 条 execute IO，现全部作为可用数据加载；默认不再截断为三十二条测试 IO。显式 --eval_size N 仍可限制加载。直接 IO baseline 会遍历所加载测试集；生成 instruction 的 pipeline 仍用语义 oracle，不把完整 execute IO 偷传给 oracle，也不伪称已经逐条执行这些 IO。

SCAN 每个 split 加载全部原始 train/test 行，保留原生重复样本；oracle 仍使用完整 20,910 条独立命令。不同划分之间样本有大量重叠，104 不是新增了更多语言规则。

训练可用池没有截断，但默认每个 run 的训练预算仍为 384，且保留 equivalent 后早停。新支持 --tot_train_size 0 使用完整可用训练池作为 std 调度上限；这不保证早停前每条 IO 都会进入 prompt。日志记录 TrainingPoolSize、ScheduledTrainingBudget、EvaluationPoolSize，避免把预算和数据总量混淆。全部原始训练池直接塞进单个 prompt 并不是扩大测试点数量的方法。

## 完整 ID 清单

以下每个 all ID 恰好出现一次；每档同时列出默认 balanced 选中的 ID。完整逐点指标见 all_setting_difficulty.csv，程序可读的权威分类见 datasets/benchmarks/difficulty_catalog.json。

### Cryptography

**L1：all 100 个；balanced 2 个。**

all：`atbash:12`, `atbash:19`, `atbash:21`, `atbash:22`, `atbash:25`, `atbash:28`, `atbash:31`, `atbash:32`, `atbash:34`, `atbash:35`, `atbash:37`, `atbash:4`, `atbash:46`, `atbash:49`, `atbash:5`, `atbash:54`, `atbash:57`, `atbash:60`, `atbash:64`, `atbash:68`, `atbash:73`, `atbash:77`, `atbash:78`, `atbash:8`, `atbash:80`, `atbash:83`, `atbash:85`, `atbash:9`, `atbash:90`, `atbash:92`, `atbash:93`, `atbash:94`, `atbash:97`, `caesar:0`, `caesar:10`, `caesar:13`, `caesar:15`, `caesar:16`, `caesar:17`, `caesar:18`, `caesar:2`, `caesar:20`, `caesar:26`, `caesar:28`, `caesar:35`, `caesar:39`, `caesar:42`, `caesar:43`, `caesar:44`, `caesar:48`, `caesar:49`, `caesar:55`, `caesar:57`, `caesar:61`, `caesar:63`, `caesar:69`, `caesar:7`, `caesar:70`, `caesar:72`, `caesar:76`, `caesar:81`, `caesar:82`, `caesar:83`, `caesar:87`, `caesar:88`, `caesar:92`, `caesar:95`, `keyboard:1`, `keyboard:12`, `keyboard:14`, `keyboard:19`, `keyboard:2`, `keyboard:22`, `keyboard:23`, `keyboard:25`, `keyboard:28`, `keyboard:31`, `keyboard:32`, `keyboard:35`, `keyboard:36`, `keyboard:42`, `keyboard:45`, `keyboard:47`, `keyboard:51`, `keyboard:7`, `keyboard:72`, `keyboard:73`, `keyboard:75`, `keyboard:76`, `keyboard:78`, `keyboard:79`, `keyboard:83`, `keyboard:84`, `keyboard:85`, `keyboard:86`, `keyboard:87`, `keyboard:95`, `keyboard:96`, `keyboard:97`, `keyboard:98`。

balanced：`caesar:87`, `keyboard:73`。

**L2：all 100 个；balanced 2 个。**

all：`atbash:0`, `atbash:1`, `atbash:11`, `atbash:14`, `atbash:15`, `atbash:2`, `atbash:20`, `atbash:23`, `atbash:24`, `atbash:27`, `atbash:29`, `atbash:3`, `atbash:30`, `atbash:36`, `atbash:40`, `atbash:41`, `atbash:42`, `atbash:43`, `atbash:50`, `atbash:51`, `atbash:53`, `atbash:58`, `atbash:6`, `atbash:61`, `atbash:63`, `atbash:69`, `atbash:72`, `atbash:75`, `atbash:76`, `atbash:87`, `atbash:88`, `atbash:89`, `atbash:96`, `atbash:99`, `caesar:11`, `caesar:14`, `caesar:21`, `caesar:23`, `caesar:24`, `caesar:25`, `caesar:27`, `caesar:3`, `caesar:31`, `caesar:32`, `caesar:33`, `caesar:34`, `caesar:37`, `caesar:38`, `caesar:47`, `caesar:5`, `caesar:52`, `caesar:59`, `caesar:60`, `caesar:62`, `caesar:66`, `caesar:67`, `caesar:71`, `caesar:73`, `caesar:74`, `caesar:75`, `caesar:79`, `caesar:80`, `caesar:84`, `caesar:86`, `caesar:90`, `caesar:94`, `caesar:98`, `keyboard:10`, `keyboard:17`, `keyboard:18`, `keyboard:24`, `keyboard:26`, `keyboard:27`, `keyboard:3`, `keyboard:38`, `keyboard:43`, `keyboard:44`, `keyboard:46`, `keyboard:48`, `keyboard:5`, `keyboard:52`, `keyboard:53`, `keyboard:54`, `keyboard:56`, `keyboard:57`, `keyboard:59`, `keyboard:6`, `keyboard:60`, `keyboard:62`, `keyboard:65`, `keyboard:66`, `keyboard:67`, `keyboard:68`, `keyboard:70`, `keyboard:77`, `keyboard:88`, `keyboard:9`, `keyboard:90`, `keyboard:94`, `keyboard:99`。

balanced：`atbash:69`, `caesar:47`。

**L3：all 100 个；balanced 2 个。**

all：`atbash:10`, `atbash:13`, `atbash:16`, `atbash:17`, `atbash:18`, `atbash:26`, `atbash:33`, `atbash:38`, `atbash:39`, `atbash:44`, `atbash:45`, `atbash:47`, `atbash:48`, `atbash:52`, `atbash:55`, `atbash:56`, `atbash:59`, `atbash:62`, `atbash:65`, `atbash:66`, `atbash:67`, `atbash:7`, `atbash:70`, `atbash:71`, `atbash:74`, `atbash:79`, `atbash:81`, `atbash:82`, `atbash:84`, `atbash:86`, `atbash:91`, `atbash:95`, `atbash:98`, `caesar:1`, `caesar:12`, `caesar:19`, `caesar:22`, `caesar:29`, `caesar:30`, `caesar:36`, `caesar:4`, `caesar:40`, `caesar:41`, `caesar:45`, `caesar:46`, `caesar:50`, `caesar:51`, `caesar:53`, `caesar:54`, `caesar:56`, `caesar:58`, `caesar:6`, `caesar:64`, `caesar:65`, `caesar:68`, `caesar:77`, `caesar:78`, `caesar:8`, `caesar:85`, `caesar:89`, `caesar:9`, `caesar:91`, `caesar:93`, `caesar:96`, `caesar:97`, `caesar:99`, `keyboard:0`, `keyboard:11`, `keyboard:13`, `keyboard:15`, `keyboard:16`, `keyboard:20`, `keyboard:21`, `keyboard:29`, `keyboard:30`, `keyboard:33`, `keyboard:34`, `keyboard:37`, `keyboard:39`, `keyboard:4`, `keyboard:40`, `keyboard:41`, `keyboard:49`, `keyboard:50`, `keyboard:55`, `keyboard:58`, `keyboard:61`, `keyboard:63`, `keyboard:64`, `keyboard:69`, `keyboard:71`, `keyboard:74`, `keyboard:8`, `keyboard:80`, `keyboard:81`, `keyboard:82`, `keyboard:89`, `keyboard:91`, `keyboard:92`, `keyboard:93`。

balanced：`atbash:39`, `keyboard:8`。

### List Functions

**L1：all 50 个；balanced 2 个。**

参考 AST score 范围：7–16。

all：`c001`, `c002`, `c003`, `c004`, `c006`, `c007`, `c008`, `c009`, `c011`, `c012`, `c013`, `c014`, `c017`, `c018`, `c029`, `c030`, `c037`, `c038`, `c041`, `c042`, `c043`, `c045`, `c046`, `c048`, `c049`, `c058`, `c062`, `c070`, `c075`, `c077`, `c078`, `c079`, `c080`, `c081`, `c082`, `c083`, `c084`, `c090`, `c091`, `c100`, `c102`, `c103`, `c104`, `c106`, `c108`, `c120`, `c126`, `c127`, `c182`, `c219`。

balanced：`c002`, `c014`。

**L2：all 50 个；balanced 2 个。**

参考 AST score 范围：16–23。

all：`c005`, `c010`, `c016`, `c025`, `c026`, `c044`, `c047`, `c050`, `c051`, `c052`, `c053`, `c056`, `c061`, `c063`, `c064`, `c065`, `c066`, `c068`, `c069`, `c071`, `c074`, `c085`, `c092`, `c093`, `c095`, `c096`, `c097`, `c101`, `c110`, `c114`, `c117`, `c121`, `c122`, `c123`, `c124`, `c130`, `c132`, `c133`, `c135`, `c160`, `c169`, `c187`, `c193`, `c196`, `c213`, `c217`, `c225`, `c244`, `c246`, `c247`。

balanced：`c123`, `c124`。

**L3：all 50 个；balanced 2 个。**

参考 AST score 范围：23–36。

all：`c015`, `c019`, `c020`, `c021`, `c022`, `c057`, `c072`, `c073`, `c098`, `c105`, `c109`, `c111`, `c112`, `c116`, `c128`, `c129`, `c136`, `c137`, `c138`, `c140`, `c145`, `c149`, `c151`, `c154`, `c155`, `c162`, `c163`, `c164`, `c165`, `c167`, `c168`, `c170`, `c174`, `c186`, `c188`, `c195`, `c198`, `c208`, `c212`, `c221`, `c222`, `c229`, `c232`, `c236`, `c238`, `c240`, `c241`, `c243`, `c245`, `c248`。

balanced：`c021`, `c208`。

**L4：all 50 个；balanced 2 个。**

参考 AST score 范围：36–59。

all：`c032`, `c033`, `c034`, `c039`, `c040`, `c054`, `c059`, `c060`, `c067`, `c076`, `c086`, `c087`, `c099`, `c107`, `c113`, `c115`, `c118`, `c119`, `c134`, `c139`, `c141`, `c143`, `c146`, `c147`, `c148`, `c156`, `c157`, `c161`, `c171`, `c172`, `c176`, `c177`, `c184`, `c185`, `c189`, `c194`, `c201`, `c202`, `c203`, `c205`, `c206`, `c207`, `c209`, `c211`, `c214`, `c218`, `c220`, `c223`, `c224`, `c230`。

balanced：`c054`, `c146`。

**L5：all 50 个；balanced 2 个。**

参考 AST score 范围：59–173。

all：`c023`, `c024`, `c027`, `c028`, `c031`, `c035`, `c036`, `c055`, `c088`, `c089`, `c094`, `c125`, `c131`, `c142`, `c144`, `c150`, `c152`, `c153`, `c158`, `c159`, `c166`, `c173`, `c175`, `c178`, `c179`, `c180`, `c181`, `c183`, `c190`, `c191`, `c192`, `c197`, `c199`, `c200`, `c204`, `c210`, `c215`, `c216`, `c226`, `c227`, `c228`, `c231`, `c233`, `c234`, `c235`, `c237`, `c239`, `c242`, `c249`, `c250`。

balanced：`c024`, `c215`。

### Instruction Induction

**L1：all 8 个；balanced 8 个。**

all：`diff`, `first_word_letter`, `letters_list`, `num_to_verbal`, `orthography_starts_with`, `second_word_letter`, `sum`, `taxonomy_animal`。

balanced：`diff`, `first_word_letter`, `letters_list`, `num_to_verbal`, `orthography_starts_with`, `second_word_letter`, `sum`, `taxonomy_animal`。

**L2：all 8 个；balanced 8 个。**

all：`antonyms`, `larger_animal`, `negation`, `rhymes`, `sentiment`, `singular_to_plural`, `synonyms`, `word_in_context`。

balanced：`antonyms`, `larger_animal`, `negation`, `rhymes`, `sentiment`, `singular_to_plural`, `synonyms`, `word_in_context`。

**L3：all 8 个；balanced 8 个。**

all：`active_to_passive`, `cause_and_effect`, `common_concept`, `informal_to_formal`, `sentence_similarity`, `translation_en-de`, `translation_en-es`, `translation_en-fr`。

balanced：`active_to_passive`, `cause_and_effect`, `common_concept`, `informal_to_formal`, `sentence_similarity`, `translation_en-de`, `translation_en-es`, `translation_en-fr`。

### SCAN

**L1：all 26 个；balanced 2 个。**

all：`addprim_turn_left`, `fewshot_num1024_rep1`, `fewshot_num1024_rep2`, `fewshot_num1024_rep3`, `fewshot_num1024_rep4`, `fewshot_num1024_rep5`, `fewshot_num128_rep1`, `fewshot_num128_rep2`, `fewshot_num128_rep3`, `fewshot_num128_rep4`, `fewshot_num128_rep5`, `fewshot_num256_rep1`, `fewshot_num256_rep2`, `fewshot_num256_rep3`, `fewshot_num256_rep4`, `fewshot_num256_rep5`, `fewshot_num512_rep1`, `fewshot_num512_rep2`, `fewshot_num512_rep3`, `fewshot_num512_rep4`, `fewshot_num512_rep5`, `filler_num1`, `filler_num2`, `filler_num3`, `simple`, `template_jump_around_right`。

balanced：`fewshot_num1024_rep2`, `filler_num1`。

**L2：all 26 个；balanced 2 个。**

all：`fewshot_num16_rep1`, `fewshot_num16_rep2`, `fewshot_num16_rep3`, `fewshot_num16_rep4`, `fewshot_num16_rep5`, `fewshot_num2_rep2`, `fewshot_num2_rep3`, `fewshot_num2_rep4`, `fewshot_num32_rep1`, `fewshot_num32_rep2`, `fewshot_num32_rep3`, `fewshot_num32_rep4`, `fewshot_num32_rep5`, `fewshot_num4_rep1`, `fewshot_num4_rep2`, `fewshot_num4_rep5`, `fewshot_num64_rep1`, `fewshot_num64_rep2`, `fewshot_num64_rep3`, `fewshot_num64_rep4`, `fewshot_num64_rep5`, `fewshot_num8_rep1`, `fewshot_num8_rep2`, `fewshot_num8_rep3`, `fewshot_num8_rep4`, `fewshot_num8_rep5`。

balanced：`fewshot_num16_rep5`, `fewshot_num32_rep2`。

**L3：all 26 个；balanced 2 个。**

all：`addprim_complex_jump_num16_rep1`, `addprim_complex_jump_num32_rep1`, `addprim_complex_jump_num32_rep2`, `addprim_complex_jump_num32_rep3`, `addprim_complex_jump_num32_rep4`, `addprim_complex_jump_num32_rep5`, `fewshot_num1_rep1`, `fewshot_num1_rep2`, `fewshot_num1_rep3`, `fewshot_num1_rep4`, `fewshot_num1_rep5`, `fewshot_num2_rep1`, `fewshot_num2_rep5`, `fewshot_num4_rep3`, `fewshot_num4_rep4`, `filler_num0`, `simple_p1`, `simple_p16`, `simple_p2`, `simple_p32`, `simple_p4`, `simple_p64`, `simple_p8`, `template_around_right`, `template_opposite_right`, `template_right`。

balanced：`fewshot_num2_rep1`, `template_right`。

**L4：all 26 个；balanced 2 个。**

all：`addprim_complex_jump_num16_rep2`, `addprim_complex_jump_num16_rep3`, `addprim_complex_jump_num16_rep4`, `addprim_complex_jump_num16_rep5`, `addprim_complex_jump_num1_rep1`, `addprim_complex_jump_num1_rep2`, `addprim_complex_jump_num1_rep3`, `addprim_complex_jump_num1_rep4`, `addprim_complex_jump_num1_rep5`, `addprim_complex_jump_num2_rep1`, `addprim_complex_jump_num2_rep2`, `addprim_complex_jump_num2_rep3`, `addprim_complex_jump_num2_rep4`, `addprim_complex_jump_num2_rep5`, `addprim_complex_jump_num4_rep1`, `addprim_complex_jump_num4_rep2`, `addprim_complex_jump_num4_rep3`, `addprim_complex_jump_num4_rep4`, `addprim_complex_jump_num4_rep5`, `addprim_complex_jump_num8_rep1`, `addprim_complex_jump_num8_rep2`, `addprim_complex_jump_num8_rep3`, `addprim_complex_jump_num8_rep4`, `addprim_complex_jump_num8_rep5`, `addprim_jump`, `length`。

balanced：`addprim_complex_jump_num8_rep5`, `length`。

### ILP

**L1：all 4 个；balanced 2 个。**

all：`has_father`, `less_than`, `predecessor`, `undirected_edge`。

balanced：`predecessor`, `undirected_edge`。

**L2：all 8 个；balanced 2 个。**

all：`adj_to_red`, `connectedness`, `cyclic`, `grandparent`, `grandparent_nlm`, `graph_coloring`, `son`, `two_children`。

balanced：`grandparent`, `graph_coloring`。

**L3：all 7 个；balanced 2 个。**

all：`buzz`, `even_odd`, `even_succ`, `fizz`, `length`, `member`, `relatedness`。

balanced：`fizz`, `length`。

## 文件、日志和复现

- datasets/benchmarks/difficulty_catalog.json：全部 697 个 setting 的分类、指标、限制和源文件 SHA-256。
- analysis/benchmark_difficulty/all_setting_difficulty.csv：每个 ID 一行，含默认 balanced 是否选中。
- analysis/benchmark_difficulty/all_settings.json：完整 all 运行计划和难度。
- analysis/benchmark_difficulty/balanced_settings.json：默认 seed=43 的 balanced 运行计划和难度。
- 新实验日志 TaskSettings.Difficulty：该 setting 的档位、轴、指标和目录版本；不改变传给 learner 的 prompt。
- 正常 run_suite 运行会在 outdir 自动保存选点 manifest；dry-run 默认不落盘，显式 --manifest 时才写清单。历史实验日志不修改。

更新数据或规则后重新生成目录；手工分组遇到新 ID 会报错，避免漏标。

```bash
python -m benchmarks.difficulty
python -m benchmarks.run_suite --runner icl_gen --suite all --dry-run --manifest analysis/benchmark_difficulty/all_settings.json
python -m benchmarks.run_suite --runner icl_gen --suite balanced --dry-run --manifest analysis/benchmark_difficulty/balanced_settings.json
```

查看验证结果：validation.json。此次没有调用 LLM、执行正式 benchmark 或改变 oracle 的检查域。


完整原数据覆盖说明与逐任务样本数：native_dataset_coverage_zh.md；验证：native_coverage_validation.json。


默认预算更新：benchmark 样本预算现等于完整训练池，std 通过十轮几何调度到达全池；CE 仍按反例最多十轮，equivalent 后早停。此前文档中的默认 384 限制或“到 384 后重复”方案已被此默认替代。原始数据清单和难度标签不变。
