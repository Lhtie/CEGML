# gpt-oss extrx: CE vs STD

分析目录：`logs/scaleup/icl_gen_extrx/model=gpt-oss`

说明：
- 这份表按 `datasets/scaleup/regex_list.json` 里的 `extrx` 元信息整理，格式对齐 `simplyrx` report。
- 单元格格式：`solved/total [progress]; s=<med #solved samples>; r=<med #solved rounds>`
- `s`：只在 solved runs 上统计，取首次解出时 `NumTrainingSamples` 的中位数
- `r`：只在 solved runs 上统计，取首次解出轮次的中位数，按 1-based 轮次计数
- `NA` 表示该格子在 `regex_list.json` 中没有定义 regex；`MISSING` 表示定义了 regex，但当前日志里没有对应结果。

## 总体观察

- `CE / agentic_reflection` 整体更强，solve rate = `0.852`，对应 `46/54` runs；`STD` 为 `0.611`，对应 `33/54` runs。
- 在当前已覆盖的 `18` 个格子上，完全解出（`3/3`）的格子数量：`CE` 为 `14/18`，`STD` 为 `11/18`。
- `regex_list.json` 为 `extrx` 定义了 `20` 个有效格子；当前日志覆盖 `18` 个，还有 `2` 个定义过但未跑到的格子，另有 `8` 个模板位置本身未定义 regex。

## STD

| #States | StarDepth=0 | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---|---|---|---|
| 3 | 3/3 [##########]; s=192; r=7 | 3/3 [##########]; s=3; r=1 | MISSING | NA |
| 4 | 3/3 [##########]; s=96; r=6 | 3/3 [##########]; s=24; r=4 | 3/3 [##########]; s=48; r=5 | NA |
| 5 | 3/3 [##########]; s=6; r=2 | MISSING | 0/3 [----------]; s=NA; r=NA | NA |
| 6 | 3/3 [##########]; s=3; r=1 | 3/3 [##########]; s=12; r=3 | 0/3 [----------]; s=NA; r=NA | NA |
| 7 | 3/3 [##########]; s=3; r=1 | 0/3 [----------]; s=NA; r=NA | 0/3 [----------]; s=NA; r=NA | NA |
| 8 | 3/3 [##########]; s=3; r=1 | 0/3 [----------]; s=NA; r=NA | 0/3 [----------]; s=NA; r=NA | NA |
| 9 | 3/3 [##########]; s=6; r=2 | 0/3 [----------]; s=NA; r=NA | NA | NA |

## CE / agentic_reflection

| #States | StarDepth=0 | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---|---|---|---|
| 3 | 3/3 [##########]; s=9; r=2 | 3/3 [##########]; s=8; r=1 | MISSING | NA |
| 4 | 3/3 [##########]; s=10; r=2 | 3/3 [##########]; s=8; r=1 | 3/3 [##########]; s=11; r=2 | NA |
| 5 | 3/3 [##########]; s=8; r=1 | MISSING | 3/3 [##########]; s=167; r=4 | NA |
| 6 | 3/3 [##########]; s=8; r=1 | 3/3 [##########]; s=258; r=2 | 3/3 [##########]; s=1095; r=7 | NA |
| 7 | 3/3 [##########]; s=8; r=1 | 0/3 [----------]; s=NA; r=NA | 0/3 [----------]; s=NA; r=NA | NA |
| 8 | 3/3 [##########]; s=15; r=2 | 3/3 [##########]; s=625; r=9 | 2/3 [#######---]; s=457; r=3 | NA |
| 9 | 3/3 [##########]; s=8; r=1 | 2/3 [#######---]; s=90.5; r=6.5 | NA | NA |
