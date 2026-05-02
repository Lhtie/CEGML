# SimplyRx: Success Rate by Model and Star Depth

- Dataset: `SimplyRx`
- Columns: `StarDepth=1,2,3`
- Rows: each model with `Standard` and `Agentic Reflection`
- Cell format: `success rate (successful runs / observed runs)`.
- Only the first rerun (`run-0`) is counted for each regex.
- `gpt5` is marked `NA` for now because `model=gpt5` has not finished running.

| Model | Method | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---|---:|---:|---:|
| gpt-oss | Standard | 52.4 (11/21) | 14.3 (3/21) | 19.0 (4/21) |
| gpt-oss | Agentic Reflection | 100.0 (21/21) | 85.7 (18/21) | 76.2 (16/21) |
| gpt5 | Standard | NA | NA | NA |
| gpt5 | Agentic Reflection | NA | NA | NA |
| qw3-235b | Standard | 28.6 (6/21) | 14.3 (3/21) | 14.3 (3/21) |
| qw3-235b | Agentic Reflection | 61.9 (13/21) | 38.1 (8/21) | 38.1 (8/21) |

Coverage notes:
- `gpt-oss / Standard`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `gpt-oss / Agentic Reflection`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `gpt5 / Standard`: SD1: forced NA; SD2: forced NA; SD3: forced NA
- `gpt5 / Agentic Reflection`: SD1: forced NA; SD2: forced NA; SD3: forced NA
- `qw3-235b / Standard`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `qw3-235b / Agentic Reflection`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
