# SimplyRx: Success Rate by Model and Star Depth

- Dataset: `SimplyRx`
- Columns: `StarDepth=1,2,3`
- Rows: each model with `Standard` and `Agentic Reflection`
- Cell format: `success rate (successful runs / observed runs)`.
- For `gpt-oss`, only the first rerun (`run-0`) is counted for each regex.
- `qw3` is marked `NA` because no matching `model=qw3` logs were found under `logs/scaleup/icl_gen_simplyrx`.

| Model | Method | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---|---:|---:|---:|
| gpt-oss | Standard | 52.4 (11/21) | 14.3 (3/21) | 19.0 (4/21) |
| gpt-oss | Agentic Reflection | 100.0 (21/21) | 85.7 (18/21) | 76.2 (16/21) |
| gpt5 | Standard | 38.1 (8/21) | 14.3 (3/21) | 14.3 (3/21) |
| gpt5 | Agentic Reflection | 90.5 (19/21) | 47.6 (10/21) | 57.1 (12/21) |
| qw3 | Standard | NA | NA | NA |
| qw3 | Agentic Reflection | NA | NA | NA |

Coverage notes:
- `gpt-oss / Standard`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `gpt-oss / Agentic Reflection`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `gpt5 / Standard`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `gpt5 / Agentic Reflection`: SD1: 21/21 regex logs, 21 counted runs; SD2: 21/21 regex logs, 21 counted runs; SD3: 21/21 regex logs, 21 counted runs
- `qw3 / Standard`: SD1: model dir missing; SD2: model dir missing; SD3: model dir missing
- `qw3 / Agentic Reflection`: SD1: model dir missing; SD2: model dir missing; SD3: model dir missing
