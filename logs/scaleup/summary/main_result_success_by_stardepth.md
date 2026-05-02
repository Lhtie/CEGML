# Main Result: Run-Level Success Rate by Star Depth

- Each cell is computed directly from the existing log files under `logs/scaleup`.
- Each regex is run up to 3 times.
- `SimpleRx` aggregates the first 3 regexes in each `(#States, StarDepth)` cell from `datasets/scaleup/regex_list.json`.
- `ExtendedRx` aggregates the first 3 regexes in each `(#States, StarDepth)` cell from `datasets/scaleup/regex_list.json`.
- Cell format: `run-level success rate (successful runs / observed runs)`.

| Method | SimpleRx StarDepth=0 | SimpleRx StarDepth=1 | SimpleRx StarDepth=2 | SimpleRx StarDepth=3 | SimpleRx StarDepth=4 | ExtendedRx StarDepth=0 | ExtendedRx StarDepth=1 | ExtendedRx StarDepth=2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 100.0 (63/63) | 50.8 (32/63) | 23.8 (15/63) | 27.0 (17/63) | 3.2 (2/63) | 100.0 (63/63) | 66.7 (42/63) | 38.9 (21/54) |
| Single Inference | 100.0 (63/63) | 92.1 (58/63) | 73.0 (46/63) | 61.9 (39/63) | 23.8 (15/63) | 100.0 (63/63) | 77.8 (49/63) | 68.5 (37/54) |
| Agentic Reflection | 100.0 (63/63) | 100.0 (63/63) | 82.5 (52/63) | 77.8 (49/63) | 38.1 (24/63) | 100.0 (63/63) | 84.1 (53/63) | 74.1 (40/54) |

Notes:
- `SimpleRx / Standard`, `SimpleRx / Single Inference`, and `SimpleRx / Agentic Reflection` each have full coverage: `7 states x 3 regexes x 3 reruns = 63` runs per star depth.
- `ExtendedRx / StarDepth=0,1` has full coverage: `7 states x 3 regexes x 3 reruns = 63` runs.
- `ExtendedRx / StarDepth=2` has full coverage over the defined regexes: `6 states x 3 regexes x 3 reruns = 54` runs, because `regex_list.json` has no regexes for `#States=9, StarDepth=2`.
