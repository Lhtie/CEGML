# ExtRx: Agentic Clustered vs Non-Clustered

- Model: `gpt-oss`
- Comparison: `agentic_no_clustered` vs `agentic_reflection`
- Evaluation uses `run-0` only.
- Cell format: `success rate (successful run-0s / observed run-0s)`.

| Setting | StarDepth=0 | StarDepth=1 | StarDepth=2 |
|---|---:|---:|---:|
| agentic_no_clustered | 100.0 (21/21) | 76.2 (16/21) | 61.1 (11/18) |
| agentic_reflection | 100.0 (21/21) | 81.0 (17/21) | 77.8 (14/18) |

Coverage notes:
- `agentic_no_clustered`: SD0: 21 observed; SD1: 21 observed; SD2: 18 observed
- `agentic_reflection`: SD0: 21 observed; SD1: 21 observed; SD2: 18 observed
