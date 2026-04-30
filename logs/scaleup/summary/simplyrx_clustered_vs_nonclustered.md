# Clustered vs Nonclustered on SimplyRx

- Model: `gpt-oss`
- Method: `agentic_reflection`
- Evaluation uses `run-0` only.
- Cell format: `success rate (successful run-0s / observed run-0s)`.

| Setting | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---:|---:|---:|
| clustered | 100.0 (21/21) | 85.7 (18/21) | 76.2 (16/21) |
| nonclustered | NA | NA | NA |

Coverage notes:
- `clustered`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
- `nonclustered`: SD1: 0 observed; SD2: 0 observed; SD3: 0 observed
