| Method | Prompt | Input | Output | CoT | Strategy | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | Naive | × | × | × | × | 9.5 (2/21) | 0.0 (0/21) | 4.8 (1/21) |
| Standard | Input+Output | ✓ | ✓ | × | × | 42.9 (9/21) | 14.3 (3/21) | 14.3 (3/21) |
| Standard | Zero | ✓ | ✓ | ✓ | × | 47.6 (10/21) | 23.8 (5/21) | 23.8 (5/21) |
| Standard | Full | ✓ | ✓ | ✓ | ✓ | 47.6 (10/21) | 19.0 (4/21) | 23.8 (5/21) |
| Single Inference | Naive | × | × | × | × | 14.3 (3/21) | 0.0 (0/21) | 4.8 (1/21) |
| Single Inference | Input+Output | ✓ | ✓ | × | × | 71.4 (15/21) | 38.1 (8/21) | 33.3 (7/21) |
| Single Inference | Zero | ✓ | ✓ | ✓ | × | 71.4 (15/21) | 33.3 (7/21) | 33.3 (7/21) |
| Single Inference | Full | ✓ | ✓ | ✓ | ✓ | 81.0 (17/21) | 33.3 (7/21) | 33.3 (7/21) |
