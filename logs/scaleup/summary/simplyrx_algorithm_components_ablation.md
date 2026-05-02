# Algorithm Components Ablation on SimplyRx

- Model: `gpt-oss`
- Evaluation uses `run-0` only for every regex log.
- Dataset slice: first 3 regexes per `(#States, StarDepth)` cell from `datasets/scaleup/regex_list.json`.
- Columns `StarDepth=1,2,3` report `success rate (successful run-0s / observed run-0s)`.

| Ablation | Regularization | Reflection | Repair Loop | StarDepth=1 | StarDepth=2 | StarDepth=3 |
|---|---|---|---|---:|---:|---:|
| standard | Y | N | N | 52.4 (11/21) | 14.3 (3/21) | 19.0 (4/21) |
| single_inference | Y | N | N | 85.7 (18/21) | 71.4 (15/21) | 61.9 (13/21) |
| noreg/agentic | N | Y | Y | 100.0 (21/21) | 85.7 (18/21) | 71.4 (15/21) |
| agentic_no_reflection | Y | N | Y | 95.2 (20/21) | 76.2 (16/21) | 76.2 (16/21) |
| agentic_no_repair_loop | Y | Y | N | 85.7 (18/21) | 66.7 (14/21) | 66.7 (14/21) |
| agentic_reflection | Y | Y | Y | 100.0 (21/21) | 85.7 (18/21) | 76.2 (16/21) |

Coverage notes:
- `standard`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
- `single_inference`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
- `noreg/agentic`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
- `agentic_no_reflection`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
- `agentic_no_repair_loop`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
- `agentic_reflection`: SD1: 21 observed; SD2: 21 observed; SD3: 21 observed
