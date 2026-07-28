# SimplyRx Ablation Success Rate by Star Depth

Only regexes from `datasets/scaleup/run_scripts/simplyrx_ablation.sh` are
included. A success for Standard/Single/Agentic means
`summary.run-0.final_accuracy == 1`; a success for the classical baselines
means that the learned DFA is strictly equivalent to the target DFA.

| Method | Protocol | Coverage | SD=1 | SD=2 | SD=3 |
|---|---|---:|---:|---:|---:|
| Standard | LLM, run-0 | 63/63 | 52.4% (11/21) | 14.3% (3/21) | 19.0% (4/21) |
| Single | LLM + CE, run-0 | 63/63 | 85.7% (18/21) | 71.4% (15/21) | 61.9% (13/21) |
| Agentic | LLM + iterative CE, run-0 | 63/63 | 100.0% (21/21) | 85.7% (18/21) | 76.2% (16/21) |
| RPNI | Passive, 1500 examples | Running | — | — | — |
| Blue-Fringe | Passive, 1500 examples | Running | — | — | — |
| L* | Active, exact EQ | 63/63 | 100.0% (21/21) | 100.0% (21/21) | 100.0% (21/21) |
| CVC5-CEGIS | SMT + exact CE, max 30 rounds | 23/63 completed | 38.1% (8/21) | 33.3% (7/21) | 23.8% (5/21) |

For CVC5-CEGIS, every unfinished regex is counted as a failure, so all three
star-depth columns use the full denominator of 21. RPNI and Blue-Fringe are
still running and are intentionally left blank.
