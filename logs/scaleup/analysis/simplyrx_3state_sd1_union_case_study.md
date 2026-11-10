# Case A: b-only repetition plus two singletons

## Ground truth: natural-language interpretation

Accept the empty string, any nonempty string consisting only of b, and the two singleton strings a and c. Reject every mixed-letter string and every longer string containing a or c.

Original target: `(((b)*+a)+c)`

Equivalent compact regex: `b* + a + c`

Accepted: empty string, `b`, `bb`, `a`, `c`. Rejected: `aa`, `cc`, `ab`, `bc`, `acb`.

This case has **3 states and recorded star depth 1**, compared with the original case’s 5 states / star depth 2. Model: **gpt-oss**. Main trajectory: **run-0**. Standard uses `std/reg`; full uses `ce/reg/agentic_reflection/dfs`, clustered counterexamples, and configured `ceBatch=250`.

Notation: `+` is union, spaces are concatenation, `*` is Kleene star, and `epsilon` is the empty string. Bracket classes in training data denote grouped **single-character** alternatives, not literal bracket text. Counts are logged training entries and can include duplicates or grouped classes.

## Results

| Method | run-0 epochs / LLM calls | run-0 training entries | run-0 equivalent? | Successful runs | Training entries at termination, runs 0/1/2 |
|---|---|---:|---|---|---|
| Standard | 7 / 7 | 192 | Yes | 3/3 | 192 / 192 / 192 |
| Full pipeline | 2 / 2 | 10 | Yes | 3/3 | 10 / 10 / 10 |

## Epoch summary tables

Each table reproduces the logged **epoch-selected `CurrentGuessReasoning` and `CurrentGuess`**. These are the model’s hypotheses, not endorsed descriptions of the ground truth. The brief Reasoning field is not the complete analysis. Missing outputs are shown explicitly; no regex has been reconstructed for them.

### Standard (std)

| Epoch | Training entries | Summary reasoning (verbatim) | Selected regex |
|---:|---:|---|---|
| 0 | 3 | All positive strings consist only of the symbol b and contain at least one b; no other symbols appear in any positive example. | `b b*` |
| 1 | 6 | All positive examples are strings made only of the symbol b (any length ≥ 1). The negative examples contain a or c, so the language is exactly “one or more b’s”. | `b b*` |
| 2 | 12 | All positive examples consist solely of the letter b and have length ≥ 2, while every negative example contains a letter other than b. Hence the language is “at least two b’s”. | `b b b*` |
| 3 | 24 | All positive examples are non‑empty strings composed exclusively of the letter b, while every negative example contains at least one a or c. Hence the target language is “one or more b’s”. | `b b*` |
| 4 | 48 | All positive examples consist solely of one or more ‘b’ characters, while every negative example contains at least one character other than ‘b’. Hence the language is exactly one‑or‑more b’s. | `b b*` |
| 5 | 96 | All positive strings are either empty, the single character “a”, or consist solely of one or more “b” characters. Every negative string contains a letter other than b (or a mix of letters), so any string made only of b’s (including the empty string) and the isolated “a” are the only allowed cases. | `(b *) + a` |
| 6 | 192 | All positive examples are either the empty string or consist solely of the letter b (any length), while the only other positive strings are the single characters a and c. No other combination of letters appears as a positive. | `b* + a + c` |

### Full pipeline

| Epoch | Training entries | Summary reasoning (verbatim) | Selected regex |
|---:|---:|---|---|
| 0 | 8 | All positive examples consist solely of the letter b and contain at least one b; every negative example contains a character other than b, so the language is “one or more b’s”. | `b b*` |
| 1 | 10 | All positive strings are either empty, a single ‘a’, a single ‘c’, or consist solely of any number of ‘b’s (including zero). No other combinations appear positively. | `b * + a + c` |

## Full pipeline: changes in reasoning and their connection to counterexamples

There are only two LLM calls in run-0, one per epoch. There are **no within-epoch retries** in this trace, so there is exactly one candidate-to-candidate transition to explain.

| Transition | Previous hypothesis → revised hypothesis | Counterexamples actually supplied | How the evidence relates to the change | Strength and limit of attribution |
|---|---|---|---|---|
| Initialization → epoch 0 | No previous candidate → accept one or more b’s: `b b*`. | Initial 8 labeled entries: five positive entries are nonempty b-only strings, and three negatives contain other letters. The positive entries include a duplicate. | The observed positive coverage encourages a b-only explanation. The initial data do not require accepting the empty string, a, or c. | This is initial inference, not a demonstrated correction driven by a previous-candidate CE. |
| Epoch 0 → epoch 1 | One or more b’s → zero or more b’s, **or singleton a, or singleton c**: `b * + a + c`. | Exactly two added positive entries: `, 1` (the empty string) and `[ac], 1` (a grouped single-character class representing a or c). No new negative entries. | The empty-string CE contradicts the mandatory first b, prompting a change from one-or-more to zero-or-more. The grouped CE contradicts the b-only restriction and prompts two singleton union branches. | **Very strong correspondence.** Both new entries are rejected by the old candidate, and both changes appear explicitly in the next summary and regex. The candidate then passes formal equivalence. This is observational evidence, not an ablation isolating CE from reflection and an extra call. |

### What the full analysis adds beyond the short summary

At epoch 0, the analysis considers whether the minimum b-count should be one or two and chooses one-or-more because the examples do not establish a stricter minimum. It explicitly keeps the empty string out. At epoch 1, it recognizes the empty string as a positive and correctly interprets `[ac]` as a **single-character class**, rather than a literal four-character string. It then combines the empty string and b-only repetitions through b-star and adds the singleton branches a and c.

The revision **preserves the valid b-only component while extending the language in precisely the missing directions**. It does not relax the regex to arbitrary a/b/c strings. No negative CE is needed in this particular transition, because the previous candidate is an under-approximation of the target: all of its accepted strings are already valid.

### Comparison with standard

Standard spends epochs 0–4 on b-only hypotheses, even temporarily requiring at least two b’s at epoch 2. At epoch 5 it adds the empty string and a, but still misses c. It reaches the correct union at epoch 6 with 192 cumulative entries. Full supplies the missing cases directly and succeeds at epoch 1 with 10 cumulative entries.

This example illustrates **coverage repair**: targeted positive CEs reveal missing disconnected branches of a language. The improvement comes from seeing the cases that distinguish b-plus from the target, rather than from a more sophisticated repetition argument. Both methods succeed in all three runs; full run-0 also uses fewer calls and fewer analysis characters here, but this is a case-specific observation.

## Per-call analysis length and verification

The logs contain text, not API reasoning-token usage or token IDs. **Analysis characters are not token counts.** They count Unicode characters before `assistantfinal`, excluding the initial `analysis` marker. The full analysis was inspected to check the interpretation above. “Equivalent” is formal language equivalence; a finite eval accuracy of 1.0 does not establish it.

| Method | Epoch.retry (zero-based) | Training entries | Analysis characters | Train accuracy | Eval accuracy | Equivalent | Witness / error |
|---|---|---:|---:|---:|---:|---|---|
| std | 0.0 | 3 | 1233 | 1.0000 | 0.9688 | False |  |
| std | 1.0 | 6 | 2253 | 1.0000 | 0.9688 | False |  |
| std | 2.0 | 12 | 2279 | 1.0000 | 0.9062 | False |  |
| std | 3.0 | 24 | 2343 | 1.0000 | 0.9688 | False |  |
| std | 4.0 | 48 | 36878 | 1.0000 | 0.9688 | False |  |
| std | 5.0 | 96 | 3256 | 1.0000 | 1.0000 | False | c |
| std | 6.0 | 192 | 4275 | 1.0000 | 1.0000 | True | — |
| full | 0.0 | 8 | 2146 | 1.0000 | 0.9688 | False |  |
| full | 1.0 | 10 | 3029 | 1.0000 | 1.0000 | True | — |

## Evidence and attribution limits

- The CE-to-rule correspondences above use examples actually present in the next prompt, not merely the logged Witness field.
- Outer-loop CEs come from disagreement between the target and the selected candidate. These cases require no within-epoch repair retries in run-0; their success demonstrates that such retries are not necessary for every task.
- The new prompt also contains accumulated training data, the previous regex, and its short reasoning summary. Thus the traces support an evidence-linked interpretation, not isolated causal attribution to CE independent of reflection or an additional model call.
- Initial data differ between standard and full; this is not a matched-initialization experiment. Comparisons at the same epoch do not imply equal evidence or equal compute.

## Source artifacts

- [std original log](</mnt/c/users/lhxlh/documents/uw-madison/research/cegml/logs/scaleup/icl_gen_simplyrx/model=gpt-oss/std/reg/msgdict_regex=(((b)*+a)+c)_totTrain=3000_startSize=3_scaleFactor=2.0.json>)
- [full original log](</mnt/c/users/lhxlh/documents/uw-madison/research/cegml/logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs/msgdict_regex=(((b)*+a)+c)_ceEpochs=12_ceBatch=250_clustered.json>)
- [All-run per-call CSV](simplyrx_3state_sd1_union_trajectory.csv)
- Implementation: `teacher.py::_generate_counterexamples_worker`; `train_icl_gen.py::build_reflection_prompt`, `build_retry_prompt`, and the epoch/retry loop.
