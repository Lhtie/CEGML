# Case B: arbitrary a/c prefix followed by bc or bac

## Ground truth: natural-language interpretation

Accept exactly the strings consisting of any number of a’s and c’s (possibly none), followed by exactly one b, followed by either c or ac. In other words, the suffix starting at the unique b must be exactly bc or bac. The empty prefix is allowed; the empty string is rejected.

Original target: `(((a)* (((((a)* (c)*))*+a) b)) (((a+a) c)+c))`

Equivalent compact regex: `(a + c)* b (a + epsilon) c`

Accepted: `bc`, `bac`, `abc`, `cbac`, `aacbc`. Rejected: `c`, `bb`, `baa`, `abcc`, `abaac`, `abacc`.

This case has **4 states and recorded star depth 2**, compared with the original case’s 5 states / star depth 2. Model: **gpt-oss**. Main trajectory: **run-0**. Standard uses `std/reg`; full uses `ce/reg/agentic_reflection/dfs`, clustered counterexamples, and configured `ceBatch=250`.

Notation: `+` is union, spaces are concatenation, `*` is Kleene star, and `epsilon` is the empty string. Bracket classes in training data denote grouped **single-character** alternatives, not literal bracket text. Counts are logged training entries and can include duplicates or grouped classes.

## Results

| Method | run-0 epochs / LLM calls | run-0 training entries | run-0 equivalent? | Successful runs | Training entries at termination, runs 0/1/2 |
|---|---|---:|---|---|---|
| Standard | 11 / 11 | 3000 | No | 1/3 | 3000 / 24 / 3000 |
| Full pipeline | 2 / 2 | 12 | Yes | 3/3 | 12 / 8 / 8 |

## Epoch summary tables

Each table reproduces the logged **epoch-selected `CurrentGuessReasoning` and `CurrentGuess`**. These are the model’s hypotheses, not endorsed descriptions of the ground truth. The brief Reasoning field is not the complete analysis. Missing outputs are shown explicitly; no regex has been reconstructed for them.

### Standard (std)

| Epoch | Training entries | Summary reasoning (verbatim) | Selected regex |
|---:|---:|---|---|
| 0 | 3 | All positive strings begin with a, end with the suffix b c, and contain only a or c between the leading a and the final b c. No b appears elsewhere, which excludes the negative “bb”. | `a (a + c)* b c` |
| 1 | 6 | Every positive string ends with the symbol c, while the negative examples end with a b or a. Thus the language consists of all strings over {a,b,c} that terminate with c. | `(a + b + c)* c` |
| 2 | 12 | Every positive string ends with the symbol c, while none of the negative strings does. No other consistent restriction is required to separate the two sets. | `(a + b + c)* c` |
| 3 | 24 | All accepted strings must contain at least one b, and every b appears only as the substring “bc” or “bac”.  No other b‑patterns (e.g., “bb” or “ba” not followed by c) are allowed, while any number of isolated a or c letters may appear anywhere. | `(a + c + b c + b a c)* (b c + b a c) (a + c + b c + b a c)*` |
| 4 | 48 | All positive strings end with the symbol c and contain at least one b somewhere before that final c. No negative string satisfies both conditions. | `(a + b + c)* b (a + b + c)* c` |
| 5 | 96 | All positive strings contain exactly one ‘b’ and always end with ‘c’. The parts before and after that single ‘b’ consist only of ‘a’ and ‘c’, guaranteeing no other ‘b’ appears. | `(a + c)* b (a + c)* c` |
| 6 | 192 | All positive strings contain at least one b and must finish with either “ac” or “bc”.  Negatives either lack a b or do not end with those two‑letter suffixes. | `(a + b + c)* b (c + (a + b + c)* a c)` |
| 7 | 384 | All positive strings end with the symbol c, contain exactly one b, and never have the substring bb. This is captured by allowing any number of a’s or c’s before the single b, any number of a’s or c’s after it, and then a final c. | `(a + c)* b (a + c)* c` |
| 8 | 768 | No summary reasoning extracted. | No regex extracted; logged error. |
| 9 | 1536 | All positive strings contain exactly one ‘b’ and always end with ‘c’. No positive string has a second ‘b’, and every negative either lacks a ‘b’ or has more than one ‘b’ (or does not end with ‘c’). Hence the language is “strings over {a,b,c} with a single ‘b’ and a final ‘c’”. | `(a + c)* b (a + c)* c` |
| 10 | 3000 | No summary reasoning extracted. | No regex extracted; logged error. |

### Full pipeline

| Epoch | Training entries | Summary reasoning (verbatim) | Selected regex |
|---:|---:|---|---|
| 0 | 8 | All positive strings consist only of a and c except for a single b that never appears at the start or the end; every string ends with c. Hence the language is “one or more a/c, then one b, then any a/c, finally a c”. | `(a + c) (a + c)* b (a + c)* c` |
| 1 | 12 | All positives consist of any number of a/c, then a single b, optionally followed by a, and must end with c. No other characters appear after this final c. | `(a + c)* b (a + epsilon) c` |

## Full pipeline: changes in reasoning and their connection to counterexamples

Run-0 has two LLM calls, one per epoch, and **no within-epoch retries**. The sole candidate-to-candidate transition simultaneously fixes an under-approximation before b and an over-approximation after b.

| Transition | Previous hypothesis → revised hypothesis | Counterexamples actually supplied | How the evidence relates to the change | Strength and limit of attribution |
|---|---|---|---|---|
| Initialization → epoch 0 | No previous candidate → a **nonempty** a/c prefix, one b, then an **arbitrary** a/c suffix ending in c: `(a + c) (a + c)* b (a + c)* c`. | Six initial positives have nonempty a/c prefixes and end in bc or bac. The two initial negatives are bb and baa. | The model invents a nonempty-prefix condition from the observed starts, while generalizing the post-b part beyond the two suffixes present in positives. The two initial negatives do not distinguish this guess from the target. | Initial inference, not proof of a CE-driven correction. The candidate fits all eight initial entries but is not equivalent. |
| Epoch 0 → epoch 1: prefix revision | “b must not be the first character” → the a/c prefix may be empty. | Positive CEs `bac, 1` and `bc, 1`. | Both start with b, so the old mandatory prefix rejects them. The new regex changes the prefix from one-or-more to zero-or-more a/c. | **Direct contradiction and matching repair.** The two positive CEs specifically expose the incorrect minimum-prefix-length assumption. |
| Epoch 0 → epoch 1: suffix revision | “arbitrary a/c after b, then final c” → after b, allow only an optional a followed by c. | Negative CEs `[ac]bcc, 0` and `[ac]ba[ac]c, 0`. | `[ac]bcc` represents abcc/cbcc; `[ac]ba[ac]c` represents abaac/abacc/cbaac/cbacc. All are accepted by the old regex but have invalid post-b suffixes. The new suffix `(a + epsilon) c` rejects them while retaining the observed bc/bac positives. | **Strong correspondence.** The negative families expose post-b overgeneralization. They motivate restricting the suffix, although these finite examples alone do not mathematically force a unique regex. The final candidate is verified equivalent. |

### What the full analysis adds beyond the short summary

At epoch 0, the analysis notices that all observed positives begin with a or c and that b occurs once. It chooses a nonempty prefix, although the data do not establish that b-leading strings are invalid. It permits arbitrary a/c after b as long as the string ends in c, even though the observed positive suffixes are only bc or bac.

At epoch 1, the analysis explicitly recognizes `bc` and `bac` as positive strings with no a/c prefix. It interprets the bracketed negatives as grouped strings, proposes an optional-a suffix, and checks that extra characters after b prevent a full match. The short summary then expresses precisely the new structure: arbitrary a/c prefix, one b, optional a, final c.

This is a **two-sided boundary repair**: positive CEs widen the prefix condition, while negative CEs narrow the suffix condition. It is not simply “allow more strings” or “be more conservative.” The same update broadens one component and restricts another.

### Why this regex is simpler than its generated syntax suggests

The target is syntactically nested, but `((a)* (c)*)*` denotes all strings over a/c: any individual a or c can be produced in a repetition. Its union with a changes nothing, and the extra leading a-star changes nothing. The final union `((a+a) c)+c` is just ac or c. Therefore the target reduces to `(a + c)* b (a + epsilon) c`. This example has 4 states rather than 5, although its recorded star depth is still 2; its semantic rule has no local parity constraint.

### Comparison with standard

Standard oscillates among “ends in c,” “contains at least one b,” “every b belongs to bc/bac,” and “exactly one b and final c.” Several hypotheses fit the current training set but fail formal equivalence. Its repeated candidate `(a + c)* b (a + c)* c` still permits an arbitrary post-b segment, which is exactly the distinction the full pipeline’s negative CEs target. In run-0, no regex is extracted at epochs 8 and 10, so the final logged result also includes output/extraction failures; those failures should not be interpreted solely as a conceptual inability to learn the suffix.

Full succeeds after four targeted CE entries are added, at 12 accumulated entries. Standard run-0 fails after 3000 entries, but **standard run-1 succeeds at 24 entries**; across the three runs its success count is 1/3, compared with full’s 3/3. The evidence supports a targeted boundary-repair interpretation, not a claim that standard can never solve this task.

## Per-call analysis length and verification

The logs contain text, not API reasoning-token usage or token IDs. **Analysis characters are not token counts.** They count Unicode characters before `assistantfinal`, excluding the initial `analysis` marker. The full analysis was inspected to check the interpretation above. “Equivalent” is formal language equivalence; a finite eval accuracy of 1.0 does not establish it.

| Method | Epoch.retry (zero-based) | Training entries | Analysis characters | Train accuracy | Eval accuracy | Equivalent | Witness / error |
|---|---|---:|---:|---:|---:|---|---|
| std | 0.0 | 3 | 3331 | 1.0000 | 0.5938 | False | abac |
| std | 1.0 | 6 | 6937 | 1.0000 | 0.8750 | False | bcc |
| std | 2.0 | 12 | 6848 | 1.0000 | 0.8750 | False | bcc |
| std | 3.0 | 24 | 11827 | 1.0000 | 1.0000 | False | bcc |
| std | 4.0 | 48 | 7888 | 1.0000 | 0.9375 | False | bcc |
| std | 5.0 | 96 | 49857 | 1.0000 | 1.0000 | False | bcc |
| std | 6.0 | 192 | 17946 | 0.9635 | 0.9375 | False | bbc |
| std | 7.0 | 384 | 16796 | 1.0000 | 1.0000 | False | bcc |
| std | 8.0 | 768 | 107685 | — | — | — | Error compiling regex: 'NoneType' object has no attribute 'strip' |
| std | 9.0 | 1536 | 18314 | 0.9980 | 1.0000 | False | bcc |
| std | 10.0 | 3000 | 98601 | — | — | — | Error compiling regex: 'NoneType' object has no attribute 'strip' |
| full | 0.0 | 8 | 6992 | 1.0000 | 0.9688 | False | bc |
| full | 1.0 | 12 | 8250 | 1.0000 | 1.0000 | True | — |

## Evidence and attribution limits

- The CE-to-rule correspondences above use examples actually present in the next prompt, not merely the logged Witness field.
- Outer-loop CEs come from disagreement between the target and the selected candidate. These cases require no within-epoch repair retries in run-0; their success demonstrates that such retries are not necessary for every task.
- The new prompt also contains accumulated training data, the previous regex, and its short reasoning summary. Thus the traces support an evidence-linked interpretation, not isolated causal attribution to CE independent of reflection or an additional model call.
- Initial data differ between standard and full; this is not a matched-initialization experiment. Comparisons at the same epoch do not imply equal evidence or equal compute.

## Source artifacts

- [std original log](</mnt/c/users/lhxlh/documents/uw-madison/research/cegml/logs/scaleup/icl_gen_simplyrx/model=gpt-oss/std/reg/msgdict_regex=(((a)* (((((a)* (c)*))*+a) b)) (((a+a) c)+c))_totTrain=3000_startSize=3_scaleFactor=2.0.json>)
- [full original log](</mnt/c/users/lhxlh/documents/uw-madison/research/cegml/logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs/msgdict_regex=(((a)* (((((a)* (c)*))*+a) b)) (((a+a) c)+c))_ceEpochs=12_ceBatch=250_clustered.json>)
- [All-run per-call CSV](simplyrx_4state_sd2_suffix_trajectory.csv)
- Implementation: `teacher.py::_generate_counterexamples_worker`; `train_icl_gen.py::build_reflection_prompt`, `build_retry_prompt`, and the epoch/retry loop.
