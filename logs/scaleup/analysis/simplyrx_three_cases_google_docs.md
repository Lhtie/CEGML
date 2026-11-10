# How Counterexamples Change LLM Reasoning: Three SimplyRx Cases

Model: gpt-oss. All trajectories below use full pipeline, run-0: regularization, DFS counterexample generation, clustered counterexamples, reflection, and the repair loop when needed. The examples are ordered from simpler to more complex.

Reading guide: + means union, spaces mean concatenation, * means zero or more repetitions, and epsilon means the empty string. A positive counterexample is a valid string the current candidate rejects; a negative counterexample is an invalid string the candidate accepts. Bracket classes such as [ac] represent a single-character alternative, not literal bracket text. Epoch and retry indices start at zero.

Evidence guide: “Logged summary” reproduces the brief Reasoning field of the epoch-selected candidate. The descriptions of the model’s work summarize the complete analysis text in Response, rather than only that brief field. The logs contain no API reasoning-token counts or token IDs, so no exact token totals are claimed. The observed correspondences support an interpretation of how feedback influenced the revision; they do not isolate counterexamples causally from reflection and extra model calls.

Selection matters: the candidate carried to the next epoch is chosen using compilation and training-fit feedback. It is not always the last retry. Reflection carries the previous regex and short reasoning summary, not the complete preceding analysis.

## Example 1. Repetition plus singleton branches — 3 states, star depth 1

Regex: b* + a + c

Meaning: Accept any number of b’s, including zero, or exactly one a or exactly one c.

Run-0 outcome: 2 epochs, 2 LLM calls, 10 accumulated training entries; the final candidate is formally equivalent to the target.

### Full pipeline: reasoning at each epoch

- Epoch 0 — selected retry 0; 8 accumulated training entries. Logged summary: “All positive examples consist solely of the letter b and contain at least one b; every negative example contains a character other than b, so the language is “one or more b’s”.” Selected regex: b b*

- Epoch 1 — selected retry 0; 10 accumulated training entries. Logged summary: “All positive strings are either empty, a single ‘a’, a single ‘c’, or consist solely of any number of ‘b’s (including zero). No other combinations appear positively.” Selected regex: b * + a + c

### Transition: epoch 0 → epoch 1

- Change: “one or more b’s only” becomes “zero or more b’s, or singleton a, or singleton c.” The valid b-only component is preserved, while the missing empty-string case and two singleton branches are added.

- Specific counterexamples: the new positive entries are the empty string (label 1) and [ac] (label 1), representing the two singletons a and c. These are two logged entries, not three separate rows. There are no new negative counterexamples in this batch.

- Work visible in the analysis: the model first revisits the positive examples; recognizes that the empty string must be accepted; interprets [ac] as a grouped single-character class; combines the empty string with b-only repetitions using b-star; adds a and c through union; and checks that the existing mixed-letter negatives are still excluded.

- Connection to the counterexamples: strong and direct. The empty string contradicts the mandatory first b, and [ac] contradicts the b-only restriction. Both corrections appear in the next summary and regex, which passes formal equivalence checking.

- Limit and interpretation: no local repair retries occur in this run. This is coverage repair driven by positive counterexamples, rather than discovery of a complicated repetition rule. The initial candidate is an under-approximation, so the required correction is to add valid cases without accepting arbitrary mixtures.

### What happened before the transition?

- At epoch 0, all observed positive entries are nonempty b-only strings. The analysis debates whether the minimum length should be one or two and chooses one-or-more b’s. It explicitly excludes the empty string. Thus the initial error comes from incomplete positive coverage, not a failure to understand b repetition.

Source log: logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs/msgdict_regex=(((b)*+a)+c)_ceEpochs=12_ceBatch=250_clustered.json

## Example 2. Flexible prefix and restricted suffix — 4 states, star depth 2

Regex: (a + c)* b (a + epsilon) c

Meaning: Accept any a/c prefix, possibly empty, followed by exactly one b and then either c or ac.

The generated target looks more nested, but reduces to this expression: its repeated a-star/c-star component allows any a/c prefix, and its final union is just ac or c. There is no parity constraint in this example.

Run-0 outcome: 2 epochs, 2 LLM calls, 12 accumulated training entries; the final candidate is formally equivalent to the target.

### Full pipeline: reasoning at each epoch

- Epoch 0 — selected retry 0; 8 accumulated training entries. Logged summary: “All positive strings consist only of a and c except for a single b that never appears at the start or the end; every string ends with c. Hence the language is “one or more a/c, then one b, then any a/c, finally a c”.” Selected regex: (a + c) (a + c)* b (a + c)* c

- Epoch 1 — selected retry 0; 12 accumulated training entries. Logged summary: “All positives consist of any number of a/c, then a single b, optionally followed by a, and must end with c. No other characters appear after this final c.” Selected regex: (a + c)* b (a + epsilon) c

### Transition: epoch 0 → epoch 1

- Change: the a/c prefix becomes optional instead of mandatory, while the segment after b becomes tightly restricted instead of arbitrary. The update broadens one part of the language and narrows another.

- Positive counterexamples: bc and bac (both label 1). They have an empty prefix before b, directly contradicting the old claim that b cannot occur at the beginning.

- Negative counterexamples: [ac]bcc and [ac]ba[ac]c (both label 0). The first represents abcc/cbcc. The second represents abaac/abacc/cbaac/cbacc. All are accepted by the old candidate but have an invalid suffix after b.

- Work visible in the analysis: the model lists positives and negatives; interprets the grouped negative examples; recognizes that bc and bac require allowing a zero-length prefix; restricts the post-b segment to an optional a followed by a final c; checks several positives against that structure; and tests the negatives by explaining why extra post-b characters leave unmatched symbols or violate the expected final c.

- Connection to the counterexamples: strong for both edits. The positive examples explain the prefix relaxation, while the negative families explain why “any a/c after b, then c” is too broad. The revised summary expresses both changes, and the regex passes formal equivalence checking.

- Limit and interpretation: the analysis contains repetitive enumeration when expanding grouped negatives; it is not a clean, consistently executed symbolic algorithm. The finite examples motivate this structure but do not uniquely prove it. Equivalence checking establishes the final correctness. There are no within-epoch repair retries in this run.

### What happened before the transition?

- At epoch 0, the observed positives all have nonempty a/c prefixes and end in bc or bac. The model infers a mandatory prefix from the observed starts, yet permits arbitrary a/c after b. It tests only the available negatives bb and baa, which do not expose either boundary error. The new counterexamples specifically target those two blind spots.

Source log: logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs/msgdict_regex=(((a)* (((((a)* (c)*))*+a) b)) (((a+a) c)+c))_ceEpochs=12_ceBatch=250_clustered.json

## Example 3. Local c-run parity — 5 states, star depth 2

Regex: (a + c c)* (b + c) + a

Meaning: Accept any sequence of a or cc blocks followed by one final b or c, or accept the singleton a.

The key distinction is local structure: every internal c-run must have even length; a final c adds one c to the final run. Checking only the total number of c’s is insufficient.

Run-0 outcome: 5 epochs, 14 LLM calls, 197 accumulated training entries; the final candidate is formally equivalent to the target.

### Full pipeline: reasoning at each epoch

- Epoch 0 — selected retry 2; 8 accumulated training entries. Logged summary: “All positives contain only a and c. The language consists of (i) only c’s, (ii) the single string a, or (iii) strings with at least three alternating a‑runs and c‑runs, always ending with a c. Thus strings with exactly two a‑runs (e.g., “ccccacca”, “ccccccaaaccccccccacccccc”) are rejected.” Selected regex: ( a + c c* + a a* (c c* a a*) (c c* a a*) (c c* a a*)* c c* + c c* a a* (c c* a a*) (c c* a a*) (c c* a a*)* c c* )

- Epoch 1 — selected retry 1; 26 accumulated training entries. Logged summary: “All positives are either a single a, b, ab, ac; a run of at least three c’s; or strings that may start with c’s, contain one or more “acc” blocks and end with b, ab, ac or with at least two a’s followed by a c (allowing extra c’s between later “ac” pieces). Negatives violate these mandatory pieces.” Selected regex: ( a + b + a b + a c + c c c* + c c* a c c (a c c)* ( b + a b + a c + a a a* c c* (c* a c)* ) + c c* a a a* c c* (c* a c)* )

- Epoch 2 — selected retry 1; 61 accumulated training entries. Logged summary: “All positives consist of runs of c’s (except the isolated “cc”), and any ‘a’ or ‘b’ that follows a single c must be preceded by another c (i.e., only “cca” and “ccb” are allowed).  The language is therefore built from the tokens a, cca, ccb, cc and optional trailing c‑runs (single c, ≥3 c) or a ≥2‑c run ending in b.” Selected regex: ( c + c c c c* ) + ( (a + c c a + c c b + c c)* b ) + ( (a + c c a + c c b + c c)* (a + c c a + c c b) (epsilon + c + c c c c* + c c c* b) )

- Epoch 3 — selected retry 1; 186 accumulated training entries. Logged summary: “Positives are either an odd‑length run of only c, or contain at least one a (or just b) where every a after the start is preceded by “cc”.  Between a‑clusters there must be at least two c’s, and a final b may appear only at the end.” Selected regex: ( c (c c)* ) + ( ( epsilon + c c c* ) ( a a* (c c (c c)*) )* a a* c* (b + epsilon) )

- Epoch 4 — selected retry 1; 197 accumulated training entries. Logged summary: “All positives consist of an even number (possibly zero) of leading c’s, then any number of “a” separated by an even number of c’s; the string may end with a single b, or with a single c (giving an overall odd count of c’s). The lone string “a” is also allowed.” Selected regex: ( (c c)* ( a (c c)* )* b ) + ( (c c)* ( a (c c)* )* c ) + a

### Initial epoch: what the repair loop changes before epoch 1

- Initial attempt, epoch 0 retry 0: the model proposes alternating nonempty a-runs and c-runs ending in c, with extra branches for pure c strings and singleton a. It does not allow b because no initial positive requires b.

- Specific repair counterexample: ccccccaaaccccccccacccccc (label 0). It has two a-runs and is incorrectly accepted by the early candidates. The repair prompt highlights it in both subsequent retries.

- Changes across retries: retry 1 tries to require at least three a-runs for c-leading strings; retry 2 extends the minimum-three-a-runs condition to both a-leading and c-leading strings. Retry 1’s stated restriction and actual regex do not fully align; retry 2 makes the restriction explicit and is selected.

- Work visible in the analysis: the model splits strings into a-runs and c-runs; compares starting and ending symbols; counts a-runs; tries several explanations for why the highlighted negative differs from positives; and encodes a minimum block count. It also speculates about the label being wrong before settling on a hypothesis.

- Interpretation: the repair clearly responds to the negative, but invents the wrong explanation. Fitting the highlighted failure does not mean the discovered rule matches the target.

### Transition: epoch 0 → epoch 1

- Change: the selected explanation changes from “a/c-only strings with many a-runs” to short a/b cases and longer acc-based templates, allowing b-ending strings.

- Specific new positive counterexamples: b, ab, ac, accb, accab, accac, accaccb, accaccab. Specific new negatives include cc, accaccacc, and acacac. The batch adds 18 entries, taking the total from 8 to 26.

- Work visible in the analysis: the model re-lists the expanded positive and negative sets; examines repeated acc substrings; separates short cases from long cases; compares suffixes after acc repetitions; and adds alternative endings and a-run conditions to cover the long positives without accepting the new negatives.

- Connection to the counterexamples: strong for abandoning the a/c-only and minimum-a-run assumptions. b and ab directly contradict the alphabet restriction, ac contradicts the requirement for many a-runs, and cc contradicts unrestricted pure-c repetition. However, the replacement acc template is an additional guess; the examples do not require that representation.

- Within-epoch repair: retry 0 has invalid syntax. Retry 1 receives error feedback with no concrete repair examples and adjusts the template; it is selected. Retry 2 receives missed positives, including accb/accab/accac, and negative cc, then loosens the explanation toward “contains acc plus allowed endings”; it is invalid and not selected. Thus not every immediate change is attributable to a newly highlighted counterexample.

### Transition: epoch 1 → epoch 2

- Change: the model moves from acc-centered templates to reusable a/cca/ccb/cc tokens, together with special trailing c-runs. It starts constraining where c may occur, instead of requiring a particular acc-containing template.

- Specific new positive counterexamples include c, ccb, ccab, aab, and aac. Specific new negatives include caac, caccab, ccaacc, and ccaacac. The batch adds 35 entries, taking the total from 26 to 61.

- Work visible in the analysis: the model compares short accepted and rejected strings; notices that a lone c is accepted but cc is rejected; looks for repeated cc/cca structures; considers whether a or b after c needs preceding cc; tries token decompositions; and adds separate endings to cover strings such as ccaaccb while rejecting ccaacc.

- Connection to the counterexamples: strong for broadening and position constraints. c refutes the previous minimum-three-c restriction, aab/aac refute the mandatory acc template, and c-leading negatives contrast with cc-leading positives. The proposed token system still fails to enforce the complete language structure.

- Within-epoch repair: retry 1 sees eight highlighted negatives, including ccccacca, abc, cc, accaccacc, and ccaacc, and expands/reorganizes the token system; it is selected with about 96.7% training accuracy. Retry 2 focuses on ccccacca and ccccccaaaccccccccacccccc, switches back to cca/acc templates with enumerated endings, and drops to about 50.8% training accuracy. The last retry is not selected: responding to fewer errors can damage the broader rule.

### Transition: epoch 2 → epoch 3

- Change: b becomes explicitly terminal; pure-c strings are described using odd length; and the mixed-string explanation shifts toward a-clusters separated by c-runs instead of freely reusable cca/ccb tokens. The resulting separation rule is still imperfect.

- Specific new negative counterexamples include cccc, ccbb, ccbc, ccba, cca, and ccacccc, along with many related strings and grouped classes. All 125 added entries in this batch are negative; the total rises from 61 to 186.

- Work visible in the analysis: the model observes that positive b occurrences are at the end; contrasts b-followed-by-more-symbols negatives with terminal-b positives; compares pure-c lengths such as c and ccccccc against cc and cccc; infers odd pure-c length; studies c-runs between a-clusters; and constructs alternatives for pure-c and mixed strings.

- Connection to the counterexamples: strong for restricting b placement and rejecting the previous “all long c-runs” treatment. cccc directly challenges the old pure-c rule, and ccba/ccbc-like strings challenge internal b blocks. The exact mixed-string condition is still a hypothesis rather than a reliably derived invariant.

- Within-epoch repair: retry 0 has invalid syntax. Retry 1 receives syntax/error feedback with no concrete repair examples, makes the odd-pure-c rule explicit, and is selected. Retry 2 receives 16 highlighted errors, including positive b/ccb and negative c-run variants; it loosens the internal structure toward “start with a or cc, then a/c, end in c or b,” but training accuracy decreases. It is not selected.

### Epoch 3 → 4: initial generation and the single repair

- Call structure: epoch 4 has two LLM outputs: retry 0 (initial generation) and retry 1 (the only repair). These are not two separate repairs. The starting point is the selected epoch-3 retry 1, rather than epoch-3’s last output.

### Stage 1 — Epoch 4, retry 0: initial generation from the new CE batch

- Starting reasoning: pure-c strings have odd length; mixed strings contain a-clusters separated by c-runs; b is terminal. The epoch-3 selected regex still allows leading c-runs of length at least two without requiring even length, and a permissive final c-star before optional b.

- Counterexamples supplied: b (label 1); ccca, cccab, cccac, cccacb, cca, ccacb, acb, acc, aa, aacb (label 0). These are the 11 new outer-loop entries; the complete accumulated training set is also present.

- LLM CoT, step 1 — reorganize the evidence: the model lists positive strings, groups them by initial symbol, notes that positive b occurrences are terminal, and compares c-only positives c/ccccccc with negatives cc/cccc.

- LLM CoT, step 2 — try candidate restrictions: it explores starting-prefix patterns, terminal-b restrictions, c-run lengths, and whether a-ending strings should be excluded. It compares ac versus acc and acb versus accb, and also considers aab/aac as positives in this attempt.

- LLM CoT, step 3 — choose global parity: it moves toward “no b means an odd total number of c’s and a final c,” while treating singleton a as an exception. For b-ending strings, it adds alternatives immediately before b: a, cc, or the standalone b case. These are still approximate rules.

- LLM CoT, step 4 — encode and spot-check: it expresses c-ending strings as a* (c a* c a*)* c, which pairs c occurrences even when a’s occur between them; constructs b-ending alternatives using the same paired-count scaffold; and manually checks examples such as cc, acc, acb, aacb, accb, ab and ccac.

- Reasoning change: the broad c-run explanation becomes an explicit GLOBAL c-count parity rule for c-ending strings, plus immediate-pre-b restrictions. It still does not require every internal c-run to be even. For example, acacac has three c’s and is accepted by this approximation.

- Output and feedback: the candidate is (a* (c a* c a*)* c + a + (a* (c a* c a*)* a b + a* (c a* c a*)* c c b + b)). Training accuracy is 193/197 (about 97.97%); eval accuracy is 100%; formal equivalence is false. The logged formal witness is cacab. This witness is not one of the four highlighted repair examples.

### Stage 2 — Epoch 4, retry 1: the only repair

- Repair input: the previous regex and its short reasoning summary, training accuracy 0.980, eval accuracy 1.000, and four highlighted existing training errors: ccacacac, acacac, caccaacac, caacac (all label 0). No additional outer-loop samples are added: the accumulated total remains 197.

- Why these counterexamples matter: each ends in c and has an odd total c-count, yet contains an odd-length internal c-run. The previous global-parity regex therefore accepts them. They isolate the difference between pairing c occurrences globally and keeping c’s paired locally within runs.

- LLM CoT, step 1 — re-analyze rather than simply patch: the model starts again by listing representative positives and revisiting pure-c parity. It explores several explanations for the a/c/b patterns. It initially misremembers aab and aac as negative even though the prompt labels both positive; those intermediate claims are errors, not valid constraints.

- LLM CoT, step 2 — infer internal evenness: it compares accac, accab and accaccab, observes two c’s between a’s, and examines ccaccaab to recognize that consecutive a’s have zero intervening c’s. It proposes that the c-count between consecutive a’s must be even, including zero.

- LLM CoT, step 3 — infer leading evenness: it compares caccab with accab and cc-leading positives, then contrasts caccac with ccac. It proposes that leading c’s before the first a must also occur in even numbers. Pure-c strings remain covered separately through the final-c case.

- LLM CoT, step 4 — combine local constraints with the ending: it revisits ac versus acc and accc versus acccc. It builds an even-c prefix and body from (c c)* and repeated a (c c)* blocks. It then creates one branch ending in b and one ending in a single extra c; the latter supplies odd final-run length. Singleton a stays a separate branch.

- LLM CoT, step 5 — test the reconstructed expression: it checks positives including b, ab, ac, accb, accab, ccab, ccac, ccaccc, c and ccccccc. It tests negatives including acc, cc, cccc, caccac, caccab, acacac and ccacacac. Some checks contain counting or segmentation mistakes, so this is not an exhaustive or error-free verification.

- Reasoning change: “the TOTAL number of c’s must be odd” becomes “the LEADING c-run and c-runs BETWEEN a’s must be even, including zero; then append b or one c.” The final summary now explicitly states the even leading and intervening c-counts. Global oddness for c-ending strings becomes a consequence of the local construction, rather than the sole constraint.

- Final regex: ((c c)* (a (c c)*)* b) + ((c c)* (a (c c)*)* c) + a. Training and eval accuracies are both 100%, and formal equivalence is true. This repair is selected.

- Interpretation: the observed revision matches the highlighted failures and replaces the representation, rather than adding four string-specific exceptions. However, the analysis largely reconstructs the rule from accumulated data and does not consistently narrate each step as a direct response to the four repair examples. The evidence supports a strong correspondence, not an isolated causal proof or a flawless repair algorithm.

Source log: logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/reg/agentic_reflection/dfs/msgdict_regex=(((((a)*+(c c)))* (b+c))+(a+c))_ceEpochs=12_ceBatch=250_clustered.json

## Cross-case interpretation

- Example 1: counterexamples reveal missing coverage. The model preserves b repetition and adds precisely the omitted empty-string and singleton cases.

- Example 2: counterexamples locate separate boundary errors. Positive examples relax the prefix; negative examples restrict the suffix. Improvement requires changing different components in opposite directions.

- Example 3: counterexamples distinguish competing explanations. A total-count rule fits much of the data, but highlighted errors push the representation toward reusable local cc blocks.

- Across the complete analysis outputs, recurring activities include organizing positive/negative examples, comparing short strings and near-misses, proposing blocks or boundary rules, manually testing candidates, and translating the explanation into regex syntax. These activities are observable tendencies, not a fixed algorithm executed correctly on every call.

- The useful contribution of counterexamples is to make the current hypothesis’s blind spot concrete. They do not directly provide the replacement rule, and some responses overfit, introduce invalid syntax, or regress. Formal verification and candidate selection are essential parts of the successful pipeline.
