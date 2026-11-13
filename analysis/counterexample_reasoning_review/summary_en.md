The saved reasoning traces suggest that LLMs use counterexamples through **iterative hypothesis revision**: they infer a structural explanation from labeled strings, identify where new evidence contradicts that explanation, and revise the corresponding regular expression. Successful revisions turn a counterexample into a change in the underlying rule. Less successful revisions add exceptions while preserving an incorrect explanation.

This account draws on the extraction of 2,843 JSON files from `icl_gen_simplyrx` and `icl_gen_extrx`, together with close reading of representative trajectories across tasks, models, and strategies. The files contain 37,905 log records, including 2,582 copied candidate-evaluation records; these are not all independent model calls. The six steps below are an analytical reconstruction of recurring behavior, rather than a fixed procedure explicitly followed in every response. Saved explanations provide evidence about the model's stated hypotheses, but do not reveal all of its internal computation. In particular, the inspected GPT-5.4 records mainly contain explicit reasoning summaries.

**Step 1: Interpret the examples and extract relevant features.**

The model first needs to determine what each example represents and which properties might explain its label. Relevant properties include string length, character categories, prefixes and suffixes, repeated blocks, occurrence counts, and whether the empty string is accepted. This converts raw strings into possible explanatory features. For example, several strings containing repeated `ca` pairs may suggest a repeating unit `ca`, rather than unrelated individual sequences.

Correct interpretation also requires distinguishing a concrete string from a symbolic family of strings. In ExtRX, an example such as `[a-z][A-Za-z]` represents two-character strings whose first character is lowercase and whose second character is a letter. It is not the literal text containing brackets and hyphens. The current acceptance checker evaluates character-class examples over their expansions. Some saved responses struggle with this distinction and dismiss bracketed examples as containing characters outside the alphabet. Such a mistake can prevent a useful counterexample from constraining the next hypothesis.

The output of this step is a collection of candidate features, not an established rule. If all observed positives happen to have even length, evenness is a possible explanation, but the observations alone do not establish that it defines the target language.

**Step 2: Form a structural hypothesis that explains the labels.**

The model combines those features into a proposed description of the language. Typical descriptions include “a run of `b` symbols followed by complete `ca` blocks,” “exactly two alphabetic characters,” or “contains `ion` but does not contain `old`.” This description compresses many examples into a rule and provides a basis for constructing a regex.

There are two levels to this hypothesis: the semantic explanation in natural language and the symbolic expression intended to implement it. Keeping these levels separate helps explain the logs: an explanation can be plausible while its regex uses the wrong operator, and a syntactically valid regex can implement an incorrect explanation.

For example, one ExtRX trajectory initially explains its examples as nonempty strings of even length and proposes `([A-Za-z0-9#]{2})+`. Later evidence leads it to replace that explanation with “contains at least two digits.” The initial hypothesis therefore selected the wrong explanatory feature. It was not simply missing a small adjustment to a length bound.

**Step 3: Compare the counterexample with the previous hypothesis.**

A counterexample identifies an input on which the candidate and target disagree. A **positive counterexample** belongs to the target language but is rejected by the candidate: the candidate excludes something it should allow. A **negative counterexample** is accepted by the candidate but rejected by the target: the candidate allows something it should exclude. These labels describe the target's decision, not whether the feedback is beneficial.

The model's task is to explain that disagreement relative to its previous rule. In reflection configurations, the prompt explicitly provides the previous reasoning summary, previous regex, and new counterexamples, alongside accumulated training examples. This gives the revision a concrete reference point. In single-inference configurations, the model can instead infer a new rule from the accumulated dataset without explicitly discussing its previous explanation.

For example, a SimplyRX run proposes `b + c c*`: either a single `b` or one or more `c` symbols. The verifier returns the empty string as a disagreement witness, and the next round supplies it as a positive example. Comparing this example with the old regex shows precisely what is missing: the candidate has no way to accept zero `c` symbols. The revision becomes `b + c*`.

**Step 4: Diagnose which assumption caused the disagreement.**

Detecting a disagreement does not by itself determine how to repair it. The model must decide which assumption is responsible: a character class may be too narrow, a repetition bound may be wrong, a prefix may be unnecessary, blocks may need to mix, or an additional global condition may be required. A useful diagnosis explains a family of errors rather than treating the witness as an isolated exception.

Consider the ExtRX target consisting of exactly two letters. One trajectory initially proposes `[A-Z][A-Za-z]`, which requires an uppercase first character. In the corresponding `noreg/agentic_reflection` trajectory, the positive counterexample `[a-z][A-Za-z]` directly challenges that restriction. The appropriate diagnosis is that the first position can be lowercase as well. The two-character length and alphabetic-character requirements can remain intact.

Diagnosis sometimes requires replacing the explanation more substantially. In the even-length trajectory, the later rule counts digits rather than total characters. This illustrates a broader revision: the model changes what it treats as relevant to membership in the language.

**Step 5: Revise the rule and express it in the required symbolic syntax.**

Once the model has diagnosed the conflict, it constructs a new regex. Local revisions include expanding a character class, allowing zero repetitions, changing a count bound, or adding a missing alternative. Broader revisions include reorganizing concatenation and repetition, introducing an intersection or complement, or replacing the entire structural hypothesis.

For example, an ExtRX run initially accepts any string containing `ion`. Negative counterexamples containing both `ion` and `old` show that this condition is insufficient. The updated explanation requires the presence of `ion` and the absence of `old`. Using `Σ` as mathematical shorthand for the alphabet, the new language is:

`contains(ion) ∩ complement(contains(old))`

The actual candidate expresses these conditions using `[A-Za-z0-9#]*`, intersection `&`, and complement `~(...)`. This is a rule-level repair: it rejects an entire class of invalid strings rather than enumerating individual counterexamples.

Symbolic syntax is a separate source of difficulty. In SimplyRX, `+` denotes union and concatenated symbols are separated by spaces. In ExtRX, `|` denotes union and `+` denotes one or more repetitions. A model that correctly describes repetition but uses the wrong dialect can still produce an invalid or incorrect candidate. The shorthand used to discuss a hypothesis should therefore be distinguished from executable task syntax.

**Step 6: Check the revised candidate and use external feedback to continue.**

The visible long responses often attempt to check whether the candidate accepts positive examples, rejects negative examples, and handles boundaries such as short strings and the empty string. These textual checks can reveal inconsistencies, but they are not reliable proof that the candidate is correct.

The surrounding program performs a separate check: it attempts to compile the candidate, measures its agreement with available examples, and checks equivalence with the target. Repair loops can use compilation errors or misclassified training examples to request another attempt. If the candidate remains nonequivalent, a subsequent counterexample round supplies further distinguishing evidence.

The distinction between sample agreement and equivalence is essential. In the `b + c c*` example, the initial candidate has training accuracy 1 and evaluation accuracy 1, yet is not equivalent to the target because it rejects the empty string. An equivalence check exposes a difference that those finite datasets missed. Consequently, the overall process combines LLM hypothesis generation with external symbolic verification; the model's claim that it has checked the examples is only one part of that process.

**What serves as the model's reference during revision?**

The traces suggest four main sources of reference:

| Reference | Role in the revision |
| --- | --- |
| Accumulated positive and negative examples | Constrain which candidate languages remain compatible with observed labels. |
| Previous reasoning summary and regex | Preserve an explicit account of the current hypothesis and provide a starting point for revision. |
| New counterexamples and repair feedback | Identify disagreements or execution problems that the next candidate should address. |
| Prompt instructions and regex conventions | Supply preferences for block structure, boundary checks, compactness, and valid symbolic expression. |

In the current reflection implementation, the previous final reasoning summary is passed forward, rather than the entire previous analysis text. It functions as an explicit memory of the explanation. This can support coherent revisions, but can also anchor the model to an incorrect assumption. Moreover, several inference strategies are explicitly suggested in the prompts, so their appearance in responses should not automatically be attributed to spontaneous discovery.

**A recurring failure: preserving an incorrect explanation through local patches.**

One SimplyRX trajectory makes this problem particularly clear. The target is `b* (c a)*`: any number of `b` symbols followed by any number of complete `ca` blocks, with the two counts independent. The model initially imposes additional relationships between the counts. It later introduces cases based on the number of `b` symbols modulo three and repeatedly increases the allowed number of `ca` blocks for other cases. Several intermediate candidates fit every accumulated training example while remaining nonequivalent to the target.

The eventual successful revision drops those unsupported relationships and adopts `b* (c a)*`. The important change is the recognition that the counts are independent. Raising a finite bound addresses the latest examples; removing the false dependency changes the underlying rule. This case shows why counterexamples can either drive structural discovery or lead to a growing collection of patches.

Across the inspected trajectories, SimplyRX emphasizes sequence order, repeated blocks, and empty-string behavior. ExtRX additionally emphasizes character categories, counting thresholds, substring requirements, intersection, and complement. In both settings, the strongest revisions explain what a counterexample reveals about the language as a whole. Common failures include accidental correlations, historical anchoring, literal enumeration, and a mismatch between the verbal explanation and the generated regex. These observations describe representative behaviors, rather than measured frequencies or a controlled ranking of models and strategies.

The Chinese report, `summary_zh.md`, provides the exact source paths and run identifiers for the cases discussed here. `reasoning_index.csv` preserves the extracted reasoning summaries and record locations, while `inventory.json` records the scan inventory.
