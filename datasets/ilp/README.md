# HRI EvenSucc ILP setting

Source: [Glanois et al., Neuro-Symbolic Hierarchical Rule Induction, ICML 2022](https://proceedings.mlr.press/v162/glanois22a.html), Appendix B.1/E, and the [author repository](https://github.com/claireaoi/hierarchical-rule-induction).

The checked-in author files are pinned to revision `1a3a0465a5a5305ad8fec958901710dc6ead76e4` under `source/`, with the author's CC0 license. They are provenance references, not imported by the benchmark runtime. `even_succ.json` records their SHA-256 hashes. Regenerate with:

```bash
python datasets/ilp/prepare.py
```

The deterministic facts and labels reproduce `EvenSuccDataset.getData`: constants `0..N-1`, `zero(0)`, consecutive `succ(i,i+1)`, and positive even-index target atoms. The preserved paper contexts use N=11/six inference rounds and N=15/eight rounds, matching `HRI/EvenSucc.py`. All 11/15 unary target atoms, including negatives, are retained. The vendored EvenOdd generator has the same target, but is not registered as a second independent setting.

The reference program is teacher/verification data, not included in learner prompts. Learners see the background declaration, labeled ground atoms, hypothesis syntax, and requested inference budgets. Hypotheses are positive unary/binary function-free Horn clauses with recursion and auxiliary predicate invention, not Python classifiers. Numeric constants in rules and arithmetic shortcuts are excluded. `true`/`false` are exposed as nullary Boolean literals; the only supplied relational predicates are `zero/1` and `succ/2`.

Our method is an LLM+CE adaptation. It does not reproduce HRI's differentiable optimizer, embeddings, Gumbel parameter noise, R-star parameterization, hierarchy or training-iteration budget. The Boolean interpreter uses explicit synchronous forward chaining, not HRI's fuzzy operations or its layer-update implementation. Although inference counts match the author settings, this is not a bit-for-bit reproduction of HRI evaluation. Rule resource limits (32 clauses, 12 invented predicates, three variables/two body atoms per clause) are explicit adapter choices. `paper_max_depth=4` and `paper_num_runs=10` record reference-paper settings; they do not claim these architectural constraints or ten runs are automatically enforced.

The runtime `scaleup` section extends the same deterministic generator to N=384 for training and N=512 for evaluation. Positive Horn inference runs to the finite least fixed point, rather than imposing a representation-dependent six/eight-round cutoff. The finite oracle checks all 512 evaluation atoms. Acceptance proves equality on that finite context, not unbounded Horn-program equivalence. CE on N=512 can expose evaluation-domain labels to the learner, an extension of the original learning method. This larger domain is an experimental extension, not the original paper setting. Hard Boolean MSE is recorded alongside accuracy; zero error satisfies the paper's 1e-4 threshold. Ten independent LLM runs can be requested explicitly with `--rerun 10`; this is not the original HRI optimization experiment.

Standard generation now reuses the regex sample schedule (default 3 → 6 → 12 → 24 → 48 → 96 → 192 → 384). CE starts with three examples and allows eight epochs, with up to eight CE per round by default. Both retain early stopping. Twelve distinct target tasks are registered locally; balanced selects six using structural difficulty tiers.


## Locally implemented task inventory

- graph_coloring: an outgoing neighbor has the same color.
- length: list length, with mutually recursive auxiliary construction.
- even_succ: even-index constants.
- predecessor: inverse successor relation.
- less_than: strict numerical order.
- fizz: multiples of three, including zero.
- buzz: multiples of five, including zero; preserves the author two-step/three-step background predicates.
- connectedness: nonempty directed reachability.
- cyclic: nodes on directed cycles.
- undirected_edge: symmetrized immediate edge relation.
- grandparent: composition of two father/mother parent links.
- has_father: existence of an incoming father link.

The vendored Task.py registers 19 synthetic task configurations plus WN. Some configurations share target semantics (EvenOdd/EvenSucc, Grandparent/Grandparent_NLM). We implement twelve different target definitions, not reruns or domain sizes. All twelve use 384 training atoms, with the unchanged regex geometric schedule and CE early stopping. Binary training contexts have 20 constants (384 of 400 possible atoms); evaluation has 24 constants and checks all 576 atoms. Unary training/evaluation has 384/512 constants and atoms. Generated labels are computed independently of the Horn interpreter.

Graph/family tasks use reproducible seeded sparse background constructions, including positive and negative labels, rather than the author's random graph/family distribution. Background facts for training and evaluation contexts are visible to the learner, while target labels and reference programs are hidden. Relational examples are ordered to interleave positive and negative labels, making the initial three-example prefix informative. These are scaleup adaptations of the author task definitions, not reproduction of the paper's sample distribution or optimizer. The numeric generator preserves the author zero/successor and Buzz offset backgrounds. No extra EvenOdd duplicate is counted.

Balanced selects predecessor/undirected_edge (easy), grandparent/graph_coloring (medium), fizz/length (hard). The structural inventory covers all 19 synthetic author configurations, but these tiers are not measured LLM difficulty. Full selection rationale: `analysis/ilp_integration/balanced_six_selection_zh.md`.
