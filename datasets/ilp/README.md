# HRI EvenSucc ILP setting

Source: [Glanois et al., Neuro-Symbolic Hierarchical Rule Induction, ICML 2022](https://proceedings.mlr.press/v162/glanois22a.html), Appendix B.1/E, and the [author repository](https://github.com/claireaoi/hierarchical-rule-induction).

The checked-in author files are pinned to revision `1a3a0465a5a5305ad8fec958901710dc6ead76e4` under `source/`, with the author's CC0 license. They are provenance references, not imported by the benchmark runtime. `even_succ.json` records their SHA-256 hashes. Regenerate with:

```bash
python datasets/ilp/prepare.py
```

The deterministic facts and labels reproduce `EvenSuccDataset.getData`: constants `0..N-1`, `zero(0)`, consecutive `succ(i,i+1)`, and positive even-index target atoms. Training uses N=11 and six inference rounds; evaluation uses N=15 and eight rounds, matching `HRI/EvenSucc.py`. All 11/15 unary target atoms, including negatives, are retained. The vendored EvenOdd generator has the same target, but is not registered as a second independent setting.

The reference program is teacher/verification data, not included in learner prompts. Learners see the background declaration, labeled ground atoms, hypothesis syntax, and requested inference budgets. Hypotheses are positive unary/binary function-free Horn clauses with recursion and auxiliary predicate invention, not Python classifiers. Numeric constants in rules and arithmetic shortcuts are excluded. `true`/`false` are exposed as nullary Boolean literals; the only supplied relational predicates are `zero/1` and `succ/2`.

Our method is an LLM+CE adaptation. It does not reproduce HRI's differentiable optimizer, embeddings, Gumbel parameter noise, R-star parameterization, hierarchy or training-iteration budget. The Boolean interpreter uses explicit synchronous forward chaining, not HRI's fuzzy operations or its layer-update implementation. Although inference counts match the author settings, this is not a bit-for-bit reproduction of HRI evaluation. Rule resource limits (32 clauses, 12 invented predicates, three variables/two body atoms per clause) are explicit adapter choices. `paper_max_depth=4` and `paper_num_runs=10` record reference-paper settings; they do not claim these architectural constraints or ten runs are automatically enforced.

The finite oracle compares all 15 target atoms under the evaluation background and eight-round semantics. Its true verdict is scoped to this finite context/budget and is not unbounded Horn-program equivalence. CE on N=15 can expose evaluation-domain labels to the learner, an extension of the original learning method. Hard Boolean MSE is recorded alongside accuracy; zero error satisfies the paper's 1e-4 threshold. Ten independent LLM runs can be requested explicitly with `--rerun 10`; this is not the original HRI optimization experiment.
