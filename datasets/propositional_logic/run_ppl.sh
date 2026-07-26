#!/usr/bin/env bash

# Propositional logic scale-up w/ counterexamples
# 45 formulas selected with seed 42; 9 formulas per syntax depth.

# FormulaDepth=1
# formula_idx=1
python train_icl_gen_ppl.py --task_type ppl --formula '(p -> r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
python train_icl_gen_ppl.py --task_type ppl --formula '(q -> q)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
python train_icl_gen_ppl.py --task_type ppl --formula '(r | p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
python train_icl_gen_ppl.py --task_type ppl --formula '!q' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
python train_icl_gen_ppl.py --task_type ppl --formula '(q | p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=6
python train_icl_gen_ppl.py --task_type ppl --formula '(r -> r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=7
python train_icl_gen_ppl.py --task_type ppl --formula '(r -> q)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=8
python train_icl_gen_ppl.py --task_type ppl --formula '(p | r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=9
python train_icl_gen_ppl.py --task_type ppl --formula '(r | r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=2
# formula_idx=1
python train_icl_gen_ppl.py --task_type ppl --formula '((r & r) -> (q | r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
python train_icl_gen_ppl.py --task_type ppl --formula '((q | q) | (q & r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
python train_icl_gen_ppl.py --task_type ppl --formula '(r -> (r -> p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
python train_icl_gen_ppl.py --task_type ppl --formula '((r -> q) <-> (p & p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
python train_icl_gen_ppl.py --task_type ppl --formula '(!p & (q & p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=6
python train_icl_gen_ppl.py --task_type ppl --formula '((r -> q) & q)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=7
python train_icl_gen_ppl.py --task_type ppl --formula '((p | r) | (q & q))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=8
python train_icl_gen_ppl.py --task_type ppl --formula '((p | r) & !p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=9
python train_icl_gen_ppl.py --task_type ppl --formula '((q <-> r) | p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=3
# formula_idx=1
python train_icl_gen_ppl.py --task_type ppl --formula '(((q & r) <-> (r <-> p)) <-> (p <-> r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
python train_icl_gen_ppl.py --task_type ppl --formula '(((p | p) <-> (p -> p)) <-> !(p -> r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
python train_icl_gen_ppl.py --task_type ppl --formula '(r <-> (q | (q -> q)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
python train_icl_gen_ppl.py --task_type ppl --formula '(((r | r) <-> p) | p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
python train_icl_gen_ppl.py --task_type ppl --formula '(((p & q) & q) <-> (!p & p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=6
python train_icl_gen_ppl.py --task_type ppl --formula '(p & (q <-> (r | p)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=7
python train_icl_gen_ppl.py --task_type ppl --formula '(((r & r) | (q -> r)) -> (!r <-> !p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=8
python train_icl_gen_ppl.py --task_type ppl --formula '(((q <-> q) | (r -> r)) | !r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=9
python train_icl_gen_ppl.py --task_type ppl --formula '((p & r) | (!r <-> (q -> r)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=4
# formula_idx=1
python train_icl_gen_ppl.py --task_type ppl --formula '(((r & p) | p) -> (((r & p) <-> q) & (p & q)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
python train_icl_gen_ppl.py --task_type ppl --formula '(!(q <-> (r | q)) | !(!p | (r | r)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
python train_icl_gen_ppl.py --task_type ppl --formula '!(r -> ((r <-> r) <-> r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
python train_icl_gen_ppl.py --task_type ppl --formula '((((r | r) <-> (q -> q)) -> q) -> !(p & p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
python train_icl_gen_ppl.py --task_type ppl --formula '((((r -> q) <-> (p & r)) <-> ((r -> r) | q)) -> !p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=6
python train_icl_gen_ppl.py --task_type ppl --formula '((!(p & r) & (r | p)) | (r | q))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=7
python train_icl_gen_ppl.py --task_type ppl --formula '!!(!r | r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=8
python train_icl_gen_ppl.py --task_type ppl --formula '(((!r -> (r -> q)) & ((r -> p) & (q | r))) -> !!r)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=9
python train_icl_gen_ppl.py --task_type ppl --formula '(!!(p <-> r) & ((p -> p) -> (q & q)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=5
# formula_idx=1
python train_icl_gen_ppl.py --task_type ppl --formula '(r <-> (p -> ((r & (p <-> p)) | !(r | p))))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
python train_icl_gen_ppl.py --task_type ppl --formula '(!(p -> (q & (p -> p))) <-> (((q -> r) & (!p | (p | q))) <-> (!(p <-> p) <-> ((r -> r) <-> p))))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
python train_icl_gen_ppl.py --task_type ppl --formula '((!((q -> p) & r) <-> (!(q -> q) & ((p <-> q) -> (r <-> p)))) | q)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
python train_icl_gen_ppl.py --task_type ppl --formula '(!(((r <-> r) -> p) <-> !p) | (r | !q))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
python train_icl_gen_ppl.py --task_type ppl --formula '(((((r -> r) | r) <-> (!p <-> !r)) <-> q) | (((q & q) <-> r) <-> (p <-> r)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=6
python train_icl_gen_ppl.py --task_type ppl --formula '((p | p) <-> (((r <-> q) -> (r -> q)) & ((q & !p) | (!r <-> (p <-> q)))))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=7
python train_icl_gen_ppl.py --task_type ppl --formula '((!!!q & ((r <-> p) -> !q)) -> (q <-> p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=8
python train_icl_gen_ppl.py --task_type ppl --formula '((p & r) | ((q -> r) -> !!!p))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=9
python train_icl_gen_ppl.py --task_type ppl --formula '!(!!!q | !(q <-> !r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 8 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
