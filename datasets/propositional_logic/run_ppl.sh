#!/usr/bin/env bash

# Propositional logic: CE and standard runs
# 25 five-variable formulas selected with seed 42; 5 formulas per syntax depth.

# FormulaDepth=3
# formula_idx=1
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((s | t) <-> r) | ((q -> p) <-> (s & t)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((s | t) <-> r) | ((q -> p) <-> (s & t)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((p -> q) | (r & q)) & ((t & p) <-> (s -> t)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((p -> q) | (r & q)) & ((t & p) <-> (s -> t)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((p & r) -> q) -> ((s -> t) & (p <-> s)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((p & r) -> q) -> ((s -> t) & (p <-> s)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(r | ((t <-> p) | (q | s)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(r | ((t <-> p) | (q | s)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((s | (t & p)) <-> (q & r))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((s | (t & p)) <-> (q & r))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=4
# formula_idx=1
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '!((t & r) & ((q & q) <-> (s -> p)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '!((t & r) & ((q & q) <-> (s -> p)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((r <-> s) -> (p <-> q)) <-> (!(q & q) & ((t -> q) & q)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((r <-> s) -> (p <-> q)) <-> (!(q & q) & ((t -> q) & q)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((r <-> q) <-> (!r <-> r)) <-> ((p | p) | !(t & s)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((r <-> q) <-> (!r <-> r)) <-> ((p | p) | !(t & s)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((t | r) <-> (p <-> p)) -> (s & (q | (s -> t))))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((t | r) <-> (p <-> p)) -> (s & (q | (s -> t))))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((!q | !p) -> (((s <-> q) -> q) | ((t -> r) <-> !r)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((!q | !p) -> (((s <-> q) -> q) | ((t -> r) <-> !r)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=5
# formula_idx=1
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(q <-> ((p | (r | (s & r))) | (t & s)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(q <-> ((p | (r | (s & r))) | (t & s)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '!(((p | t) -> (r & s)) & ((!r <-> t) | q))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '!(((p | t) -> (r & s)) & ((!r <-> t) | q))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((!p & !!p) -> (!(s & q) & !(t & r))) & !(q & ((s -> p) <-> !q)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((!p & !!p) -> (!(s & q) & !(t & r))) & !(q & ((s -> p) <-> !q)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((((q | t) -> t) | (((q | p) -> (r -> s)) | (r & s))) | !(!r | (r | p)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((((q | t) -> t) | (((q | p) -> (r -> s)) | (r & s))) | !(!r | (r | p)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '!(!((q | p) | !t) & (r & (q <-> s)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '!(!((q | p) | !t) & (r & (q <-> s)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=6
# formula_idx=1
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((!((p | t) <-> s) <-> (((s | p) <-> (s | t)) <-> ((t | t) | (p <-> r)))) -> !!!q) & ((r & p) | (q -> r)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((!((p | t) <-> s) <-> (((s | p) <-> (s | t)) <-> ((t | t) | (p <-> r)))) -> !!!q) & ((r & p) | (q -> r)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((((p | t) <-> (!r | (r | p))) <-> ((!q | ((s | t) <-> !r)) <-> ((!q -> q) <-> !(t | q)))) -> ((!s -> r) -> (p | r)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((((p | t) <-> (!r | (r | p))) <-> ((!q | ((s | t) <-> !r)) <-> ((!q -> q) <-> !(t | q)))) -> ((!s -> r) -> (p | r)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((!(p & s) -> ((!(r & q) | (t -> t)) -> r)) <-> !!((q & r) <-> !p))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((!(p & s) -> ((!(r & q) | (t -> t)) -> r)) <-> !!((q & r) <-> !p))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((q & q) <-> (s -> p)) -> ((((t -> q) | t) -> (q <-> p)) & (!!!t | (!!s <-> (s | r)))))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((q & q) <-> (s -> p)) -> ((((t -> q) | t) -> (q <-> p)) & (!!!t | (!!s <-> (s | r)))))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((((q | r) <-> (!q -> ((q -> q) -> (s | p)))) | (t <-> s)) & !!((t | p) | s))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((((q | r) <-> (!q -> ((q -> q) -> (s | p)))) | (t <-> s)) & !!((t | p) | s))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic

# FormulaDepth=7
# formula_idx=1
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '!(((p <-> !(q -> q)) | !(!(s & s) <-> (s -> (p <-> r)))) <-> (!!s <-> (t | t)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '!(((p <-> !(q -> q)) | !(!(s & s) <-> (s -> (p <-> r)))) <-> (!!s <-> (t | t)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=2
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(((!s | r) -> !(q -> q)) & (!(r & p) -> (((q <-> r) -> (((s -> q) <-> q) <-> (t -> s))) & (((r -> t) | (r <-> p)) -> ((!s <-> (s <-> t)) -> ((t -> r) & r))))))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(((!s | r) -> !(q -> q)) & (!(r & p) -> (((q <-> r) -> (((s -> q) <-> q) <-> (t -> s))) & (((r -> t) | (r <-> p)) -> ((!s <-> (s <-> t)) -> ((t -> r) & r))))))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=3
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((q | !(!(p -> (p & r)) -> !((q & p) <-> (p | r)))) & (!(!r <-> (t & q)) | (q & (s & t))))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((q | !(!(p -> (p & r)) -> !((q & p) <-> (p | r)))) & (!(!r <-> (t & q)) | (q & (s & t))))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=4
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '(!(((r <-> ((s -> t) & (q | p))) <-> (p <-> !(q | r))) -> (!t <-> p)) & ((q -> p) | !(p -> t)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '(!(((r <-> ((s -> t) & (q | p))) <-> (p <-> !(q | r))) -> (!t <-> p)) & ((q -> p) | !(p -> t)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# formula_idx=5
# CE
python train_icl_gen_ppl.py --task_type ppl --formula '((p & r) -> ((!((r & t) | (!p | (q <-> t))) | q) <-> (!(t -> q) -> s)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 5 --ce_batch_size 8 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
# Standard (no CE)
python train_icl_gen_ppl.py --task_type ppl --formula '((p & r) -> ((!((r & t) | (!p | (q <-> t))) | q) <-> (!(t -> q) -> s)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 32 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/propositional_logic/truth_tables --outdir logs/propositional_logic
