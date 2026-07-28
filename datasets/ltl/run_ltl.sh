#!/usr/bin/env bash

# 1 variable, depth 3, example 1
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '((p R (p | p)) | F(X(p)))' --variables p --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '((p R (p | p)) | F(X(p)))' --variables p --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 1 variable, depth 4, example 1
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '!(G(((p <-> p) R (p | p))))' --variables p --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '!(G(((p <-> p) R (p | p))))' --variables p --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 1 variable, depth 3, example 2
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(!((p | p)) -> G(p))' --variables p --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(!((p | p)) -> G(p))' --variables p --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 1 variable, depth 4, example 2
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(X(((p R p) <-> X(p))) | (F(p) & (p & p)))' --variables p --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(X(((p R p) <-> X(p))) | (F(p) & (p & p)))' --variables p --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 1 variable, depth 3, example 3
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(((p & p) & (p R p)) | F(p))' --variables p --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(((p & p) & (p R p)) | F(p))' --variables p --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 1 variable, depth 4, example 3
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(((p U p) <-> ((p -> p) <-> (p <-> p))) R X(G((p <-> p))))' --variables p --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(((p U p) <-> ((p -> p) <-> (p <-> p))) R X(G((p <-> p))))' --variables p --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 2 variables, depth 3, example 1
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(q U F((p <-> q)))' --variables p q --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(q U F((p <-> q)))' --variables p q --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 2 variables, depth 4, example 1
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(q U (((p R q) R q) -> ((p | p) R (p | q))))' --variables p q --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(q U (((p R q) R q) -> ((p | p) R (p | q))))' --variables p q --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 2 variables, depth 3, example 2
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '((q -> (q & p)) & p)' --variables p q --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '((q -> (q & p)) & p)' --variables p q --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 2 variables, depth 4, example 2
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '!(X(X((p U q))))' --variables p q --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '!(X(X((p U q))))' --variables p q --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 2 variables, depth 3, example 3
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(((p -> p) -> (q | p)) -> q)' --variables p q --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(((p -> p) -> (q | p)) -> q)' --variables p q --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 2 variables, depth 4, example 3
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(((p R X(p)) | (F(p) | !(p))) & q)' --variables p q --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(((p R X(p)) | (F(p) | !(p))) & q)' --variables p q --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 3 variables, depth 3, example 1
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(((p -> r) -> (q U p)) R !(X(q)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(((p -> r) -> (q U p)) R !(X(q)))' --variables p q r --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 3 variables, depth 4, example 1
# CE
python train_icl_gen_ltl.py --task_type ltl --formula 'F(((X(q) & (q | p)) | (G(p) -> r)))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula 'F(((X(q) & (q | p)) | (G(p) -> r)))' --variables p q r --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 3 variables, depth 3, example 2
# CE
python train_icl_gen_ltl.py --task_type ltl --formula 'G(((q <-> p) & r))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula 'G(((q <-> p) & r))' --variables p q r --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 3 variables, depth 4, example 2
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '((G(p) | ((r U r) -> q)) -> !((r -> (r & p))))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '((G(p) | ((r U r) -> q)) -> !((r -> (r & p))))' --variables p q r --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 3 variables, depth 3, example 3
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(F((q U r)) U p)' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(F((q U r)) U p)' --variables p q r --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# 3 variables, depth 4, example 3
# CE
python train_icl_gen_ltl.py --task_type ltl --formula '(G(((r <-> r) <-> (q & p))) <-> X(((r R q) -> (q <-> q))))' --variables p q r --mkey gpt-oss --use_reg --use_ce --ce_epochs 10 --ce_batch_size 16 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula '(G(((r <-> r) <-> (q & p))) <-> X(((r R q) -> (q <-> q))))' --variables p q r --mkey gpt-oss --use_reg --tot_train_size 384 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
