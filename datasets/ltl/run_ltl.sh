#!/usr/bin/env bash

# CE
python train_icl_gen_ltl.py --task_type ltl --formula 'G((p & X q) -> F(r & (s U t)))' --variables p q r s t --mkey gpt-oss --use_reg --use_ce --ce_epochs 6 --ce_batch_size 24 --ce_generation_mode search --reasoning_mode agentic_reflection --retries 3 --rerun 1 --tot_train_size 128 --eval_size 128 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl

# Standard (no CE)
python train_icl_gen_ltl.py --task_type ltl --formula 'G((p & X q) -> F(r & (s U t)))' --variables p q r s t --mkey gpt-oss --use_reg --tot_train_size 128 --eval_size 128 --start_size 3 --scale_factor 2.0 --rerun 1 --min_trace_length 1 --max_trace_length 8 --indir datasets/ltl/traces --outdir logs/ltl
