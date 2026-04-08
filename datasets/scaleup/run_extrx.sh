#!/usr/bin/env bash

# Scale up w/ standard
# #States=3, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=3, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(A[A-Za-z0-9#]*)|(An[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=3, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=0
python train_icl_gen.py --task_type extrx --regex 'x[A-Za-z]{2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(Al[A-Za-z0-9#]*)&~([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z0-9#]*){3,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{4}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(([A-Za-z]+)&(how))[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){4,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[0-9]{5}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z0-9#]*G)&([A-Za-z]+))[A-Za-z0-9#]*){5}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z]{0,5}))[A-Za-z0-9#]*){5,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{5}s' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(([0-9#])*[A-Za-z]+([0-9#])*){2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z]+#[A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Z0-9]{7}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z]*[A-Za-z0-9#]*){3}&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){7,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=0
python train_icl_gen.py --task_type extrx --regex 'P[a-z0-9]{7}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([0-9#])*[A-Za-z]*aought([0-9#])*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# Scale up w/ ce
# #States=3, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=3, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(A[A-Za-z0-9#]*)|(An[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=3, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=0
python train_icl_gen.py --task_type extrx --regex 'x[A-Za-z]{2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(Al[A-Za-z0-9#]*)&~([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z0-9#]*){3,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{4}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(([A-Za-z]+)&(how))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){4,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[0-9]{5}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z0-9#]*G)&([A-Za-z]+))[A-Za-z0-9#]*){5}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z]{0,5}))[A-Za-z0-9#]*){5,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{5}s' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(([0-9#])*[A-Za-z]+([0-9#])*){2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z]+#[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Z0-9]{7}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z]*[A-Za-z0-9#]*){3}&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){7,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=0
python train_icl_gen.py --task_type extrx --regex 'P[a-z0-9]{7}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([0-9#])*[A-Za-z]*aought([0-9#])*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# Scale up w/ ce w/o agentic
# #States=3, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{2}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=3, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(A[A-Za-z0-9#]*)|(An[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=3, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=0
python train_icl_gen.py --task_type extrx --regex 'x[A-Za-z]{2}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(Al[A-Za-z0-9#]*)&~([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z0-9#]*){3,}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{4}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(([A-Za-z]+)&(how))[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){4,}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[0-9]{5}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z0-9#]*G)&([A-Za-z]+))[A-Za-z0-9#]*){5}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z]{0,5}))[A-Za-z0-9#]*){5,}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{5}s' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '(([0-9#])*[A-Za-z]+([0-9#])*){2}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z]+#[A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=0
python train_icl_gen.py --task_type extrx --regex '[A-Z0-9]{7}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z]*[A-Za-z0-9#]*){3}&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){7,}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=0
python train_icl_gen.py --task_type extrx --regex 'P[a-z0-9]{7}' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=1
python train_icl_gen.py --task_type extrx --regex '([0-9#])*[A-Za-z]*aought([0-9#])*' --mkey gpt-oss --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode single_inference --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
