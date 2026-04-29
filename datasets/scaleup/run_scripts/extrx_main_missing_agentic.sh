#!/usr/bin/env bash

# Missing agentic_reflection runs from extrx_main.sh
# Source logs checked against:
# logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/agentic_reflection

# #States=3, Stardepth=0
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'S[0-9]' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Z][A-Za-z0-9]' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=1
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*z[A-Za-z0-9#]*)&~([A-Za-z0-9#]*q[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((de[A-Za-z0-9#]*)&([A-Za-z]+))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=2
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]*y[A-Za-z]*[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*0[A-Za-z0-9#]*){2,}[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=0
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{2,3}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z][0-9]{2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=1
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*[A-Za-z]*ing[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'X[A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z0-9#]*oa[A-Za-z0-9#]*))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=2
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*){3,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){3,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=0
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{4}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'agde' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=1
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{2}[A-Za-z0-9#]*){2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '~([A-Za-z0-9#]*[A-Za-z0-9#]{4,}[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=2
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*c[A-Za-z0-9#]*){4,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*ly[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=0
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'AEIOU' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'Ex[0-9]{3}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
