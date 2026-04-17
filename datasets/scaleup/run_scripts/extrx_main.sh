#!/usr/bin/env bash

# Scale up w/ standard
# #States=3, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'S[0-9]' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Z][A-Za-z0-9]' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(A[A-Za-z0-9#]*)|(An[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*z[A-Za-z0-9#]*)&~([A-Za-z0-9#]*q[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((de[A-Za-z0-9#]*)&([A-Za-z]+))[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]*y[A-Za-z]*[A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*0[A-Za-z0-9#]*){2,}[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex 'x[A-Za-z]{2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{2,3}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z][0-9]{2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(Al[A-Za-z0-9#]*)&~([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*[A-Za-z]*ing[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'X[A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z0-9#]*oa[A-Za-z0-9#]*))[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z0-9#]*){3,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*){3,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){3,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{4}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{4}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'agde' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(([A-Za-z]+)&(how))[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{2}[A-Za-z0-9#]*){2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '~([A-Za-z0-9#]*[A-Za-z0-9#]{4,}[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){4,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*c[A-Za-z0-9#]*){4,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*ly[A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[0-9]{5}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'AEIOU' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'Ex[0-9]{3}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z0-9#]*G)&([A-Za-z]+))[A-Za-z0-9#]*){5}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((run)|(hat))[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[0-9#]*[A-Za-z]+(([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*)&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*))[A-Za-z]+[0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z]{0,5}))[A-Za-z0-9#]*){5,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{1,3}[A-Za-z0-9#]*){5,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*[A-Za-z][A-Za-z0-9#]*){5,}[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{5}s' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{6}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'Beaker' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(([0-9#])*[A-Za-z]+([0-9#])*){2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]{5})&~([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'xyz([A-Za-z0-9#]*xyz)?' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z]+#[A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]*c[A-Za-z0-9#]*){6,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{3,}[A-Za-z0-9#]*){2,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Z0-9]{7}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'country' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Z]{6}8' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z]*[A-Za-z0-9#]*){3}&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((er)|(let))[A-Za-z0-9#]*wig[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*a[A-Za-z0-9#]*)&([A-Za-z0-9#]*b[A-Za-z0-9#]*)&([A-Za-z0-9#]*c[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){7,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){7,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z0-9#]*){7,}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex 'P[a-z0-9]{7}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'Facebook' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z]{4}){2}' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([0-9#])*[A-Za-z]*aought([0-9#])*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*ug[A-Za-z0-9#]*){4}[A-Za-z0-9#]*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '~([A-Za-z0-9#]*old[A-Za-z0-9#]*)&([A-Za-z0-9#]*ion[A-Za-z0-9#]*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# Scale up w/ ce
# #States=3, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'S[0-9]' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Z][A-Za-z0-9]' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(A[A-Za-z0-9#]*)|(An[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*z[A-Za-z0-9#]*)&~([A-Za-z0-9#]*q[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((de[A-Za-z0-9#]*)&([A-Za-z]+))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]*y[A-Za-z]*[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*0[A-Za-z0-9#]*){2,}[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex 'x[A-Za-z]{2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{2,3}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z][0-9]{2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(Al[A-Za-z0-9#]*)&~([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*[A-Za-z]*ing[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'X[A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z0-9#]*oa[A-Za-z0-9#]*))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z0-9#]*){3,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*){3,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){3,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{4}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{4}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'agde' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(([A-Za-z]+)&(how))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{2}[A-Za-z0-9#]*){2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '~([A-Za-z0-9#]*[A-Za-z0-9#]{4,}[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){4,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*c[A-Za-z0-9#]*){4,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*ly[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[0-9]{5}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'AEIOU' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'Ex[0-9]{3}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z0-9#]*G)&([A-Za-z]+))[A-Za-z0-9#]*){5}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((run)|(hat))[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[0-9#]*[A-Za-z]+(([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*)&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*))[A-Za-z]+[0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=6, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*(([A-Za-z]+)&([A-Za-z]{0,5}))[A-Za-z0-9#]*){5,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{1,3}[A-Za-z0-9#]*){5,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*[A-Za-z][A-Za-z0-9#]*){5,}[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]{5}s' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z]{6}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'Beaker' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '(([0-9#])*[A-Za-z]+([0-9#])*){2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]{5})&~([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex 'xyz([A-Za-z0-9#]*xyz)?' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*#[A-Za-z]+#[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]*c[A-Za-z0-9#]*){6,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]{3,}[A-Za-z0-9#]*){2,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '[A-Z0-9]{7}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'country' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '[A-Z]{6}8' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z]*[A-Za-z0-9#]*){3}&([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*((er)|(let))[A-Za-z0-9#]*wig[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*a[A-Za-z0-9#]*)&([A-Za-z0-9#]*b[A-Za-z0-9#]*)&([A-Za-z0-9#]*c[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[0-9][A-Za-z0-9#]*){7,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){7,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z0-9#]*[A-Za-z][A-Za-z0-9#]*){7,}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=0
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex 'P[a-z0-9]{7}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex 'Facebook' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '([A-Za-z]{4}){2}' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type extrx --regex '([0-9#])*[A-Za-z]*aought([0-9#])*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type extrx --regex '[A-Za-z0-9#]*([A-Za-z0-9#]*ug[A-Za-z0-9#]*){4}[A-Za-z0-9#]*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type extrx --regex '~([A-Za-z0-9#]*old[A-Za-z0-9#]*)&([A-Za-z0-9#]*ion[A-Za-z0-9#]*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

