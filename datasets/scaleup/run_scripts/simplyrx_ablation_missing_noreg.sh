#!/usr/bin/env bash

# Missing ce/noreg agentic_reflection runs aligned to simplyrx_ablation.sh
# Source logs checked against:
# logs/scaleup/icl_gen_simplyrx/model=gpt-oss/ce/noreg/agentic_reflection

# #States=3, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((b)*+a)+c)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((b)* ((c a))*)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((c)*+b)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((c)* a) ((b)* (b)*))+((a)* ((a b) (a)*))))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((((c)* (c c))+(b)*))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((a ((b)*+c)))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=3, Stardepth=3
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((b)*+c)+b) (((c ((a)* c)))*+(c+c))))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((((((b)* a)+(b)*))* a))*+(b)*)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((b)* (((b (c)*))* (((b)*+b) b))))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((b)*+c) (((a)*+(b)*) (b)*))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((b (b)*) ((c)* a))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((c)* (a)*) (((c)*+a)+b))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((b a))*+a))*+(c)*)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((a)* (((((a)* (c)*))*+a) b)) (((a+a) c)+c))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((b (b)*) (c c)) ((((((c (a+c)))*+c) (b)*) (a)*))*)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=4, Stardepth=3
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '((c ((((a)* ((b (c)*))*)+((a+b)+(b)*)) (a)*)))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((((a)*+(a+b))+((c ((a (c c)) (b)*)))*))*' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((c a) (((b)* ((b)* (((b)* (b (a)*)))*)))*)' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '((((b)* c)+(c (a+b)))+((a+c) ((c)*+c)))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((((b)*+(b)*)+b) ((a)* ((a)*+b))) ((c)*+b))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((((c)* b)+c)+((b)* ((c)* b)))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=5, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((((c b) b))* ((c)*+a)))*+c)+(((b)*+a)+(c)*))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((((a)*+(c c)))* (b+c))+(a+c))' --mkey gpt-oss --retries 3 --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
