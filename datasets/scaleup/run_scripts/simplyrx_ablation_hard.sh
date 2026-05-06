#!/usr/bin/env bash

# Scale up w/ standard

# Stardepth=4
# regex_idx=1
# #States=3, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((a)* (b (c)*)))*+(a (c)*)))* ((a+a) (b)*)))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((a)*+a))* (((((c)* (c)*)+(b b)))* (b c))))* a))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(b (((c (((((b)*+b))* (c)*))*) (b b)))*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((a ((a)* c)) c) (((((((a)* c))* c))*+((c)* (b)*)))*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((c)* b) (((a+c) (c ((a (a)*))*)))*))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((b)* (((c)* b))*))* (((b+b)+a) a)))*+(c)*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((((b)* a))*+b) c))*+(a+c)))*+(b)*) ((b (c)*)+b))' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# regex_idx=2
# #States=3, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)*+a))* c))*+(c c))+((b+c) (a)*)))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((c (b)*)+(c)*) (((((((a)*+(c)*))*+a))* c))*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((b a) (b)*))* ((a)* (a+c))))* (a)*))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((a)* (((c+c)+c))*) ((((a ((a (c)*))*))*+b))*)' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((c)* (((b)*+a))*))* (((((a c) a)+b) a)+(b (b)*))))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((a)* (a c)) (((((a b)+(c)*))* c))*))* (b+c))' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((a (b (a+a))) ((((a (c)*))* a)+(a)*)))*+(b)*))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# regex_idx=3
# #States=3, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((b (((a+a) (((((a+c)+(c)*))*+b)+((a (a)*))*)))*))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((((b)* a))*+c) c))*+(b)*))* (b+c))' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)* b)+b)+a) c) a)+(((((a)* (a+b)))* b))*))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((b ((c)*+c)) (((((a a) ((a c)+(b)*)))* (b)*))*))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)* b))* (b)*))*+(a)*) ((((b)*+a) a) a)))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((a a)+c) (((b)* b))*))* ((a+b) (c b))))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)*+c))*+b))* ((a c) (c (a)*))) a))*' --mkey gpt-oss --use_reg --tot_train_size 3000 --start_size 3 --scale_factor 2.0 --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# Scale up w/ ce

# Stardepth=4
# regex_idx=1
# #States=3, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((a)* (b (c)*)))*+(a (c)*)))* ((a+a) (b)*)))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((a)*+a))* (((((c)* (c)*)+(b b)))* (b c))))* a))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(b (((c (((((b)*+b))* (c)*))*) (b b)))*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((a ((a)* c)) c) (((((((a)* c))* c))*+((c)* (b)*)))*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((c)* b) (((a+c) (c ((a (a)*))*)))*))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((b)* (((c)* b))*))* (((b+b)+a) a)))*+(c)*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((((b)* a))*+b) c))*+(a+c)))*+(b)*) ((b (c)*)+b))' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# regex_idx=2
# #States=3, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)*+a))* c))*+(c c))+((b+c) (a)*)))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((c (b)*)+(c)*) (((((((a)*+(c)*))*+a))* c))*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((b a) (b)*))* ((a)* (a+c))))* (a)*))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((a)* (((c+c)+c))*) ((((a ((a (c)*))*))*+b))*)' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((c)* (((b)*+a))*))* (((((a c) a)+b) a)+(b (b)*))))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((a)* (a c)) (((((a b)+(c)*))* c))*))* (b+c))' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((a (b (a+a))) ((((a (c)*))* a)+(a)*)))*+(b)*))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# regex_idx=3
# #States=3, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((b (((a+a) (((((a+c)+(c)*))*+b)+((a (a)*))*)))*))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=4, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((((((((b)* a))*+c) c))*+(b)*))* (b+c))' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=5, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)* b)+b)+a) c) a)+(((((a)* (a+b)))* b))*))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=6, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '(((b ((c)*+c)) (((((a a) ((a c)+(b)*)))* (b)*))*))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=7, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)* b))* (b)*))*+(a)*) ((((b)*+a) a) a)))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=8, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((a a)+c) (((b)* b))*))* ((a+b) (c b))))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# #States=9, Stardepth=4
python train_icl_gen.py --task_type simplyrx --regex '((((((((b)*+c))*+b))* ((a c) (c (a)*))) a))*' --mkey gpt-oss --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
