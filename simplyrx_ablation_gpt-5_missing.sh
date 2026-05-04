#!/usr/bin/env bash

# Scale up w/ ce

# #States=6, Stardepth=2
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((((a)*+(b)*)+a)+((a)* ((((c)* b) (((a c))*+a)))*))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=7, Stardepth=3
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((a ((((((b c)+(c)*)+(a)*) (a (c)*)) (a)*))*) a))*' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '((((c)* (a)*) ((c b)+c)) (((a)*+b)+(a)*))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((b)*+(c ((b a) ((b (a c)) (((b+c))*+c)))))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((((b)*+a) (b c))+(a)*) (((a+b) (b)*)+c))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((a)*+(b c)) (((c)* c) a))+(a c)))*' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((((((c (a+b)) (c)*) b))* (a (a+b)))+(a)*)+(b+b))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((a+c) (((((a (b a))+b))*+c))*)+(a ((b (a)*)+c)))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=8, Stardepth=3
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((((((c)*+b))* (b)*)+(b)*) a) b))*+((a+c) a))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((b (a c)) (((((b)*+(c)*))* ((c)* ((b b) ((a)*+a)))))*)' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((b ((((((((c)* b))*+(b)*) b))*+(a)*)+c)) a)' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=1
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '(((((a)*+(b c)) a)+((a)* ((c)* (a)*)))+((a b) b))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((c)* (((c (b+c)) ((a)* b))+(b)*)) (a+c))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '(((a b)+(b)*) ((((b a) a) (a+c))+a))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=2
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '((((b)* c))*+(((b+b) ((c b) (c a)))+(a (a)*)))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '(((((((a a))* b))*+(a (a)*))+(((a)* (a+c)))*) a)' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((((((c)* (b)*) a))*+((c b)+b))+(((b)* (c)*) ((b+b))*))' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup

# #States=9, Stardepth=3
# regex_idx=1
python train_icl_gen.py --task_type simplyrx --regex '((((b)* a) (((((c+c) (a c)) (c (c)*)))*+((b)*+c))))*' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=2
python train_icl_gen.py --task_type simplyrx --regex '((((((b)* ((b (a c)) b)) ((b)* c)))*+((b+c) (a+c))))*' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
# regex_idx=3
python train_icl_gen.py --task_type simplyrx --regex '((((((b ((c)* (c)*)))*+(a (c)*)) (((a)* c) c)))* b)' --mkey gpt5 --retries 3 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 250 --ce_clustered --reasoning_mode agentic_reflection --rerun 1 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup
