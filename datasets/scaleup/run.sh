#!/usr/bin/env bash

# Scale up w/ standard
# #States=3, Stardepth=0
python train_icl_gen.py --regex '(a ((a+a)+a))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=3, Stardepth=1
python train_icl_gen.py --regex '(((b)*+a)+c)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=3, Stardepth=2
python train_icl_gen.py --regex '(((((c)* a) ((b)* (b)*))+((a)* ((a b) (a)*))))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=3, Stardepth=3
python train_icl_gen.py --regex '(((((b)*+c)+b) (((c ((a)* c)))*+(c+c))))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=3, Stardepth=4
python train_icl_gen.py --regex '(((((((a)* (b (c)*)))*+(a (c)*)))* ((a+a) (b)*)))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=4, Stardepth=0
python train_icl_gen.py --regex '((c (b+b)) c)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=4, Stardepth=1
python train_icl_gen.py --regex '(((b)*+c) (((a)*+(b)*) (b)*))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=4, Stardepth=2
python train_icl_gen.py --regex '(((((b a))*+a))*+(c)*)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=4, Stardepth=3
python train_icl_gen.py --regex '((c ((((a)* ((b (c)*))*)+((a+b)+(b)*)) (a)*)))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=4, Stardepth=4
python train_icl_gen.py --regex '(((((((a)*+a))* (((((c)* (c)*)+(b b)))* (b c))))* a))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=5, Stardepth=0
python train_icl_gen.py --regex '(((c a) a) c)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=5, Stardepth=1
python train_icl_gen.py --regex '((((b)* c)+(c (a+b)))+((a+c) ((c)*+c)))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=5, Stardepth=2
python train_icl_gen.py --regex '(((((((c b) b))* ((c)*+a)))*+c)+(((b)*+a)+(c)*))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=5, Stardepth=3
python train_icl_gen.py --regex '(((((c)* ((b)* (b c))))* b))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=5, Stardepth=4
python train_icl_gen.py --regex '(b (((c (((((b)*+b))* (c)*))*) (b b)))*)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=6, Stardepth=0
python train_icl_gen.py --regex '((c (a (a b))) b)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=6, Stardepth=1
python train_icl_gen.py --regex '((((a b))*+a)+(c b))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=6, Stardepth=2
python train_icl_gen.py --regex '((((((a+a)+(c a)))* a))*+(c)*)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=6, Stardepth=3
python train_icl_gen.py --regex '((((((c)*+c))* ((a)* a)) (a ((b b) c))))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=6, Stardepth=4
python train_icl_gen.py --regex '(((a ((a)* c)) c) (((((((a)* c))* c))*+((c)* (b)*)))*)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=7, Stardepth=0
python train_icl_gen.py --regex '(((a c)+(c b)) ((a a) a))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=7, Stardepth=1
python train_icl_gen.py --regex '((((b)*+c) ((c)*+(c)*))+((b c))*)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=7, Stardepth=2
python train_icl_gen.py --regex '(((((((b+c)+a) (a ((b)*+b))) (b)*))*+c)+(b c))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=7, Stardepth=3
python train_icl_gen.py --regex '((((c)* ((b)* c)) ((c (a (((b)* b))*)) (b)*)))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=7, Stardepth=4
python train_icl_gen.py --regex '((((c)* b) (((a+c) (c ((a (a)*))*)))*))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=8, Stardepth=0
python train_icl_gen.py --regex '(((((a a)+a) c) (c (b+c))) ((c+c) a))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=8, Stardepth=1
python train_icl_gen.py --regex '((((c)* (a)*) ((c b)+c)) (((a)*+b)+(a)*))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=8, Stardepth=2
python train_icl_gen.py --regex '(((((a)*+(b c)) (((c)* c) a))+(a c)))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=8, Stardepth=3
python train_icl_gen.py --regex '(((((((((c)*+b))* (b)*)+(b)*) a) b))*+((a+c) a))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=8, Stardepth=4
python train_icl_gen.py --regex '((((((b)* (((c)* b))*))* (((b+b)+a) a)))*+(c)*)' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=9, Stardepth=0
python train_icl_gen.py --regex '(((((a b) ((a c)+a)) ((b+c)+c)) (a+b))+(c c))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=9, Stardepth=1
python train_icl_gen.py --regex '(((((a)*+(b c)) a)+((a)* ((c)* (a)*)))+((a b) b))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=9, Stardepth=2
python train_icl_gen.py --regex '((((b)* c))*+(((b+b) ((c b) (c a)))+(a (a)*)))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=9, Stardepth=3
python train_icl_gen.py --regex '((((b)* a) (((((c+c) (a c)) (c (c)*)))*+((b)*+c))))*' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
# #States=9, Stardepth=4
python train_icl_gen.py --regex '((((((((((b)* a))*+b) c))*+(a+c)))*+(b)*) ((b (c)*)+b))' --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0 --indir datasets/scaleup/regex_datasets --outdir logs/scaleup/icl_gen
