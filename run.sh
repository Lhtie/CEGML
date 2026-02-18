# generation setting:
# noreg; std
# python train_icl_gen.py --regex "((a+b)c(a+b))*" --mkey gpt5 --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "((a a)*+(b b)*+(c c)*)*" --mkey gpt5 --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "(b(b+c))* a* (cc)*" --mkey gpt5 --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "((a+b)(b+c)(a+c)a(b+c)(a+b+c))*" --mkey gpt5 --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "(c b(b+a c)a b)* (a+(b+c)*a)*" --mkey gpt5 --tot_train_size 1280 --start_size 3 --scale_factor 2.0

# noreg; ce
# python train_icl_gen.py --regex "((a+b)c(a+b))*" --mkey gpt5 --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "((a a)*+(b b)*+(c c)*)*" --mkey gpt5 --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "(b(b+c))* a* (cc)*" --mkey gpt5 --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "((a+b)(b+c)(a+c)a(b+c)(a+b+c))*" --mkey gpt5 --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "(c b(b+a c)a b)* (a+(b+c)*a)*" --mkey gpt5 --use_ce --ce_epochs 12 --ce_batch_size 128

# reg; std
# python train_icl_gen.py --regex "((a+b)c(a+b))*" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "((a a)*+(b b)*+(c c)*)*" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "(b(b+c))* a* (cc)*" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "((a+b)(b+c)(a+c)a(b+c)(a+b+c))*" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "(c b(b+a c)a b)* (a+(b+c)*a)*" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "((a* b)* c)*" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0
# python train_icl_gen.py --regex "((a*(b+c))*c + c((a+c)*b)*)* a" --mkey gpt5 --use_reg --tot_train_size 1280 --start_size 3 --scale_factor 2.0

# reg; ce
# python train_icl_gen.py --regex "((a+b)c(a+b))*" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "((a a)*+(b b)*+(c c)*)*" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "(b(b+c))* a* (cc)*" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "((a+b)(b+c)(a+c)a(b+c)(a+b+c))*" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "(c b(b+a c)a b)* (a+(b+c)*a)*" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "((a* b)* c)*" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 128
# python train_icl_gen.py --regex "((a*(b+c))*c + c((a+c)*b)*)* a" --mkey gpt5 --use_reg --use_ce --ce_epochs 12 --ce_batch_size 1024

# python train_icl_gen_extrx.py --regex "[A-Za-z0-9#]*z[A-Za-z]*[A-Za-z0-9#]*" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
# python train_icl_gen_extrx.py --regex "[A-Za-z0-9#]*z[A-Za-z]*[A-Za-z0-9#]*" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250
# python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){2,}" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
# python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*[A-Za-z]+[A-Za-z0-9#]*){2,}" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250
# python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*x[A-Za-z0-9#]*)&([A-Za-z0-9#]*y[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
# python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*x[A-Za-z0-9#]*)&([A-Za-z0-9#]*y[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250
# python train_icl_gen_extrx.py --regex "~([A-Za-z0-9#]*[A-Za-z0-9#]{4,}[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
# python train_icl_gen_extrx.py --regex "~([A-Za-z0-9#]*[A-Za-z0-9#]{4,}[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250

python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*th[A-Za-z]*[A-Za-z0-9#]*)&([A-Za-z0-9#]*7[0-9]*[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*th[A-Za-z]*[A-Za-z0-9#]*)&([A-Za-z0-9#]*7[0-9]*[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250
python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]{5})&~([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*)" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]{5})&~([A-Za-z0-9#]*[AEIOUaeiou][A-Za-z0-9#]*)" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250
python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*Mr[A-Za-z0-9#]*)&([A-Za-z0-9#]*Mrs[A-Za-z0-9#]*)&~([A-Za-z0-9#]*((Ms)|(Miss))[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --rerun 1 --tot_train_size 3000 --start_size 3 --scale_factor 2.0
python train_icl_gen_extrx.py --regex "([A-Za-z0-9#]*Mr[A-Za-z0-9#]*)&([A-Za-z0-9#]*Mrs[A-Za-z0-9#]*)&~([A-Za-z0-9#]*((Ms)|(Miss))[A-Za-z0-9#]*)" --mkey gpt5 --use_reg --use_ce --rerun 1 --ce_epochs 12 --ce_batch_size 250
