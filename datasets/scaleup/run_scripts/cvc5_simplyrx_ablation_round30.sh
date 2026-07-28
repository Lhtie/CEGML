#!/usr/bin/env bash
set -euo pipefail

/home/lhtie/anaconda3/envs/cegml/bin/python -B baselines/run_cvc5.py \
  --max_train_examples 20 \
  --max_cegis_rounds 30 \
  --max_states 12 \
  --out logs/summary/cvc5_simplyrx_ablation_cegis_round30.json
