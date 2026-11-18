#!/bin/bash
# Module ablation (Table 7, Fig. 5): official split, two-stage training, both datasets.
source "$(dirname "$0")/common.sh"; export SPLIT=off
for D in kdd xt; do
  export DATA=$D; H=$([ $D = kdd ] && echo 30 || echo 37); CK=$([ $D = xt ] && echo 1 || echo 0)
  for cfg in "0 1 1 _noattn" "1 0 1 _notcn" "1 1 0 _nojk" "0 0 0 _base"; do set -- $cfg
    for f in signet full; do for s in 0 1 2 3 4; do
      python3 src/csl_gnn.py --variant user_c --H $H --feat $f --seed $s --attn $1 --temporal $2 --jk $3 --wd 0.0001 \
        --fullfit 1 --ckpt $CK --out fullfit --tag $4 > "$LOGS/abl_${D}$4_${f}_s$s.log" 2>&1
    done; done
  done
  H=$H python3 src/ablation_compare.py | tee "$LOGS/ablation_$D.txt"
done
