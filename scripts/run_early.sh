#!/bin/bash
# Early prediction (Table 8, Fig. 6): KDD Cup 2015, official split, H = 10 and 20 (H = 30 comes from run_main.sh kdd off).
source "$(dirname "$0")/common.sh"; export DATA=kdd SPLIT=off
for v in user_c user_p mlp; do for H in 10 20; do for f in signet full; do for s in 0 1 2 3 4; do
  python3 src/csl_gnn.py --variant $v --H $H --feat $f --seed $s $CSL_OPTS --tag _plus > "$LOGS/early_${v}_H${H}_${f}_s$s.log" 2>&1
done; done; done; done
python3 src/early_compare.py | tee "$LOGS/early_compare.txt"
