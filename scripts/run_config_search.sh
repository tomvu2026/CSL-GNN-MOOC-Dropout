#!/bin/bash
# Configuration search (Table 3): KDD Cup 2015, official split, SIG-Net-style features, seed 0, VALIDATION AUC only.
source "$(dirname "$0")/common.sh"; export DATA=kdd SPLIT=off
run() { name=$1; shift; python3 src/csl_gnn.py --variant user_c --feat signet --seed 0 --out tune_sig --tag "_$name" "$@" > "$LOGS/tune_$name.log" 2>&1; }
run base
run attn --attn 1
run temporal --temporal 1
run jk --jk 1
run all --attn 1 --temporal 1 --jk 1
run all_de02 --attn 1 --temporal 1 --jk 1 --dropedge 0.2
run all_wd1e4 --attn 1 --temporal 1 --jk 1 --wd 0.0001
run all_d02 --attn 1 --temporal 1 --jk 1 --dropout 0.2
run all_lr1e3 --attn 1 --temporal 1 --jk 1 --lr 0.001
run base_h192 --hidden 192
run attn_temporal --attn 1 --temporal 1
python3 - <<'PY'
import os, pandas as pd
r = os.environ.get("CSL_RESULTS", "results")
d = pd.read_csv(f"{r}/kdd_off/tune_sig/gnn_results.csv").drop_duplicates("model", keep="last").sort_values("val_auc", ascending=False)
print(d[["model", "val_auc", "epochs"]].to_string(index=False))
PY
