#!/bin/bash
# Main experiments for one dataset and split: CSL-GNN and the controls (both feature modes, 5 seeds, two-stage training)
# and the SIG-Net re-runs (labeled / unlabeled, 5 runs).
# Usage: bash scripts/run_main.sh kdd|xt off|lr
source "$(dirname "$0")/common.sh"
export DATA=${1:-kdd} SPLIT=${2:-off}; H=$([ "$DATA" = xt ] && echo 37 || echo 30); CK=$([ "$DATA" = xt ] && echo 1 || echo 0)
VARIANTS="user_c user_p mlp"; [ "$SPLIT" = off ] && VARIANTS="user_c user_p res mlp"   # 'res' was run on the official split only
for v in $VARIANTS; do for f in signet full; do for s in 0 1 2 3 4; do
  python3 src/csl_gnn.py --variant $v --H $H --feat $f --seed $s $CSL_OPTS --ckpt $CK --tag _plus > "$LOGS/${DATA}_${SPLIT}_${v}_${f}_s$s.log" 2>&1
done; done; done
for m in nolabel label; do
  STUDENT_EDGE=$m python3 src/signet/signet_pyg.py --runs 5 --workers 8 > "$LOGS/${DATA}_${SPLIT}_signet_$m.log" 2>&1
done
H=$H python3 src/compare.py | tee "$LOGS/${DATA}_${SPLIT}_compare.txt"
