#!/bin/bash
# Everything in the paper, in order (several days on one GPU). Data must be prepared first (scripts/prepare_data.sh).
set -e; cd "$(dirname "$0")"
bash run_config_search.sh
for d in kdd xt; do for s in off lr; do bash run_main.sh $d $s; done; done
bash run_ablation.sh
bash run_early.sh
bash run_extra_metrics.sh
cd ..; for d in kdd xt; do DATA=$d python3 src/data_stats.py; done
