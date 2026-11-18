#!/bin/bash
# Classification metrics at the 0.5 threshold (Table 5) for the four dataset/split settings, from saved test predictions.
source "$(dirname "$0")/common.sh"
for cfg in "kdd off 30" "kdd lr 30" "xt off 37" "xt lr 37"; do set -- $cfg
  DATA=$1 SPLIT=$2 H=$3 python3 src/extra_metrics.py | tee -a "$LOGS/extra_metrics.md"
done
