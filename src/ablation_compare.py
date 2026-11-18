"""Module ablation (official split, two-stage training; Table 7): full three-module model vs removing one module at a
time and vs the backbone without modules. Paired bootstrap on seed-averaged test predictions.
Usage: DATA=kdd|xt [H=30|37] python3 ablation_compare.py
Output: <results>/ablation_<DATA>.csv"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.environ.setdefault("SPLIT", "off"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K

H = int(os.environ.get("H", 37 if K.DATA == "xt" else 30)); te = K.split == 2; y = K.y[te]
avg = lambda t: np.mean([np.load(K.R + f"fullfit/preds/{t}_H{H}_s{s}.npy") for s in K.SEEDS], 0)
print(f"DATA={K.DATA} H={H}"); out = []
for fn, sx in [("SIG-Net-style", "_sig"), ("full", "")]:
    full = avg(f"user_c{sx}_plus_full"); print(f"[{fn}] full model AUC {roc_auc_score(y, full):.4f}")
    for ab, nm in [("_noattn", "attention"), ("_notcn", "temporal encoding"), ("_nojk", "layer combination"), ("_base", "all three (backbone only)")]:
        p = avg(f"user_c{sx}{ab}_full"); d, lo, hi = K.bootstrap(y, full, p)
        print(f"  removed: {nm:26s} AUC {roc_auc_score(y, p):.4f} | full - this: {d:+.4f} ({lo:+.4f}; {hi:+.4f})")
        out.append(dict(features=fn, removed=nm, auc_full=roc_auc_score(y, full), auc=roc_auc_score(y, p), delta=d, lo=lo, hi=hi))
pd.DataFrame(out).to_csv(os.path.join(K.RESULTS_ROOT, f"ablation_{K.DATA}.csv"), index=False)
