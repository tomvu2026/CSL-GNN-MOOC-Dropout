"""Early prediction on KDD Cup 2015, official split (Table 8): AUC of the averaged prediction at H = 10, 20, 30 and paired
bootstrap of CSL-GNN against the placebo graph and the no-graph MLP.
Usage: python3 early_compare.py   (DATA=kdd, SPLIT=off)
Output: <results>/early_compare.csv"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.environ.setdefault("DATA", "kdd"); os.environ.setdefault("SPLIT", "off"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K

te = K.split == 2; y = K.y[te]
avg = lambda f: np.mean([np.load(f.format(s=s)) for s in K.SEEDS], 0)
out = []
for H in [10, 20, 30]:
    P = {}
    for fn, sx in [("SIG-Net-style", "_sig"), ("full", "")]:
        for v, nm in [("user_c", "CSL-GNN"), ("user_p", "Placebo graph"), ("mlp", "MLP")]:
            P[f"{nm} ({fn})"] = avg(K.R + f"fullfit/preds/{v}{sx}_plus_full_H{H}_s{{s}}.npy")
    print(f"=== H={H}")
    for k, v in P.items():
        print(f"  {k:28s} AUC {roc_auc_score(y, v):.4f}"); out.append(dict(H=H, comparison=k, auc=roc_auc_score(y, v)))
    for fn in ["SIG-Net-style", "full"]:
        for b in ["Placebo graph", "MLP"]:
            d, lo, hi = K.bootstrap(y, P[f"CSL-GNN ({fn})"], P[f"{b} ({fn})"])
            print(f"  CSL-GNN ({fn}) - {b} ({fn}): {d:+.4f} ({lo:+.4f}; {hi:+.4f})")
            out.append(dict(H=H, comparison=f"CSL-GNN ({fn}) - {b} ({fn})", delta=d, lo=lo, hi=hi))
pd.DataFrame(out).to_csv(os.path.join(K.RESULTS_ROOT, "early_compare.csv"), index=False)
