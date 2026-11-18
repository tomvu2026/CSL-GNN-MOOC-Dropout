"""Test AUC and paired-bootstrap comparisons for one dataset / split / horizon (Tables 4 and 6 of the paper).

Reads the saved test predictions of CSL-GNN and the controls (csl_gnn.py --fullfit 1 --out fullfit --tag _plus) and of the
SIG-Net re-runs (signet/signet_pyg.py). For every model: mean ± sample SD of the per-seed (per-run) AUCs and the AUC of the
averaged prediction. Comparisons: paired bootstrap (B = 2000) of the AUC difference on the averaged predictions.
Usage: DATA=kdd|xt SPLIT=off|lr [H=30|37] python3 compare.py
Output: <results>/compare_<DATA>_<SPLIT>_H<H>_auc.csv and ..._delta.csv
"""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K

H = int(os.environ.get("H", 37 if K.DATA == "xt" else 30))
te = K.split == 2; eid_te = K.E.enrollment_id.values[te]; y = K.y[te]


def ours(variant, feat):
    sx = "_sig" if feat == "signet" else ""
    fs = [K.R + f"fullfit/preds/{variant}{sx}_plus_full_H{H}_s{s}.npy" for s in K.SEEDS]
    return [np.load(f).ravel() for f in fs if os.path.exists(f)]


def signet(mode):
    out = []
    for f in sorted(glob.glob(os.path.join(K.signet_dir(K.DATA, K.SPLIT), mode, "preds_run*.npy"))):
        a = np.load(f); out.append(pd.Series(a[:, 1], index=a[:, 0].astype(int)).loc[eid_te].values)
    return out


M = {"Unlabeled SIG-Net": signet("nolabel"), "Labeled SIG-Net": signet("label")}
for feat, fn in [("signet", "SIG-Net-style"), ("full", "full")]:
    for v, nm in [("mlp", "No graph (MLP)"), ("res", "No same-learner edges"), ("user_p", "Placebo graph"), ("user_c", "CSL-GNN")]:
        M[f"{nm} ({fn})"] = ours(v, feat)

rows, avg = [], {}
for k, P in M.items():
    if not P: continue
    a = [roc_auc_score(y, p) for p in P]; avg[k] = np.mean(P, 0)
    rows.append(dict(model=k, n=len(P), mean=np.mean(a), sd=np.std(a, ddof=1) if len(a) > 1 else 0.0, avg_pred=roc_auc_score(y, avg[k])))
T = pd.DataFrame(rows)
print(f"DATA={K.DATA} SPLIT={K.SPLIT} H={H} test n={te.sum()}"); print(T.round(4).to_string(index=False))

C = [("CSL-GNN (SIG-Net-style)", "Unlabeled SIG-Net"), ("CSL-GNN (SIG-Net-style)", "Labeled SIG-Net"),
     ("CSL-GNN (full)", "Labeled SIG-Net"), ("Labeled SIG-Net", "Unlabeled SIG-Net")]
for fn in ["SIG-Net-style", "full"]:
    C += [(f"CSL-GNN ({fn})", f"Placebo graph ({fn})"), (f"CSL-GNN ({fn})", f"No same-learner edges ({fn})"),
          (f"CSL-GNN ({fn})", f"No graph (MLP) ({fn})"), (f"No same-learner edges ({fn})", f"No graph (MLP) ({fn})")]
D = []
for a, b in C:
    if a in avg and b in avg:
        d, lo, hi = K.bootstrap(y, avg[a], avg[b]); D.append(dict(comparison=f"{a} - {b}", delta=d, lo=lo, hi=hi))
        print(f"{a} - {b}: {d:+.4f} ({lo:+.4f}; {hi:+.4f})")
base = os.path.join(K.RESULTS_ROOT, f"compare_{K.DATA}_{K.SPLIT}_H{H}")
T.to_csv(base + "_auc.csv", index=False); pd.DataFrame(D).to_csv(base + "_delta.csv", index=False)
