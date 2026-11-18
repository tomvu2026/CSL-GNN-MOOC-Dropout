"""Threshold-dependent and PR metrics for the main models (Table 5), computed from saved TEST predictions (no retraining).

For one (DATA, SPLIT, H) setting it reports, per model, mean ± sd over seeds/runs and the seed-averaged ensemble:
  AUC, PR-AUC (dropout = positive class), PR-AUC (retained = positive class),
  Accuracy, Precision, Recall, F1 (dropout class), Macro-F1  -- all at a fixed threshold of 0.5.
Validation predictions were not saved, so the threshold is fixed at 0.5 for every model (no tuning on test).
A majority-class row ("always dropout") is added as a reference for Accuracy.

Usage (one setting):  DATA=kdd SPLIT=off H=30 python3 extra_metrics.py
All four settings:    bash scripts/run_extra_metrics.sh
Output: <results>/extra_metrics_<DATA>_<SPLIT>.csv (per run) and ..._summary.csv, plus a Markdown table on stdout."""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K

H = int(os.environ.get("H", 37 if K.DATA == "xt" else 30)); TH = float(os.environ.get("TH", 0.5))
te = K.split == 2; eid_te = K.E.enrollment_id.values[te]; y = K.y[te].astype(int)
SROOT = os.path.join(K.RESULTS_ROOT, ""); sdir = K.signet_dir(K.DATA, K.SPLIT)


def to_prob(p, name):
    p = np.asarray(p, dtype=float).ravel()
    if p.min() < 0 or p.max() > 1:
        print(f"  [warn] {name}: values outside [0,1] -> applying sigmoid"); p = 1 / (1 + np.exp(-p))
    return p


def ours(rel):
    """Seed files of CSL-GNN / controls; returns list of (run_id, prob)."""
    out = []
    for s in range(5):
        f = K.R + rel.format(H=H, s=s)
        if os.path.exists(f): out.append((f"s{s}", to_prob(np.load(f), rel)))
        else: print(f"  [missing] {f}")
    return out


def signet(mode):
    out = []
    for f in sorted(glob.glob(f"{sdir}/{mode}/preds_run*.npy")):
        a = np.load(f); s = pd.Series(a[:, 1], index=a[:, 0].astype(int))
        if not np.isin(eid_te, s.index).all():
            print(f"  [skip] {f}: test enrollment ids do not match"); continue
        if a.shape[1] > 2:   # column 2 holds SIG-Net's own label; check alignment with ours
            yy = pd.Series(a[:, 2], index=a[:, 0].astype(int)).loc[eid_te].values
            if (np.round(yy) != y).mean() > 0: print(f"  [warn] {f}: {(np.round(yy) != y).mean():.4%} label mismatch")
        out.append((os.path.basename(f)[6:-4], to_prob(s.loc[eid_te].values, f)))
    if not out: print(f"  [missing] {sdir}/{mode}/preds_run*.npy")
    return out


MODELS = {
    "CSL-GNN (SIG-Net feats)": lambda: ours("fullfit/preds/user_c_sig_plus_full_H{H}_s{s}.npy"),
    "CSL-GNN (full feats)":    lambda: ours("fullfit/preds/user_c_plus_full_H{H}_s{s}.npy"),
    "SIG-Net label":           lambda: signet("label"),
    "SIG-Net nolabel":         lambda: signet("nolabel"),
    "Placebo (SIG-Net feats)": lambda: ours("fullfit/preds/user_p_sig_plus_full_H{H}_s{s}.npy"),
    "No graph MLP (SIG-Net feats)": lambda: ours("fullfit/preds/mlp_sig_plus_full_H{H}_s{s}.npy"),
}


def metrics(p):
    yh = (p >= TH).astype(int)
    return dict(AUC=roc_auc_score(y, p), PR_AUC_dropout=average_precision_score(y, p),
                PR_AUC_retained=average_precision_score(1 - y, 1 - p), Accuracy=accuracy_score(y, yh),
                Precision=precision_score(y, yh, zero_division=0), Recall=recall_score(y, yh, zero_division=0),
                F1=f1_score(y, yh, zero_division=0), Macro_F1=f1_score(y, yh, average="macro", zero_division=0))


print(f"DATA={K.DATA} SPLIT={K.SPLIT} H={H} threshold={TH} test n={len(y)} dropout rate={y.mean():.4f}")
rows, summ = [], []
for name, load in MODELS.items():
    runs = load()
    if not runs: continue
    for rid, p in runs: rows.append(dict(model=name, run=rid, **metrics(p)))
    ens = metrics(np.mean([p for _, p in runs], 0))
    df = pd.DataFrame([r for r in rows if r["model"] == name]).drop(columns=["model", "run"])
    summ.append(dict(model=name, n_runs=len(runs), **{f"{k}_mean": v for k, v in df.mean().items()},
                     **{f"{k}_sd": v for k, v in df.std(ddof=1).fillna(0).items()}, **{f"{k}_ens": v for k, v in ens.items()}))
maj = metrics(np.ones_like(y, dtype=float)); maj["AUC"] = maj["PR_AUC_dropout"] = maj["PR_AUC_retained"] = np.nan
summ.append(dict(model="Majority (always dropout)", n_runs=1, **{f"{k}_mean": v for k, v in maj.items()},
                 **{f"{k}_sd": 0.0 for k in maj}, **{f"{k}_ens": v for k, v in maj.items()}))

tag = f"{K.DATA}_{K.SPLIT}"; outdir = SROOT
pd.DataFrame(rows).to_csv(outdir + f"extra_metrics_{tag}.csv", index=False)
S = pd.DataFrame(summ); S.to_csv(outdir + f"extra_metrics_{tag}_summary.csv", index=False)

keys = ["AUC", "PR_AUC_dropout", "PR_AUC_retained", "Accuracy", "Precision", "Recall", "F1", "Macro_F1"]
fmt = lambda m, s: "–" if pd.isna(m) else f"{100 * m:.2f} ± {100 * s:.2f}"
print(f"\n### {tag}: mean ± sd over seeds/runs (%)\n")
print("| Model | n | " + " | ".join(keys) + " |"); print("|---|---|" + "---|" * len(keys))
for _, r in S.iterrows():
    print(f"| {r.model} | {r.n_runs} | " + " | ".join(fmt(r[k + '_mean'], r[k + '_sd']) for k in keys) + " |")
print(f"\n### {tag}: seed-averaged ensemble (%)\n")
print("| Model | " + " | ".join(keys) + " |"); print("|---|" + "---|" * len(keys))
for _, r in S.iterrows():
    print(f"| {r.model} | " + " | ".join("–" if pd.isna(r[k + '_ens']) else f"{100 * r[k + '_ens']:.2f}" for k in keys) + " |")
print(f"\nSaved: {outdir}extra_metrics_{tag}.csv and _summary.csv")
