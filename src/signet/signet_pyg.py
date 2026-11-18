"""SIG-Net (Roh et al., 2024) re-run on KDD Cup 2015 or XuetangX (env DATA) in PyTorch Geometric.

The official code (https://github.com/Noverse0/SIG-Net, MIT licence) uses DGL; the model is ported 1:1:
  DGL RelGraphConv(in, out, num_rels=11, self_loop=True)  ==  PyG RGCNConv(in, out, 11, aggr='add', root_weight=True, bias=True)
  (both: h_v' = W_0 h_v + b + sum_{r} sum_{u in N_r(v)} W_r h_u, no normalisation, no basis decomposition)
Everything else is copied from src/model.py::Multi_RGCN and src/trainer.py: 2 layers, hidden 64, ReLU, concatenation of
layer outputs, [target enrollment || target course] per window, 2-layer Bi-LSTM over the 3 windows, MLP head with
dropout 0.3, sigmoid, BCE, Adam 1e-3, batch 256, 4 epochs, test evaluation after the last epoch, 5 repetitions after one
seed(0). Subgraphs come from signet_fast.FastKDD (same node set / edges / edge types / masking as the original builder).
STUDENT_EDGE=label|nolabel selects the student-edge typing (see signet_fast.py).
STUDENT_EDGE=label is the original ("labeled SIG-Net"); nolabel merges the student edges into one type ("unlabeled SIG-Net").
Usage: [DATA=kdd|xt] [SPLIT=off|lr] STUDENT_EDGE=label|nolabel python3 signet_pyg.py [--runs 5] [--limit N] [--verify 40]
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGCNConv
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, os.path.dirname(__file__))
import signet_fast as SF
DATA = os.environ.get("DATA", "kdd"); SPLIT = os.environ.get("SPLIT", "off")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("CSL_DATA", os.path.join(REPO, "data"))
RESULTS_ROOT = os.environ.get("CSL_RESULTS", os.path.join(REPO, "results"))

class PyGKDD(torch.utils.data.Dataset):
    """Wraps FastKDD, returning PyG Data objects instead of DGL graphs (same content)."""
    def __init__(self, base): self.b = base
    def __len__(self): return len(self.b)
    def __getitem__(self, i):
        gs, y = self.b.build(i)
        return [Data(x=torch.from_numpy(f), edge_index=torch.tensor(np.array([s, d]), dtype=torch.long).reshape(2, -1),
                     edge_type=torch.tensor(et, dtype=torch.long), target=torch.from_numpy(t)) for s, d, et, f, t in gs], y

def collate(batch):
    g1, g2, g3 = (Batch.from_data_list([b[0][w] for b in batch]) for w in range(3))
    return g1, g2, g3, torch.stack([b[1] for b in batch])

class Multi_RGCN(nn.Module):
    def __init__(s, num_layers=2, in_feats=20, h=64, bi=True, dropout=0.3):  # in_feats: 20 on KDD, 3+12+6 on XuetangX
        super().__init__()
        s.convs = nn.ModuleList([RGCNConv(in_feats if l == 0 else h, h, 11, aggr="add", root_weight=True, bias=True) for l in range(num_layers)])
        s.lstm = nn.LSTM(h * num_layers * 2, h * num_layers * 2, num_layers=2, bidirectional=bi, dropout=dropout, batch_first=True)
        s.W1 = nn.Linear(h * num_layers * 2 * 2, h * num_layers * 2); s.W2 = nn.Linear(h * num_layers * 2, h * num_layers)
        s.W3 = nn.Linear(h * num_layers, 1); s.p = dropout
    def window(s, g):
        x, outs = g.x, []
        for conv in s.convs:
            x = F.relu(conv(x, g.edge_index, g.edge_type)); outs.append(x)
        o = torch.cat(outs, 1)
        return torch.cat([o[g.target == 1], o[g.target == 2]], 1)
    def forward(s, g1, g2, g3):
        em = torch.stack([s.window(g1), s.window(g2), s.window(g3)], 1)
        out, _ = s.lstm(em); x = out[:, -1]
        x = F.dropout(F.relu(s.W1(x)), s.p, s.training); x = F.dropout(F.relu(s.W2(x)), s.p, s.training)
        return torch.sigmoid(s.W3(x)).reshape(-1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = lambda *x: print(f"[{time.time() - t0:7.0f}s]", *x, flush=True)
    if DATA == "xt":
        import signet_xt; base = signet_xt.FastXT()
    else: base = SF.FastKDD()
    mode = base.mode; in_feats = getattr(base, "in_feats", 20); log("dataset built, mode", mode, "enrollments", len(base))
    if a.verify:
        bad, tot = SF.verify(base, a.verify); log(f"VERIFY mismatching subgraphs: {bad}/{tot}"); sys.exit(0)
    ds = PyGKDD(base)
    out = os.path.join(RESULTS_ROOT, f"signet_rerun{'_xt' if DATA == 'xt' else ''}{'_lr' if SPLIT == 'lr' else ''}", mode, ""); os.makedirs(out, exist_ok=True)
    tr_ids, te_ids = base.train_ids, base.test_ids
    if SPLIT == "lr":   # learner-level split (same hash rule as prep_kdd / prep_xuetangx): train on lr 0+1 (no validation in SIG-Net), test on lr 2
        import pandas as pd
        Ep = pd.read_parquet(os.path.join(DATA_ROOT, {"kdd": "kddcup15", "xt": "xuetangx"}[DATA], "processed", "enrollments.parquet"))
        tr_ids = Ep.enrollment_id[Ep.lr_split < 2].tolist(); te_ids = Ep.enrollment_id[Ep.lr_split == 2].tolist()
        log("learner-level split: train", len(tr_ids), "test", len(te_ids))
    if a.limit: tr_ids, te_ids = tr_ids[:a.limit], te_ids[:a.limit // 2]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); torch.manual_seed(0); np.random.seed(0)
    trl = DataLoader(ds, collate_fn=collate, sampler=SubsetRandomSampler(tr_ids), batch_size=256, num_workers=a.workers, persistent_workers=True)
    tel = DataLoader(ds, collate_fn=collate, sampler=list(te_ids), batch_size=256, num_workers=a.workers, persistent_workers=True)
    res = []
    for run in range(1, a.runs + 1):
        m = Multi_RGCN(in_feats=in_feats).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3); bce = nn.BCELoss()
        for e in range(1, a.epochs + 1):
            m.train(); tot = 0
            for g1, g2, g3, y in trl:
                p = m(g1.to(dev), g2.to(dev), g3.to(dev)); loss = bce(p, y[:, 3].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
            log(f"run {run} epoch {e} loss {tot / len(tr_ids):.4f}")
        m.eval(); P, Y = [], []
        with torch.no_grad():
            for g1, g2, g3, y in tel:
                P += m(g1.to(dev), g2.to(dev), g3.to(dev)).cpu().tolist(); Y += y[:, 3].tolist()
        auc = roc_auc_score(Y, P); f1 = f1_score(np.round(Y), np.round(P))
        res.append(dict(run=run, auc=auc, f1=f1)); log(f"RUN {run} test AUC {auc:.4f} F1 {f1:.4f}")
        np.save(out + f"preds_run{run}.npy", np.c_[te_ids, P, Y])
    aucs = [r["auc"] for r in res]
    log(f"FINAL mode={mode} AUC {100 * np.mean(aucs):.2f} ± {100 * np.std(aucs):.2f} | runs {[round(x, 4) for x in aucs]}")
    json.dump(res, open(out + "results.json", "w"), indent=1)
