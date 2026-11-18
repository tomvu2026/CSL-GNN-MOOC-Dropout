"""CSL-GNN and its component controls (full-batch training), KDD Cup 2015 or XuetangX (env DATA), split from env SPLIT.

Graph: enrollment nodes (features: 'full' or 'signet', quantile-normalised on the training part),
       learning-object nodes (learned ID embedding + category + course embedding),
       enrollment <-> object edges for objects touched in days [0, H) (edge attr: log count, first/last day, n days),
       causal same-learner edges (neighbour enrollment -> target): the message is built only from the neighbour
       enrollment's activity inside the target's own window [start_x, start_x + H) plus the neighbour's course embedding.
Variants (paper names):
  user_c  CSL-GNN
  user_p  placebo graph (same-learner neighbours replaced by enrollments of other learners in the same courses)
  res     no same-learner edges (enrollment-object graph only)
  mlp     no graph
Modules: --attn (attentive message passing), --temporal (1D-CNN over the daily activity sequence), --jk (layer combination).
--fullfit 1: two-stage training (stage 1 with validation early stopping, stage 2 refit on train + validation for the
best epoch count, replaying the stage-1 learning-rate schedule); test predictions come from stage 2.
Output: <results>/<DATA>_<SPLIT>/<out>/preds/<variant>[_sig]<tag>[_full]_H<H>_s<seed>.npy and gnn_results.csv
"""
import argparse, os, sys
import torch.utils.checkpoint
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import QuantileTransformer
sys.path.insert(0, os.path.dirname(__file__))
import common as K

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="user_c", choices=["user_c", "user_p", "res", "mlp"])
ap.add_argument("--H", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--dropedge", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--patience", type=int, default=40)
ap.add_argument("--feat", default="full", choices=["full", "signet"], help="node features: 'full' (common.features) or SIG-Net-style (common.signet_features)")
ap.add_argument("--attn", type=int, default=0, help="attention (GAT-style softmax) instead of mean aggregation")
ap.add_argument("--temporal", type=int, default=0, help="1D-CNN encoder over the daily activity sequence")
ap.add_argument("--jk", type=int, default=0, help="jumping knowledge: concatenate every layer's output in the head")
ap.add_argument("--ckpt", type=int, default=0, help="gradient checkpointing per layer (memory only; same model and results up to numerics)")
ap.add_argument("--fullfit", type=int, default=0, help="after the validation run, retrain a fresh model on train+val for the best epoch count, replaying the validation run's learning-rate trajectory; test predictions come from the refit")
ap.add_argument("--tag", default="", help="suffix for output names"); ap.add_argument("--out", default="gnn")
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"; RD = K.R + a.out + "/"; os.makedirs(RD + "preds", exist_ok=True)
split, y, N = K.split, K.y.astype(np.float32), K.N

X = (K.features(a.H).drop(columns=["course"]) if a.feat == "full" else K.signet_features(a.H)).astype(float)
qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=10 ** 9, random_state=0).fit(X[split == 0])
X = np.clip(qt.transform(X), -5, 5).astype(np.float32)
course = K.E.course.values; No = len(K.O)
a_ = K.A[(K.A.day < a.H) & (K.A.obj >= 0)]
g = a_.groupby(["enr", "obj"]).agg(cnt=("sec", "size"), fr=("day", "min"), lr=("day", "max"), nd=("day", "nunique")).reset_index()
ea = np.c_[np.log1p(g.cnt), g.fr / a.H, g.lr / a.H, np.log1p(g.nd)].astype(np.float32)
pairs = K.user_pairs(a.H, placebo_seed=1000 + a.seed if a.variant == "user_p" else None)
pa = K.pair_matrix(pairs)
K.log(f"variant={a.variant} N={N} objects={No} obj-edges={len(g)} user-edges={len(pairs)} feat={X.shape[1]}")

tt = lambda v, dt=torch.float32: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)
X_t, y_t, c_t = tt(X), tt(y), tt(course, torch.long)
oc = K.O.course.fillna(0).astype(int).values; ocat = K.O.category.cat.codes.values
oc_t, ocat_t = tt(oc, torch.long), tt(ocat, torch.long)
Es, Ed, EA = tt(g.enr.values, torch.long), tt(g.obj.values, torch.long), tt(ea)
Us, Ud = tt(pairs.enr_y.values, torch.long), tt(pairs.enr_x.values, torch.long)     # message neighbour -> target
PA, Uc = tt(pa), tt(pairs.course_y.values, torch.long)
tr = tt(split == 0, torch.bool)
# daily activity sequence (same information as SIG-Net's daily features): log1p(events per day), standardised on train
dseq = K.A[K.A.day < a.H].groupby(["enr", "day"]).size().unstack().reindex(index=range(N), columns=range(a.H)).fillna(0).values
dseq = np.log1p(dseq); dseq = (dseq - dseq[split == 0].mean(0)) / (dseq[split == 0].std(0) + 1e-6)
SEQ = tt(dseq[:, None, :].astype(np.float32))

def mean_agg(msg, idx, n):
    out = torch.zeros(n, msg.size(1), device=dev).index_add_(0, idx, msg)
    cnt = torch.zeros(n, device=dev).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32)).clamp_min(1)
    return out / cnt[:, None]

from torch_geometric.utils import softmax as seg_softmax
class Agg(nn.Module):
    """mean aggregation, or attention: alpha_ij = softmax_j(LeakyReLU(a_m^T m_ij + a_q^T q_i)) per destination i
    (GAT-style additive score, decomposed so no [m || q] edge tensor is materialised)."""
    def __init__(s, h):
        super().__init__(); s.am = nn.Linear(h, 1) if a.attn else None; s.aq = nn.Linear(h, 1, bias=False) if a.attn else None
    def forward(s, msg, idx, n, q):
        if s.am is None: return mean_agg(msg, idx, n)
        w = seg_softmax(F.leaky_relu(s.am(msg).squeeze(1) + s.aq(q).squeeze(1)[idx], 0.2), idx, num_nodes=n)
        return torch.zeros(n, msg.size(1), device=dev).index_add_(0, idx, msg * w[:, None])

class Layer(nn.Module):
    def __init__(s, h, p):
        super().__init__()
        s.e2o = nn.Sequential(nn.Linear(h + 4, h), nn.ReLU(), nn.Linear(h, h))
        s.o2e = nn.Sequential(nn.Linear(h + 4, h), nn.ReLU(), nn.Linear(h, h))
        s.u2c = nn.Sequential(nn.Linear(h + 5, h), nn.ReLU(), nn.Linear(h, h)) if a.variant in ("user_c", "user_p") else None
        s.no, s.ne, s.p = nn.LayerNorm(h), nn.LayerNorm(h), p
        s.a_eo, s.a_oe, s.a_u = Agg(h), Agg(h), Agg(h)
    def forward(s, he, ho, es, ed, ea, cemb=None):
        ho = s.no(ho + F.dropout(s.a_eo(s.e2o(torch.cat([he[es], ea], 1)), ed, ho.size(0), ho), s.p, s.training))
        msg = s.a_oe(s.o2e(torch.cat([ho[ed], ea], 1)), es, he.size(0), he)
        if s.u2c is not None: msg = msg + s.a_u(s.u2c(torch.cat([cemb(Uc), PA], 1)), Ud, he.size(0), he)
        return s.ne(he + F.dropout(msg, s.p, s.training)), ho

class Net(nn.Module):
    def __init__(s, d, h, L, p):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h))
        s.c_emb = nn.Embedding(K.NC, h); s.cat_emb = nn.Embedding(int(ocat.max()) + 2, h)
        s.o_emb = nn.Embedding(No, h) if a.variant in ("res", "user_c", "user_p") else None
        if s.o_emb is not None: nn.init.normal_(s.o_emb.weight, std=0.02)
        s.layers = nn.ModuleList([Layer(h, p) for _ in range(L)]) if a.variant != "mlp" else nn.ModuleList()
        s.tcn = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()) if a.temporal else None
        s.tproj = nn.Linear(64, h) if a.temporal else None
        nh = (L if (a.jk and a.variant != "mlp") else 1) + 1
        s.head = nn.Sequential(nn.Linear(nh * h + d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))
    def forward(s):
        he = s.inp(X_t) + s.c_emb(c_t)
        if s.tcn is not None:
            z_ = s.tcn(SEQ); he = he + s.tproj(torch.cat([z_.mean(2), z_.amax(2)], 1))
        h0 = he; hs = []
        if len(s.layers):
            ho = s.c_emb(oc_t) + s.cat_emb(ocat_t + 1) + (s.o_emb.weight if s.o_emb is not None else 0)
            es, ed, ea_ = Es, Ed, EA
            if s.training and a.dropedge > 0:
                k = torch.rand(es.size(0), device=dev) >= a.dropedge; es, ed, ea_ = es[k], ed[k], ea_[k]
            for l in s.layers:
                if a.ckpt and s.training:
                    he, ho = torch.utils.checkpoint.checkpoint(l, he, ho, es, ed, ea_, s.c_emb, use_reentrant=False)
                else: he, ho = l(he, ho, es, ed, ea_, s.c_emb)
                hs.append(he)
        hcat = torch.cat(hs, 1) if (a.jk and hs) else he
        z = s.head(torch.cat([hcat, h0, X_t], 1)).squeeze(1)
        return z

m = Net(X.shape[1], a.hidden, a.layers, a.dropout).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
best, bad, state, lrs, best_ep = -1, 0, None, [], 0
for ep in range(a.epochs):
    lrs.append(opt.param_groups[0]["lr"])
    m.train(); opt.zero_grad(); out = m()
    loss = F.binary_cross_entropy_with_logits(out[tr], y_t[tr]); loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); m.eval()
    with torch.no_grad(): p = torch.sigmoid(m()).cpu().numpy()
    v = roc_auc_score(y[split == 1], p[split == 1]); sch.step(v)
    if v > best: best, bad, state, best_ep = v, 0, {k: t_.clone() for k, t_ in m.state_dict().items()}, ep + 1
    else: bad += 1
    if ep % 20 == 0: K.log(f"ep {ep} loss {loss.item():.4f} val {v:.4f} best {best:.4f}")
    if bad >= a.patience: break
m.load_state_dict(state); m.eval()
with torch.no_grad(): p = torch.sigmoid(m()).cpu().numpy()
pv_val = p[split == 1].copy()     # validation predictions of the validation-run model (used for the F1 threshold)
if a.fullfit:
    # refit on train + validation, fixed number of epochs = best epoch of the validation run, same LR trajectory.
    # The test set is not used for anything here; preprocessing (quantile transform, sequence scaling) stays fit on train.
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    trv = tt((split == 0) | (split == 1), torch.bool)
    m = Net(X.shape[1], a.hidden, a.layers, a.dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    for e2 in range(best_ep):
        for g_ in opt.param_groups: g_["lr"] = lrs[e2]
        m.train(); opt.zero_grad(); out = m()
        loss = F.binary_cross_entropy_with_logits(out[trv], y_t[trv]); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if e2 % 20 == 0: K.log(f"refit ep {e2} loss {loss.item():.4f} lr {lrs[e2]:.2e}")
    m.eval()
    with torch.no_grad(): p = torch.sigmoid(m()).cpu().numpy()
    a.tag = a.tag + "_full"
pv, pt, yv, yt = pv_val, p[split == 2], y[split == 1], y[split == 2]
ths = np.linspace(0.05, 0.95, 91); th = ths[np.argmax([f1_score(yv, pv >= t_, average="macro") for t_ in ths])]
tag = a.variant + ("_sig" if a.feat == "signet" else "") + a.tag
r = dict(model=f"CSL-GNN[{tag}]", H=a.H, seed=a.seed, auc=roc_auc_score(yt, pt), auprc=average_precision_score(yt, pt),
         auprc_retained=average_precision_score(1 - yt, 1 - pt), macro_f1=f1_score(yt, pt >= th, average="macro"),
         val_auc=best, epochs=ep + 1, best_ep=best_ep, fullfit=a.fullfit, hidden=a.hidden, layers=a.layers, dropout=a.dropout, lr=a.lr, attn=a.attn,
         temporal=a.temporal, jk=a.jk, feat=a.feat)
K.log(r); f = RD + "gnn_results.csv"; pd.DataFrame([r]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
np.save(RD + f"preds/{tag}_H{a.H}_s{a.seed}.npy", pt)
