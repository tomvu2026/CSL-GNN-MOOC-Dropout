"""Faithful, fast re-implementation of SIG-Net's KDD Cup 2015 dataset (src/graph_builder.py::kddcup_graph).

The original builder loops over 200k enrollments with full-table filters (O(N^2)) and extracts every subgraph with
dgl.node_subgraph on three graphs of millions of edges, which takes days. This module reproduces the SAME node
features, edge types and per-target subgraphs (same node set, same induced edges, same student-edge masking),
building each subgraph directly from pre-indexed arrays. Quirks of the original are kept on purpose:
  * the daily feature is overwritten by the click count of the LAST (event, object) group of that day
    (original: feature[days] = click_sum inside an iterrows loop), rows with a missing object are ignored;
  * per-course StandardScaler over all enrollments (train + test); course features standardised over courses;
  * edge types: object-course = event (0-3), enrollment-object = 4 + event, student edge = 8 (dropout) / 9 (not).
STUDENT_EDGE (env):
  'label'   (original)  window-1/2/3 student edges typed by the SOURCE enrollment's labels1 / labels2 / labels3,
                        where labels3 is the ground-truth dropout label of the other enrollment (also for test);
                        the target's own outgoing edges are re-typed with the neighbour's status (original masking).
  'nolabel'             the same edges, all of one type (8): no status information at all.
`verify(ds, n)` rebuilds the three full graphs exactly as the original and compares node_subgraph + original masking
with the fast subgraphs for n random targets.
"""
import os
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("CSL_DATA", os.path.join(REPO, "data"))
D = os.path.join(DATA_ROOT, "kddcup15", "raw", "kddcup15", "")
CATS = ['video', 'problem', 'discussion', 'chapter', 'html', 'about']
EVMAP = {'problem': 0, 'video': 1, 'discussion': 2, 'wiki': 3, 'page_close': 3, 'navigate': 3, 'access': 3}


class FastKDD(torch.utils.data.Dataset):
    def __init__(self, mode=None):
        self.mode = mode or os.environ.get("STUDENT_EDGE", "label")
        date = pd.read_csv(D + "date.csv"); info = pd.read_csv(D + "object.csv")
        en = pd.concat([pd.read_csv(D + f"{s}/enrollment_{s}.csv") for s in ["train", "test"]])
        tr = pd.concat([pd.read_csv(D + f"{s}/truth_{s}.csv", header=None, names=["enrollment_id", "truth"]) for s in ["train", "test"]])
        self.train_ids = pd.read_csv(D + "train/truth_train.csv", header=None)[0].tolist()
        self.test_ids = pd.read_csv(D + "test/truth_test.csv", header=None)[0].tolist()
        truth = tr.merge(en, on="enrollment_id", how="left")
        log = pd.concat([pd.read_csv(D + f"{s}/log_{s}.csv") for s in ["train", "test"]], ignore_index=True)
        log = log.merge(en, on="enrollment_id", how="left").merge(date, on="course_id", how="left")
        log["days"] = (pd.to_datetime(log.time.str.split("T").str[0]) - pd.to_datetime(log["from"])).dt.days
        log["event"] = log.event.map(EVMAP)
        lo = log.dropna(subset=["object"])                                         # groupby drops NaN objects

        # ---- enrollment daily features (with the original overwrite quirk)
        cs = lo.groupby(["enrollment_id", "event", "days", "object"]).size().rename("click_sum").reset_index()
        last = cs.groupby(["enrollment_id", "days"]).click_sum.last()                # last row in groupby order wins
        F = last.unstack().reindex(index=truth.enrollment_id.values, columns=range(30)).fillna(0)
        lab = np.c_[(F.values[:, 0:10].sum(1) == 0), (F.values[:, 10:20].sum(1) == 0), (F.values[:, 20:30].sum(1) == 0)].astype(int)
        self.labels = pd.DataFrame(np.c_[lab, truth.truth.values], index=truth.enrollment_id.values, columns=["l0", "l1", "l2", "l3"])
        Fs = F.copy()
        for c, ids in truth.groupby("course_id").enrollment_id:
            Fs.loc[ids.values] = StandardScaler().fit_transform(F.loc[ids.values])
        # ---- course features
        cc = info.groupby(["course_id", "category"]).module_id.count().reset_index()
        cc = cc[cc.category.isin(CATS)].pivot(index="course_id", columns="category", values="module_id").reindex(columns=CATS).fillna(0)
        cc = cc.reindex(date.course_id).fillna(0); cc["total"] = truth.groupby("course_id").size().reindex(cc.index).values
        cfeat = pd.DataFrame(StandardScaler().fit_transform(cc), index=cc.index)

        # ---- indices
        self.eid = {e: i for i, e in enumerate(truth.enrollment_id.values)}
        self.e_course = dict(zip(truth.enrollment_id, truth.course_id)); self.e_user = dict(zip(truth.enrollment_id, truth.username))
        self.user_enr = truth.groupby("username").enrollment_id.apply(list).to_dict()
        self.featw = [Fs.values[:, 10 * w:10 * (w + 1)].astype(np.float32) for w in range(3)]
        self.cfeat = cfeat
        act = log[log.event.isin([0, 1, 2, 3])]
        oi = act.merge(truth[["enrollment_id", "course_id"]], how="left").dropna(subset=["object"])
        oi = oi[["course_id", "object", "event"]].drop_duplicates()
        self.course_obj = {c: (g.object.values, g.event.values.astype(int)) for c, g in oi.groupby("course_id")}   # parallel edges kept
        self.course_objset = {c: np.unique(v[0]) for c, v in self.course_obj.items()}
        self.win = []
        for w in range(3):
            a = act[(act.days >= 10 * w) & (act.days < 10 * (w + 1))].dropna(subset=["object"])
            a = a[["enrollment_id", "event", "object"]].drop_duplicates()
            self.win.append({e: (g.object.values, g.event.values.astype(int)) for e, g in a.groupby("enrollment_id")})
        self._oi, self._log = oi, None

    def __len__(self): return len(self.eid)

    def status(self, e, w):   # SOURCE status used to type student edges in window w (original: labels1/2/3)
        return int(self.labels.at[e, ["l1", "l2", "l3"][w]])

    def build(self, i):
        """Per-window subgraph as raw arrays: list of (src, dst, etype, feature[n,20], target[n]) and the label row."""
        c = self.e_course[i]; S = self.user_enr[self.e_user[i]]; objs = self.course_objset.get(c, np.array([], dtype=object))
        nE, nO = len(S), len(objs); lidE = {e: k for k, e in enumerate(S)}; lidO = {o: nE + k for k, o in enumerate(objs)}; cn = nE + nO
        oc_o, oc_t = self.course_obj.get(c, (np.array([]), np.array([], dtype=int)))
        out = []
        for w in range(3):
            src, dst, et = [], [], []
            for o, t in zip(oc_o, oc_t):                     # object <-> course (both directions, same type)
                src += [lidO[o], cn]; dst += [cn, lidO[o]]; et += [t, t]
            for e in S:                                      # enrollment <-> object of the target course
                if e not in self.win[w]: continue
                for o, t in zip(*self.win[w][e]):
                    if o in lidO: src += [lidE[e], lidO[o]]; dst += [lidO[o], lidE[e]]; et += [4 + t, 4 + t]
            for s_ in S:                                     # student edges, all ordered pairs
                for d in S:
                    if s_ == d: continue
                    if self.mode == "label":
                        st = self.status(d, w) if s_ == i else self.status(s_, w)   # original masking of target's out-edges
                        typ = 8 if st == 1 else 9
                    else:
                        typ = 8
                    src.append(lidE[s_]); dst.append(lidE[d]); et.append(typ)
            feat = np.zeros((cn + 1, 20), np.float32)
            feat[:nE, 0] = 1; feat[:nE, 3:13] = self.featw[w][[self.eid[e] for e in S]]
            feat[nE:cn, 1] = 1
            feat[cn, 2] = 1; feat[cn, 13:20] = self.cfeat.loc[c].values
            tgt = np.zeros(cn + 1, np.int64); tgt[lidE[i]] = 1; tgt[cn] = 2
            out.append((src, dst, et, feat, tgt))
        return out, torch.Tensor(self.labels.loc[i].values.astype(float))

def verify(ds, n=40, seed=0):
    """Rebuild the original full window graphs (same edge lists as graph_builder.py), take the node-induced subgraph
    (as dgl.node_subgraph) and apply the original student-edge masking; compare the multiset of
    (src-kind, dst-kind, etype) and the node count with the fast subgraphs for n targets (half multi-course users)."""
    import collections
    rng = np.random.default_rng(seed); allids = list(ds.eid)
    multi = [e for e in allids if len(ds.user_enr[ds.e_user[e]]) > 2]
    ids = list(rng.choice(allids, n // 2, replace=False)) + list(rng.choice(multi, n - n // 2, replace=False))
    allobj = sorted(set(ds._oi.object)); nid = {}; k = 0
    for e in allids: nid[("e", e)] = k; k += 1
    for o in allobj: nid[("o", o)] = k; k += 1
    for c in ds.cfeat.index: nid[("c", c)] = k; k += 1
    kind = np.empty(k, dtype=object)
    for key, v in nid.items(): kind[v] = key[0]
    full = []
    for w in range(3):
        s, d, t = [], [], []
        for c, o, ev in ds._oi[["course_id", "object", "event"]].itertuples(index=False):
            s.append(nid[("o", o)]); d.append(nid[("c", c)]); t.append(int(ev))
        for e, (os_, ts) in ds.win[w].items():
            for o, tt in zip(os_, ts): s.append(nid[("e", e)]); d.append(nid[("o", o)]); t.append(4 + int(tt))
        s, d = s + d, d + s; t = t + t; sd = collections.defaultdict(list)
        for u, es in ds.user_enr.items():
            for a_ in es:
                for b_ in es:
                    if a_ == b_: continue
                    st = ds.status(a_, w); t.append((8 if st == 1 else 9) if ds.mode == "label" else 8)
                    sd[nid[("e", a_)]].append(len(s)); s.append(nid[("e", a_)]); d.append(nid[("e", b_)])
        full.append((np.array(s), np.array(d), np.array(t), sd))
    bad = 0
    for i in ids:
        fast, _ = ds.build(i); c = ds.e_course[i]
        nodes = set([nid[("o", o)] for o in ds.course_objset.get(c, [])] + [nid[("e", e)] for e in ds.user_enr[ds.e_user[i]]] + [nid[("c", c)], nid[("e", i)]])
        for w in range(3):
            s, d, t, sd = full[w]; keep = np.where(np.isin(s, list(nodes)) & np.isin(d, list(nodes)))[0]
            et = {int(x): int(t[x]) for x in keep}; tid = nid[("e", i)]
            for eidx in sd[tid]:                       # original masking: target out-edge takes the reverse edge's type
                v = d[eidx]; rev = [x for x in keep if s[x] == v and d[x] == tid][0]; et[eidx] = int(t[rev])
            orig = collections.Counter((kind[s[x]], kind[d[x]], et[int(x)]) for x in keep)
            fs, fd, fe, ff, ft = fast[w]; nE = len(ds.user_enr[ds.e_user[i]]); nn_ = ff.shape[0]
            kf = lambda j: "e" if j < nE else ("c" if j == nn_ - 1 else "o")
            fastc = collections.Counter((kf(a_), kf(b_), int(x)) for a_, b_, x in zip(fs, fd, fe))
            if len(nodes) != nn_ or orig != fastc: bad += 1
    return bad, len(ids) * 3
