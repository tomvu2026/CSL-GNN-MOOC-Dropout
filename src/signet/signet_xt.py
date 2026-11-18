"""SIG-Net dataset for XuetangX, built from <CSL_DATA>/xuetangx/processed (prep_xuetangx.py) with the SAME construction as
signet_fast.FastKDD (verified against the original KDD builder):
  * 4 event classes (problem 0, video 1, forum 2, other 3) -> object-course edge type = event, enrollment-object = 4 + event,
    student edges 8 / 9 (label mode) or 8 (nolabel mode), target out-edges re-typed with the neighbour's status;
  * daily enrollment feature with the original quirk (value of the LAST (event, day, object) group of the day; records
    without an object ignored), standardised per course over all enrollments;
  * three windows. XuetangX histories span days 0..36 after the course start (H = 37), so the windows are 13 days long (KDD: 10), which makes
    the node feature 3 + 13 + course-feature dims;
  * course features: number of objects per object family (5 families; KDD: 6 module categories) + number of
    enrollments, standardised over courses.
Labels l0..l2 = window inactivity (as in the original), l3 = ground-truth dropout."""
import os
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("CSL_DATA", os.path.join(REPO, "data"))
P = os.path.join(DATA_ROOT, "xuetangx", "processed", "")
EV4 = lambda ev: np.select([ev <= 5, ev <= 10, ev >= 16], [0, 1, 2], 3)        # action codes of prep_xuetangx.ACT

class FastXT(torch.utils.data.Dataset):
    def __init__(self, mode=None, H=37):
        self.mode = mode or os.environ.get("STUDENT_EDGE", "label"); W = int(np.ceil(H / 3)); self.W = W
        E = pd.read_parquet(P + "enrollments.parquet"); A = pd.read_parquet(P + "events.parquet"); O = pd.read_parquet(P + "objects.parquet")
        A = A[A.day < H].assign(event=lambda d: EV4(d.ev.values))
        eid = E.enrollment_id.values; A["enrollment_id"] = eid[A.enr.values]; A["course"] = E.course.values[A.enr.values]
        self.train_ids = eid[E.off_split.values == 0].tolist(); self.test_ids = eid[E.off_split.values == 2].tolist()
        lo = A[A.obj >= 0]
        cs = lo.groupby(["enrollment_id", "event", "day", "obj"]).size().rename("click_sum").reset_index()
        last = cs.groupby(["enrollment_id", "day"]).click_sum.last()
        F = last.unstack().reindex(index=eid, columns=range(3 * W)).fillna(0)
        lab = np.stack([(F.values[:, W * w:W * (w + 1)].sum(1) == 0) for w in range(3)], 1).astype(int)
        self.labels = pd.DataFrame(np.c_[lab, E.y.values], index=eid, columns=["l0", "l1", "l2", "l3"])
        Fs = F.copy()
        for c, ids in pd.Series(eid).groupby(E.course.values):
            Fs.loc[ids.values] = StandardScaler().fit_transform(F.loc[ids.values])
        cats = list(O.category.cat.categories)
        cc = O.dropna(subset=["course"]).groupby(["course", "category"], observed=False).size().unstack().reindex(columns=cats).fillna(0)
        cc = cc.reindex(range(int(E.course.max()) + 1)).fillna(0); cc["total"] = E.groupby("course").size().reindex(cc.index).fillna(0).values
        self.cfeat = pd.DataFrame(StandardScaler().fit_transform(cc), index=cc.index)
        self.eid = {e: i for i, e in enumerate(eid)}
        self.e_course = dict(zip(eid, E.course.values)); self.e_user = dict(zip(eid, E.user.values))
        self.user_enr = pd.Series(eid).groupby(E.user.values).apply(list).to_dict()
        self.featw = [Fs.values[:, W * w:W * (w + 1)].astype(np.float32) for w in range(3)]
        oi = lo[["course", "obj", "event"]].drop_duplicates()
        self.course_obj = {c: (g.obj.values, g.event.values.astype(int)) for c, g in oi.groupby("course")}
        self.course_objset = {c: np.unique(v[0]) for c, v in self.course_obj.items()}
        self.win = []
        for w in range(3):
            a = lo[(lo.day >= W * w) & (lo.day < W * (w + 1))][["enrollment_id", "event", "obj"]].drop_duplicates()
            self.win.append({e: (g.obj.values, g.event.values.astype(int)) for e, g in a.groupby("enrollment_id")})
        self.in_feats = 3 + W + self.cfeat.shape[1]

    def __len__(self): return len(self.eid)
    def status(self, e, w): return int(self.labels.at[e, ["l1", "l2", "l3"][w]])

    def build(self, i):
        c = self.e_course[i]; S = self.user_enr[self.e_user[i]]; objs = self.course_objset.get(c, np.array([], dtype=np.int64))
        nE, nO = len(S), len(objs); lidE = {e: k for k, e in enumerate(S)}; lidO = {o: nE + k for k, o in enumerate(objs)}; cn = nE + nO
        oc_o, oc_t = self.course_obj.get(c, (np.array([], dtype=np.int64), np.array([], dtype=int)))
        out, W, dc = [], self.W, self.cfeat.shape[1]
        for w in range(3):
            src, dst, et = [], [], []
            for o, t in zip(oc_o, oc_t): src += [lidO[o], cn]; dst += [cn, lidO[o]]; et += [t, t]
            for e in S:
                if e not in self.win[w]: continue
                for o, t in zip(*self.win[w][e]):
                    if o in lidO: src += [lidE[e], lidO[o]]; dst += [lidO[o], lidE[e]]; et += [4 + t, 4 + t]
            for s_ in S:
                for d in S:
                    if s_ == d: continue
                    if self.mode == "label":
                        st = self.status(d, w) if s_ == i else self.status(s_, w); typ = 8 if st == 1 else 9
                    else: typ = 8
                    src.append(lidE[s_]); dst.append(lidE[d]); et.append(typ)
            feat = np.zeros((cn + 1, 3 + W + dc), np.float32)
            feat[:nE, 0] = 1; feat[:nE, 3:3 + W] = self.featw[w][[self.eid[e] for e in S]]
            feat[nE:cn, 1] = 1; feat[cn, 2] = 1; feat[cn, 3 + W:] = self.cfeat.loc[c].values
            tgt = np.zeros(cn + 1, np.int64); tgt[lidE[i]] = 1; tgt[cn] = 2
            out.append((src, dst, et, feat, tgt))
        return out, torch.Tensor(self.labels.loc[i].values.astype(float))
