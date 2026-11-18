"""Descriptive statistics for the multi-course data figure (Fig. 3) and Table 2 (no model, labels used only to describe the data).

(a) share of enrollments by the number of courses their learner is enrolled in;
(b) dropout rate by the learner's activity in OTHER courses during [course start, prediction time),
    restricted to the neighbours of the causal graph (other courses that started before the prediction time).
Usage: DATA=kdd H=30 python3 data_stats.py ; DATA=xt H=37 python3 data_stats.py
Output: <results>/data_stats_<DATA>.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K

H = int(os.environ.get("H", 37 if K.DATA == "xt" else 30))
E, A, cday = K.E, K.A, K.cday
u = E.user.values; st = cday[E.course.values]; lo, hi = st, st + H

# (a) learner course count
ncou = E.groupby("user").course.transform("nunique").values
bins_a = pd.cut(ncou, [0, 1, 2, 3, 5, 10, 10**6], labels=["1", "2", "3", "4–5", "6–10", ">10"])
ta = pd.DataFrame({"bin": bins_a, "y": E.y.values}).groupby("bin", observed=False).agg(n=("y", "size"), dropout=("y", "mean"))
ta["share"] = ta.n / ta.n.sum(); ta.insert(0, "panel", "a")

# (b) neighbours = other enrollments of the same learner whose course started before the prediction time
en = pd.DataFrame({"user": u, "enr": E.enr.values, "st": st})
pairs = en.merge(en, on="user", suffixes=("", "_o"))
pairs = pairs[(pairs.enr != pairs.enr_o)]
pairs = pairs[pairs.st_o < pairs.st + H]
n_nb = pairs.groupby("enr").size().reindex(E.enr.values).fillna(0).values

# active days in neighbour courses inside [start, start + H)
au = A[["enr", "aday"]].drop_duplicates()
au = au.merge(pairs[["enr", "enr_o", "st"]].rename(columns={"enr": "t_enr", "enr_o": "enr"}), on="enr")
au = au[(au.aday >= au.st) & (au.aday < au.st + H)]
days = au.groupby("t_enr").aday.nunique().reindex(E.enr.values).fillna(0).values

lab = np.where(n_nb == 0, "no other course",
      np.where(days == 0, "0 days", np.where(days <= 3, "1–3 days", np.where(days <= 10, "4–10 days", ">10 days"))))
order = ["no other course", "0 days", "1–3 days", "4–10 days", ">10 days"]
tb = pd.DataFrame({"bin": pd.Categorical(lab, order), "y": E.y.values}).groupby("bin", observed=False).agg(
    n=("y", "size"), dropout=("y", "mean"))
tb["share"] = tb.n / tb.n.sum(); tb.insert(0, "panel", "b")

out = pd.concat([ta, tb]).reset_index()
print(f"DATA={K.DATA} H={H} N={len(E)} overall dropout={E.y.mean():.4f} share with >=1 neighbour={np.mean(n_nb > 0):.4f}"
      f" share of learners' enrollments with >=2 courses={np.mean(ncou >= 2):.4f}")
print(out.to_string(index=False))
out.to_csv(os.path.join(K.RESULTS_ROOT, f"data_stats_{K.DATA}.csv"), index=False)
