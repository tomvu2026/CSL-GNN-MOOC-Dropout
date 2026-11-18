"""Shared data loading and feature construction for KDD Cup 2015 and XuetangX.

Prediction time = course start + H days (H = 30 on KDD Cup 2015, 37 on XuetangX; H = 10 / 20 for early prediction).
Only events with day < H are used. Cross-course (same-learner) features use the learner's OTHER enrollments' events whose
calendar time falls in [this course's start, start + H) -- nothing after the prediction time.

Environment variables
  DATA         kdd | xt                          (dataset; both are converted to the same schema by prep_*.py)
  SPLIT        off | lr                          (official split with 10% of training learners held out for validation,
                                                  or the learner-level 70/10/20 split)
  CSL_DATA     root folder of the data           (default: <repo>/data)
  CSL_RESULTS  root folder of all outputs        (default: <repo>/results)
"""
import os, time
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("CSL_DATA", os.path.join(REPO, "data"))
RESULTS_ROOT = os.environ.get("CSL_RESULTS", os.path.join(REPO, "results"))
PROCESSED = {"kdd": os.path.join(DATA_ROOT, "kddcup15", "processed", ""),
             "xt": os.path.join(DATA_ROOT, "xuetangx", "processed", "")}


def signet_dir(data, split):
    """Folder with the SIG-Net re-run predictions for one dataset / split (written by signet/signet_pyg.py)."""
    return os.path.join(RESULTS_ROOT, "signet_rerun" + ("_xt" if data == "xt" else "") + ("_lr" if split == "lr" else ""))


DATA = os.environ.get("DATA", "kdd")
P = PROCESSED[DATA]
SPLIT = os.environ.get("SPLIT", "off")
R = os.path.join(RESULTS_ROOT, f"{DATA}_{SPLIT}", ""); os.makedirs(R, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]; t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

E = pd.read_parquet(P + "enrollments.parquet"); C = pd.read_parquet(P + "courses.parquet")
O = pd.read_parquet(P + "objects.parquet"); A = pd.read_parquet(P + "events.parquet")
split = (E.off_split_v if SPLIT == "off" else E.lr_split).values      # 0 train, 1 validation, 2 test
y = E.y.values; N = len(E); NC = len(C)
cday = ((C.start - C.start.min()).dt.days).values                  # course start as absolute day index
A["cat"] = O.category.cat.codes.values[A.obj.clip(lower=0).values]; A.loc[A.obj < 0, "cat"] = -1
A["aday"] = cday[E.course.values[A.enr.values]] + A.day.values       # absolute calendar day
A["user"] = E.user.values[A.enr.values]


def features(H):
    """'Full' enrollment features: own activity, time-bounded cross-course activity, course statistics and the module
    release schedule (99 dimensions on KDD Cup 2015, 212 on XuetangX)."""
    a = A[A.day < H]; X = pd.DataFrame(index=E.enr)
    g = a.groupby("enr")
    X["n_ev"] = g.size(); X["n_obj"] = g.obj.nunique(); X["n_days"] = g.day.nunique()
    X["first_d"] = g.day.min(); X["last_d"] = g.day.max(); X["gap"] = H - 1 - X.last_d
    X = X.join(a.pivot_table(index="enr", columns="ev", values="sec", aggfunc="size").add_prefix("ev_"))
    X = X.join(a.pivot_table(index="enr", columns="src", values="sec", aggfunc="size").add_prefix("src_"))
    X = X.join(a.pivot_table(index="enr", columns="day", values="sec", aggfunc="size").add_prefix("d_"))
    a = a.assign(wk=a.day // 7)
    wk = a.groupby(["enr", "wk", "ev"]).size().unstack(["wk", "ev"])
    wk.columns = [f"w{w}_e{e}" for w, e in wk.columns]; X = X.join(wk)
    top = a[a.cat >= 0].cat.value_counts().index[:10]
    X = X.join(a[a.cat.isin(top)].pivot_table(index="enr", columns="cat", values="sec", aggfunc="size").add_prefix("cat_"))
    s = a.sort_values(["enr", "sec"]); new = (s.enr.diff() != 0) | (s.sec.diff() > 3600)
    X["n_sess"] = new.groupby(s.enr).sum()
    X["last7"] = a[a.day >= H - 7].groupby("enr").size(); X["hour_mean"] = ((a.sec % 86400) / 3600).groupby(a.enr).mean()
    # ---- same learner, other courses, calendar window [start, start+H)
    ua = A.groupby(["user", "aday"]).size()
    D = int(A.aday.max()) + 40; nu = int(E.user.max()) + 1
    cum = np.zeros((nu, D + 1), np.int32)
    np.add.at(cum, (ua.index.get_level_values(0), ua.index.get_level_values(1) + 1), ua.values); cum = cum.cumsum(1)
    act = np.zeros((nu, D + 1), np.int16)
    np.add.at(act, (ua.index.get_level_values(0), ua.index.get_level_values(1) + 1), 1); act = act.cumsum(1)
    lo = cday[E.course.values]; hi = lo + H; u = E.user.values
    own = X.n_ev.fillna(0).values
    X["u_ev_window"] = cum[u, hi] - cum[u, lo] - own
    X["u_days_window"] = act[u, hi] - act[u, lo]
    X["u_last10_ev"] = cum[u, hi] - cum[u, np.maximum(hi - 10, lo)]
    started = cday[E.course.values]
    ecs = pd.DataFrame({"u": u, "st": started})
    # number of the learner's courses that started before the prediction time
    srt = ecs.groupby("u").st.apply(np.sort)
    X["u_n_courses"] = [np.searchsorted(srt[ui], h, side="left") for ui, h in zip(u, hi)]
    # number of OTHER courses the learner was active in during the window
    au = A[["user", "aday", "enr"]].drop_duplicates(["user", "aday", "enr"])
    au = au.merge(pd.DataFrame({"user": u, "t_enr": E.enr.values, "lo": lo, "hi": hi}), on="user")
    au = au[(au.enr != au.t_enr) & (au.aday >= au.lo) & (au.aday < au.hi)]
    X["u_n_active_courses"] = au.groupby("t_enr").enr.nunique()
    # ---- course level (unsupervised, all enrollments)
    X["c_n_enr"] = E.groupby("course").enr.transform("size").values
    X["c_mean_ev"] = pd.Series(X.n_ev.fillna(0).values).groupby(E.course.values).transform("mean").values
    X["rel_ev"] = X.n_ev.fillna(0) / X.c_mean_ev.clip(lower=1)
    # ---- module release schedule (known in advance)
    od = O.dropna(subset=["course", "start_day"])
    for lo_, hi_, nm in [(H - 7, H, "mod_rel_last7"), (H, H + 10, "mod_rel_next10"), (H, 999, "mod_rel_after")]:
        k = od[(od.start_day >= lo_) & (od.start_day < hi_)].groupby("course").size()
        X[nm] = k.reindex(E.course.values).fillna(0).values
    X = X.fillna(0); X["course"] = E.course.astype("category").values
    return X


def user_pairs(H, placebo_seed=None):
    """Causal same-learner edges: pairs (target x <- neighbour y) of the same learner whose neighbour course started before
    x's prediction time, with edge attributes computed only from y's activity inside x's window [start_x, start_x + H).
    placebo_seed: replace every neighbour y by a random enrollment of the SAME course as y but a DIFFERENT learner
    (same number of edges, same neighbour-course timeline) -> placebo graph."""
    U = E[["enr", "user", "course"]]; pairs = U.merge(U, on="user"); pairs = pairs[pairs.enr_x != pairs.enr_y]
    pairs = pairs[cday[pairs.course_y.values] < cday[pairs.course_x.values] + H].reset_index(drop=True)
    if placebo_seed is not None:
        rng = np.random.default_rng(placebo_seed); by_c = {c: g.enr.values for c, g in E.groupby("course")}
        new = pairs.enr_y.values.copy()
        for c, idx in pairs.groupby("course_y").groups.items():
            pool = by_c[c]; new[idx] = pool[rng.integers(0, len(pool), len(idx))]
        bad = E.user.values[new] == pairs.user.values                    # resample the rare same-learner draws
        for i in np.where(bad)[0]:
            pool = by_c[pairs.course_y.values[i]]
            while E.user.values[new[i]] == pairs.user.values[i]: new[i] = pool[rng.integers(0, len(pool))]
        pairs["enr_y"] = new
    ed_ = A.groupby(["enr", "aday"]).size().rename("n").reset_index()
    pp = pairs[["enr_x", "enr_y"]].drop_duplicates().merge(ed_, left_on="enr_y", right_on="enr")
    lo_x = cday[E.course.values[pp.enr_x.values]]; hi_x = lo_x + H; k = (pp.aday >= lo_x) & (pp.aday < hi_x)
    pp = pp[k].assign(hi=hi_x[k])
    agg = pp.assign(last7=np.where(pp.aday >= pp.hi - 7, pp.n, 0)).groupby(["enr_x", "enr_y"]).agg(
        n=("n", "sum"), days=("aday", "nunique"), last=("aday", "max"), hi=("hi", "first"), last7=("last7", "sum"))
    pairs = pairs.merge(agg.reset_index(), on=["enr_x", "enr_y"], how="left").fillna({"n": 0, "days": 0, "last7": 0})
    pairs["rec"] = np.where(pairs["last"].notna(), (cday[pairs.course_x.values] + H - pairs["last"]) / H, 1.5)
    return pairs


def pair_matrix(pairs):
    """Edge attribute p_{e'e} (5 values): log events, log active days, recency, log events in the last 7 days, active."""
    return np.c_[np.log1p(pairs.n), np.log1p(pairs.days), pairs.rec, np.log1p(pairs.last7), (pairs.n > 0)].astype(np.float32)


def signet_features(H=30):
    """'SIG-Net-style' node features (Roh et al., 2024, Sec. 3.1): per-day count of log records over the history period
    (H daily values) for the enrollment, plus the course node's features (number of objects in the course, number of
    enrolled students), log-scaled."""
    a = A[A.day < H]
    X = a.groupby(["enr", "day"]).size().unstack().reindex(index=E.enr, columns=range(H)).fillna(0)
    X.columns = [f"d_{c}" for c in X.columns]
    n_obj = O.dropna(subset=["course"]).groupby("course").size()
    X["c_n_obj"] = np.log1p(n_obj.reindex(E.course.values).fillna(0).values)
    X["c_n_enr"] = np.log1p(E.groupby("course").enr.transform("size").values)
    return X


def bootstrap(y, a, b, B=2000, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b) on the test set: point estimate and 95% percentile interval."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    d = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    return roc_auc_score(y, a) - roc_auc_score(y, b), *np.percentile(d, [2.5, 97.5])
