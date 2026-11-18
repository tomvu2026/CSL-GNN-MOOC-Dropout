"""KDD Cup 2015 -> compact parquet tables (<CSL_DATA>/kddcup15/processed/).

enrollments.parquet : enr (0..N-1), enrollment_id, user, course, y, off_split (0 = official train, 2 = official test),
                      lr_split (learner-level 70/10/20 by an MD5 hash of the username; the salt string is kept
                      unchanged so that the splits match the paper),
                      off_split_v (official train with 10% of its USERS held out as validation = 1)
events.parquet      : enr, day (0..29 relative to course start), sec (seconds since course start), ev (event code),
                      src (0 server / 1 browser), obj (object index, -1 unknown)
objects.parquet     : obj, module_id, course, category, start_day (module release day relative to course start, nan)
courses.parquet     : course, start, end (dates from date.csv)
"""
import hashlib, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("CSL_DATA", os.path.join(REPO, "data"))
import numpy as np, pandas as pd

D = os.path.join(DATA_ROOT, "kddcup15", "raw", "kddcup15", ""); P = os.path.join(DATA_ROOT, "kddcup15", "processed", "")
os.makedirs(P, exist_ok=True)
EV = ["problem", "video", "access", "wiki", "discussion", "navigate", "page_close"]

dates = pd.read_csv(D + "date.csv", parse_dates=["from", "to"])
dates["course"] = np.arange(len(dates)); cmap = dict(zip(dates.course_id, dates.course))
dates.rename(columns={"from": "start", "to": "end"})[["course", "course_id", "start", "end"]].to_parquet(P + "courses.parquet")

en = []
for s, code in [("train", 0), ("test", 2)]:
    e = pd.read_csv(D + f"{s}/enrollment_{s}.csv")
    t = pd.read_csv(D + f"{s}/truth_{s}.csv", header=None, names=["enrollment_id", "y"])
    en.append(e.merge(t, on="enrollment_id", validate="1:1").assign(off_split=code))
E = pd.concat(en).sort_values("enrollment_id").reset_index(drop=True)
E["enr"] = np.arange(len(E)); E["course"] = E.course_id.map(cmap)
h = E.username.map(lambda u: int(hashlib.md5(("ia-hgnn-2026" + u).encode()).hexdigest(), 16) % 100)
E["lr_split"] = np.where(h < 70, 0, np.where(h < 80, 1, 2))
E["off_split_v"] = np.where((E.off_split == 0) & (h % 10 == 0), 1, E.off_split)
E["user"] = E.username.astype("category").cat.codes
E[["enr", "enrollment_id", "user", "course", "y", "off_split", "off_split_v", "lr_split"]].to_parquet(P + "enrollments.parquet")
print(E.groupby("off_split_v").y.agg(["size", "mean"]), E.groupby("lr_split").y.agg(["size", "mean"]))

O = pd.read_csv(D + "object.csv")
O = O.drop_duplicates("module_id").reset_index(drop=True); O["obj"] = np.arange(len(O)); O["course"] = O.course_id.map(cmap)
st = pd.to_datetime(O.start.replace("null", np.nan), errors="coerce")
O["start_day"] = (st - dates.set_index("course")["from"].loc[O.course.fillna(0).astype(int)].values).dt.days
O["category"] = O.category.astype("category")
O[["obj", "module_id", "course", "category", "start_day"]].to_parquet(P + "objects.parquet")
omap = dict(zip(O.module_id, O.obj))

eid2enr = dict(zip(E.enrollment_id, E.enr)); cstart = dates.set_index("course")["from"]
parts = []
for s in ["train", "test"]:
    for L in pd.read_csv(D + f"{s}/log_{s}.csv", chunksize=2_000_000):
        L["enr"] = L.enrollment_id.map(eid2enr).astype(np.int32)
        t = pd.to_datetime(L.time, format="%Y-%m-%dT%H:%M:%S")
        c0 = cstart.loc[E.course.values[L.enr.values]].values
        sec = (t.values - c0) / np.timedelta64(1, "s")
        L = pd.DataFrame({"enr": L.enr, "day": (sec // 86400).astype(np.int16), "sec": sec.astype(np.int32),
                          "ev": L.event.map({e: i for i, e in enumerate(EV)}).astype(np.int8),
                          "src": (L.source == "browser").astype(np.int8),
                          "obj": L.object.map(omap).fillna(-1).astype(np.int32)})
        parts.append(L); print(s, len(L), flush=True)
A = pd.concat(parts, ignore_index=True)
print("events", len(A), "day range", A.day.min(), A.day.max(), "unknown obj", (A.obj < 0).mean().round(3))
A.to_parquet(P + "events.parquet")
