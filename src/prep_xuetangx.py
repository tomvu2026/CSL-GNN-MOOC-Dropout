"""XuetangX (Feng et al., 2019, public prediction data) -> the same parquet schema as prep_kdd.py (<CSL_DATA>/xuetangx/processed/).

The public prediction data has no course-date or object-metadata file. Hence:
  * course start = calendar date of the first log record of the course (train + test logs, unsupervised); every
    enrollment's history lies within 35 days of that point (checked: max 35), so day = days since course start;
  * object category = family of the actions recorded on it (video / problem / courseware / forum / other);
  * no module release schedule (start_day = NaN -> the schedule features are zero).
Official split: train_truth / test_truth; 10% of train USERS held out as validation (same hash rule as KDD).
"""
import hashlib, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("CSL_DATA", os.path.join(REPO, "data"))
import numpy as np, pandas as pd

D = os.path.join(DATA_ROOT, "xuetangx", "raw", "prediction_log", ""); P = os.path.join(DATA_ROOT, "xuetangx", "processed", "")
os.makedirs(P, exist_ok=True)
ACT = ["problem_get", "problem_check", "problem_check_correct", "problem_check_incorrect", "problem_save", "reset_problem",
       "load_video", "play_video", "pause_video", "seek_video", "stop_video",
       "click_courseware", "close_courseware", "click_info", "click_about", "click_progress",
       "click_forum", "close_forum", "create_thread", "create_comment", "delete_thread", "delete_comment"]
FAM = {a: ("problem" if "problem" in a else "video" if "video" in a else "courseware" if "courseware" in a
           else "forum" if any(k in a for k in ["forum", "thread", "comment"]) else "other") for a in ACT}
amap = {a: i for i, a in enumerate(ACT)}
L = []
for s, code in [("train", 0), ("test", 2)]:
    x = pd.read_csv(D + f"{s}_log.csv", usecols=["enroll_id", "username", "course_id", "action", "object", "time"],
                    dtype={"enroll_id": np.int32, "username": str, "course_id": str, "action": str, "object": str})
    x["t"] = pd.to_datetime(x.time, format="%Y-%m-%dT%H:%M:%S"); x = x.drop(columns="time"); x["part"] = code
    L.append(x); print(s, len(x), flush=True)
L = pd.concat(L, ignore_index=True)
assert L.action.isin(ACT).all(), set(L.action) - set(ACT)
# ---- courses
cs = L.groupby("course_id").t.agg(["min", "max"]).reset_index()
cs["start"] = cs["min"].dt.normalize(); cs["end"] = cs["max"]; cs["course"] = np.arange(len(cs))
cs[["course", "course_id", "start", "end"]].to_parquet(P + "courses.parquet"); cmap = dict(zip(cs.course_id, cs.course))
# ---- enrollments
tr = pd.concat([pd.read_csv(D + f"{s}_truth.csv").assign(off_split=c) for s, c in [("train", 0), ("test", 2)]])
eu = L.drop_duplicates("enroll_id")[["enroll_id", "username", "course_id"]]
E = tr.merge(eu, on="enroll_id", how="left", validate="1:1").rename(columns={"enroll_id": "enrollment_id", "truth": "y"})
print("enrollments without logs:", E.username.isna().sum())
E = E.dropna(subset=["username"]).sort_values("enrollment_id").reset_index(drop=True)
E["enr"] = np.arange(len(E)); E["course"] = E.course_id.map(cmap)
h = E.username.map(lambda u: int(hashlib.md5(("ia-hgnn-2026" + u).encode()).hexdigest(), 16) % 100)
E["lr_split"] = np.where(h < 70, 0, np.where(h < 80, 1, 2))
E["off_split_v"] = np.where((E.off_split == 0) & (h % 10 == 0), 1, E.off_split)
E["user"] = E.username.astype("category").cat.codes
E[["enr", "enrollment_id", "user", "course", "y", "off_split", "off_split_v", "lr_split"]].to_parquet(P + "enrollments.parquet")
print(E.groupby("off_split_v").y.agg(["size", "mean"]))
# ---- objects (category = majority action family; course = course of first record)
ob = L.dropna(subset=["object"]).assign(fam=lambda d: d.action.map(FAM))
O = ob.groupby("object").agg(course_id=("course_id", "first"), category=("fam", lambda s: s.value_counts().index[0])).reset_index()
O["obj"] = np.arange(len(O)); O["course"] = O.course_id.map(cmap); O["module_id"] = O.object
O["category"] = O.category.astype("category"); O["start_day"] = np.nan
O[["obj", "module_id", "course", "category", "start_day"]].to_parquet(P + "objects.parquet"); omap = dict(zip(O.object, O.obj))
# ---- events
eid2enr = dict(zip(E.enrollment_id, E.enr)); L = L[L.enroll_id.isin(eid2enr)]
enr = L.enroll_id.map(eid2enr).values.astype(np.int32)
c0 = cs.set_index("course").start.values[E.course.values[enr]]
sec = (L.t.values - c0) / np.timedelta64(1, "s")
A = pd.DataFrame({"enr": enr, "day": (sec // 86400).astype(np.int16), "sec": sec.astype(np.int32),
                  "ev": L.action.map(amap).astype(np.int8).values, "src": np.zeros(len(L), np.int8),
                  "obj": L.object.map(omap).fillna(-1).astype(np.int32).values})
print("events", len(A), "day range", A.day.min(), A.day.max(), "unknown obj", (A.obj < 0).mean().round(3), "objects", len(O))
A.to_parquet(P + "events.parquet")
