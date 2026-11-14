"""The all-years mirror for graph building (SFZMAPS_DATA_DIR=<scratch>/data_ay2): a copy of data_ay (everything linked
to the real data except each corrected sheet's folder) with each corrected sheet's transform.json pointing at its
chosen correction: prob6 (the probabilistic fit) where it puts at least as many block-line points within 1 px of the
print as p1 and has no fold, else p1.  Every sheet is chosen the same way (until 2026-11-13, 2000 p06 and p07 used a
prob6 fitted on the page with every fitted letter erased; the owner dropped text removal from the warper).  Writes
PW_RUN/choice.json.  Warps are made after (warp_prob.sh).
usage: mirror_prob.py"""
import json
import os
import shutil
from pathlib import Path

A = Path(__file__).resolve().parent
PIPE = A.parents[2]
RUN = Path(os.environ.get("PW_RUN", PIPE / "data" / "prob-warp" / "run"))
M1 = Path(os.environ.get("PW_MIRROR", PIPE / "data" / "prob-warp" / "data_ay"))
M2 = Path(os.environ.get("PW_MIRROR2", PIPE / "data" / "prob-warp" / "data_ay2"))
if not M2.exists():
    shutil.copytree(M1, M2, symlinks=True)
rows = {}
for line in open(RUN / "probsum.jsonl"):
    r = json.loads(line)
    rows[(r["series"], r["sheet"])] = r
choice = {}
for (series, sheet), r in sorted(rows.items()):
    tj_p = M2 / "work" / series / sheet / "transform.json"
    if not tj_p.exists() or tj_p.is_symlink():
        choice[f"{series}/{sheet}"] = {"fit": None, "why": "no corrected folder in the mirror"}
        continue
    p1, pr = r.get("within1_p1", 0), r.get("within1_prob6", 0)
    use = "prob6" if (pr >= p1 and r.get("min_det_prob6", 0) > 0) else "p1"
    f = RUN / series / sheet / f"confit_{sheet}_{use}.npz"
    tj = json.loads(tj_p.read_text())
    tj["correction"] = {"file": str(f)}
    tj_p.write_text(json.dumps(tj))
    choice[f"{series}/{sheet}"] = {"fit": use, "within1_today": r.get("within1_today"), "within1_p1": p1, "within1_prob6": pr,
                                   "gcp_p99_p1": r.get("gcp_miss_px_p99_p1"), "gcp_p99_prob6": r.get("gcp_miss_px_p99_prob6"),
                                   "min_det_prob6": r.get("min_det_prob6")}
json.dump(choice, open(RUN / "choice.json", "w"), indent=1)
n = {}
for v in choice.values():
    n[v["fit"]] = n.get(v["fit"], 0) + 1
print(json.dumps(n))
