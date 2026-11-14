"""Choose each sheet's correction and write it into the stand-in data folder the warps are made in (WARP_STAND_IN_CHOSEN,
default pipeline/data/sheet-warp/stand-in-data-chosen; a copy of WARP_STAND_IN, made if missing): the probabilistic
correction where it puts at least as many block-line points within 1 px of the print as the first correction and never
folds, else the first correction.  Writes WARP_FITS/choice.json.  The warps are made after (warp_corrected_sheets.sh).
usage: choose_correction.py"""
import json
import os
import shutil
from pathlib import Path

A = Path(__file__).resolve().parent
PIPE = A.parents[2]
RUN = Path(os.environ.get("WARP_FITS", PIPE / "data" / "sheet-warp" / "fits"))
M1 = Path(os.environ.get("WARP_STAND_IN", PIPE / "data" / "sheet-warp" / "stand-in-data"))
M2 = Path(os.environ.get("WARP_STAND_IN_CHOSEN", PIPE / "data" / "sheet-warp" / "stand-in-data-chosen"))
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
        choice[f"{series}/{sheet}"] = {"fit": None, "why": "no corrected folder in the stand-in data"}
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
