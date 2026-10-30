"""Offline re-scoring of saved sheets with patched constants (no API calls; work files untouched).

Re-runs refine -> classify_changes -> score from each sheet's blocks_px.geojson + rough.json.

    uv run python scripts/tuning/harness.py 1948 1 3 7 --smooth 0.1 --coh 4 --minc 3
    uv run python scripts/tuning/harness.py 1960 all --smooth 0.03 --maxit 60

Prints, per sheet: held-out p50/p95 (the pass rule), in-sample share within --tol and within
8 ft, in-sample p95, counts, the reason, the vote status, refine iterations and the two
convergence flags (refine, classify).
"""
import argparse
import json
import warnings

import numpy as np
from shapely.geometry import shape

from _common import blocks_corners
from sfzmaps import config, linework
from sfzmaps import refine as R
from sfzmaps import score as S
from sfzmaps.transform import SheetTransform

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("series")
ap.add_argument("sheets", nargs="+", help="sheet ids, or 'all'")
ap.add_argument("--smooth", type=float, default=config.TPS_SMOOTHING)
ap.add_argument("--coarse", type=float, default=config.COARSE_TPS_SMOOTHING)
ap.add_argument("--tol", type=float, default=config.TOLERANCE_FT)
ap.add_argument("--buffer", type=float, default=0.0, help="px; + grows every extracted block")
ap.add_argument("--coh", type=float, default=4.0, help="classify_changes coherence_ft")
ap.add_argument("--minc", type=int, default=3, help="classify_changes min_cluster")
ap.add_argument("--rcfrac", type=float, default=config.MAX_REAL_CHANGE_FRAC)
ap.add_argument("--maxit", type=int, default=30)
ap.add_argument("--shift", type=float, default=R.SHIFT_TOL_FT)
ap.add_argument("--rounds", type=int, default=R.MAX_CLASSIFY_ROUNDS)
a = ap.parse_args()
config.TPS_SMOOTHING, config.COARSE_TPS_SMOOTHING = a.smooth, a.coarse
config.MAX_REAL_CHANGE_FRAC, R.SHIFT_TOL_FT, R.MAX_CLASSIFY_ROUNDS = a.rcfrac, a.shift, a.rounds
blocks, corners = blocks_corners()
sheets = a.sheets
if sheets == ["all"]:
    sheets = sorted((p.name for p in (config.WORK_DIR / a.series).iterdir() if (p / "rough.json").exists()),
                    key=lambda s: [int(t) if t.isdigit() else t for t in __import__("re").split(r"(\d+)", s)])
print(f"smooth={a.smooth} coarse={a.coarse} tol={a.tol} buffer={a.buffer} coh={a.coh} minc={a.minc} "
      f"rcfrac={a.rcfrac} maxit={a.maxit} shift={a.shift} rounds={a.rounds}")
for sh in sheets:
    d = config.WORK_DIR / a.series / sh
    rough = json.loads((d / "rough.json").read_text())["transform"]
    if rough is None:
        print(f"{sh:8s} no rough fit")
        continue
    bj = json.loads((d / "blocks_px.geojson").read_text())
    if a.buffer:
        P = linework.block_corners_px([shape(f["geometry"]).buffer(a.buffer, join_style="mitre")
                                       for f in bj["features"]])
    else:
        P = np.array([[np.nan if v is None else v for v in c] for c in bj["corners"]], float).reshape(-1, 2)
    res = R.refine(P, corners, SheetTransform.from_json(rough), max_iter=a.maxit)
    res = R.classify_changes(res, tol_ft=a.tol, min_cluster=a.minc, coherence_ft=a.coh)
    sc = S.score(res, tol_ft=a.tol)
    d_in = np.array([m.dist_ft for m in res.matches if m.label in ("inlier", "outlier")])
    print(f"{sh:8s} ho_p50={sc.holdout_p50_ft:.1f} ho_p95={sc.holdout_p95_ft:.1f} "
          f"in_f{a.tol:g}={sc.insample_frac_within:.3f} in_f8={np.mean(d_in <= 8) if len(d_in) else 0:.3f} "
          f"in_p95={sc.insample_p95_ft:.1f} n={sc.n_scored} rc={sc.n_real_change} noeq={sc.n_no_equivalent} "
          f"{sc.reason or 'PASS'} vote={res.vote_status} it={res.iterations} "
          f"conv={int(res.converged)}{int(res.classify_converged)}")
