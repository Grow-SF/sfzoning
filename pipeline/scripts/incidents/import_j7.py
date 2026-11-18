"""Bring J7 (the R&D solver's best run, the yardstick) into this build's lot and pair order, for comparisons only.

    uv run python scripts/incidents/import_j7.py RND_DIR LOT_MAP

RND_DIR is J7's R&D data folder (`pipeline/data/rnd/zon2000` of the checkout that ran it): its joined answer
c_bub_J7cv.npz (names, pred) and wall odds zodds_Q1x{q}.npz (lo_ab), both on J7's lots and pairs
(truth_grid.npz pairs).  LOT_MAP is an .npy giving, per lot of this build, J7's index of the same lot.

Writes into data/solver/experiments/incidents-j7/ (never an input of the package):
  runs/j7.npz        J7's code per lot, in this build's lot order (`city.save_run`; read with `load_run("j7", folder)`)
  walls_{q}.npz      J7's held-out wall odds on this build's pairs (lo_ab); a pair J7 lacks gets 0 (neutral: no cost
                     either way), as the solver plan's Task 10 check did; a solve reads them with --walls
  lot_of_new.npy     the lot map, copied, so later scripts need no path outside the data folder
  source.json        where each came from, the files' sha256, and the matched pair counts
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city, save_run
from sfzmaps.solver.codes import canon

J7_DIR = runs.EXPERIMENTS_DIR / "incidents-j7"


def sha256(f: Path) -> str:
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main(rnd: Path, lot_map: Path) -> None:
    C = city()
    j7_of_new = np.load(lot_map)
    if len(j7_of_new) != C.n:
        raise SystemExit(f"{lot_map} maps {len(j7_of_new):,} lots; this build has {C.n:,}")
    J7_DIR.mkdir(parents=True, exist_ok=True)
    (J7_DIR / "runs").mkdir(exist_ok=True)

    R = np.load(rnd / "c_bub_J7cv.npz", allow_pickle=True)
    j7_codes = np.array([canon(x) for x in np.asarray(R["names"]).astype(str)])[R["pred"]]
    ok = j7_of_new >= 0
    codes = np.where(ok, j7_codes[np.maximum(j7_of_new, 0)], "")
    save_run(J7_DIR / "runs" / "j7.npz", codes)
    right = C.scored & (codes == C.table)
    print(f"J7's answer: {C.n:,} lots, {int((~ok).sum())} unmatched; scored lots right "
          f"{int(right.sum()):,} of {int(C.scored.sum()):,}; wrong {int((C.scored & ~right).sum()):,}")

    j7_pairs = np.load(rnd / "truth_grid.npz")["pairs"][:, :2].astype(np.int64)
    n7 = int(j7_pairs.max()) + 1
    key7 = np.minimum(j7_pairs[:, 0], j7_pairs[:, 1]) * n7 + np.maximum(j7_pairs[:, 0], j7_pairs[:, 1])
    order = np.argsort(key7)
    ja, jb = j7_of_new[C.a], j7_of_new[C.b]
    k = np.minimum(ja, jb) * n7 + np.maximum(ja, jb)
    pos = np.clip(np.searchsorted(key7[order], k), 0, len(key7) - 1)
    hit = (ja >= 0) & (jb >= 0) & (key7[order][pos] == k)
    row = np.where(hit, order[pos], -1)
    source = {"rnd": str(rnd), "lot_map": str(lot_map), "c_bub_J7cv.npz": sha256(rnd / "c_bub_J7cv.npz"),
              "lot_map_sha256": sha256(lot_map), "pairs": len(C.a), "pairs_found_in_j7": int(hit.sum()), "walls": {}}
    for q in paths.QUARTERS:
        f = rnd / f"zodds_Q1x{q}.npz"
        lo = np.load(f)["lo_ab"].astype(np.float64)
        odds = np.where(hit, lo[np.maximum(row, 0)], 0.0)
        odds = np.nan_to_num(odds, nan=0.0)
        np.savez_compressed(J7_DIR / f"walls_{q}.npz", lo_ab=odds.astype(np.float32))
        source["walls"][q] = {"file": f.name, "sha256": sha256(f), "odds_above_0": int((odds > 0).sum())}
        print(f"J7's walls {q}: {int(hit.sum()):,} of {len(C.a):,} pairs found in J7 ({hit.mean():.2%}); the rest 0")
    shutil.copyfile(lot_map, J7_DIR / "lot_of_new.npy")
    (J7_DIR / "source.json").write_text(json.dumps(source, indent=1) + "\n")
    print(f"wrote {J7_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
