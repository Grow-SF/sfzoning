"""The neighbour pairs with border px in a map window, grouped by their codes: the table's and each run's, touching
or across a street, with the median wall probability under each walls folder.

    uv run python scripts/incidents/window_pairs.py Q X0 Y0 X1 Y1 [--run RUN ...] [--walls DIR ...]

Q is the quarter whose held-out walls are read (walls_{Q}.npz); X0 Y0 X1 Y1 the window in map ft.  Runs default to
the stage's (final) and J7 (`import_j7.py`); walls to the stage's (runs.WALLS_DIR).  Reads only."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from common import has_run, run_codes
from sfzmaps.solver import grid, paths, runs
from sfzmaps.solver.city import city


def main(q, window, run_names, walls_dirs) -> None:
    C = city()
    a, b = C.a, C.b
    x0, y0, x1, y1 = window
    codes = {r: run_codes(r) for r in run_names}
    lo = {str(w): np.load(Path(w) / f"walls_{q}.npz")["lo_ab"].astype(float) for w in walls_dirs}
    touching = np.load(paths.LOTS_DIR / "lots.npz")["pairs"][:, 2]
    blklot = C.lots["blklot"].astype(str).values
    c0, r0 = np.floor(grid.to_grid([x0, y1])).astype(int)
    c1, r1 = np.ceil(grid.to_grid([x1, y0])).astype(int)
    G = C.grown_raster[r0:r1, c0:c1]
    keys = a.astype(np.int64) * C.n + b
    found = set()
    for gA, gB in ((G[:, :-1], G[:, 1:]), (G[:-1], G[1:])):
        m = gA != gB
        lo_, hi_ = np.minimum(gA[m], gB[m]).astype(np.int64), np.maximum(gA[m], gB[m]).astype(np.int64)
        k = np.minimum(np.searchsorted(keys, lo_ * C.n + hi_), len(keys) - 1)
        found |= set(k[keys[k] == lo_ * C.n + hi_].tolist())
    rows = defaultdict(list)
    for k in sorted(found):
        i, j = a[k], b[k]
        rows[(f"{C.table[i]}|{C.table[j]}",) + tuple(f"{codes[r][i]}|{codes[r][j]}" for r in run_names)
             + ("touch" if touching[k] else "street",)].append(k)
    print(f"{len(found)} pairs in the window; columns: table, {', '.join(run_names)}, kind; median wall p under "
          + ", ".join(str(w) for w in walls_dirs))
    for key, ks in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        ks = np.array(ks)
        ps = " ".join(f"{np.median(expit(v[ks])):.3f}" for v in lo.values())
        print(f"{len(ks):4} pairs  " + "  ".join(f"{s:18}" for s in key) + f"  p {ps}  e.g. {blklot[a[ks[0]]]}-"
              f"{blklot[b[ks[0]]]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("q", choices=paths.QUARTERS)
    ap.add_argument("window", type=float, nargs=4, metavar="FT")
    ap.add_argument("--run", action="append")
    ap.add_argument("--walls", action="append", type=Path)
    args = ap.parse_args()
    main(args.q, args.window, args.run or [runs.FINAL_RUN] + (["j7"] if has_run("j7") else []),
         args.walls or [runs.WALLS_DIR])
