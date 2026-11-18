"""A picture of a series' walls over its merged page ($0, reads only): each border between neighbouring lots drawn
green where the walls see a printed district line (odds > 0) and the published map has a border there, red where the
published map has a border the walls miss, blue where the walls see a line the published map does not.

    uv run python scripts/all_years/walls_picture.py SERIES X Y [--size PX] [--quarter nw]

X, Y: the centre, EPSG:2227 ft.  Writes <series' folder>/crops/walls_<x>_<y>.png."""
import argparse

import cv2
import numpy as np

from sfzmaps.solver import grid, paths
from sfzmaps.solver.city import city


def main():
    import sys
    sys.path.insert(0, str(paths.config.PIPELINE_DIR / "scripts" / "all_years"))
    from score_series import published_codes
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--quarter", default="nw")
    a = ap.parse_args()
    folder = paths.solver_dir(a.series)
    C = city()
    pub = published_codes(a.series, C.lots["blklot"].astype(str).values)
    lo = np.load(folder / "walls" / f"walls_{a.quarter}.npz")["lo_ab"]
    grown = np.load(paths.LOTS_DIR / "lots.npz")["grown"]
    ink = cv2.imread(str(folder / "page" / "ink.png"), cv2.IMREAD_GRAYSCALE)
    c, r = np.floor(grid.to_grid(np.array([[a.x, a.y]]))).astype(int)[0]
    h = a.size // 2
    r0, r1, c0, c1 = max(r - h, 0), min(r + h, ink.shape[0]), max(c - h, 0), min(c + h, ink.shape[1])
    g = grown[r0:r1, c0:c1]
    view = cv2.cvtColor(np.where(ink[r0:r1, c0:c1] < 128, 120, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(C.a, C.b))}
    for dy, dx in ((0, 1), (1, 0)):
        A, B = g[:g.shape[0] - dy, :g.shape[1] - dx], g[dy:, dx:]
        rr, cc = np.nonzero(A != B)
        for y, x in zip(rr, cc):
            i, j = int(A[y, x]), int(B[y, x])
            k = key.get((min(i, j), max(i, j)))
            if k is None:
                continue
            border = pub[i] != "" and pub[j] != "" and pub[i] != pub[j]
            line = lo[k] > 0
            if line and border:
                view[y, x] = (0, 160, 0)
            elif border:
                view[y, x] = (0, 0, 230)
            elif line:
                view[y, x] = (230, 80, 0)
    out = folder / "crops" / f"walls_{int(a.x)}_{int(a.y)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), view)
    print(out)


if __name__ == "__main__":
    main()
