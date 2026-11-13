"""Line up each straight side of each block outline with the print, on its own.

Block outlines: the 2000 lots merged by block, on the page.  Each ring is split into sides: runs of
segments turning less than TURN deg.  A side with ON_PRINT or more of its length on ink is left alone;
otherwise it is slid across itself (along its normal) by up to MAX_OFF px either way, and the offset with
the most ink along it is kept (ties: the smaller move).  A printed line is solid ink; letters are broken.

usage: sidealign.py OUTDIR [x,y ...]  -> SP/walls.json (page px segments [x0,y0,x1,y1]),
       OUTDIR/align_r03.png (close-ups: lot outline green, lined-up sides orange)"""
import json
import math
import sys

import cv2
import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

SP = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad"
OUT = sys.argv[1]
show = [tuple(map(int, v.split(","))) for v in sys.argv[2:]]
TURN, ON_PRINT, MAX_OFF = 20.0, 0.80, 6
import os
d = config.WORK_DIR / os.environ["SERIES"] / os.environ["SHEET"]
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
gray = cv2.imread(str(d / "page.png"), cv2.IMREAD_GRAYSCALE)
ink = gray < 128
H, W = ink.shape
lots = read.load_parcels(year=read.series_year(os.environ["SERIES"]))
lots = lots[lots.intersects(shapely.Polygon(t.to_world(np.array([[0, 0], [W, 0], [W, H], [0, H]], float))))]
blocks = lots.dissolve(by="block_num").geometry


def sides(P):
    """Split a ring/polyline into runs of segments turning less than TURN deg from the run's first one."""
    out, cur = [], [P[0]]
    base = None
    for p0, p1 in zip(P[:-1], P[1:]):
        dv = p1 - p0
        if np.hypot(*dv) < 0.3:
            continue
        a = math.degrees(math.atan2(dv[1], dv[0]))
        if base is not None and min(abs(a - base) % 360, 360 - abs(a - base) % 360) > TURN:
            out.append(np.asarray(cur))
            cur, base = [p0], None
        if base is None:
            base = a
        cur.append(p1)
    if len(cur) > 1:
        out.append(np.asarray(cur))
    return out


def ink_along(S, off, nrm):
    q = np.round(S + nrm * off).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < W) & (q[:, 1] < H)
    return float(ink[q[ok, 1], q[ok, 0]].mean()) if ok.any() else 0.0


walls, kept, moved, nomove, drawn = [], 0, 0, 0, []
for blk, g in blocks.items():
    for ring in shapely.get_parts(g.boundary):
        P = t.to_pixel(np.asarray(ring.coords)[:, :2])
        for sd in sides(P):
            L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sd, axis=0), axis=1))]
            if L[-1] < 1:
                continue
            s = np.arange(0.5, L[-1] - 0.5, 0.5) if L[-1] > 1.5 else np.array([L[-1] / 2])
            S = np.c_[np.interp(s, L, sd[:, 0]), np.interp(s, L, sd[:, 1])]
            dv = sd[-1] - sd[0]
            u = dv / max(1e-9, np.linalg.norm(dv))
            nrm = np.array([-u[1], u[0]])
            before = ink_along(S, 0, nrm)
            off = 0
            if before >= ON_PRINT:
                kept += 1
            else:
                best = max(((ink_along(S, o, nrm), -abs(o), o) for o in range(-MAX_OFF, MAX_OFF + 1)))
                if best[0] > before:
                    off = best[2]
                    moved += 1
                else:
                    nomove += 1
            Q = sd + nrm * off
            walls += [[*p0, *p1] for p0, p1 in zip(Q[:-1], Q[1:])]
            drawn.append((sd, Q, off))
import os
json.dump(np.round(np.asarray(walls), 2).tolist(), open(os.environ.get("WALLS_OUT") or f"{SP}/walls.json", "w"))
print(f"sides: {kept + moved + nomove} | on the print already: {kept} | slid: {moved} | below {ON_PRINT:.0%}, no better spot: {nomove} | wall segments {len(walls)}")

rows = []
for X, Y in show:
    x0, y0, x1, y1 = X - 60, Y - 60, X + 130, Y + 90
    Z = 4
    img = cv2.resize(cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    img = cv2.addWeighted(img, 0.6, np.full_like(img, 255), 0.4, 0)
    tp = lambda p: (int((p[0] - x0) * Z + Z / 2), int((p[1] - y0) * Z + Z / 2))
    for sd, Q, off in drawn:
        if not ((sd[:, 0] > x0 - 20) & (sd[:, 0] < x1 + 20) & (sd[:, 1] > y0 - 20) & (sd[:, 1] < y1 + 20)).any():
            continue
        cv2.polylines(img, [np.array([tp(p) for p in sd], np.int32)], False, (40, 170, 40), 2)
        if off:
            cv2.polylines(img, [np.array([tp(p) for p in Q], np.int32)], False, (0, 120, 255), 2)
    head = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"around {X},{Y}: lot outline (green), sides slid onto the print (orange)", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    rows.append(np.vstack([head, img, np.full((10, img.shape[1], 3), 255, np.uint8)]))
if rows:
    Wd = max(r.shape[1] for r in rows)
    stamp = np.full((40, Wd, 3), 235, np.uint8)
    cv2.putText(stamp, "RUN: align r03 - each block side lined up on its own (only where < 80% on ink)", (6, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 160), 2, cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/align_r03.png", np.vstack([stamp] + [np.hstack([r, np.full((r.shape[0], Wd - r.shape[1], 3), 255, np.uint8)]) for r in rows]))
