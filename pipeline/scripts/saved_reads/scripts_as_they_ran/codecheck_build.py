"""Build the code-check page data: every 2000 use code, crops packed 25 to a sprite.
Order: codes Terra or Sol read differently first, then the rest shuffled (seed 7), so a
partial pass is still a fair sample."""
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config

SP = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
OUT = SP / "codecheck"
(OUT / "s").mkdir(parents=True, exist_ok=True)
HALF, UP, PER, COLS = 110, 2, 25, 5
T = 2 * HALF * UP


def norm(s):
    return (s or "").upper().replace(" ", "")


rows = []
for f in sorted(glob.glob(str(SP / "leaders3/gpt-6.1-terra_*.json"))):
    sol = {(r["x"], r["y"]): r for r in json.load(open(f.replace("terra", "sol")))}
    for r in json.load(open(f)):
        o = sol[(r["x"], r["y"])]
        flag = norm(r["read"]) != norm(r["code"]) or norm(o["read"]) != norm(r["code"])
        rows.append({"sid": r["sid"], "code": r["code"], "x": r["x"], "y": r["y"], "flag": flag})
flagged = [r for r in rows if r["flag"]]
rest = [r for r in rows if not r["flag"]]
random.Random(7).shuffle(rest)
rows = flagged + rest

pages = {}
cases = []
for k in range(0, len(rows), PER):
    sprite = np.full(((PER // COLS) * T, COLS * T), 255, np.uint8)
    for j, r in enumerate(rows[k:k + PER]):
        if r["sid"] not in pages:
            pages[r["sid"]] = cv2.imread(str(config.WORK_DIR / "2000-digital-zoning" / r["sid"] / "page.png"),
                                         cv2.IMREAD_GRAYSCALE)
        img = pages[r["sid"]]
        h, w = img.shape
        x0, y0 = int(r["x"]) - HALF, int(r["y"]) - HALF
        tile = np.full((2 * HALF, 2 * HALF), 255, np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + 2 * HALF), min(h, y0 + 2 * HALF)
        tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        tile = cv2.resize(tile, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        rr, cc = divmod(j, COLS)
        sprite[rr * T:(rr + 1) * T, cc * T:(cc + 1) * T] = tile
        n = k + j
        cases.append({"id": f"r{n:04d}", "sid": r["sid"][-3:], "code": r["code"], "x": r["x"], "y": r["y"],
                      "s": k // PER, "j": j, "flag": r["flag"]})
    cv2.imwrite(str(OUT / "s" / f"{k // PER:03d}.png"), sprite, [cv2.IMWRITE_PNG_BILEVEL, 1])
(OUT / "cases.json").write_text(json.dumps(cases, separators=(",", ":")))
print(len(cases), "codes,", len(flagged), "flagged,", len(rows) // PER + 1, "sprites")
