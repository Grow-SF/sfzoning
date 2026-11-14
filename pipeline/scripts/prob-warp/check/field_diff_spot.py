"""A close-up of where two runs' prob6 fields differ most on a sheet: the scan around that spot, with the 2000
block-line points as each run places them (dm_<sheet>_prob6.npz 'px'): run A blue, run B red.  With PW_CHAIN set, the
ink text removal takes out of the page prob6 fits (ay_regprep3: noblock.png ink not in <sheet>_c1/clean.png) is orange.
Written 3x enlarged to OUT, with a key along the top (it reads A as with text removal, B as without: text_ablation.sh).  usage: [PW_CHAIN=...] field_diff_spot.py RUN_A RUN_B SERIES SHEET
OUT.png [HALF_PX]"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config

a, b, ser, sh, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
half = int(sys.argv[6]) if len(sys.argv) > 6 else 120
za, zb = np.load(a / ser / sh / f"confit_{sh}_prob6.npz"), np.load(b / ser / sh / f"confit_{sh}_prob6.npz")
m = np.hypot(*(za["Gc"] - zb["Gc"]))
iy, ix = np.unravel_index(np.argmax(m), m.shape)
sp, o = float(za["spacing"]), np.asarray(za["origin"], float)
cx, cy = int(o[0] + ix * sp), int(o[1] + iy * sp)
page = cv2.imread(str(config.WORK_DIR / ser / sh / "page.png"), cv2.IMREAD_GRAYSCALE)
x0, y0 = max(0, cx - half), max(0, cy - half)
crop = cv2.cvtColor(page[y0:cy + half, x0:cx + half], cv2.COLOR_GRAY2BGR)
crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
crop = (0.55 * crop + 0.45 * 255).astype(np.uint8)          # the scan pale, so the points read
chain = os.environ.get("PW_CHAIN")
if chain:
    nb = cv2.imread(str(Path(chain) / ser / sh / "noblock.png"), cv2.IMREAD_GRAYSCALE) < 128
    cl = cv2.imread(str(Path(chain) / ser / f"{sh}_c1" / "clean.png"), cv2.IMREAD_GRAYSCALE) < 128
    out_ink = (nb & ~cl & (page < 128))[y0:cy + half, x0:cx + half]
    out_ink = np.repeat(np.repeat(out_ink, 3, 0), 3, 1)
    crop[out_ink] = (0, 150, 255)                            # BGR orange
for run, colour in ((a, (255, 80, 0)), (b, (0, 0, 230))):  # BGR: A blue, B red
    px = np.load(run / ser / sh / f"dm_{sh}_prob6.npz")["px"]
    sel = (px[:, 0] >= x0) & (px[:, 0] < cx + half) & (px[:, 1] >= y0) & (px[:, 1] < cy + half)
    for x, y in px[sel]:
        cv2.circle(crop, (int((x - x0) * 3), int((y - y0) * 3)), 2, colour, -1)
key = np.full((34, crop.shape[1], 3), 255, np.uint8)
x = 6
for text, colour in (("with text removal", (255, 80, 0)), ("without", (0, 0, 230)),
                     ("ink text removal took out", (0, 150, 255)) if chain else ("", (0, 0, 0))):
    cv2.putText(key, text, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] + 24
cv2.imwrite(out, np.vstack([key, crop]))
print(f"{sh}: largest difference {m.max():.2f} px at page ({cx}, {cy}); crop {2 * half} px")
