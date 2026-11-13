"""Step 4a, before the letters: delete the block lines from the scan.

The 2000 block outlines (lots merged by block: the lines this map prints) are drawn onto the page at
the printed block line's width, BLOCK_FT (17 ft, the thin line), plus 1 px each side for the
projection's offset, and the ink under them is deleted.  Nothing else is touched: a letter loses only
the part of it lying on the line.  District (heavy) lines are kept: the heavy edges of the raw scan's
ink graph (the W1 split) are drawn at their own width and their ink is put back.

usage: blocklines.py SERIES SHEET GRAPHDIR OUTDIR
  GRAPHDIR: graph.json + zgraph.json made from the raw scan (for the heavy edges)
  -> OUTDIR/noblock.png (full page), OUTDIR/s4a_detect.png, s4a_after.png"""
import json
import sys

import cv2
import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

series, sheet, gdir, outdir = sys.argv[1:5]
d = config.WORK_DIR / series / sheet
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
gray = cv2.imread(str(d / "page.png"), cv2.IMREAD_GRAYSCALE)
H, W = gray.shape
ink = gray < 128
FT = json.load(open(f"{gdir}/graph.json"))["ft_per_px"]
Z = json.load(open(f"{gdir}/zgraph.json"))
BLOCK_FT = Z["thin_ft"]

corners = t.to_world(np.array([[0, 0], [W, 0], [W, H], [0, H]], float))
lots = read.load_parcels(year=read.series_year(series))
lots = lots[lots.intersects(shapely.Polygon(corners).buffer(200))]
blocks = lots.dissolve(by="block_num").geometry.boundary
paint = np.zeros((H, W), np.uint8)
th = int(round(BLOCK_FT / FT)) + 2
for g in blocks:
    for part in shapely.get_parts(g):
        P = np.round(t.to_pixel(np.asarray(part.coords)[:, :2])).astype(np.int32)
        cv2.polylines(paint, [P], False, 1, th)
keep = np.zeros((H, W), np.uint8)
for e in Z["edges"]:
    if e["kind"] != "heavy":
        continue
    P = np.round(t.to_pixel(np.asarray(e["coords"])[:, :2])).astype(np.int32)
    cv2.polylines(keep, [P], False, 1, max(1, int(round(float(np.median(e["width_ft"])) / FT)) + 2))
dele = ink & (paint > 0) & (keep == 0)
out = gray.copy()
out[dele] = 255
cv2.imwrite(f"{outdir}/noblock.png", out)
det = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
det[dele] = (0, 110, 255)
cv2.imwrite(f"{outdir}/s4a_detect.png", det)
cv2.imwrite(f"{outdir}/s4a_after.png", out)
st = {"blocks": len(blocks), "line_px": th, "ink_px": int(ink.sum()), "deleted_px": int(dele.sum())}
json.dump(st, open(f"{outdir}/blocklines_stats.json", "w"))
print(st)
