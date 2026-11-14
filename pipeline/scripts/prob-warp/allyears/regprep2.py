"""Inputs for the direction-aware direct fit (regfit2.py), sheet NN:
  centres  block-line-weight ink centreline pixels (as regprep.py), each with the direction its ink runs
           locally (structure tensor of the ink, 2 px smoothing)
  targets  one distance map per 10-degree direction bin (18 bins over 0-180): distance to the nearest centre
           running within 20 deg of the bin's direction; cropped to the record points' extent + 60 px; uint8,
           capped at 255
  points   the 2000 block-line points (as regprep.py) with their own direction (the block line's tangent
           there, on the page) and its bin
-> reg2_NN.npz.  usage: regprep2.py NN"""
import json
import sys

import cv2
import numpy as np
import shapely
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

from ay import *  # noqa: F403  (NN, S, D, PART, year, half)
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
H, W = page.shape
ink = page < 128
T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
lots = read.load_parcels(year=year())
part = PART()
area = part.difference(lots[lots["block_num"] == "1300"].union_all().buffer(8.0))
pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [W, 0], [W, H], [0, H]], float)))
blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
lines = shapely.unary_union(list(blocks.boundary)).intersection(area)
pts, tang = [], []
for ln in shapely.get_parts(lines):
    if ln.geom_type != "LineString" or ln.length < 4:
        continue
    for v in np.arange(0, ln.length, 2.0):
        a = np.array(ln.interpolate(max(0, v - 3)).coords[0][:2]); b = np.array(ln.interpolate(min(ln.length, v + 3)).coords[0][:2])
        pts.append(ln.interpolate(v).coords[0][:2]); tang.append((a, b))
pts = np.array(pts)
# speed E8: only the crop extent is used downstream (later steps read targets, origin, shape), and it depends only on the
# extreme points: those are found with the fit's linear part and placed exactly (cropext.py; the same extent as placing
# every point on all 70 sheets checked).  points / pbin below hold just those placed points (nothing reads them).
from cropext import extent  # noqa: E402
x0, y0, x1, y1, _sel, px, pdir = extent(T0, pts, np.array([a for a, _ in tang]), np.array([b for _, b in tang]), W, H)
sub = ink[y0:y1, x0:x1]
half = ndimage.distance_transform_edt(sub)
sk = skeletonize(sub)
import ay as _ay
HALF = _ay.half(ink)
centres = sk & (half <= HALF)
f = ndimage.gaussian_filter(sub.astype(np.float32), 1.0)
gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
phi = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))          # dominant gradient direction
cdir = (phi + 90.0) % 180                                        # the ink runs across it
BINS = np.arange(0, 180, 10.0)
targets = np.zeros((len(BINS), y1 - y0, x1 - x0), np.uint8)
for i, b in enumerate(BINS):
    dd = np.abs((cdir - b + 90) % 180 - 90)
    m = centres & (dd <= 20.0)
    targets[i] = np.minimum(ndimage.distance_transform_edt(~m), 255).astype(np.uint8)
pbin = (np.round(pdir / 10.0).astype(int)) % len(BINS)
np.savez_compressed(S + f"reg2_{NN}.npz", targets=targets, points=(px - [x0, y0]).astype(np.float32), pbin=pbin,
                    origin=np.array([x0, y0]), shape=np.array([H, W]))
own = targets[pbin, np.clip(np.round(px[:, 1] - y0).astype(int), 0, y1 - y0 - 1), np.clip(np.round(px[:, 0] - x0).astype(int), 0, x1 - x0 - 1)]
print(f"p{NN}: crop {x1 - x0}x{y1 - y0} at ({x0},{y0}); centres {int(centres.sum())}; points {len(px)}; "
      f"extreme points only (E8): within 1 px of a same-direction centre {np.mean(own <= 1):.1%}, within 2 px {np.mean(own <= 2):.1%}")
