"""Facing-aware targets (sheet NN), same crop as reg2:
  street      white areas of the print mostly (over half their pixels) outside the 2000 blocks as today's fit places
              them; block insides are white areas mostly inside
  facing      each block-weight centre pixel (as reg2) looks 3-5 px to each side across its own direction: street on
              exactly one side -> it faces that way (the direction of its normal pointing into the street); street on
              both sides or neither -> not a block line, left out
  targets     one distance map per 10 deg of FACING (36 bins over 360): distance to the nearest centre facing within
              20 deg of the bin
  sides       for each 2000 block-side point (dm_sides_NN.npz), the unit normal pointing out of its block, in map feet
-> reg3_NN.npz (targets, origin, shape), dm_sides_out_NN.npy.  usage: regprep3.py NN"""
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

NN = sys.argv[1]
S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
D = config.WORK_DIR / "2000-digital-zoning" / f"2000-digital-zoning-p{NN}"
T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
R2 = np.load(S + f"reg2_{NN}.npz")
x0, y0 = (int(v) for v in R2["origin"])
_, h, w = R2["targets"].shape
Hp, Wp = (int(v) for v in R2["shape"])
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
sub = page[y0:y0 + h, x0:x0 + w] < 128
lots = read.load_parcels(year=2000)
pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], float)))
blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
bmask = np.zeros((h, w), np.uint8)
for g in blocks:
    for p in shapely.get_parts(g):
        if p.geom_type != "Polygon":
            continue
        cv2.fillPoly(bmask, [np.round(T0.to_pixel(np.asarray(p.exterior.coords)[:, :2]) - [x0, y0]).astype(np.int32)], 1)
        for r in p.interiors:
            cv2.fillPoly(bmask, [np.round(T0.to_pixel(np.asarray(r.coords)[:, :2]) - [x0, y0]).astype(np.int32)], 0)
lab, n = ndimage.label(~sub)
tot = np.bincount(lab.ravel(), minlength=n + 1)
inb = np.bincount(lab.ravel(), weights=bmask.ravel().astype(float), minlength=n + 1)
street_comp = inb / np.maximum(tot, 1) < 0.5
street_comp[0] = False
street = street_comp[lab]
half = ndimage.distance_transform_edt(sub)
sk = skeletonize(sub)
centres = sk & (half <= 2.0)
f = ndimage.gaussian_filter(sub.astype(np.float32), 1.0)
gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180
ys, xs = np.nonzero(centres)
th = np.radians(cdir[ys, xs])
nxv, nyv = -np.sin(th), np.cos(th)
plus = np.zeros(len(xs), bool); minus = np.zeros(len(xs), bool)
for s in (3, 4, 5):
    for sign, acc in ((1, plus), (-1, minus)):
        qx = np.clip(np.round(xs + sign * s * nxv).astype(int), 0, w - 1)
        qy = np.clip(np.round(ys + sign * s * nyv).astype(int), 0, h - 1)
        acc |= street[qy, qx]
one = plus ^ minus
face = np.degrees(np.arctan2(np.where(plus, nyv, -nyv), np.where(plus, nxv, -nxv))) % 360
print(f"p{NN}: white areas {n}, street {int(street_comp.sum())}; block-weight centre pixels {len(xs)}: facing one way {one.mean():.1%}, "
      f"street both sides {np.mean(plus & minus):.1%}, neither {np.mean(~plus & ~minus):.1%}")
BINS = np.arange(0, 360, 10.0)
targets = np.zeros((len(BINS), h, w), np.uint8)
for i, b in enumerate(BINS):
    dd = np.abs((face - b + 180) % 360 - 180)
    m = np.zeros((h, w), bool)
    sel = one & (dd <= 20.0)
    m[ys[sel], xs[sel]] = True
    targets[i] = np.minimum(ndimage.distance_transform_edt(~m), 255).astype(np.uint8)
np.savez_compressed(S + f"reg3_{NN}.npz", targets=targets, origin=R2["origin"], shape=R2["shape"], points=R2["points"], pbin=R2["pbin"])
SD = np.load(S + f"dm_sides_{NN}.npz")
t = SD["tb"] - SD["ta"]
t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
nw = np.column_stack([-t[:, 1], t[:, 0]])
U = shapely.unary_union(list(blocks))
probe = SD["pts"] + 3.0 * nw
inside = shapely.contains_xy(U, probe[:, 0], probe[:, 1])
out = np.where(inside[:, None], -nw, nw)            # pointing out of the block, into the street
np.save(S + f"dm_sides_out_{NN}.npy", out)
print(f"sides points {len(out)}; block on the +normal side for {inside.mean():.1%}")
