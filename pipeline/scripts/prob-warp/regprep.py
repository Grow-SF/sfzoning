"""Inputs for the direct fit (regfit.py), sheet NN, page px:
  target  distance (px) from each page pixel to the nearest printed block-line centre.  Block-line centres
          = the ink's centreline (skeleton) where the ink is block-line weight: distance to paper <= 2 px
          there, i.e. at most ~3-4 px thick (the sheet's measured p90 is 3 px); heavy zoning lines (6 px,
          distance to paper >= 3) are left out so nothing is drawn to them.
  points  every 2 ft along each 2000 block outline inside the drawn map (the sheet's part, no Presidio),
          drawn onto the page by the current fit.
-> reg_NN.npz (target float32 H x W, points N x 2, page shape).  usage: regprep.py NN"""
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
SH = f"2000-digital-zoning-p{NN}"
S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
D = config.WORK_DIR / "2000-digital-zoning" / SH
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
H, W = page.shape
ink = page < 128
T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
half = ndimage.distance_transform_edt(ink)
sk = skeletonize(ink)
centres = sk & (half <= 2.0)
target = ndimage.distance_transform_edt(~centres).astype(np.float32)
lots = read.load_parcels(year=2000)
part = shapely.from_wkt(json.load(open(S + "stepview/city_m2/parts.json"))[NN])
area = part.difference(lots[lots["block_num"] == "1300"].union_all().buffer(8.0))
pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [W, 0], [W, H], [0, H]], float)))
blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
lines = shapely.unary_union(list(blocks.boundary)).intersection(area)
pts = []
for ln in shapely.get_parts(lines):
    if ln.geom_type != "LineString" or ln.length < 4:
        continue
    for v in np.arange(0, ln.length, 2.0):
        pts.append(ln.interpolate(v).coords[0][:2])
pts = np.array(pts)
px = T0.to_pixel(pts)
ok = np.isfinite(px).all(1) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
np.savez_compressed(S + f"reg_{NN}.npz", target=target, points=px[ok].astype(np.float32), world=pts[ok], shape=np.array([H, W]))
print(f"p{NN}: page {W}x{H}; block-line centre px {int(centres.sum())} (heavy-line centre px left out {int((sk & (half > 2.0)).sum())}); "
      f"record points {int(ok.sum())}; at today's fit: within 1 px of a centre {np.mean(target[np.round(px[ok, 1]).astype(int), np.round(px[ok, 0]).astype(int)] <= 1):.1%}, "
      f"within 2 px {np.mean(target[np.round(px[ok, 1]).astype(int), np.round(px[ok, 0]).astype(int)] <= 2):.1%}")
