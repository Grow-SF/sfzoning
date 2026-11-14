"""What the fits aim at: the printed block lines, by direction.  On the scan, the centre line of every ink stroke no
wider than a block line (warp_common.half), each with the direction its ink runs there (structure tensor, 2 px
smoothing).  For each of 18 directions (every 10 degrees), a map of the distance from every pixel to the nearest such
centre running within 20 degrees of it (capped at 255 px).  The maps cover the crop the fits work on: the box round
the sheet's block lines as today's placement puts them on the page, plus 60 px.
-> fit_targets.npz (targets: 18 x h x w; origin: the crop's top-left page pixel; shape: the page's)."""
import json

import cv2
import numpy as np
import shapely
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps.transform import SheetTransform

from warp_common import D, S, SHEET, half, lots_and_area


def crop(T0, pts, ta, tb, W, H):
    """the box round the points that land on the page with a usable direction, plus 60 px.  Only its extreme points
    matter, so only candidates for them are placed exactly: each point placed by the placement's linear part alone
    (to_pixel's own first guess), those within 200 px of the page, the 3000 most extreme each way along x and y."""
    T0._ensure_fit()
    lin = T0._pm + (pts - T0.to_world(T0._pm[None])) @ T0._lin_inv.T
    near = np.flatnonzero((lin[:, 0] > -200) & (lin[:, 0] < W + 200) & (lin[:, 1] > -200) & (lin[:, 1] < H + 200))
    pick = set()
    for c in (0, 1):
        o = np.argsort(lin[near][:, c], kind="stable")
        pick.update(o[:3000].tolist()); pick.update(o[-3000:].tolist())
    sel = near[np.array(sorted(pick))]
    px = T0.to_pixel(pts[sel])
    dv = T0.to_pixel(tb[sel]) - T0.to_pixel(ta[sel])
    pdir = np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180
    ok = np.isfinite(px).all(1) & np.isfinite(pdir) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    p = px[ok]
    return (max(0, int(p[:, 0].min()) - 60), max(0, int(p[:, 1].min()) - 60),
            min(W, int(p[:, 0].max()) + 60), min(H, int(p[:, 1].max()) + 60))


page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
H, W = page.shape
ink = page < 128
T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
lots, area = lots_and_area()
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
x0, y0, x1, y1 = crop(T0, np.array(pts), np.array([a for a, _ in tang]), np.array([b for _, b in tang]), W, H)
sub = ink[y0:y1, x0:x1]
centres = skeletonize(sub) & (ndimage.distance_transform_edt(sub) <= half(ink))
f = ndimage.gaussian_filter(sub.astype(np.float32), 1.0)
gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180     # the ink runs across its gradient
targets = np.zeros((18, y1 - y0, x1 - x0), np.uint8)
for i, b in enumerate(np.arange(0, 180, 10.0)):
    m = centres & (np.abs((cdir - b + 90) % 180 - 90) <= 20.0)
    targets[i] = np.minimum(ndimage.distance_transform_edt(~m), 255).astype(np.uint8)
np.savez_compressed(S + "fit_targets.npz", targets=targets, origin=np.array([x0, y0]), shape=np.array([H, W]))
print(f"{SHEET}: crop {x1 - x0} x {y1 - y0} px at ({x0}, {y0}); {int(centres.sum())} block-line centre pixels")
