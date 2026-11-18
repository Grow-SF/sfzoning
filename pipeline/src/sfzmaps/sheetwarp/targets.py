"""What the fits aim at: the block-side points of a sheet, the printed block lines by direction, and how far a
placement of the points is from them."""
from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps.sheetwarp.sheet import NothingToFit, Sheet, distance_transform
from sfzmaps.transform import Correction

DIRECTIONS = np.arange(0, 180, 10.0)       # the 18 directions of the maps (degrees)


@dataclass
class SidePoints:
    """A point every 2 ft along every side of every block of the sheet inside its drawn map: `pts`, the `sid` of
    the side it is on, and the ends `ta`, `tb` of its tangent (map ft); `px`, `pa`, `pb` are the same placed on the
    page by today's placement."""
    pts: np.ndarray
    sid: np.ndarray
    ta: np.ndarray
    tb: np.ndarray
    px: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    wt: np.ndarray | None = None
    kind: np.ndarray | None = None


@dataclass
class InkCentres:
    """The centre line (skeleton) of the ink on a crop and, at each of its pixels (in `skeleton`'s row-major order),
    the ink's half width (px: the distance to the nearest paper pixel) and the direction the ink runs there
    (degrees, 0-180: the structure tensor, 2 px smoothing)."""
    skeleton: np.ndarray
    half_width: np.ndarray
    direction: np.ndarray


@dataclass
class Targets:
    """For each of 18 directions (every 10 degrees), the distance (px, capped at 255) from every pixel of the crop
    to the nearest printed block-line centre running that way.  `origin` is the crop's top-left page pixel;
    `ink_centres`, the crop's ink centres the maps were made from (the block-line weights use them too)."""
    maps: np.ndarray
    origin: np.ndarray
    page_shape: tuple
    ink_centres: InkCentres | None = None


def block_side_points(sheet: Sheet) -> SidePoints:
    """Sides are cut where they turn more than 30 degrees; each point has its tangent's two ends, 3 ft either way."""
    T0 = sheet.transform
    lots, area = sheet.lots, sheet.area(cuts=2)
    Hp, Wp = sheet.page.shape
    pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], float)))
    blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
    pts, sid_, ta, tb = [], [], [], []
    sid = 0
    for g in blocks.boundary:
        for ring in shapely.get_parts(g):
            for ln in shapely.get_parts(ring.intersection(area)):
                if ln.geom_type != "LineString" or ln.length < 4:
                    continue
                c = np.asarray(shapely.simplify(ln, 2.0).coords)[:, :2]
                if len(c) < 2:
                    continue
                ang = np.degrees(np.arctan2(*np.diff(c, axis=0)[:, ::-1].T))
                turn = np.abs((np.diff(ang) + 180) % 360 - 180)
                cuts = [0] + [i + 1 for i in np.flatnonzero(turn > 30)] + [len(c) - 1]
                for a, b in zip(cuts[:-1], cuts[1:]):
                    sl = shapely.LineString(c[a:b + 1])
                    if sl.length < 2:
                        continue
                    for v in np.arange(0, sl.length, 2.0):
                        pts.append(sl.interpolate(v).coords[0][:2]); sid_.append(sid)
                        ta.append(sl.interpolate(max(0, v - 3)).coords[0][:2])
                        tb.append(sl.interpolate(min(sl.length, v + 3)).coords[0][:2])
                    sid += 1
    if not pts:
        raise NothingToFit(f"{sheet.series}/{sheet.name}: no block sides inside the drawn map")
    pts, sid_, ta, tb = np.array(pts), np.array(sid_), np.array(ta), np.array(tb)
    # Placed through the grid (grid_to_pixel), always one call on each whole array: its iteration stops on the
    # batch's largest step, so placing the points in other batches would change the values slightly.
    page_shape = sheet.page.shape
    return SidePoints(pts=pts, sid=sid_, ta=ta, tb=tb, px=T0.grid_to_pixel(pts, page_shape),
                      pa=T0.grid_to_pixel(ta, page_shape), pb=T0.grid_to_pixel(tb, page_shape))


def _crop(T0, pts, ta, tb, W, H):
    """the box round the points that land on the page with a usable direction, plus 60 px.  Only its extreme points
    matter, so only candidates for them are placed (through the grid, `grid_to_pixel`): each point placed by the
    placement's linear part alone (to_pixel's own first guess), those within 200 px of the page, the 3000 most
    extreme each way along x and y."""
    T0._ensure_fit()
    lin = T0._pm + (pts - T0.to_world(T0._pm[None])) @ T0._lin_inv.T
    near = np.flatnonzero((lin[:, 0] > -200) & (lin[:, 0] < W + 200) & (lin[:, 1] > -200) & (lin[:, 1] < H + 200))
    pick = set()
    for c in (0, 1):
        o = np.argsort(lin[near][:, c], kind="stable")
        pick.update(o[:3000].tolist()); pick.update(o[-3000:].tolist())
    sel = near[np.array(sorted(pick))]
    px = T0.grid_to_pixel(pts[sel], (H, W))
    dv = T0.grid_to_pixel(tb[sel], (H, W)) - T0.grid_to_pixel(ta[sel], (H, W))
    pdir = np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180
    ok = np.isfinite(px).all(1) & np.isfinite(pdir) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    p = px[ok]
    if not len(p):
        raise NothingToFit("no block-side point lands on the page")
    return (max(0, int(p[:, 0].min()) - 60), max(0, int(p[:, 1].min()) - 60),
            min(W, int(p[:, 0].max()) + 60), min(H, int(p[:, 1].max()) + 60))


def ink_centres(ink: np.ndarray) -> InkCentres:
    """The ink's centre line on a crop, with its half width and direction there."""
    skeleton = skeletonize(ink)
    f = ndimage.gaussian_filter(ink.astype(np.float32), 1.0)
    gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
    Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0)[skeleton] for v in (gx * gx, gx * gy, gy * gy))
    direction = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180     # the ink runs across its gradient
    return InkCentres(skeleton=skeleton, half_width=distance_transform(ink)[skeleton], direction=direction)


def runs_within_20_degrees(direction: np.ndarray, b: float) -> np.ndarray:
    """whether each direction (degrees, 0-180) is within 20 degrees of b, either way round"""
    return np.abs((direction - b + 90) % 180 - 90) <= 20.0


def direction_maps(centres: InkCentres, block_line_half_width: float) -> np.ndarray:
    """(18, h, w) uint8: per direction of DIRECTIONS, each pixel's distance (px, capped at 255) to the nearest ink
    centre no wider than a block line running within 20 degrees of it"""
    rows, cols = np.nonzero(centres.skeleton)
    thin = centres.half_width <= block_line_half_width
    maps = np.zeros((len(DIRECTIONS), *centres.skeleton.shape), np.uint8)
    for i, b in enumerate(DIRECTIONS):
        on = thin & runs_within_20_degrees(centres.direction, b)
        centre = np.zeros(centres.skeleton.shape, bool)
        centre[rows[on], cols[on]] = True
        maps[i] = distance_transform(~centre, cap=255).astype(np.uint8)
    return maps


def fit_targets(sheet: Sheet) -> Targets:
    """On the scan, the centre line of every ink stroke no wider than a block line (`sheet.block_line_half_width`),
    each with the direction its ink runs there (structure tensor, 2 px smoothing); per direction, the distance to
    the nearest such centre running within 20 degrees of it (`direction_maps`).  The maps cover the box round the
    sheet's block lines as today's placement puts them on the page, plus 60 px."""
    T0 = sheet.transform
    H, W = sheet.page.shape
    ink = sheet.ink
    lots, area = sheet.lots, sheet.area(cuts=2)
    pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [W, 0], [W, H], [0, H]], float)))
    blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
    lines = shapely.unary_union(list(blocks.boundary)).intersection(area)
    pts, tang = [], []
    for ln in shapely.get_parts(lines):
        if ln.geom_type != "LineString" or ln.length < 4:
            continue
        for v in np.arange(0, ln.length, 2.0):
            a = np.array(ln.interpolate(max(0, v - 3)).coords[0][:2])
            b = np.array(ln.interpolate(min(ln.length, v + 3)).coords[0][:2])
            pts.append(ln.interpolate(v).coords[0][:2]); tang.append((a, b))
    if not pts:
        raise NothingToFit(f"{sheet.series}/{sheet.name}: no block lines inside the drawn map")
    x0, y0, x1, y1 = _crop(T0, np.array(pts), np.array([a for a, _ in tang]), np.array([b for _, b in tang]), W, H)
    centres = ink_centres(ink[y0:y1, x0:x1])
    return Targets(maps=direction_maps(centres, sheet.block_line_half_width), origin=np.array([x0, y0]),
                   page_shape=(H, W), ink_centres=centres)


def distances(targets: Targets, px, pa, pb) -> np.ndarray:
    """Per point placed at `px` with its tangent from `pa` to `pb`, the distance (px) to the nearest printed
    block-line centre running its way; inf off the crop."""
    B, h, w = targets.maps.shape
    x0, y0 = (int(v) for v in targets.origin)
    dv = pb - pa
    b = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
    q = px - [x0, y0]
    ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
    d = np.full(len(px), np.inf)
    d[ok] = targets.maps[b[ok], np.round(q[ok, 1]).astype(int), np.round(q[ok, 0]).astype(int)].astype(float)
    return d


def measure(targets: Targets, sides: SidePoints, correction: Correction | None = None) -> np.ndarray:
    """How well a placement fits the print: `distances` of today's placement, or of today's plus the correction
    added where today's placement puts each point."""
    px, pa, pb = sides.px, sides.pa, sides.pb
    if correction is not None:
        px, pa, pb = px + correction.u(px), pa + correction.u(pa), pb + correction.u(pb)
    return distances(targets, px, pa, pb)
