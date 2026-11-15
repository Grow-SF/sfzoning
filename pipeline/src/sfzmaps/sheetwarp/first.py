"""The first correction of a sheet's placement: which block sides the print already holds, where the others should
move, and the smooth displacement field that does both."""
import heapq
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree

from sfzmaps import linework
from sfzmaps.sheetwarp.sheet import NothingToFit, Sheet
from sfzmaps.sheetwarp.targets import SidePoints, Targets
from sfzmaps.transform import Correction

log = logging.getLogger(__name__)

HOLD, HOLD_CP, RATIO, RIP, CAP = 100.0, 1000.0, 2.0, 1000.0, 3.0
GRID_SPACING = 50


def held_sides(sheet: Sheet, sides: SidePoints, d_today: np.ndarray) -> np.ndarray:
    """The sides the print already holds, as a bool per side: a block side is held if every one of its points is
    within 1 px of its printed line under today's placement (`d_today`, from `measure`), leaving out points where
    the printed line is hidden under heavy ink (ink wider than a block line within 3 px) and points within 2 px of
    the side's ends (the corner, where the line running its way stops).  The first correction keeps held sides in
    place."""
    d, sid, p = d_today, sides.sid, sides.px
    heavy = ndimage.maximum_filter((sheet.distance_to_paper > sheet.block_line_half_width).astype(np.uint8), size=7) > 0
    H, W = sheet.ink.shape
    ok = np.isfinite(p).all(1)
    xi = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
    on_heavy = ok & heavy[yi, xi]
    n = sid.max() + 1                     # position along the side: the points are every 2 ft, in order
    first = np.full(n, len(sid))
    last = np.zeros(n, int)
    idx = np.arange(len(sid))
    np.minimum.at(first, sid, idx)
    np.maximum.at(last, sid, idx)
    ft_per_px = sheet.ft_per_px
    near_end = ((idx - first[sid]) * 2.0 / ft_per_px <= 2.0) | ((last[sid] - idx) * 2.0 / ft_per_px <= 2.0)
    counted = ok & ~on_heavy & ~near_end
    bad = np.zeros(n, bool)
    np.logical_or.at(bad, sid, counted & (d > 1.0))
    held = ~bad                           # a side with no measurable point lies under heavy ink: on the ink, held
    length = np.bincount(sid, minlength=n) * 2.0
    log.info("%s/%s: sides %d; held sides %d (%.1f%% of the length); free %d", sheet.series, sheet.name, n,
             held.sum(), 100 * length[held].sum() / length.sum(), (~held).sum())
    return held


@dataclass
class _Inputs:
    """What the first correction holds and what it moves, on the crop of the targets (crop px): the points of the
    held sides with their normals (held across the line, free to slide along it), the held control points (the
    placement's control points with no free side within 4 px), and the thrown-out corners (block corners the
    placement matched 15-60 ft away and threw out, with where the scan prints them)."""
    held_line: np.ndarray
    held_n: np.ndarray
    held_cp: np.ndarray
    corner_rec: np.ndarray
    corner_scan: np.ndarray


def _inputs(sheet: Sheet, sides: SidePoints, targets: Targets, held_side: np.ndarray) -> _Inputs:
    T0 = sheet.transform
    x0, y0 = (int(v) for v in targets.origin)
    B, h, w = targets.maps.shape
    sid = sides.sid
    cp = sides.px - [x0, y0]
    dv = sides.pb - sides.pa
    pdir = np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180
    ok = (np.isfinite(cp).all(1) & (cp[:, 0] >= 0) & (cp[:, 0] < w) & (cp[:, 1] >= 0) & (cp[:, 1] < h)
          & np.isfinite(pdir))
    cp, sid, pdir = cp[ok], sid[ok], pdir[ok]
    held = held_side[sid]
    th = np.radians(pdir)
    nrm = np.column_stack([-np.sin(th), np.cos(th)])
    gc_all = [g for g in json.loads((sheet.folder / "transform.json").read_text())["gcps"] if g.get("enabled", True)]
    gc = np.array([[g["px"], g["py"]] for g in gc_all], float)
    gcr = T0.to_pixel(np.array([[g["wx"], g["wy"]] for g in gc_all], float)) - [x0, y0]
    kt = cKDTree(cp)
    gc_free = np.array([bool(np.any(~held_side[sid[kt.query_ball_point(q, 4.0)]]))
                        if len(kt.query_ball_point(q, 4.0)) else False for q in gcr])
    area = sheet.area(cuts=2)
    matches = json.loads((sheet.folder / "refine.json").read_text())["matches"]
    rc = np.array([(m["px"], m["py"], m["wx"], m["wy"]) for m in matches
                   if m["label"] in ("no_equivalent", "outlier") and m["wx"] is not None
                   and np.isfinite(m["dist_ft"]) and 15.0 < m["dist_ft"] <= 60.0], float).reshape(-1, 4)
    rc = rc[shapely.contains_xy(area, rc[:, 2], rc[:, 3])] if len(rc) else rc
    corner_rec = T0.to_pixel(rc[:, 2:4]) if len(rc) else np.zeros((0, 2))
    gch = gc[~gc_free]
    return _Inputs(held_line=cp[held].astype(np.float32), held_n=nrm[held].astype(np.float32),
                   held_cp=(gch - [x0, y0]).astype(np.float32),
                   corner_rec=(corner_rec - [x0, y0]).astype(np.float32),
                   corner_scan=(rc[:, :2] - [x0, y0]).astype(np.float32))


def side_members(sid: np.ndarray, keep: np.ndarray, n_sides: int) -> list[np.ndarray]:
    """per side 0 .. n_sides - 1, the indices of its points where `keep` is True, ascending (one stable sort of the
    side ids, not a mask of every point per side)"""
    order = np.argsort(sid, kind="stable")
    order = order[keep[order]]
    starts = np.searchsorted(sid[order], np.arange(n_sides + 1))
    return [order[a:b] for a, b in zip(starts[:-1], starts[1:])]


@dataclass
class _Shifts:
    """per free side point (crop px): its normal and how far it should move along it"""
    free: np.ndarray
    free_n: np.ndarray
    free_off: np.ndarray


def propagated_shifts(sheet: Sheet, sides: SidePoints, targets: Targets, held_side: np.ndarray) -> _Shifts:
    """Where each free side (not held) should move, carried outward from what the print already holds.
      start    held sides are solved, with shift (0, 0)
      order    repeatedly the unsolved side nearest the solved ones (by its closest point)
      prior    the shift of its 30 nearest solved points, weighted 1 / distance^2
      search   each of its points looks across its own line, from where the prior puts it, for the nearest printed
               block-line centre running its way, up to just under half a street width each way (the line across
               the street is a full street width away)
      solved   prior + the median of the found offsets along the side's normal; if none of its points finds a
               line, the prior"""
    TG = targets.maps
    x0, y0 = (int(v) for v in targets.origin)
    B, h, w = TG.shape
    sid = sides.sid
    P = sides.px - [x0, y0]
    dv = sides.pb - sides.pa
    L = np.linalg.norm(dv, axis=1)
    tng = dv / np.maximum(L, 1e-9)[:, None]
    nrm = np.column_stack([-tng[:, 1], tng[:, 0]])
    pbin = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
    ok = np.isfinite(P).all(1) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
    ns = sid.max() + 1
    members = side_members(sid, ok, ns)
    shift = np.full((len(P), 2), np.nan)
    solved_side = np.zeros(ns, bool)
    for s in range(ns):
        if held_side[s] and len(members[s]):
            shift[members[s]] = 0.0
            solved_side[s] = True
    pending = [s for s in range(ns) if not solved_side[s] and len(members[s])]
    if not pending:
        raise NothingToFit(f"{sheet.series}/{sheet.name}: every block side is held, none is left to move")
    if not solved_side.any():
        raise NothingToFit(f"{sheet.series}/{sheet.name}: no held block side on the crop to carry the shifts from")
    blocks = json.loads((sheet.folder / "blocks_px.geojson").read_text())
    SW, _ = linework.estimate_street_px([shapely.geometry.shape(f["geometry"]) for f in blocks["features"]])
    reach = np.arange(-(SW / 2 - 0.5), SW / 2 - 0.5 + 0.01, 0.5)

    def solved_tree():
        idx = np.flatnonzero(~np.isnan(shift[:, 0]))
        return cKDTree(P[idx]), idx

    def nearest_first(tree, unsolved):
        """the heap of the unsolved sides by their closest point's distance to the solved points in `tree` (one
        query of all their points, on every core: the same distances as a query per side)"""
        if not unsolved:
            return []
        points = np.concatenate([members[t] for t in unsolved])
        starts = np.cumsum([0] + [len(members[t]) for t in unsolved[:-1]])
        heap = list(zip(np.minimum.reduceat(tree.query(P[points], workers=-1)[0], starts).tolist(), unsolved))
        heapq.heapify(heap)
        return heap

    tree, sidx = solved_tree()
    heap = nearest_first(tree, pending)
    k = 0
    while heap:
        _, s = heapq.heappop(heap)
        if solved_side[s]:
            continue
        m = members[s]
        dd, ii = tree.query(P[m].mean(0)[None], k=min(30, len(sidx)))
        dd, ii = np.atleast_1d(dd[0]), np.atleast_1d(ii[0])
        wts = 1.0 / np.maximum(dd, 1.0) ** 2
        prior = (shift[sidx[ii]] * wts[:, None]).sum(0) / wts.sum()
        q = P[m][:, None, :] + prior[None, None, :] + nrm[m][:, None, :] * reach[None, :, None]
        xi = np.clip(np.round(q[..., 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(q[..., 1]).astype(int), 0, h - 1)
        dmap = TG[pbin[m][:, None], yi, xi].astype(float)
        best = np.argmin(dmap + 1e-3 * np.abs(reach)[None, :], axis=1)
        hit = dmap[np.arange(len(m)), best] <= 1.0
        if hit.any():
            off = float(np.median(reach[best[hit]]))
            nn_ = np.median(nrm[m], axis=0)
            sv = prior + off * nn_ / max(np.linalg.norm(nn_), 1e-9)
        else:
            sv = prior
        shift[m] = sv
        solved_side[s] = True
        k += 1
        if k % 50 == 0:
            tree, sidx = solved_tree()
            heap = nearest_first(tree, [t for t in range(ns) if not solved_side[t] and len(members[t])])
    free = ok & ~held_side[sid]
    tgt = np.sum(shift[free] * nrm[free], axis=1)          # target offset along each point's normal
    return _Shifts(free=P[free].astype(np.float32), free_n=nrm[free].astype(np.float32),
                   free_off=tgt.astype(np.float32))


def first_correction(sheet: Sheet, sides: SidePoints, targets: Targets, held: np.ndarray) -> Correction:
    """A smooth displacement field (a grid every 50 px over the crop of the targets, bilinear between grid points)
    added to today's placement, the one that best does all of:
      free sides     move each free side point along its normal by its propagated shift
      free corners   bring each thrown-out corner to where the scan prints it, the miss capped softly at 3 px
      held sides     don't move held side points across their line (weight 100; they may slide along it)
      held corners   don't move the held control points at all (weight 1000)
      smooth         keep neighbouring grid points' corrections close
      no spikes      no grid point's correction more than twice its neighbours' average (weight 1000)
    Every term is a mean over its points (smooth and no spikes: per free point), so each weighs the same on every
    sheet.  Fitted with PyTorch on `fit_device()`; the grid comes back as float32 on the CPU."""
    import torch

    from sfzmaps.sheetwarp.device import PointField, fit_device

    shifts = propagated_shifts(sheet, sides, targets, held)
    C = _inputs(sheet, sides, targets, held)
    if not len(C.held_line):          # the mean of an empty term is NaN: the whole fit would be NaN
        raise NothingToFit(f"{sheet.series}/{sheet.name}: no held block-side points on the crop for the first correction")
    device = fit_device()
    if device.type == "cpu":
        torch.set_num_threads(8)
    Hp, HN, HC = C.held_line, C.held_n, C.held_cp
    Fp, FN, FO = shifts.free, shifts.free_n, shifts.free_off
    CR, CS = C.corner_rec, C.corner_scan
    x0, y0 = (int(v) for v in targets.origin)
    B, h, w = targets.maps.shape
    SP = GRID_SPACING
    gh, gw = int(np.ceil((h - 1) / SP)) + 1, int(np.ceil((w - 1) / SP)) + 1
    field_at_free, field_at_held = PointField(Fp, (gh, gw), SP, device), PointField(Hp, (gh, gw), SP, device)
    field_at_corners, field_at_held_cp = PointField(CR, (gh, gw), SP, device), PointField(HC, (gh, gw), SP, device)
    CRt, CSt, HNt, FNt, FOt = (torch.tensor(a).to(device) for a in (CR, CS, HN, FN, FO))
    G = torch.zeros(1, 2, gh, gw, device=device, requires_grad=True)
    t0 = time.time()
    opt = torch.optim.Adam([G], lr=0.25)
    for it in range(1500):
        opt.zero_grad()
        data = torch.mean((torch.sum(field_at_free.field(G) * FNt, dim=1) - FOt) ** 2)
        dc = (torch.linalg.norm(CRt + field_at_corners.field(G) - CSt, dim=1) if len(CR)
              else torch.zeros(1, device=device))
        cor = torch.mean(CAP * CAP * dc * dc / (dc * dc + CAP * CAP)) / (CAP * CAP)
        held_term = torch.mean(torch.sum(field_at_held.field(G) * HNt, dim=1) ** 2)
        heldcp = torch.mean(torch.sum(field_at_held_cp.field(G) ** 2, dim=1))
        smooth = (torch.sum((G[:, :, 1:, :] - G[:, :, :-1, :]) ** 2)
                  + torch.sum((G[:, :, :, 1:] - G[:, :, :, :-1]) ** 2)) / len(Fp)
        mag = torch.sqrt(torch.sum(G[0] ** 2, dim=0) + 1e-6)
        nb = 0.25 * (mag[:-2, 1:-1] + mag[2:, 1:-1] + mag[1:-1, :-2] + mag[1:-1, 2:])
        rip = torch.sum(torch.relu(mag[1:-1, 1:-1] - RATIO * nb) ** 2) / len(Fp)
        (data + cor + HOLD * held_term + HOLD_CP * heldcp + smooth + RIP * rip).backward()
        opt.step()
    with torch.no_grad():
        uh = field_at_held.field(G).cpu().numpy()
        umax = float(torch.linalg.norm(G[0], dim=0).max())
    hm = np.abs(np.sum(uh * HN, axis=1))
    log.info("%s/%s: first correction (on %s): free side points %d, thrown-out corners %d, held side points %d, held "
             "control points %d; biggest correction %.1f px; held sides moved across their line px p99 %.2f; %.0fs",
             sheet.series, sheet.name, device, len(Fp), len(CR), len(Hp), len(HC), umax, np.percentile(hm, 99),
             time.time() - t0)
    return Correction(grid=G.detach().cpu().numpy()[0], spacing=SP, origin=np.array([x0, y0], float))
