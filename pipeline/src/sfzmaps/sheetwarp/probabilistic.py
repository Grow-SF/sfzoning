"""The probabilistic correction of a sheet's placement.

Every block-side point (placed by today's placement) is explained by the print with a probability:
  printed      it lies on a printed block line running its way: the density across the line, N(d; 0, s), summed over
               every such line's centre pixels (`block_line_weights`: each weighted by how likely it is a block line),
               weight 1 - eps
  not printed  missing from the print or hidden under heavy ink: uniform over every distance a point can have, weight eps
The placement's control points are evidence too: a 2-D Gaussian around each one's matched page pixel.  The correction is
a smooth displacement field (a grid every 50 px, bilinear) that makes all of it together most probable, with a prior
keeping each grid point's correction near zero where no printed line is near, a prior on how much the field may
stretch and shear (the spread of today's own placement), and a steep penalty where it would fold the scan over.  The
spread s starts at half a street width and halves each stage; after each stage eps and sigma (the printed lines' own
spread) are re-estimated from the points; the last stage is at s = sigma.
Points with weights (street edges and later blocks) each count by their weight; only the block sides (kind 0)
measure how well the print is fitted."""
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from sfzmaps import linework
from sfzmaps.sheetwarp.sheet import Sheet
from sfzmaps.sheetwarp.targets import SidePoints, Targets
from sfzmaps.transform import Correction

log = logging.getLogger(__name__)

ITERS = 800                        # optimiser steps per stage
GRID_SPACING = 50


@dataclass
class ProbabilisticInputs:
    """`P`: the points on the crop (float32); `bin`: their direction (one of 18); `street_px`: the street width;
    `origin`: the crop's top-left page pixel; `gbase`, `gpx`: the control points where today's placement puts them
    and where they are matched on the scan; `tau_j`: how much today's placement stretches and shears; `wt`, `kind`:
    each point's weight and kind, for street edges and later blocks."""
    P: np.ndarray
    bin: np.ndarray
    street_px: float
    origin: np.ndarray
    gbase: np.ndarray
    gpx: np.ndarray
    tau_j: float
    wt: np.ndarray | None = None
    kind: np.ndarray | None = None


def probabilistic_inputs(sheet: Sheet, points: SidePoints, targets: Targets) -> ProbabilisticInputs:
    """The inputs on the crop of the targets: every point placed by today's placement, with its direction (one of
    18, every 10 degrees); the street width (blocks_px.geojson); the placement's control points, where today's
    placement puts them and where they are matched on the scan; and how much today's placement stretches and shears
    across the sheet (the spread the correction's own stretch is held to)."""
    T0 = sheet.transform
    TJ = json.loads((sheet.folder / "transform.json").read_text())
    x0, y0 = (int(v) for v in targets.origin)
    B, h, w = targets.maps.shape
    blocks_px = json.loads((sheet.folder / "blocks_px.geojson").read_text())["features"]
    SW, _ = linework.estimate_street_px([shapely.geometry.shape(f["geometry"]) for f in blocks_px])
    pts, ta, tb = points.px, points.pa, points.pb
    P = pts - [x0, y0]
    dv = tb - ta
    pbin = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
    ok = np.isfinite(P).all(1) & np.isfinite(dv).all(1) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
    gc = [q for q in TJ["gcps"] if q.get("enabled", True)]
    GBASE = T0.to_pixel(np.array([[q["wx"], q["wy"]] for q in gc])) - [x0, y0]
    GPX = np.array([[q["px"], q["py"]] for q in gc]) - [x0, y0]
    # stretch and shear: today's placement's local Jacobian on a 50 px grid near the block sides, against its median
    gy_, gx_ = np.mgrid[0:h:50, 0:w:50]
    gq = np.c_[gx_.ravel(), gy_.ravel()].astype(float)
    weighted = points.kind is not None
    near = cKDTree(P[ok & (points.kind == 0)] if weighted else P[ok]).query(gq)[0] <= 100
    gq = gq[near] + [x0, y0]
    J = np.stack([(T0.to_world(gq + [0.5, 0]) - T0.to_world(gq - [0.5, 0])),
                  (T0.to_world(gq + [0, 0.5]) - T0.to_world(gq - [0, 0.5]))], -1)
    K = J @ np.linalg.inv(np.median(J, axis=0)) - np.eye(2)
    TAU_J = float(np.sqrt(np.mean(K ** 2)))
    extra = dict(wt=points.wt[ok].astype(np.float32), kind=points.kind[ok]) if weighted else {}
    log.info("%s/%s: points %d of %d; control points %d; street width %.1f px; stretch spread %.4f; crop %d x %d at "
             "(%d, %d)", sheet.series, sheet.name, ok.sum(), len(ok), len(gc), SW, TAU_J, w, h, x0, y0)
    return ProbabilisticInputs(P=P[ok].astype(np.float32), bin=pbin[ok], street_px=SW, origin=np.array([x0, y0]),
                               gbase=GBASE.astype(np.float32), gpx=GPX.astype(np.float32), tau_j=TAU_J, **extra)


def block_line_weights(sheet: Sheet, targets: Targets) -> np.ndarray:
    """How likely each inked centre pixel is to be a block line, from its stroke width: the sheet's stroke widths on
    the crop of the targets, fitted as two groups (thin block lines and heavier ink); each centre pixel's weight is
    its chance of being in the thin group.  All ink counts as printed, text included (its width speaks for it).
    Kept per direction (18, every 10 degrees: the ink's direction within 20 degrees).  Returns (18, h, w) float16."""
    page = sheet.page < 128
    x0, y0 = (int(v) for v in targets.origin)
    B, h, w = targets.maps.shape
    sub = page[y0:y0 + h, x0:x0 + w]
    half = ndimage.distance_transform_edt(sub)
    sk = skeletonize(sub)
    wid = 2.0 * half[sk]
    m = np.array([np.percentile(wid, 25), np.percentile(wid, 95)])
    s = np.array([1.0, 1.0])
    pi = np.array([0.7, 0.3])
    for _ in range(200):
        lk = pi[None] * np.exp(-(wid[:, None] - m[None]) ** 2 / (2 * s[None] ** 2)) / (np.sqrt(2 * np.pi) * s[None])
        r = lk / np.maximum(lk.sum(1, keepdims=True), 1e-300)
        pi = r.mean(0)
        m = (r * wid[:, None]).sum(0) / r.sum(0)
        s = np.sqrt((r * (wid[:, None] - m[None]) ** 2).sum(0) / r.sum(0)) + 1e-3
    thin = int(np.argmin(m))
    weight = np.zeros(sub.shape, np.float32)
    weight[sk] = r[:, thin]
    f = ndimage.gaussian_filter(sub.astype(np.float32), 1.0)
    gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
    Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
    cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180
    W8 = np.zeros((B, h, w), np.float16)
    for i, b in enumerate(np.arange(0, 180, 10.0)):
        dd = np.abs((cdir - b + 90) % 180 - 90)
        W8[i] = (weight * (dd <= 20.0)).astype(np.float16)
    log.info("%s/%s: block line width %.2f px, heavy ink %.2f px, block share %.2f", sheet.series, sheet.name,
             m[thin], m[1 - thin], pi[thin])
    return W8


def probabilistic_correction(inputs: ProbabilisticInputs, targets: Targets, weights: np.ndarray) -> Correction:
    """The displacement field that makes every point and control point together most probable (PyTorch)."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(3)           # fixed: the same threads every run
    T8 = targets.maps
    B, h, w = T8.shape
    Pp, PB, SW = inputs.P.astype(np.float32), inputs.bin, float(inputs.street_px)
    N = len(Pp)
    GB_, GX_ = inputs.gbase.astype(np.float32), inputs.gpx.astype(np.float32)
    NG = len(GB_)
    x0, y0 = (int(v) for v in inputs.origin)
    # every point is one observation, counted by its weight; only the block sides (kind 0) measure how well the print
    # is fitted
    WPT = torch.from_numpy(inputs.wt.astype(np.float32)) if inputs.wt is not None else torch.ones(N)
    NOBS = float(WPT.sum()) + NG
    BASE = (inputs.kind == 0) if inputs.kind is not None else np.ones(N, bool)
    DMAX = float(T8.max())
    SP = GRID_SPACING
    gh, gw = int(np.ceil((h - 1) / SP)) + 1, int(np.ceil((w - 1) / SP)) + 1
    ext = torch.tensor([(gw - 1) * SP, (gh - 1) * SP], dtype=torch.float32)
    groups = [torch.as_tensor(np.flatnonzero(PB == b)) for b in range(B)]

    def gp(xy):
        xy = torch.as_tensor(xy)
        return torch.stack([xy[:, 0] / ext[0] * 2 - 1, xy[:, 1] / ext[1] * 2 - 1], -1)[None, None]

    def field(G, g):
        return F.grid_sample(G, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T

    def sample(xy, maps1):
        """each point's value on the density map of its direction (B, 1, h, w); differentiable in xy"""
        g = torch.stack([xy[:, 0] / (w - 1) * 2 - 1, xy[:, 1] / (h - 1) * 2 - 1], -1)
        vals, idx = [], []
        for b, ii in enumerate(groups):
            if len(ii):
                vals.append(F.grid_sample(maps1[b:b + 1], g[ii][None, None], mode="bilinear", padding_mode="zeros",
                                          align_corners=True)[0, 0, 0, :])
                idx.append(ii)
        return torch.zeros(xy.shape[0]).index_copy(0, torch.cat(idx), torch.cat(vals))

    def jdet(G):
        """determinant of the map p -> p + u(p) on each grid cell (forward differences averaged over the cell's
        sides): below 0 the scan folds over itself"""
        ux_x = (G[0, 0, :, 1:] - G[0, 0, :, :-1]) / SP
        uy_x = (G[0, 1, :, 1:] - G[0, 1, :, :-1]) / SP
        ux_y = (G[0, 0, 1:, :] - G[0, 0, :-1, :]) / SP
        uy_y = (G[0, 1, 1:, :] - G[0, 1, :-1, :]) / SP
        ax, ay = 0.5 * (ux_x[1:, :] + ux_x[:-1, :]), 0.5 * (uy_x[1:, :] + uy_x[:-1, :])
        bx, by = 0.5 * (ux_y[:, 1:] + ux_y[:, :-1]), 0.5 * (uy_y[:, 1:] + uy_y[:, :-1])
        return (1 + ax) * (1 + by) - bx * ay

    def nearest_d(xy, m=None):
        xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        return T8[PB if m is None else PB[m], yi, xi].astype(np.float64)

    def em(d, eps, sig, rounds=50):
        """eps, sigma of the two-part model on the distances d to the nearest same-way centre (half-normal + uniform
        over 0..DMAX)"""
        for _ in range(rounds):
            pl = (1 - eps) * 2 * np.exp(-d * d / (2 * sig * sig)) / (np.sqrt(2 * np.pi) * sig)
            pu = eps / DMAX
            r = pl / np.maximum(pl + pu, 1e-300)
            eps = float(1 - r.mean())
            sig = float(np.sqrt((r * d * d).sum() / max(r.sum(), 1e-9)))
        return eps, max(sig, 0.5)

    def share(xy):
        d = nearest_d(xy[BASE], BASE)
        return np.mean(d <= 1), np.mean(d <= 2), np.mean(d <= 4)

    eps, sig = em(nearest_d(Pp[BASE], BASE), 0.5, SW / 4)
    TAU = sig          # prior on the correction: today's fit is believed close; spread = today's measured misfit
    AREA = float(h * w)

    def em2(r, eps, sig, rounds=50):
        """control points: 2-D Gaussian of spread sig around their matched px, or anywhere in the crop (eps)"""
        for _ in range(rounds):
            pl = (1 - eps) * np.exp(-r * r / (2 * sig * sig)) / (2 * np.pi * sig * sig)
            pu = eps / AREA
            q = pl / np.maximum(pl + pu, 1e-300)
            eps = float(np.clip(1 - q.mean(), 1e-4, 0.5))
            sig = float(np.sqrt((q * r * r).sum() / max(2 * q.sum(), 1e-9)))
        return eps, max(sig, 0.25)

    geps, gsig = em2(np.linalg.norm(GB_ - GX_, axis=1), 0.05, 2.0)
    log.info("control points %d: today eps %.3f, sigma %.2f px", NG, geps, gsig)
    log.info("points %d; street width %.1f px; grid %d px; start (today): eps %.2f, sigma %.2f px; within 1/2/4 px "
             "%.1f%% %.1f%% %.1f%%", N, SW, SP, eps, sig, *(100 * a for a in share(Pp)))
    cent = weights
    TAU_J = float(inputs.tau_j)         # how much today's placement stretches and shears across the sheet
    G = torch.zeros(1, 2, gh, gw, requires_grad=True)
    gP = gp(Pp)
    gG = gp(GB_)
    GBt, GXt = torch.from_numpy(GB_), torch.from_numpy(GX_)
    Pt = torch.from_numpy(Pp)
    t0 = time.time()
    s = SW / 2          # the two printed edges of a street (one street width apart) stay two ridges, not one
    stage = 0
    while True:
        last = s <= sig
        if last:
            s = sig
        maps = np.zeros((B, 1, h, w), np.float32)
        for b in range(B):
            maps[b, 0] = gaussian_filter(np.asarray(cent[b], np.float32), s)   # summed 2-D Gaussians: along a line, N(d; 0, s)
        maps = torch.from_numpy(maps)
        opt = torch.optim.Adam([G], lr=0.25)
        for it in range(ITERS):
            opt.zero_grad()
            v = sample(Pt + field(G, gP), maps)
            nll = -torch.log((1 - eps) * torch.clamp(v, min=0) + eps / DMAX)
            rg = torch.sum((GBt + field(G, gG) - GXt) ** 2, dim=1)
            nllg = -torch.log((1 - geps) * torch.exp(-rg / (2 * gsig * gsig)) / (2 * np.pi * gsig * gsig) + geps / AREA)
            data = (torch.sum(nll * WPT) + torch.sum(nllg)) / NOBS * 2 * s * s
            prior = torch.sum(G ** 2) / (2 * TAU * TAU) / NOBS * 2 * s * s
            fold = torch.sum(torch.relu(-jdet(G)) ** 2) * 1e6
            ux_x = (G[0, 0, :, 1:] - G[0, 0, :, :-1]) / SP
            uy_x = (G[0, 1, :, 1:] - G[0, 1, :, :-1]) / SP
            ux_y = (G[0, 0, 1:, :] - G[0, 0, :-1, :]) / SP
            uy_y = (G[0, 1, 1:, :] - G[0, 1, :-1, :]) / SP
            shape = (torch.sum(ux_x ** 2) + torch.sum(uy_x ** 2) + torch.sum(ux_y ** 2) + torch.sum(uy_y ** 2)) / (2 * TAU_J ** 2) / NOBS * 2 * s * s
            (data + prior + shape + fold).backward()
            opt.step()
        del maps
        with torch.no_grad():
            u = field(G, gP).numpy()
            umax = float(torch.linalg.norm(G[0], dim=0).max())
            dmin = float(jdet(G).min())
            ug = field(G, gG).numpy()
        eps, sig = em(nearest_d((Pp + u)[BASE], BASE), eps, sig)
        a1, a2, a4 = share(Pp + u)
        log.info("  stage %d: s %.2f px; within 1/2/4 px %.1f%% %.1f%% %.1f%%; eps %.2f, sigma %.2f px; control points "
                 "off p99 %.1f px; moved p50 %.1f p90 %.1f px, biggest correction %.1f px; smallest cell determinant "
                 "%.2f; %.0fs", stage, s, 100 * a1, 100 * a2, 100 * a4, eps, sig,
                 np.percentile(np.linalg.norm(GB_ + ug - GX_, axis=1), 99), np.median(np.linalg.norm(u, axis=1)),
                 np.percentile(np.linalg.norm(u, axis=1), 90), umax, dmin, time.time() - t0)
        stage += 1
        if last:
            break
        s = s / 2
    return Correction(grid=G.detach().numpy()[0], spacing=SP, origin=np.array([x0, y0]))
