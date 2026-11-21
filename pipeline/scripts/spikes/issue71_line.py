"""Issue #71 audit, spike 1: frame-LINE evidence instead of region chances.  Throwaway measurement (brief:
.superpowers/sdd/2026-11-20-frame-crop/spike-line-brief.md).

A "fold 16" net (the current defaults: FOLD 16, LEVELS 5, WIDTH 32, BUCKET 1024) with ONE output per px is trained on
fold 0's 16 sheets (seed 0, 4000 steps, learn.train's cut, scale and augmentation, cpu/device split) to say where the
map outline is: target h = exp(-d^2 / 2 sigma^2), d the distance in px to the label's outline (`class_mask != 1`,
removed blocks counted as map), outline px within EDGE_TOL px of the page's own edge left out (a frameless side has no
line); binary cross-entropy with logits, weight 1 on the page, 0 on the padding.  The target is computed on the cut
and scaled page (the cpu side), so a cut that takes a side's frame away takes its line away too.

From the predicted heat H a straight rectangle is fitted per page (one angle, four sides by the mean of H along
them, a tilt and bow per side, notches by three inner sides of mean H > 0.5), and its kept region scored like the
region net's per-pixel cut.

    uv run python scripts/spikes/issue71_line.py train 8      # train sigma 8 (one GPU turn), predict the heat
    uv run python scripts/spikes/issue71_line.py train 32
    uv run python scripts/spikes/issue71_line.py fit 8        # fit, score, pictures (cpu)
    uv run python scripts/spikes/issue71_line.py region       # the region net's per-pixel cut, same convention
    uv run python scripts/spikes/issue71_line.py compare 8    # the owner's combined picture
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage

from sfzmaps import gpu
from sfzmaps.frame import data, decide, folds, learn, model, scores
from sfzmaps.frame import run as fr_run

log = logging.getLogger("line")
Image.MAX_IMAGE_PIXELS = None

_spec = importlib.util.spec_from_file_location("issue71_shape", Path(__file__).with_name("issue71_shape.py"))
shp = importlib.util.module_from_spec(_spec)
sys.modules["issue71_shape"] = shp
_spec.loader.exec_module(shp)
cv2.setNumThreads(4)

OUT = data.DATA_DIR / "frame-line"
DEV = "mps"
EDGE_TOL = 3              # px: outline px this near the page's edge are the page edge, not a line
EXTRA = [("1948", "14"), ("1948", "3"), ("1948", "5a"), ("1948", "6a"), ("1948", "8"), ("1948", "1"),
         ("1921", "20180102094204401")]
COMPARE = [("1948", "14"), ("1948", "3"), ("1948", "5a"), ("1948", "6a"), ("1948", "8")]
CHECK_EVERY = 500

# the fit
PHI_RANGE, PHI_STEP = 2.5, 0.1      # deg
REACH = 0.4                         # a side's offset runs from the canvas edge to this share of the page inward
SPAN_INSET = 0.10                   # each end of a side's span left out of its mean (corner arcs, notch corners)
TILT_MAX, TILT_STEP = 0.45, 0.05    # deg
SAG_MAX, SAG_STEP = 22, 2           # px
LINE_MIN = 0.5                      # a side's best mean H below this: no line, the page edge
NOTCH_MIN = 0.5                     # each inner side of a notch: mean H above this
NOTCH_CELL = 4                      # px, the notch search's grid
NOTCH_MIN_PX = 64                   # px, a notch's least depth and span
NOTCH_CAP = 6
BAND = 2.5                          # x sigma: an along side within this of a frame side is that side, not a notch
NOTCH_REACH = 0.5                   # a notch reaches at most to the page's middle
SIDES = shp.SIDES


def fold0():
    return folds.split(data.labelled_sheets(), 4)[0]


def sdir(sigma) -> Path:
    return OUT / f"sigma{sigma}"


def heat_path(sigma, sid) -> Path:
    return sdir(sigma) / "heat" / f"{sid[0]}__{sid[1]}.npy"


# ----------------------------------------------------------------------------------------------------------------
# The target and the training
# ----------------------------------------------------------------------------------------------------------------

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def outline_px(cls: np.ndarray) -> np.ndarray:
    """Bool: the map region's (cls != 1) boundary px (4-neighbour), less those within EDGE_TOL of the page's edge."""
    m = (np.asarray(cls) != 1).astype(np.uint8)
    er = cv2.erode(m, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=1)
    ol = (m == 1) & (er == 0)
    t = EDGE_TOL
    ol[:t], ol[-t:], ol[:, :t], ol[:, -t:] = False, False, False, False
    return ol


def dist_u8(cls: np.ndarray) -> np.ndarray:
    """Uint8: the distance in px to the outline (`outline_px`), rounded, at most 255."""
    ol = outline_px(cls)
    if not ol.any():
        return np.full(cls.shape, 255, np.uint8)
    d = cv2.distanceTransform(np.where(ol, 0, 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.minimum(d + 0.5, 255).astype(np.uint8)


def _cpu(src, cut, scale, aug):
    lum, cls = learn._cut_and_scale(src, cut, scale)
    return np.stack((learn._photometric(aug, lum), dist_u8(cls))), aug


@torch.no_grad()
def _device(planes, aug, sigma, device):
    moved = learn._move(aug, torch.from_numpy(planes).to(device)).contiguous()
    _, h, w = moved.shape
    size = (1, 1, *model.padded_shape(h, w))
    x = torch.full(size, model.PAPER, dtype=torch.float32, device=device)
    t = torch.zeros(size, dtype=torch.float32, device=device)
    wt = torch.zeros(size, dtype=torch.float32, device=device)
    x[0, 0, :h, :w] = moved[0].float() / 255.0
    d = moved[1].float()
    t[0, 0, :h, :w] = torch.exp(-d * d / (2.0 * sigma * sigma))
    wt[0, 0, :h, :w] = 1.0
    return x, t, wt


def train(pages, sigma, recipe: learn.Recipe, device: str):
    rng = learn._settle(recipe.seed)
    model.CLASSES = 1
    net = model.new_net().to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)
    sources = [learn._Source.of(pg) for pg in pages]
    depths = [learn.side_depths(pg.cls) for pg in pages]

    def draw():
        k = int(rng.integers(len(sources)))
        cut = learn.draw_cut(rng, depths[k])
        scale = 1.0 + float(rng.uniform(-learn.SCALE_JITTER, learn.SCALE_JITTER))
        aug = learn.draw_aug(rng)
        return _cpu, (sources[k], cut, scale, aug)

    curve, total, since = [], torch.zeros((), device=device), 0
    t0 = t_start = time.perf_counter()
    with learn._batches(draw, recipe.steps, recipe.prefetch) as batches:
        for step, (planes, aug) in enumerate(batches, 1):
            x, t, w = _device(planes, aug, sigma, device)
            z = net(x)
            loss = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach()
            since += 1
            if step % CHECK_EVERY:
                continue
            mean = float(total) / since
            ms = (time.perf_counter() - t0) / since * 1000.0
            curve.append([step, mean, ms])
            log.info("sigma %s step %d: train %.5f (%.0f ms/step)", sigma, step, mean, ms)
            total.zero_()
            since = 0
            t0 = time.perf_counter()
    return net.eval().cpu(), {"curve": curve, "train_s": time.perf_counter() - t_start}


def cmd_train(sigma):
    sigma = float(sigma)
    tag = f"{sigma:g}"
    out = sdir(tag)
    (out / "heat").mkdir(parents=True, exist_ok=True)
    f0 = fold0()
    pages = list(fr_run.labelled_pages(list(f0.train)).values())
    for pg in pages:     # removed blocks are map here: the line is the polygons' outline only
        pg.cls = np.where(pg.cls == 2, 0, pg.cls).astype(np.uint8)
    sids = list(f0.held) + EXTRA
    with gpu.gpu_turn(True, DEV):
        net, info = train(pages, sigma, learn.Recipe(), DEV)
        torch.save(learn._state(net), out / "net.pt")
        pred = {}
        for sid in sids:
            lum = data.load_luminance(data.page_path(*sid))
            t0 = time.perf_counter()
            z = model.logits(net, lum, DEV)[0]
            pred[f"{sid[0]}/{sid[1]}"] = time.perf_counter() - t0
            H = 1.0 / (1.0 + np.exp(-z))
            np.save(heat_path(tag, sid), np.round(H * 255).astype(np.uint8))
            log.info("heat %s/%s", *sid)
    info["predict_s"] = pred
    (out / "train.json").write_text(json.dumps(info, indent=1))


# ----------------------------------------------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------------------------------------------

TIE = 0.01     # numerical: means within this of the best tie (the heat is stored in 1/255 steps)


TIE_SHAPE = 1e-3   # numerical: a tilt or bow is taken only when its mean beats the straighter one's by this (with TIE,
                   # the label's own sigma-32 heat lost its bows: a 2 px bow moves that mean by less than 0.01)


def plateau(p) -> int:
    """The middle of the run of near-best values (within TIE of the max) around the argmax: the heat's top is flat
    (sigma 32: 0.70 over ~70 px), and argmax alone takes the run's first index, the outermost line."""
    p = np.asarray(p)
    k = int(np.argmax(p))
    ok = p >= p[k] - TIE
    lo = hi = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(p) - 1 and ok[hi + 1]:
        hi += 1
    return (lo + hi) // 2


def _profiles(A, Bv, cv, h, w, s, spans):
    """Per side: (offset cells k, mean H along the line at row k of the side view over its span cols)."""
    out = {}
    for side in SIDES:
        Av, Vv = shp.side_view(A, side), shp.side_view(Bv, side)
        a0, a1 = spans[side]
        half = (h if side in ("top", "bottom") else w) / 2.0
        E = cv.EA(side)[0]
        kmax = min(Av.shape[0] - 1, int((E - half + REACH * 2 * half) // s))
        num = Av[:kmax + 1, a0:a1].sum(1, dtype=np.float64)
        den = Vv[:kmax + 1, a0:a1].sum(1, dtype=np.float64)
        out[side] = np.where(den > 0.25 * (a1 - a0) * s * s, num / np.maximum(den, 1e-9), 0.0)
    return out


def _spans(cv, s, beta, line, h, w):
    """Columns (cells of s) of each side's view its mean is taken over: between the perpendicular sides (or the page
    edge where they have no line), SPAN_INSET of it left out at each end."""
    spans = {}
    for side in SIDES:
        E, Aoff = cv.EA(side)
        if side in ("top", "bottom"):
            lo = beta["left"] if line.get("left") else -w / 2.0
            hi = -beta["right"] if line.get("right") else w / 2.0
        else:
            lo = beta["top"] if line.get("top") else -h / 2.0
            hi = -beta["bottom"] if line.get("bottom") else h / 2.0
        L = hi - lo
        lo, hi = lo + SPAN_INSET * L, hi - SPAN_INSET * L
        spans[side] = (int(max(0, (lo + Aoff) // s)), int(max(1, (hi + Aoff) // s)))
    return spans


def _coarse(H, h, w, s=4):
    """The best angle and offsets on cells of s px: per angle two rounds (spans the page's middle, then between the
    found sides), scored by the sum of the four sides' best means."""
    Hs, Vs = shp.pool(H, s), shp.pool(np.ones_like(H), s)
    best = None
    for phi in shp._centre_first(np.round(np.arange(-PHI_RANGE, PHI_RANGE + 1e-9, PHI_STEP), 4), 0.0):
        cv = shp.Canvas.page(phi, h, w)
        A, Bv = cv.warp(Hs, s, s), cv.warp(Vs, s, s)
        beta, line = {sd: 0.0 for sd in SIDES}, {}
        for _ in range(2):
            prof = _profiles(A, Bv, cv, h, w, s, _spans(cv, s, beta, line, h, w))
            for side, p in prof.items():
                k = plateau(p)
                beta[side] = s * k + s / 2.0 - cv.EA(side)[0]
                line[side] = p[k] >= LINE_MIN
            score = sum(float(p.max()) for p in prof.values())
        if best is None or score > best[0] + 1e-9:
            best = (score, float(phi), dict(beta), dict(line))
    return best


def _fine_side(view, vv, E, Aoff, beta, ends, side_lo_hi, R=24):
    """The side's best (offset, tilt, sag, mean) at 1 px on the full-res side view: offset within +-R of beta, tilt
    within +-TILT_MAX, sag within +-SAG_MAX, the mean of H along the curve over the span columns (every 2nd)."""
    lo, hi = ends
    m, hw = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
    s_lo, s_hi = side_lo_hi
    L = s_hi - s_lo
    a0 = max(0, int(math.ceil(s_lo + SPAN_INSET * L + Aoff)))
    a1 = min(view.shape[1], int(s_hi - SPAN_INSET * L + Aoff))
    cols = np.arange(a0, a1, 2)
    al = cols - Aoff + 0.5
    x = (al - m) / hw
    offs = np.arange(-R, R + 1)
    best = (-1.0, beta, 0.0, 0.0)
    grid = [(td, sag) for td in np.arange(-TILT_MAX, TILT_MAX + 1e-9, TILT_STEP)
            for sag in np.arange(-SAG_MAX, SAG_MAX + 1e-9, SAG_STEP)]
    grid.sort(key=lambda g: (abs(g[0]) / TILT_MAX + abs(g[1]) / SAG_MAX, abs(g[0]), abs(g[1])))   # straight first
    for td, sag in grid:
        tt = math.tan(math.radians(td))
        base = np.floor(beta + E + tt * (al - m) + sag * (1.0 - x * x)).astype(np.int64)
        rows = base[None, :] + offs[:, None]
        ok = (rows >= 0) & (rows < view.shape[0])
        rr = np.clip(rows, 0, view.shape[0] - 1)
        num = np.where(ok, view[rr, cols[None, :]], 0).sum(1)
        den = np.where(ok, vv[rr, cols[None, :]], 0).sum(1)
        mean = np.where(den > 0.5 * len(cols), num / np.maximum(den, 1e-9), 0.0)
        k = plateau(mean)
        if mean[k] > best[0] + TIE_SHAPE:   # a tilt or bow must beat the straighter line by more than a tie
            best = (float(mean[k]), beta + offs[k], tt, float(sag))
    return best


def fit_frame(H, timing=None):
    """The rectangle (shp.Frame) and per side whether it has a line, its best mean."""
    h, w = H.shape
    t0 = time.perf_counter()
    _, phi, beta, line = _coarse(H, h, w)
    t1 = time.perf_counter()
    cv = shp.Canvas.page(phi, h, w)
    A1, V1 = cv.warp(H, 1, 1), cv.warp(np.ones_like(H), 1, 1)
    fr = shp.Frame(h, w, phi, {sd: (beta[sd] if line[sd] else -cv.EA(sd)[0]) for sd in SIDES})
    means = {}
    for _ in range(2):
        for side in SIDES:
            E, Aoff = cv.EA(side)
            b0 = beta[side]          # always refit around the best line (also a "no line" side, to report its mean)
            # the span: between the perpendicular sides (page edge where they have no line)
            if side in ("top", "bottom"):
                span = (fr.beta["left"] if line["left"] else -w / 2.0, -fr.beta["right"] if line["right"] else w / 2.0)
            else:
                span = (fr.beta["top"] if line["top"] else -h / 2.0, -fr.beta["bottom"] if line["bottom"] else h / 2.0)
            mean, b, tt, sag = _fine_side(shp.side_view(A1, side), shp.side_view(V1, side), E, Aoff, b0,
                                          fr.ends(side), span)
            beta[side] = b
            means[side] = mean
            line[side] = mean >= LINE_MIN
            if line[side]:
                fr.beta[side], fr.tilt[side], fr.sag[side] = b, tt, sag
            else:
                fr.beta[side], fr.tilt[side], fr.sag[side] = -E, 0.0, 0.0
    t2 = time.perf_counter()
    if timing is not None:
        timing["coarse_s"], timing["fine_s"] = t1 - t0, t2 - t1
    return fr, line, means, (A1, V1)


NOTCH_DEG, NOTCH_DEG_STEP = 2.5, 0.25   # deg: a notch's angle within this of the frame's (labelled: up to 1.64 off)
NOTCH_SEARCH = 8                        # px, the notch search's cells


def _halves_ok(P, Q, lo, hi, axis_rows=True):
    """Bool per column: the mean of the column's values between prefix-rows lo and hi (per column arrays), and over
    each half, all > NOTCH_MIN.  P, Q are prefix sums along rows with a leading zero row."""
    cols = np.arange(P.shape[1])
    mid = (lo + hi) // 2
    ok = np.ones(P.shape[1], bool)
    for a, b in ((lo, mid), (mid, hi)):
        q = Q[b, cols] - Q[a, cols]
        ok &= (q > 0) & ((P[b, cols] - P[a, cols]) > NOTCH_MIN * np.maximum(q, 1e-9))
    return ok & (hi - lo >= 2)


def _notch_candidates(fr: shp.Frame, H, K0, band_px):
    """Every notch candidate: (area px, side, psi, a0, a1, d) in the psi frame about the page centre (along in [a0,
    a1), e < d), its three inner sides each passing (`_halves_ok`: the mean of H > NOTCH_MIN over each half of the
    side, so a side whose line runs along only part of it fails; an across side at the frame's own end is not
    checked), for angles psi within NOTCH_DEG of the frame's, on cells of NOTCH_SEARCH px.  Per (psi, side, depth)
    the largest few."""
    s = NOTCH_SEARCH
    h, w = H.shape
    Hs, Vs, Ks = shp.pool(H, s), shp.pool(np.ones_like(H), s), shp.pool(K0.astype(np.float32), s)
    # an along side must be a line of its own, not the frame side's: its mean is taken with the heat within band_px
    # of the rectangle's sides set to 0 (first run: tilted slivers along the frame's sides passed as notches)
    edge = (K0 & ~ndimage.binary_erosion(K0, border_value=1)).astype(np.uint8)
    near = cv2.distanceTransform(np.where(edge > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3) < band_px
    Hz = shp.pool(np.where(near, 0.0, H).astype(np.float32), s)
    del edge, near
    cands = []
    for psi in fr.phi + np.arange(-NOTCH_DEG, NOTCH_DEG + 1e-9, NOTCH_DEG_STEP):
        cv = shp.Canvas.page(psi, h, w)
        A, B, KK, AZ = cv.warp(Hs, s, s), cv.warp(Vs, s, s), cv.warp(Ks, s, s), cv.warp(Hz, s, s)
        for side in SIDES:
            view, vv, kv = shp.side_view(A, side), shp.side_view(B, side), shp.side_view(KK, side)
            vz = shp.side_view(AZ, side)
            E, Aoff = cv.EA(side)
            inside = kv > 0.5 * np.maximum(vv, 1e-9) * 1.0
            inside &= vv > 0.5 * s * s
            colin = inside.any(0)
            if colin.sum() < 4:
                continue
            c0, c1 = int(np.argmax(colin)), int(len(colin) - np.argmax(colin[::-1]))
            r0 = np.where(colin, inside.argmax(0), 0)              # first kept row per column (the side line)
            half = (h if side in ("top", "bottom") else w) / 2.0
            r1 = min(view.shape[0], int((E - half + NOTCH_REACH * 2 * half) / s))
            P = np.zeros((view.shape[0] + 1, view.shape[1]))
            Q = np.zeros_like(P)
            P[1:], Q[1:] = np.cumsum(view, 0), np.cumsum(vv, 0)
            dmin = NOTCH_MIN_PX // s
            span_min = NOTCH_MIN_PX // s
            for d in range(int(r0[c0:c1].min()) + dmin, r1):
                lo = np.minimum(r0, d)
                ok = _halves_ok(P, Q, lo, np.full_like(lo, d)) & colin
                ok &= d - r0 >= dmin
                ok[c0] = ok[c1 - 1] = True
                idx = np.flatnonzero(ok[c0:c1]) + c0
                if len(idx) < 2:
                    continue
                Pa = np.concatenate([[0.0], np.cumsum(vz[d], dtype=np.float64)])
                Qa = np.concatenate([[0.0], np.cumsum(vv[d], dtype=np.float64)])
                i0, i1 = np.meshgrid(idx, idx, indexing="ij")
                good = (i1 - i0 >= span_min) & ~((i0 == c0) & (i1 == c1 - 1))   # not the whole side
                if not good.any():
                    continue
                a, b = i0[good], i1[good] + 1
                mid = (a + b) // 2
                sel = np.ones(len(a), bool)
                for u, v in ((a, mid), (mid, b)):
                    q = Qa[v] - Qa[u]
                    sel &= (q > 0) & ((Pa[v] - Pa[u]) > NOTCH_MIN * np.maximum(q, 1e-9))
                if not sel.any():
                    continue
                a, b = a[sel], b[sel]
                depth = d - np.minimum(r0[a], r0[b - 1])
                area = (b - a) * depth * s * s
                for j in np.argsort(-area)[:3]:
                    # along: the notch starts after the across line at col a (its peak is map), ends at col b - 1
                    al0 = (a[j] + 1) * s - Aoff if a[j] != c0 else -2.0 * E
                    al1 = (b[j] - 1) * s - Aoff if b[j] - 1 != c1 - 1 else 2.0 * E
                    cands.append((int(area[j]), side, float(psi), float(al0), float(al1), float(d * s + s / 2 - E)))
    return cands


def _coarse_mask(n: shp.Notch, h, w, s=16):
    cv = shp.Canvas.page(n.psi, h, w)
    E, _ = cv.EA(n.side)
    pts = []
    for al in (max(n.a0, -2 * E), min(n.a1, 2 * E)):
        for e in (-E - 2.0, n.d):
            u, v = {"top": (al, e), "bottom": (al, -e), "left": (e, al), "right": (-e, al)}[n.side]
            pts.append(cv.xy(u, v))
    pts = [pts[0], pts[1], pts[3], pts[2]]
    img = Image.new("L", (-(-w // s), -(-h // s)), 0)
    ImageDraw.Draw(img).polygon([(x / s, y / s) for x, y in pts], fill=1)
    return np.asarray(img, bool)


def fit_notches(fr: shp.Frame, H, K0, band_px, R=24):
    """Notches: the candidates (`_notch_candidates`) taken greedily by area, each not overlapping one taken (on a
    16 px page grid), at most NOTCH_CAP; each taken notch's three inner sides then moved to their best mean of H at
    1 px (within +-R) on its own angle's full-res canvas."""
    h, w = H.shape
    cands = sorted(_notch_candidates(fr, H, K0, band_px), key=lambda c: -c[0])
    taken, masks = [], []
    for area, side, psi, a0, a1, d in cands:
        if len(taken) >= NOTCH_CAP:
            break
        n = shp.Notch(side, psi, a0, a1, d)
        m = _coarse_mask(n, h, w)
        if any((m & t).sum() > 0.05 * min(m.sum(), t.sum()) for t in masks):
            continue
        n = _refine_notch(n, H, K0, R)
        taken.append(n)
        masks.append(_coarse_mask(n, h, w))
    return taken


def _refine_notch(n: shp.Notch, H, K0, R):
    h, w = H.shape
    cv = shp.Canvas.page(n.psi, h, w)
    E, Aoff = cv.EA(n.side)
    view = shp.side_view(cv.warp(H, 1, 1), n.side)
    vv = shp.side_view(cv.warp(np.ones_like(H), 1, 1), n.side)
    kv = shp.side_view(cv.warp(K0.astype(np.float32), 1, 1), n.side)
    nc = view.shape[1]
    ca = int(round(n.a0 + Aoff)) - 1 if n.a0 > -2 * E + 1 else None     # the across lines' cols (None: frame end)
    cb = int(round(n.a1 + Aoff)) if n.a1 < 2 * E - 1 else None
    lo_c = ca + 1 if ca is not None else 0
    hi_c = cb if cb is not None else nc
    rd = int(round(n.d + E))
    colk = kv[:, max(0, lo_c):min(nc, hi_c)] > 0.5 * np.maximum(vv[:, max(0, lo_c):min(nc, hi_c)], 1e-9)
    r0 = int(np.median(np.where(colk.any(0), colk.argmax(0), 0))) if colk.size else 0

    def mean_rows(rows, c_lo, c_hi):
        c_lo, c_hi = max(0, c_lo), min(nc, c_hi)
        num, den = view[rows, c_lo:c_hi].sum(1), vv[rows, c_lo:c_hi].sum(1)
        return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

    def mean_cols(cols, r_lo, r_hi):
        num, den = view[r_lo:r_hi, cols].sum(0), vv[r_lo:r_hi, cols].sum(0)
        return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

    rows = np.arange(max(0, rd - R), min(view.shape[0], rd + R + 1))
    rd = int(rows[plateau(mean_rows(rows, lo_c, hi_c))])
    a0, a1 = n.a0, n.a1
    if ca is not None:
        cols = np.arange(max(0, ca - R), min(nc, ca + R + 1))
        ca = int(cols[plateau(mean_cols(cols, r0, rd))])
        a0 = ca + 1 - Aoff
    if cb is not None:
        cols = np.arange(max(0, cb - R), min(nc, cb + R + 1))
        cb = int(cols[plateau(mean_cols(cols, r0, rd))])
        a1 = cb - Aoff
    return shp.Notch(n.side, n.psi, float(a0), float(a1), float(rd - E))


INWARD = 4      # px, the diagnostic arm's inward move (the region cut's own median offset from the label: +4)


def moved_inward(fr: shp.Frame, notches, line, px=INWARD):
    """Diagnostic: every side with a line moved px inward, every notch grown by px on its inner sides."""
    fr = fr.copy()
    for sd in SIDES:
        if line[sd]:
            fr.beta[sd] += px
    ns = []
    for n in notches:
        E = shp.Canvas.page(n.psi, fr.h, fr.w).EA(n.side)[0]
        ns.append(shp.Notch(n.side, n.psi, n.a0 - px if n.a0 > -2 * E + 1 else n.a0,
                            n.a1 + px if n.a1 < 2 * E - 1 else n.a1, n.d + px))
    return fr, ns


def kept_region(fr: shp.Frame, notches):
    fr = fr.copy()
    fr.notches = list(notches)
    for c in fr.arc:
        fr.arc[c] = 0.0
    return fr.kept()


# ----------------------------------------------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------------------------------------------

def side_errors(K: np.ndarray, cls0: np.ndarray):
    """Per side: the label side's kind (line / no line: the label's map reaches the page edge along the middle 70%)
    and the median |first kept px - first label map px| (px from that page edge) over the side's middle 70%."""
    lab = cls0 == 0
    h, w = lab.shape
    out = {}
    rows, cols = np.flatnonzero(lab.any(1)), np.flatnonzero(lab.any(0))
    for side in SIDES:
        if side in ("top", "bottom"):
            lo, hi = cols[0], cols[-1]
            L = hi - lo
            sl = slice(int(lo + 0.15 * L), int(hi - 0.15 * L))
            a, k = lab[:, sl], K[:, sl]
            if side == "bottom":
                a, k = a[::-1], k[::-1]
        else:
            lo, hi = rows[0], rows[-1]
            L = hi - lo
            sl = slice(int(lo + 0.15 * L), int(hi - 0.15 * L))
            a, k = lab[sl].T, K[sl].T
            if side == "right":
                a, k = a[::-1], k[::-1]
        fa = np.where(a.any(0), a.argmax(0), a.shape[0])
        fk = np.where(k.any(0), k.argmax(0), k.shape[0])
        out[side] = {"label_line": bool(np.median(fa) > EDGE_TOL),
                     "label_depth": float(np.median(fa)),
                     "err_px": float(np.median(np.abs(fk - fa))),
                     "signed_px": float(np.median(fk - fa))}    # < 0: the cut keeps px outside the label (frame)
    return out


def score(deleted, cls0, lum, p=None):
    if p is None:
        p = np.broadcast_to(np.float32(1.0 / 3.0), (3, *lum.shape))
    sc = scores.sheet_scores(p, deleted, cls0, lum)
    return {k: sc[k] for k in ("frame_kept", "map_cut", "cost", "frame_kept_share", "map_cut_share", "hd95_px")}


def cls0_of(sid, shape):
    return data.class_mask(data.load_label(*sid), [], shape)


# ----------------------------------------------------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------------------------------------------------

LONG = 1400


def _edge(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=1)


def picture(lum, H, K, cls0=None, side=LONG, box=None):
    """Grey page, red where H is high (alpha = H), the fitted kept region's boundary in blue, the label's in green."""
    if box is not None:
        x0, y0, x1, y1 = box
        lum, H, K = lum[y0:y1, x0:x1], H[y0:y1, x0:x1], K[y0:y1, x0:x1]
        cls0 = None if cls0 is None else cls0[y0:y1, x0:x1]
    h, w = lum.shape
    s = min(1.0, side / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    g = np.asarray(Image.fromarray(lum.astype(np.float32)).resize(size, Image.BOX), np.float32)
    if s < 1:   # the heat's max over each shrunk px, so a thin line stays visible
        H = ndimage.maximum_filter(H.astype(np.float32), size=max(1, int(round(1 / s))))
    a = np.clip(np.asarray(Image.fromarray(H.astype(np.float32)).resize(size, Image.BOX), np.float32), 0, 1)[..., None]
    base = np.repeat((0.35 + 0.65 * g)[..., None], 3, 2) * 255
    rgb = (1 - 0.85 * a) * base + 0.85 * a * np.array([255, 0, 0], np.float32)
    k = np.asarray(Image.fromarray(K.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    if cls0 is not None:
        lm = np.asarray(Image.fromarray((cls0 == 0).astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
        rgb[ndimage.binary_dilation(_edge(lm)) if s < 1 else _edge(lm)] = (0, 170, 0)
    rgb[ndimage.binary_dilation(_edge(k)) if s < 1 else _edge(k)] = (0, 60, 255)
    return Image.fromarray(np.round(rgb).astype(np.uint8), "RGB")


# ----------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------

def fit_page(tag, sid, held):
    lum = data.load_luminance(data.page_path(*sid))
    H = np.load(heat_path(tag, sid)).astype(np.float32) / 255.0
    timing = {}
    t0 = time.perf_counter()
    fr, line, means, (A1, V1) = fit_frame(H, timing)
    t1 = time.perf_counter()
    del A1, V1
    notches = fit_notches(fr, H, kept_region(fr, []), BAND * float(tag))
    t2 = time.perf_counter()
    K = kept_region(fr, notches)
    timing.update(frame_s=t1 - t0, notch_s=t2 - t1, total_s=time.perf_counter() - t0)
    row = {"sid": list(sid), "phi": fr.phi, "beta": fr.beta, "line": line, "mean_H": means,
           "tilt_deg": {k: math.degrees(math.atan(v)) for k, v in fr.tilt.items()}, "sag": fr.sag,
           "notches": [n.__dict__ for n in notches], "timing": timing}
    cls0 = None
    if held:
        cls0 = cls0_of(sid, lum.shape)
        row["scores"] = score(~K, cls0, lum)
        row["sides"] = side_errors(K, cls0)
        R = rect_only = kept_region(fr, [])
        row["scores_no_notches"] = score(~R, cls0, lum)
        Ki = kept_region(*moved_inward(fr, notches, line))
        row["scores_inward"] = score(~Ki, cls0, lum)
        row["sides_inward"] = side_errors(Ki, cls0)
        del Ki
        del rect_only, R
        # the ceiling of this fit: the same fit on the label's own target heat (what a perfect net would give)
        dl = dist_u8(cls0).astype(np.float32)
        sg = float(tag)
        Hl = np.exp(-dl * dl / (2 * sg * sg)).astype(np.float32)
        frl, linel, meansl, _ = fit_frame(Hl)
        nl = fit_notches(frl, Hl, kept_region(frl, []), BAND * sg)
        del _, Hl, dl
        Kl = kept_region(frl, nl)
        Kli = kept_region(*moved_inward(frl, nl, linel))
        li = score(~Kli, cls0, lum)
        del Kli
        row["label_heat"] = {"scores": score(~Kl, cls0, lum), "sides": side_errors(Kl, cls0), "line": linel,
                             "notches": len(nl), "scores_inward": li}
        del Kl
    pic = sdir(tag) / "look" / f"{sid[0]}__{sid[1]}.png"
    pic.parent.mkdir(parents=True, exist_ok=True)
    picture(lum, H, K, cls0).save(pic)
    np.save(sdir(tag) / "kept" / f"{sid[0]}__{sid[1]}.npy", np.packbits(K))
    log.info("%s %s/%s line %s means %s notches %d %.1fs %s", tag, *sid, line,
             {k: round(v, 2) for k, v in means.items()}, len(notches), timing["total_s"],
             row.get("scores", {}).get("cost", ""))
    return row


def cmd_fit(sigma):
    import concurrent.futures
    import multiprocessing
    tag = f"{float(sigma):g}"
    (sdir(tag) / "kept").mkdir(parents=True, exist_ok=True)
    f0 = fold0()
    jobs = [(tag, sid, True) for sid in f0.held] + [(tag, sid, False) for sid in EXTRA]
    with concurrent.futures.ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_worker_init) as ex:
        rows = list(ex.map(fit_page, *zip(*jobs)))
    (sdir(tag) / "fit.json").write_text(json.dumps(rows, indent=1, default=float))


def _worker_init():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cv2.setNumThreads(2)


def cmd_region():
    rows = []
    for sid in fold0().held:
        lum = data.load_luminance(data.page_path(*sid))
        p = np.load(shp.CACHE / f"{sid[0]}__{sid[1]}.npy")
        out = decide.decide(p, lum).outside
        cls0 = cls0_of(sid, lum.shape)
        rows.append({"sid": list(sid), "scores": score(out, cls0, lum, p), "sides": side_errors(~out, cls0)})
        log.info("region %s/%s %s", *sid, rows[-1]["scores"]["cost"])
        del p
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "region.json").write_text(json.dumps(rows, indent=1, default=float))


def cmd_pictures(sigma):
    """The pictures again, from the saved heat and kept regions."""
    tag = f"{float(sigma):g}"
    f0 = fold0()
    for sid in list(f0.held) + EXTRA:
        lum = data.load_luminance(data.page_path(*sid))
        H = np.load(heat_path(tag, sid)).astype(np.float32) / 255.0
        K = np.unpackbits(np.load(sdir(tag) / "kept" / f"{sid[0]}__{sid[1]}.npy"))[:lum.size].reshape(lum.shape) > 0
        cls0 = cls0_of(sid, lum.shape) if sid in f0.held else None
        picture(lum, H, K, cls0).save(sdir(tag) / "look" / f"{sid[0]}__{sid[1]}.png")
        if sid == ("1921", "20180102094204401"):
            picture(lum, H, K, None, side=10_000, box=(0, 0, 700, 700)).save(sdir(tag) / "closeup_1921_204401_corner.png")


def cmd_compare(sigma):
    tag = f"{float(sigma):g}"
    from sfzmaps.frame import look
    font = look._font()
    tiles = []
    for sid in COMPARE:
        tiles.append((f"{sid[0]}/{sid[1]}", Image.open(sdir(tag) / "look" / f"{sid[0]}__{sid[1]}.png")))
    sid = ("1921", "20180102094204401")
    lum = data.load_luminance(data.page_path(*sid))
    H = np.load(heat_path(tag, sid)).astype(np.float32) / 255.0
    K = np.unpackbits(np.load(sdir(tag) / "kept" / f"{sid[0]}__{sid[1]}.npy"))[:lum.size].reshape(lum.shape) > 0
    box = (0, 0, 700, 700)      # look.CLOSEUPS[0]'s (54, 82, 354, 382) cuts the frame's rule off; this shows it whole
    close = picture(lum, H, K, None, side=10_000, box=box)
    tiles.append((f"1921/204401 top-left corner, full resolution {box}", close))
    gap, cap = 8, 22
    cols = 3
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    W = max(sum(t.width for _, t in r) + gap * (len(r) + 1) for r in rows)
    Hh = sum(max(t.height for _, t in r) + cap + gap for r in rows) + gap
    canvas = Image.new("RGB", (W, Hh), (200, 200, 200))
    d = ImageDraw.Draw(canvas)
    y = gap
    for r in rows:
        x = gap
        for name, t in r:
            d.text((x, y + 2), name, fill=(0, 0, 0), font=font)
            canvas.paste(t, (x, y + cap))
            x += t.width + gap
        y += max(t.height for _, t in r) + cap + gap
    d.text((gap, Hh - 18), f"sigma {tag}: red = predicted line heat H, blue = fitted kept region's edge",
           fill=(0, 0, 0), font=font)
    canvas.save(OUT / "compare.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cmd, *args = sys.argv[1:]
    {"train": cmd_train, "fit": cmd_fit, "region": cmd_region, "pictures": cmd_pictures, "compare": cmd_compare}[cmd](*args)
