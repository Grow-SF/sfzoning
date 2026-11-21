"""Issue #71, combined plan Task 1b: the cut lies on paper (spike level).  Throwaway.

Start: the combined test's "W = 0" arm (layout chances + straight-line shape + bounded blocks, no edge net).
Then each fitted line (frame side, notch side, block side) slides along its normal within one patch width of where
the layout put it, to where it lies on paper.

    uv run python scripts/spikes/issue71_on_paper.py layout    # the 3 heads again (the combined test's temporaries
                                                                # were deleted); patch chances to tmp/
    uv run python scripts/spikes/issue71_on_paper.py start     # the W = 0 arm on every page -> pages/<key>_start.json
    uv run python scripts/spikes/issue71_on_paper.py likelihood   # label rule 1 as a likelihood (training labels)
    uv run python scripts/spikes/issue71_on_paper.py slide     # every rule (RULES) on every page -> pages/<key>_slide.json
    uv run python scripts/spikes/issue71_on_paper.py table     # the report's tables
    uv run python scripts/spikes/issue71_on_paper.py residual sum+tilt   # mistakes within / beyond a patch of the cut
    uv run python scripts/spikes/issue71_on_paper.py explain 1921 20180102094204401 "side left"   # one line's rows
    uv run python scripts/spikes/issue71_on_paper.py pictures sum+tilt   # compare.png (windows in windows.json)

The rule carried forward is "sum+tilt": each frame side's two ends move independently within +-R (a shift and a
tilt), every other line (corner arc radius, notch sides, block sides, ellipse rings) shifts within +-R; each line
minimises  E(t) + COST_FRAME x (ink in the first kept row)  with E the layout's expected cost of the window's kept
rows; ties stay nearest the layout's position; page-edge sides don't move; lines in turn, one pass.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue71_combined as cb        # noqa: E402
import issue71_dino as dino          # noqa: E402
import issue71_shape as sh           # noqa: E402
from sfzmaps.frame import data, decide   # noqa: E402

log = logging.getLogger("on_paper")
OUT = data.DATA_DIR / "frame-on-paper"
TMP = OUT / "tmp"
PAGES = OUT / "pages"
cb.TMP = TMP          # the combined test's layout_p reads TMP / "layout"
key = cb.key


def cmd_layout():
    torch.set_num_threads(4)
    from sfzmaps.frame import folds
    f0 = folds.split(data.labelled_sheets(), 4)[0]
    (TMP / "layout").mkdir(parents=True, exist_ok=True)
    fits, info = [], {}
    for seed in cb.SEEDS:
        t0 = time.perf_counter()
        fit, hist = dino.train_head(*cb.VARIANT, list(f0.train), seed=seed)
        info[f"seed{seed}"] = {"steps": len(hist), "loss": hist[-1], "train_s": time.perf_counter() - t0}
        log.info("seed %d: %d steps, loss %.4f, %.0f s", seed, len(hist), hist[-1], time.perf_counter() - t0)
        fits.append(fit)
    for sid in cb.all_pages():
        with torch.no_grad():
            ps = [torch.softmax(f.head(f.prep(sid)), 0).numpy() for f in fits]
        np.save(TMP / "layout" / f"{key(sid)}.npy", np.mean(ps, 0).astype(np.float32))
    (OUT / "layout.json").write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# scoring: the combined test's cost and split, plus each class's ink total (for shares)
# ---------------------------------------------------------------------------------------------------------------

def score(deleted, sid, lum):
    r = cb.score(deleted, sid, lum)
    cls = data.class_mask(data.load_label(*sid), data.load_remove(*sid), lum.shape)
    ink = 1.0 - lum.astype(np.float64)
    r["frame_ink"] = float(ink[cls != 0].sum())
    r["map_ink"] = float(ink[cls == 0].sum())
    r["frame_kept_pct"] = 100 * r["frame_kept"] / max(r["frame_ink"], 1e-9)
    r["map_cut_pct"] = 100 * r["map_cut"] / max(r["map_ink"], 1e-9)
    return r


# ---------------------------------------------------------------------------------------------------------------
# the start: the combined test's W = 0 arm (bounded blocks, notches, the rectangle re-run at +-0 px about (b)'s,
# sides and arcs, blocks re-scored), all on the layout chances only (no edge net anywhere)
# ---------------------------------------------------------------------------------------------------------------

def start_decision(sid, lum, pl, bounds):
    ab = json.loads((cb.PAGES / f"{key(sid)}_ab.json").read_text())
    h, w = lum.shape
    Dl, ink = sh.cost_map(pl, lum)
    rect = sh.Frame(h, w, ab["start_rect"]["phi"], dict(ab["start_rect"]["beta"]))
    K0 = rect.kept()
    shapes = cb.block_shapes_b(pl, Dl, K0, rect.phi, bounds)
    B0, _ = sh.accept_blocks(shapes, Dl, K0, cb.LAM_B)
    Dz = np.where(B0, 0, Dl).astype(np.float32)
    inkz = np.where(B0, 0, ink).astype(np.float32)
    notches = cb.greedy_notches_c(Dz, Dz, inkz, K0, rect, cb.BW, cb.LAM_N, 16)
    Nm = sh.notches_mask(notches, h, w)
    Dzz = np.where(Nm, 0, Dz).astype(np.float32)
    fr_b = cb.fr_load(ab["frame_b"])
    centre = sh.Frame(h, w, fr_b.phi, dict(fr_b.beta))
    fr = sh.fit_sides_arcs(cb.rect_given_w(centre, Dzz, 0.0), Dzz, notches)
    K = fr.kept()
    removed, kept = sh.accept_blocks(shapes, Dl, K, cb.LAM_B)
    return fr, kept, K, removed & K, Dl, ink


def blocks_mask(blocks, K, h, w):
    m = np.zeros((h, w), bool)
    for b in blocks:
        y0, y1, x0, x1, bm = sh.block_mask_box(b, h, w)
        m[y0:y1, x0:x1] |= bm
    return m & K


def start_page(sid, bounds):
    path = PAGES / f"{key(sid)}_start.json"
    if path.exists():
        return json.loads(path.read_text())
    lum = data.load_luminance(data.page_path(*sid))
    pl = cb.layout_p(sid, lum.shape)
    t0 = time.perf_counter()
    fr, kept, K, removed, _, _ = start_decision(sid, lum, pl, bounds)
    out = {"sid": list(sid), "seconds": time.perf_counter() - t0, "frame": cb.fr_dump(fr), "blocks": cb.blk_dump(kept)}
    if cb.labelled(sid):
        out["score"] = score(~K | removed, sid, lum)
    PAGES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    log.info("start %s/%s %.1f s %s", *sid, out["seconds"],
             f"cost {out['score']['cost']:.0f}" if "score" in out else "")
    return out


def cmd_start():
    wb = json.loads((cb.OUT / "wb.json").read_text())
    with cb._pool() as ex:
        rows = [f.result() for f in [ex.submit(start_page, sid, wb["bounds"]) for sid in cb.all_pages()]]
    held = [r for r in rows if tuple(r["sid"]) in set(cb.held())]
    print("pooled held", {k: round(sum(r["score"][k] for r in held)) for k in ("cost", "frame_kept", "map_cut")})



# ---------------------------------------------------------------------------------------------------------------
# the lines: every fitted line as (signed distance s, kept side positive, 0 at the line now; the px it controls)
# ---------------------------------------------------------------------------------------------------------------

def reach(shape):
    """One patch width: page px per patch = the page's long side / 111."""
    return max(shape) / 111.0


def _box(xs, ys, h, w, pad):
    x0, x1 = max(0, int(math.floor(min(xs) - pad))), min(w, int(math.ceil(max(xs) + pad)) + 1)
    y0, y1 = max(0, int(math.floor(min(ys) - pad))), min(h, int(math.ceil(max(ys) + pad)) + 1)
    return y0, y1, x0, x1


class State:
    """The decision: the frame (sides, arcs, notches) and the kept blocks."""

    def __init__(self, fr, blocks):
        self.fr, self.blocks = fr, list(blocks)

    def kept(self, y0, y1, x0, x1, skip_side=None, skip_notch=None, skip_block=None, skip_arc=None):
        """Kept px on the crop with one element left out (a side: pushed off the page, its corners' arcs off)."""
        fr = self.fr
        K = _kept_rows(fr, y0, y1, x0, x1, skip_side, skip_arc)
        h, w = fr.h, fr.w
        for k, n in enumerate(fr.notches):
            if k != skip_notch:
                K &= ~_notch_crop(n, h, w, y0, y1, x0, x1)
        for k, b in enumerate(self.blocks):
            if k != skip_block:
                by0, by1, bx0, bx1, bm = sh.block_mask_box(b, h, w)
                ya, yb, xa, xb = max(y0, by0), min(y1, by1), max(x0, bx0), min(x1, bx1)
                if ya < yb and xa < xb:
                    K[ya - y0:yb - y0, xa - x0:xb - x0] &= ~bm[ya - by0:yb - by0, xa - bx0:xb - bx0]
        return K

    def deleted(self, chunk=1024):
        h, w = self.fr.h, self.fr.w
        out = np.zeros((h, w), bool)
        for y0 in range(0, h, chunk):
            out[y0:y0 + chunk] = ~self.kept(y0, min(h, y0 + chunk), 0, w)
        return out

    def copy(self):
        fr = self.fr.copy()
        fr.notches = [sh.Notch(**dict(n.__dict__)) for n in self.fr.notches]
        return State(fr, [sh.Block(b.kind, b.cx, b.cy, b.theta, tuple(b.p), b.gain) for b in self.blocks])


def _kept_rows(fr, y0, y1, x0, x1, skip_side=None, skip_arc=None):
    """`Frame.kept_rows` with one side (and the arcs at its corners) left out."""
    cv = fr.canvas
    y, x = np.mgrid[y0:y1, x0:x1]
    u, v = cv.uv(x, y)
    keep = np.ones(u.shape, bool)
    ea = {}
    for side in sh.SIDES:
        e, al = sh.e_along(side, u, v)
        dist = e - fr.bound(side, al)
        ea[side] = dist
        if side != skip_side:
            keep &= dist >= 0
    for c in sh.CORNER_SIDES:
        r = fr.arc[c]
        if r <= 0 or skip_side in c or c == skip_arc:
            continue
        a, b = ea[c[0]], ea[c[1]]
        near = (a >= 0) & (b >= 0) & (a < r) & (b < r)
        if near.any():
            an, bn = a[near], b[near]
            cut = an + bn + np.sqrt(2.0 * an * bn) < r
            idx = np.flatnonzero(near.ravel())[cut]
            keep.ravel()[idx] = False
    return keep


def _notch_crop(n, h, w, y0, y1, x0, x1):
    cv = sh.Canvas.page(n.psi, h, w)
    y, x = np.mgrid[y0:y1, x0:x1]
    u, v = cv.uv(x, y)
    e, al = sh.e_along(n.side, u, v)
    return (al >= n.a0) & (al < n.a1) & (e < n.d)


def side_is_page_edge(fr, side):
    """A side whose line lies outside the page along its whole length cuts nothing: the page's own edge."""
    cv = fr.canvas
    lo, hi = fr.ends(side)
    al = np.linspace(lo, hi, 64)
    e = fr.bound(side, al)
    u, v = {"top": (al, e), "bottom": (al, -e), "left": (e, al), "right": (-e, al)}[side]
    x, y = cv.xy(u, v)
    inside = (x > 0) & (x < fr.w) & (y > 0) & (y < fr.h)
    return not inside.any()


def lines(st, R):
    """Every line of the decision: (name, crop box, function crop -> (s, zone), apply(t))."""
    fr = st.fr
    h, w = fr.h, fr.w
    out = []
    cv = fr.canvas
    for side in sh.SIDES:
        if side_is_page_edge(fr, side):
            continue
        lo, hi = fr.ends(side)
        c_lo, c_hi = cb_END[side]
        f_lo = lo + (fr.arc[c_lo] + 8 if fr.arc[c_lo] > 0 else 0)
        f_hi = hi - (fr.arc[c_hi] + 8 if fr.arc[c_hi] > 0 else 0)
        if f_hi <= f_lo:
            continue
        al = np.linspace(f_lo, f_hi, 64)
        pts = []
        for de in (-R - 2, R + 2):
            e = fr.bound(side, al) + de
            u, v = {"top": (al, e), "bottom": (al, -e), "left": (e, al), "right": (-e, al)}[side]
            pts.append(cv.xy(u, v))
        xs = np.concatenate([p[0] for p in pts]); ys = np.concatenate([p[1] for p in pts])
        box = _box(xs, ys, h, w, 2)

        def f(y0, y1, x0, x1, side=side, f_lo=f_lo, f_hi=f_hi, with_al=False):
            y, x = np.mgrid[y0:y1, x0:x1]
            u, v = cv.uv(x, y)
            e, al = sh.e_along(side, u, v)
            s = e - fr.bound(side, al)
            zone = (al >= f_lo) & (al < f_hi) & st.kept(y0, y1, x0, x1, skip_side=side)
            if with_al:
                return s, zone, (al - f_lo) / (f_hi - f_lo)
            return s, zone

        def apply(t, side=side, f_lo=f_lo, f_hi=f_hi):
            if isinstance(t, tuple):          # each end moved: t_lo at f_lo, t_hi at f_hi (a shift and a tilt)
                t_lo, t_hi = t
                lo, hi = fr.ends(side)
                m = (lo + hi) / 2.0
                b = (t_hi - t_lo) / (f_hi - f_lo)
                fr.beta[side] += t_lo + b * (m - f_lo)
                fr.tilt[side] += b
            else:
                fr.beta[side] += t
        out.append((f"side {side}", box, f, apply))
    edge_sides = {side for side in sh.SIDES if side_is_page_edge(fr, side)}
    for c in sh.CORNER_SIDES:
        if c[0] in edge_sides or c[1] in edge_sides:
            continue
        s1, s2 = c
        r0 = fr.arc[c]
        span = r0 + 2 * R + 48
        uc = fr.beta["left"] if s2 == "left" else -fr.beta["right"]
        vc = fr.beta["top"] if s1 == "top" else -fr.beta["bottom"]
        du = 1 if s2 == "left" else -1
        dv = 1 if s1 == "top" else -1
        pts = [cv.xy(uc + a * du, vc + b * dv) for a in (-8, span) for b in (-8, span)]
        xs, ys = zip(*pts)
        box = _box(xs, ys, h, w, 2)

        def f(y0, y1, x0, x1, c=c):
            y, x = np.mgrid[y0:y1, x0:x1]
            u, v = cv.uv(x, y)
            e1, al1 = sh.e_along(c[0], u, v)
            e2, al2 = sh.e_along(c[1], u, v)
            a = e1 - fr.bound(c[0], al1)
            b = e2 - fr.bound(c[1], al2)
            ok = (a >= 0) & (b >= 0)
            rho = np.where(ok, a + b + np.sqrt(2 * np.maximum(a, 0) * np.maximum(b, 0)), -1e9)
            s = rho - fr.arc[c]
            return s, ok & st.kept(y0, y1, x0, x1, skip_arc=c)

        def apply(t, c=c):
            fr.arc[c] = max(0.0, fr.arc[c] + t)
        out.append((f"arc {c[0]}-{c[1]}", box, f, apply))
    for k, n in enumerate(fr.notches):
        ncv = sh.Canvas.page(n.psi, h, w)
        E, _ = ncv.EA(n.side)
        corners = []
        for a in (n.a0 - R - 2, n.a1 + R + 2):
            for e in (-E - 2.0, n.d + R + 2):
                u, v = {"top": (a, e), "bottom": (a, -e), "left": (e, a), "right": (-e, a)}[n.side]
                corners.append(ncv.xy(u, v))
        xs, ys = zip(*corners)
        box = _box(xs, ys, h, w, 2)
        for which in ("d", "a0", "a1"):
            def f(y0, y1, x0, x1, n=n, k=k, which=which, ncv=ncv):
                y, x = np.mgrid[y0:y1, x0:x1]
                u, v = ncv.uv(x, y)
                e, al = sh.e_along(n.side, u, v)
                if which == "d":
                    s, zone = e - n.d, (al >= n.a0) & (al < n.a1)
                elif which == "a0":
                    s, zone = n.a0 - al, e < n.d
                else:
                    s, zone = al - n.a1, e < n.d
                return s, zone & st.kept(y0, y1, x0, x1, skip_notch=k)

            def apply(t, n=n, which=which):
                if which == "d":
                    n.d += t
                elif which == "a0":
                    n.a0 -= t
                else:
                    n.a1 += t
            out.append((f"notch {k} {n.side} {which}", box, f, apply))
    for k, b in enumerate(st.blocks):
        t_ = math.radians(b.theta)
        c, sn = math.cos(t_), math.sin(t_)
        if b.kind == "rect":
            u0, u1, v0, v1 = b.p
            pts = [(u, v) for u in (u0 - R - 2, u1 + R + 2) for v in (v0 - R - 2, v1 + R + 2)]
        else:
            a_, bb = b.p
            pts = [(u, v) for u in (-a_ - R - 2, a_ + R + 2) for v in (-bb - R - 2, bb + R + 2)]
        xs = [b.cx + c * u - sn * v for u, v in pts]
        ys = [b.cy + sn * u + c * v for u, v in pts]
        box = _box(xs, ys, h, w, 2)
        whiches = ("u0", "u1", "v0", "v1") if b.kind == "rect" else ("ring",)
        for which in whiches:
            def f(y0, y1, x0, x1, b=b, k=k, which=which, c=c, sn=sn):
                y, x = np.mgrid[y0:y1, x0:x1]
                dx, dy = x + 0.5 - b.cx, y + 0.5 - b.cy
                u, v = c * dx + sn * dy, -sn * dx + c * dy
                if b.kind == "rect":
                    u0, u1, v0, v1 = b.p
                    inu, inv = (u >= u0) & (u < u1), (v >= v0) & (v < v1)
                    s, zone = {"u0": (u0 - u, inv), "u1": (u - u1, inv), "v0": (v0 - v, inu),
                               "v1": (v - v1, inu)}[which]
                else:
                    s, zone = _ellipse_s(u, v, *b.p), np.ones(u.shape, bool)
                return s, zone & st.kept(y0, y1, x0, x1, skip_block=k)

            def apply(t, b=b, which=which):
                if b.kind == "rect":
                    u0, u1, v0, v1 = b.p
                    b.p = {"u0": (u0 - t, u1, v0, v1), "u1": (u0, u1 + t, v0, v1), "v0": (u0, u1, v0 - t, v1),
                           "v1": (u0, u1, v0, v1 + t)}[which]
                else:
                    b.p = (max(2.0, b.p[0] + t), max(2.0, b.p[1] + t))
            out.append((f"block {k} {b.kind} {which}", box, f, apply))
    return out


cb_END = sh.END_CORNERS


def _ellipse_s(u, v, a, b):
    """The t with the px on the ellipse of semi-axes (a + t, b + t) (kept side: outside it, t < 0 inside...): s =
    the px's signed distance in that family, + outside the ellipse (kept), - inside (removed)."""
    lo = np.full(u.shape, -min(a, b) + 0.5)
    hi = np.full(u.shape, 4.0 * max(a, b) + 10)
    for _ in range(40):
        m = (lo + hi) / 2
        f = (u / (a + m)) ** 2 + (v / (b + m)) ** 2
        inside = f <= 1.0            # px inside the ellipse of offset m: the boundary offset is below m
        hi = np.where(inside, m, hi)
        lo = np.where(inside, lo, m)
    return (lo + hi) / 2


def row_stats(f, box, R, ink, D, q):
    """Per row r in [-R, R) (s in [r, r + 1)) over the line's zone: px count, ink, D, q (sum)."""
    y0, y1, x0, x1 = box
    s, zone = f(y0, y1, x0, x1)
    r = np.floor(s).astype(np.int64)
    sel = zone & (r >= -R) & (r < R)
    idx = r[sel] + R
    n = 2 * R
    cnt = np.bincount(idx, minlength=n).astype(np.float64)
    ik = np.bincount(idx, weights=ink[y0:y1, x0:x1][sel].astype(np.float64), minlength=n)
    dd = np.bincount(idx, weights=D[y0:y1, x0:x1][sel].astype(np.float64), minlength=n)
    qq = np.bincount(idx, weights=q[y0:y1, x0:x1][sel].astype(np.float64), minlength=n)
    return cnt, ik, dd, qq

# ---------------------------------------------------------------------------------------------------------------
# label rule 1 as a likelihood: the ink under a labelled side against the ink under any row within a patch of it,
# measured on fold 0's training labels (the held sheets are not used)
# ---------------------------------------------------------------------------------------------------------------

BINS = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, np.inf])   # ink under the row, above the window's
                                                                         # lightest row (log-spaced, not fitted)


def rel_ink(cnt, ik):
    f = ik / np.maximum(cnt, 1.0)
    ok = cnt > 0
    if not ok.any():
        return np.zeros_like(f), ok
    return np.where(ok, f - f[ok].min(), 0.0), ok


def label_rows(sid):
    """Per labelled side (not on the page edge): the row ink at r in [-R, R) about the label's side line (r = 0:
    the first row on the map side), over the middle 70% of the side."""
    lum = data.load_luminance(data.page_path(*sid))
    h, w = lum.shape
    ink = 1.0 - lum
    R = int(math.ceil(reach(lum.shape)))
    out = []
    for poly in data.load_label(*sid):
        P = np.asarray(poly, float)
        if len(P) <= 4:
            continue
        sides = sh.label_sides(P, (h, w))
        edge = sh.page_edge_sides(sides, (h, w))
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        for side, (a, b) in sides.items():
            if side in edge:
                continue
            if side in ("top", "bottom"):
                xs = np.arange(int(x0 + 0.15 * (x1 - x0)), int(x1 - 0.15 * (x1 - x0)))
                yc = a * (xs + 0.5) + b
                ylo, yhi = max(0, int(yc.min()) - R - 3), min(h, int(yc.max()) + R + 4)
                yy, xx = np.mgrid[ylo:yhi, xs[0]:xs[-1] + 1]
            else:
                ys = np.arange(int(y0 + 0.15 * (y1 - y0)), int(y1 - 0.15 * (y1 - y0)))
                xc = a * (ys + 0.5) + b
                xlo, xhi = max(0, int(xc.min()) - R - 3), min(w, int(xc.max()) + R + 4)
                yy, xx = np.mgrid[ys[0]:ys[-1] + 1, xlo:xhi]
            pts = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)
            sgn = sh._inward(side, (a, b), pts).reshape(xx.shape)
            r = np.floor(sgn).astype(np.int64)
            sel = (r >= -R) & (r < R)
            cnt = np.bincount(r[sel] + R, minlength=2 * R).astype(np.float64)
            ik = np.bincount(r[sel] + R, weights=ink[yy[sel], xx[sel]].astype(np.float64), minlength=2 * R)
            g, ok = rel_ink(cnt, ik)
            out.append({"sid": list(sid), "side": side, "R": R, "g": g.tolist(), "ok": ok.tolist(),
                        "f": (ik / np.maximum(cnt, 1)).tolist()})
    return out


def cmd_likelihood():
    from sfzmaps.frame import folds
    train = list(folds.split(data.labelled_sheets(), 4)[0].train)
    with cb._pool() as ex:
        rows = [r for f in [ex.submit(label_rows, sid) for sid in train] for r in f.result()]
    nb = len(BINS) - 1
    pos, allr = np.zeros(nb), np.zeros(nb)
    for r in rows:
        g, ok, R = np.asarray(r["g"]), np.asarray(r["ok"]), r["R"]
        b = np.clip(np.searchsorted(BINS, g, side="right") - 1, 0, nb - 1)
        allr += np.bincount(b[ok], minlength=nb)
        pos[b[R]] += 1
    L = ((pos + 1) / (pos.sum() + nb)) / ((allr + 1) / (allr.sum() + nb))
    out = {"bins": BINS.tolist(), "pos": pos.tolist(), "all": allr.tolist(), "L": L.tolist(), "n_sides": len(rows),
           "profiles": rows}
    (OUT / "likelihood.json").write_text(json.dumps(out, default=float))
    print("sides", len(rows))
    for k in range(nb):
        print(f"g in [{BINS[k]:.0e}, {BINS[k + 1]:.0e}): label row {pos[k]:.0f}, all rows {allr[k]:.0f}, L {L[k]:.3f}")


def load_L():
    d = json.loads((OUT / "likelihood.json").read_text())
    return np.asarray(d["L"])


def L_of(g, L):
    nb = len(BINS) - 1
    return L[np.clip(np.searchsorted(BINS, g, side="right") - 1, 0, nb - 1)]


# ---------------------------------------------------------------------------------------------------------------
# where to put one line, given its rows (r in [-R, R); the line at t: rows < t deleted, row t the first kept)
# ---------------------------------------------------------------------------------------------------------------

def choose(rule, cnt, ik, dd, qq, L):
    n = len(cnt)
    R = n // 2
    ts = np.arange(-R, R)                     # t = -R .. R - 1 (row t under the line)
    # the layout's expected cost (sum of D over the kept rows in the window), relative
    E = np.array([dd[t + R:].sum() for t in ts])

    def best(J):                              # ties (paper on a 1-bit scan is exactly 0): stay nearest the layout's
        return int(ts[np.lexsort((np.abs(ts), np.round(J, 6)))[0]])
    if rule == "layout":
        return best(E), {}
    under = ik[ts + R]
    if rule == "sum":                         # the ink in the first kept row charged as frame ink left in
        return best(E + decide.COST_FRAME * under), {}
    if rule == "sum1":                        # sensitivity: charged as map ink cut instead
        return best(E + decide.COST_MAP * under), {}
    if rule == "sum2":                        # both rows touching the line: kept one as frame left in, cut one as map cut
        before = np.concatenate([[0.0], ik[:-1]])[ts + R]
        return best(E + decide.COST_FRAME * under + decide.COST_MAP * before), {}
    g, ok = rel_ink(cnt, ik)
    if rule == "lex":                         # on paper first (the lightest row), the layout's cost among those
        cand = ok & (g <= g[ok].min() + 1e-12) if ok.any() else np.ones(n, bool)
        Jc = np.where(cand, E, np.inf)
        return best(Jc), {}
    # bayes: the layout's row chances as the prior on the first map row, rule 1 as the likelihood
    Q = np.where(cnt > 0, qq / np.maximum(cnt, 1.0), np.nan)
    Q = np.interp(np.arange(n), np.flatnonzero(cnt > 0), Q[cnt > 0]) if (cnt > 0).any() else np.full(n, 0.5)
    Q = np.minimum.accumulate(np.clip(Q, 0, 1))          # outside chance falls inward (monotone)
    m = np.zeros(n + 1)                                   # mass on the first map row at r = -R .. R (R: beyond)
    m[0] = 1 - Q[0]
    m[1:n] = Q[:-1] - Q[1:]
    m[n] = Q[-1]
    lik = np.ones(n + 1)
    lik[1:n] = np.where(ok[1:], L_of(g[1:], L), 1.0)
    post = m * lik
    post /= max(post.sum(), 1e-300)
    C = np.concatenate([[0.0], np.cumsum(ik)])            # C[i] = ink of rows -R .. -R + i - 1
    J = np.zeros(n)
    for i, t in enumerate(ts):
        ti = t + R
        rs = np.arange(n + 1)
        kept_frame = np.where(rs > ti, C[rs.clip(max=n)] - C[ti], 0.0)   # rows t .. r - 1 kept, outside
        cut_map = np.where(rs < ti, C[ti] - C[rs.clip(max=n)], 0.0)      # rows r .. t - 1 cut, map
        J[i] = (post * (decide.COST_FRAME * kept_frame + decide.COST_MAP * cut_map)).sum()
    return best(J), {"post_max_row": int(np.argmax(post)) - R}


RULES = ("layout", "sum", "sum1", "sum2", "lex", "bayes", "layout+tilt", "sum+tilt")


def slide(st, lum, pl, rule, L, log_lines=None):
    """Every line in turn (sides, notch sides, block sides), each moved to its chosen t given the ones before."""
    h, w = lum.shape
    R = int(math.ceil(reach(lum.shape)))
    D, ink = sh.cost_map(pl, lum)
    q = (pl[1] + pl[2]).astype(np.float32)
    moves = []
    base = rule.replace("+tilt", "")
    for name, box, f, apply in lines(st, R):
        if rule.endswith("+tilt") and name.startswith("side"):
            (t_lo, t_hi), info = choose_ends(base, f, box, R, ink, D)
            moves.append({"line": name, "t": [t_lo, t_hi], **info})
            if t_lo or t_hi:
                apply((t_lo, t_hi))
            continue
        cnt, ik, dd, qq = row_stats(f, box, R, ink, D, q)
        if cnt.sum() == 0:
            continue
        t, info = choose(base, cnt, ik, dd, qq, L)
        g, ok = rel_ink(cnt, ik)
        moves.append({"line": name, "t": t, "under_before": float(ik[R]), "under_after": float(ik[t + R]),
                      **info})
        if t:
            apply(t)
    return moves


ALONG_BIN = 32      # px: the side's columns are binned along it; the line's shift is taken at each bin's middle


def choose_ends(rule, f, box, R, ink, D):
    """A frame side's two ends each moved within +-R (a shift and a tilt), minimising the summed per-bin cost: the
    layout's expected cost of the kept rows, plus (rule "sum") COST_FRAME x the ink in each bin's first kept row."""
    y0, y1, x0, x1 = box
    s, zone, x = f(y0, y1, x0, x1, with_al=True)
    r = np.floor(s).astype(np.int64)
    sel = zone & (r >= -R) & (r < R)
    length = None
    xs = x[sel]
    nb = 64
    if sel.any():
        span = float(np.ptp(xs)) if len(xs) else 1.0
    b = np.clip((x[sel] * nb).astype(np.int64), 0, nb - 1)
    idx = b * (2 * R) + r[sel] + R
    ik = np.bincount(idx, weights=ink[y0:y1, x0:x1][sel].astype(np.float64), minlength=nb * 2 * R).reshape(nb, 2 * R)
    dd = np.bincount(idx, weights=D[y0:y1, x0:x1][sel].astype(np.float64), minlength=nb * 2 * R).reshape(nb, 2 * R)
    keptD = np.cumsum(dd[:, ::-1], 1)[:, ::-1]           # keptD[b, i] = sum of D over rows >= i - R in bin b
    xc = (np.arange(nb) + 0.5) / nb
    ts = np.arange(-R, R)
    best = None
    for tl in ts:
        k = np.clip(np.round(tl + (ts[:, None] - tl) * xc[None, :]).astype(np.int64) + R, 0, 2 * R - 1)  # (th, b)
        J = keptD[np.arange(nb)[None, :], k].sum(1)
        if rule == "sum":
            J = J + decide.COST_FRAME * ik[np.arange(nb)[None, :], k].sum(1)
        for th, j in zip(ts, J):
            cand = (round(float(j), 6), abs(int(tl)) + abs(int(th)), int(tl), int(th))
            if best is None or cand < best:
                best = cand
    k0 = np.full(nb, R)
    under0 = float(ik[np.arange(nb), k0].sum())
    kk = np.clip(np.round(best[2] + (best[3] - best[2]) * xc).astype(np.int64) + R, 0, 2 * R - 1)
    return (best[2], best[3]), {"under_before": under0, "under_after": float(ik[np.arange(nb), kk].sum())}


def load_state(sid):
    d = json.loads((PAGES / f"{key(sid)}_start.json").read_text())
    fr = cb.fr_load(d["frame"])
    blocks = [sh.Block(b["kind"], b["cx"], b["cy"], b["theta"], tuple(b["p"]), b["gain"]) for b in d["blocks"]]
    return State(fr, blocks)


def state_dump(st):
    return {"frame": cb.fr_dump(st.fr), "blocks": cb.blk_dump(st.blocks)}


def slide_page(sid, rules):
    path = PAGES / f"{key(sid)}_slide.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in rules if r not in old]
    if not todo:
        return old
    lum = data.load_luminance(data.page_path(*sid))
    pl = cb.layout_p(sid, lum.shape)
    L = load_L()
    out = dict(old)
    st0 = load_state(sid)
    if "start" not in out:
        out["start"] = {"state": state_dump(st0)}
        if cb.labelled(sid):
            out["start"]["score"] = score(st0.deleted(), sid, lum)
    for rule in todo:
        t0 = time.perf_counter()
        st = st0.copy()
        moves = slide(st, lum, pl, rule, L)
        secs = time.perf_counter() - t0
        out[rule] = {"moves": moves, "seconds": secs, "state": state_dump(st)}
        if cb.labelled(sid):
            out[rule]["score"] = score(st.deleted(), sid, lum)
        log.info("slide %s/%s %s %.1f s %s moves %s", *sid, rule, secs,
                 f"cost {out[rule]['score']['cost']:.0f}" if cb.labelled(sid) else "",
                 [m["t"] for m in moves])
    path.write_text(json.dumps(out, indent=1, default=float))
    return out


def cmd_slide(rules=",".join(RULES)):
    rules = rules.split(",")
    with cb._pool() as ex:
        rows = {tuple(sid): f for sid, f in [(sid, ex.submit(slide_page, sid, rules)) for sid in cb.all_pages()]}
        rows = {k: f.result() for k, f in rows.items()}
    table(rows)


def table(rows=None):
    if rows is None:
        rows = {sid: json.loads((PAGES / f"{key(sid)}_slide.json").read_text()) for sid in cb.all_pages()}
    held = cb.held()
    arms = [a for a in ("start",) + RULES if all(a in rows[s] for s in held)]
    print("| sheet | arm | cost | frame/legend ink left in | map ink cut |\n|---|---|---:|---:|---:|")
    for sid in held:
        for a in arms:
            r = rows[sid][a]["score"]
            print(f"| {sid[0]}/{sid[1]} | {a} | {r['cost']:,.0f} | {r['frame_kept_pct']:.2f}% ({r['frame_kept']:,.0f}) | "
                  f"{r['map_cut_pct']:.3f}% ({r['map_cut']:,.0f}) |")
    print("\n| pooled (6 held) | cost | frame/legend ink left in | map ink cut | edge | layout |\n|---|---:|---:|---:|---:|---:|")
    for a in arms:
        t = {k: sum(rows[s][a]["score"][k] for s in held) for k in
             ("cost", "frame_kept", "map_cut", "frame_ink", "map_ink", "cost_edge", "cost_layout")}
        print(f"| {a} | {t['cost']:,.0f} | {100 * t['frame_kept'] / t['frame_ink']:.2f}% ({t['frame_kept']:,.0f}) | "
              f"{100 * t['map_cut'] / t['map_ink']:.3f}% ({t['map_cut']:,.0f}) | {t['cost_edge']:,.0f} | "
              f"{t['cost_layout']:,.0f} |")
    for sid in [s for s in cb.PICS if cb.labelled(s)]:
        for a in [a for a in ("start",) + RULES if a in rows[sid]]:
            r = rows[sid][a]["score"]
            print(f"in-sample {sid[0]}/{sid[1]} {a}: cost {r['cost']:,.0f} frame {r['frame_kept_pct']:.2f}% "
                  f"map {r['map_cut_pct']:.3f}%")


# ---------------------------------------------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------------------------------------------

def crop_img(sid, lum, deleted, win):
    """1:1: ink coloured by the decision (kept blue, deleted red), the kept region's edge magenta, the label green."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    x0, y0, x1, y1 = win
    L = lum[y0:y1, x0:x1]
    Dl = deleted[y0:y1, x0:x1]
    ink = np.clip(1.0 - L, 0, 1)[..., None]
    col = np.where(Dl[..., None], np.array([255, 0, 0], np.float32), np.array([0, 0, 255], np.float32))
    rgb = (1 - ink) * 255 + ink * col
    kept = ~Dl
    edge = kept & ~ndimage.binary_erosion(kept, border_value=1) & ndimage.binary_dilation(Dl)
    rgb[edge] = (255, 0, 255)
    img = Image.fromarray(np.round(rgb).astype(np.uint8), "RGB")
    if cb.labelled(sid):
        d = ImageDraw.Draw(img)
        for poly in data.load_label(*sid) + data.load_remove(*sid):
            d.line([(x - x0, y - y0) for x, y in poly + poly[:1]], fill=(0, 170, 0), width=1)
    return img


def state_from(d, h, w):
    fr = cb.fr_load(d["frame"])
    return State(fr, [sh.Block(b["kind"], b["cx"], b["cy"], b["theta"], tuple(b["p"]), b["gain"]) for b in d["blocks"]])


def cmd_pictures(rule="bayes", windows_json=None):
    """compare.png: full-resolution close-ups, start | after; then 1948/14 and 1948/4 whole, start | after."""
    from PIL import Image, ImageDraw
    wins = json.loads((OUT / "windows.json").read_text())
    rows = []
    cache = {}
    for sid_s, win, cap in wins:
        sid = tuple(sid_s)
        if sid not in cache:
            lum = data.load_luminance(data.page_path(*sid))
            d = json.loads((PAGES / f"{key(sid)}_slide.json").read_text())
            h, w = lum.shape
            cache[sid] = (lum, state_from(d["start"]["state"], h, w).deleted(), state_from(d[rule]["state"], h, w).deleted())
        lum, d0, d1 = cache[sid]
        rows.append((f"{cap}: start | on paper ({rule}), 1:1; kept ink blue, cut red, cut edge magenta, label green",
                     [crop_img(sid, lum, d0, win), crop_img(sid, lum, d1, win)], sid))
    for sid in (("1948", "14"), ("1948", "4")):
        lum = cache[sid][0] if sid in cache else data.load_luminance(data.page_path(*sid))
        d = json.loads((PAGES / f"{key(sid)}_slide.json").read_text())
        h, w = lum.shape
        pl = cb.layout_p(sid, lum.shape)
        q = pl[1] + pl[2]
        del pl
        ts = []
        for arm in ("start", rule):
            dd = state_from(d[arm]["state"], h, w).deleted()
            img = cb.look.sheet_picture(lum, q, dd, 1100)
            ts.append(img)
        rows.append((f"{sid[0]}/{sid[1]} whole (unlabelled): start | on paper ({rule}); red = layout says not map, "
                     "magenta = the cut", ts, sid))
    cb._grid_picture(rows, OUT / "compare.png", lambda t: t)


def explain(series, sheet, line_name):
    """One line's rows at the start: runs of ink / paper, the layout's row chance, the prior, the likelihood, the
    posterior, and each rule's choice."""
    sid = (series, sheet)
    lum = data.load_luminance(data.page_path(*sid))
    pl = cb.layout_p(sid, lum.shape)
    D, ink = sh.cost_map(pl, lum)
    q = (pl[1] + pl[2]).astype(np.float32)
    del pl
    st = load_state(sid)
    R = int(math.ceil(reach(lum.shape)))
    L = load_L()
    for name, box, f, apply in lines(st, R):
        if name != line_name:
            continue
        cnt, ik, dd, qq = row_stats(f, box, R, ink, D, q)
        g, ok = rel_ink(cnt, ik)
        Q = qq / np.maximum(cnt, 1)
        n = len(cnt)
        res = {rule: choose(rule, cnt, ik, dd, qq, L)[0] for rule in RULES}
        print(f"{series}/{sheet} {name}: R {R}, px per row ~{cnt.mean():.0f}; choices {res}")
        print("rows (r: first kept row = r; r < 0 outward), ink per row above the lightest, layout outside chance, L")
        for r in range(-R, R):
            i = r + R
            tag = "".join(k[0].upper() for k, t in res.items() if t == r)
            print(f"{r:4d} g {g[i]:.4f} Q {Q[i]:.3f} L {L_of(np.array([g[i]]), L)[0]:6.2f} {tag}")
        return


def residual_page(sid, arm="start"):
    """Where the arm's mistakes lie: frame/legend ink left in and map ink cut within one patch of the arm's cut
    (a slide can reach it) or beyond (a layout miss)."""
    lum = data.load_luminance(data.page_path(*sid))
    h, w = lum.shape
    d = json.loads((PAGES / f"{key(sid)}_slide.json").read_text()) if (PAGES / f"{key(sid)}_slide.json").exists() \
        else {"start": {"state": json.loads((PAGES / f"{key(sid)}_start.json").read_text())}}
    dele = state_from(d[arm]["state"], h, w).deleted()
    R = reach(lum.shape)
    edge = np.zeros((h, w), bool)
    edge[:-1] |= dele[:-1] != dele[1:]
    edge[:, :-1] |= dele[:, :-1] != dele[:, 1:]
    near = cv2.distanceTransform(np.where(edge, 0, 1).astype(np.uint8), cv2.DIST_L2, 5) <= R
    cls = data.class_mask(data.load_label(*sid), data.load_remove(*sid), lum.shape)
    ink = 1.0 - lum.astype(np.float64)
    fk = (cls != 0) & ~dele
    mc = (cls == 0) & dele
    return {"sid": list(sid), "R": R, "fk_near": float(ink[fk & near].sum()), "fk_far": float(ink[fk & ~near].sum()),
            "mc_near": float(ink[mc & near].sum()), "mc_far": float(ink[mc & ~near].sum())}


def cmd_residual(arm="start"):
    with cb._pool() as ex:
        rows = [f.result() for f in [ex.submit(residual_page, sid, arm) for sid in cb.held()]]
    for r in rows:
        print(r["sid"], f"R {r['R']:.0f}", {k: round(v) for k, v in r.items() if k[:2] in ("fk", "mc")})
    print("pooled", {k: round(sum(r[k] for r in rows)) for k in ("fk_near", "fk_far", "mc_near", "mc_far")})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"layout": cmd_layout, "start": cmd_start, "likelihood": cmd_likelihood, "slide": cmd_slide,
     "table": table, "pictures": cmd_pictures, "explain": explain, "residual": cmd_residual}[sys.argv[1]](*sys.argv[2:])
