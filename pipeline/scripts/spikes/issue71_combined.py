"""Issue #71, combined plan Task 1: the combined test at spike level.  Throwaway.

Layout chances: the DINOv2 head (1554 px, last 4 layers, conv3 on a 128-dim PCA) trained on fold 0's 16 sheets,
seeds 0-2, softmax averaged at the patch grid, bilinearly upsampled to the page (`issue71_dino`).  Then three decisions
at the shape spike's fitted numbers (lambda_notch 1e4, beta 3, lambda_block 1e3):

  (a) the per-pixel cut (`decide.decide`) on the layout chances;
  (b) the shape decision (`issue71_shape`, unchanged) on the layout chances;
  (c) from (b)'s winning start: block candidates and notch candidates on the layout chances (blocks bounded by the
      labelled `remove` sizes x 1.5); the refinements (notch sides at full res, the rectangle within +-0.5 deg and
      +-W of (b)'s fitted rectangle, per-side tilt/bow/offset, fillets, the final block re-scoring) on blended chances:
      the full-resolution fold-0 net's within W of any of (b)'s fitted sides, fillets or notch sides, the layout's
      elsewhere.  W = the largest side-position error of (b) against the labels on the 6 held sheets x 1.5.

    uv run --with transformers --with safetensors python scripts/spikes/issue71_combined.py extract
    uv run python scripts/spikes/issue71_combined.py layout     # 3 heads, patch chances of every page (tmp)
    uv run python scripts/spikes/issue71_combined.py ab         # (a), (b); the edge chances meanwhile (GPU turn)
    uv run python scripts/spikes/issue71_combined.py wb         # W and the block bounds
    uv run python scripts/spikes/issue71_combined.py c          # (c)
    uv run python scripts/spikes/issue71_combined.py pictures   # compare.png, compare_held.png, thin_rule.png
    uv run python scripts/spikes/issue71_combined.py table      # the report's tables
    uv run python scripts/spikes/issue71_combined.py clean      # delete the temporaries (per-pixel chances)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue71_dino as dino          # noqa: E402
import issue71_shape as sh           # noqa: E402
from sfzmaps.frame import data, decide, folds, look   # noqa: E402

log = logging.getLogger("combined")
Image.MAX_IMAGE_PIXELS = None

OUT = data.DATA_DIR / "frame-combined"
TMP = OUT / "tmp"
PAGES = OUT / "pages"
TILES = OUT / "tiles"
CROPS = OUT / "crops"
VARIANT = (1554, "last4", "conv3")
SEEDS = (0, 1, 2)
LAM_N, BW, LAM_B = 1e4, 3.0, 1e3
PICS = [("1948", s) for s in ("14", "3", "5a", "6a", "8", "2", "9", "4")]
IN_SAMPLE = {("1948", "2"), ("1948", "9")}
THIN = ("1921", "20180102094204401")
TILE_SIDE = 850
WORKERS = 3
METHODS = ("a", "b", "c")


def key(sid):
    return f"{sid[0]}__{sid[1]}"


def held():
    return list(folds.split(data.labelled_sheets(), 4)[0].held)


def all_pages():
    return held() + PICS + [THIN]


def _init_worker():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def _pool():
    import concurrent.futures
    import multiprocessing
    return concurrent.futures.ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"),
                                                  initializer=_init_worker)


# ---------------------------------------------------------------------------------------------------------------
# features for the one page missing from the cache, and the layout chances
# ---------------------------------------------------------------------------------------------------------------

def cmd_extract():
    """DINOv2 features (as `issue71_dino.cmd_extract`, cpu) for any of our pages missing from the cache."""
    missing = [s for s in all_pages() if not dino.feat_path(1554, s).exists()]
    log.info("missing features: %s", missing)
    if not missing:
        return
    from transformers import Dinov2WithRegistersModel
    m = Dinov2WithRegistersModel.from_pretrained(dino.MODEL, use_safetensors=True).eval()
    nreg = m.config.num_register_tokens
    for sid in missing:
        t0 = time.perf_counter()
        lum = data.load_luminance(data.page_path(*sid))
        h, w = lum.shape
        gh, gw = dino.grid(h, w, 1554)
        small = np.asarray(Image.fromarray(lum).resize((gw * dino.PATCH, gh * dino.PATCH), Image.BOX), np.float32)
        x = (np.repeat(small[..., None], 3, -1) - dino.MEAN) / dino.STD
        x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
        with torch.no_grad():
            hs = m(pixel_values=x, output_hidden_states=True).hidden_states[-4:]
            toks = [m.layernorm(t)[0, 1 + nreg:] for t in hs]
            f = torch.cat(toks, -1).T.reshape(-1, gh, gw)
        np.save(dino.feat_path(1554, sid), f.numpy().astype(np.float16))
        log.info("features %s/%s %dx%d %.1f s", *sid, gh, gw, time.perf_counter() - t0)


def cmd_layout():
    """Three heads on fold 0's 16 training sheets; each page's seed-averaged patch chances to TMP/layout (small).  A
    reproduction check: seed 0 alone, per-pixel cut, on the held sheets (the DINO spike: 205,036)."""
    torch.set_num_threads(4)
    f0 = folds.split(data.labelled_sheets(), 4)[0]
    (TMP / "layout").mkdir(parents=True, exist_ok=True)
    fits, info = [], {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        fit, hist = dino.train_head(*VARIANT, list(f0.train), seed=seed)
        info[f"seed{seed}"] = {"steps": len(hist), "loss": hist[-1], "train_s": time.perf_counter() - t0}
        log.info("seed %d: %d steps, loss %.4f, %.0f s", seed, len(hist), hist[-1], time.perf_counter() - t0)
        fits.append(fit)
    for sid in all_pages():
        with torch.no_grad():
            ps = [torch.softmax(f.head(f.prep(sid)), 0).numpy() for f in fits]
        np.save(TMP / "layout" / f"{key(sid)}.npy", np.mean(ps, 0).astype(np.float32))
        np.save(TMP / "layout" / f"{key(sid)}.seed0.npy", ps[0].astype(np.float32))
    repro = {}
    for sid in held():
        lum = data.load_luminance(data.page_path(*sid))
        p = upsample(np.load(TMP / "layout" / f"{key(sid)}.seed0.npy"), lum.shape)
        d = decide.decide(p, lum, 1)
        repro[key(sid)] = score(d.deleted, sid, lum)
        log.info("seed0 per-pixel %s/%s cost %.0f", *sid, repro[key(sid)]["cost"])
    info["seed0_pixel_pooled"] = sum(r["cost"] for r in repro.values())
    info["seed0_pixel"] = repro
    log.info("seed 0 per-pixel pooled %.0f (DINO spike 205,036)", info["seed0_pixel_pooled"])
    (OUT / "layout.json").write_text(json.dumps(info, indent=1))


def upsample(pp, shape):
    with torch.no_grad():
        return F.interpolate(torch.from_numpy(pp)[None], size=shape, mode="bilinear", align_corners=False)[0].numpy()


def layout_p(sid, shape):
    return upsample(np.load(TMP / "layout" / f"{key(sid)}.npy"), shape)


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------

def labelled(sid):
    return sid in set(data.labelled_sheets())


def score(deleted, sid, lum):
    """cost, frame_kept, map_cut and the cost split: edge (ink within 100 px of a class change in the label's class
    mask, the DINO spike's `near_mask`) and layout (the rest)."""
    cls = data.class_mask(data.load_label(*sid), data.load_remove(*sid), lum.shape)
    near = dino.near_mask(sid, cls)
    ink = 1.0 - lum.astype(np.float64)
    fk = (cls != 0) & ~deleted
    mc = (cls == 0) & deleted
    frame_kept, map_cut = float(ink[fk].sum()), float(ink[mc].sum())
    cost = decide.COST_FRAME * frame_kept + decide.COST_MAP * map_cut
    edge = decide.COST_FRAME * float(ink[fk & near].sum()) + decide.COST_MAP * float(ink[mc & near].sum())
    return {"cost": cost, "frame_kept": frame_kept, "map_cut": map_cut, "cost_edge": edge, "cost_layout": cost - edge}


# ---------------------------------------------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------------------------------------------

def tile(sid, lum, q, deleted, name):
    img = look.sheet_picture(lum, q, deleted, TILE_SIDE)
    if labelled(sid):
        s = img.width / lum.shape[1]
        d = ImageDraw.Draw(img)
        for poly in data.load_label(*sid) + data.load_remove(*sid):
            d.line([(x * s, y * s) for x, y in poly + poly[:1]], fill=(0, 170, 0), width=2)
    TILES.mkdir(parents=True, exist_ok=True)
    img.save(TILES / f"{key(sid)}_{name}.png")


def crop_windows(sid, shape):
    """Full-resolution windows on the 1921 thin rule: look.CLOSEUPS' two on the close-up page; on a labelled 1921
    sheet, the label's top-left corner and the middle of its left side."""
    if sid == THIN:
        return [(w, cap) for s, h, w, cap in look.CLOSEUPS if (s, h) == sid]
    if sid[0] != "1921" or not labelled(sid):
        return []
    P = np.asarray(max(data.load_label(*sid), key=len), float)
    x0, y0 = P.min(0)
    ym = (P[:, 1].min() + P[:, 1].max()) / 2
    x0, y0, ym = int(x0), int(y0), int(ym)
    return [((max(0, x0 - 60), max(0, y0 - 60), x0 + 240, y0 + 240), f"{sid[0]}/{sid[1]} label top-left"),
            ((max(0, x0 - 60), ym - 150, x0 + 240, ym + 150), f"{sid[0]}/{sid[1]} label mid-left")]


def crops(sid, lum, deleted, name):
    """1:1 crops: ink coloured by the decision (kept blue, deleted red), the kept region's edge 1 px magenta, the
    label's outline 1 px green."""
    CROPS.mkdir(parents=True, exist_ok=True)
    for k, ((x0, y0, x1, y1), _) in enumerate(crop_windows(sid, lum.shape)):
        L = lum[y0:y1, x0:x1]
        Dl = deleted[y0:y1, x0:x1]
        ink = np.clip(1.0 - L, 0, 1)[..., None]
        col = np.where(Dl[..., None], np.array([255, 0, 0], np.float32), np.array([0, 0, 255], np.float32))
        rgb = (1 - ink) * 255 + ink * col
        kept = ~Dl
        edge = kept & ~ndimage.binary_erosion(kept, border_value=1)
        edge &= ndimage.binary_dilation(Dl)
        rgb[edge] = (255, 0, 255)
        img = Image.fromarray(np.round(rgb).astype(np.uint8), "RGB")
        if labelled(sid):
            d = ImageDraw.Draw(img)
            for poly in data.load_label(*sid) + data.load_remove(*sid):
                d.line([(x - x0, y - y0) for x, y in poly + poly[:1]], fill=(0, 170, 0), width=1)
        img.save(CROPS / f"{key(sid)}_{name}_{k}.png")


# ---------------------------------------------------------------------------------------------------------------
# (b): the shape decision, unchanged, keeping its winning start
# ---------------------------------------------------------------------------------------------------------------

def shape_b(p, lum):
    timing = {}
    D, ink = sh.cost_map(p, lum)
    rects = sh.frame_search(D, (LAM_N,), timing)
    starts = {"plain": rects["plain"][0]}
    if rects[LAM_N][0] is not rects["plain"][0]:
        starts["notch-aware"] = rects[LAM_N][0]
    best = None
    for name, rect in starts.items():
        r = sh.pipeline(rect, p, D, ink, LAM_N, BW, LAM_B, timing)
        if best is None or r[4] < best[2][4]:
            best = (name, rect, r)
    name, rect, (K, removed, fr, kept, J) = best
    return K, removed, fr, kept, J, name, rect


def fr_dump(fr):
    return {"h": fr.h, "w": fr.w, "phi": fr.phi, "beta": fr.beta, "tilt": fr.tilt, "sag": fr.sag,
            "arc": [[a, b, r] for (a, b), r in fr.arc.items()], "notches": [dict(n.__dict__) for n in fr.notches]}


def fr_load(d):
    return sh.Frame(d["h"], d["w"], d["phi"], dict(d["beta"]), dict(d["tilt"]), dict(d["sag"]),
                    {(a, b): r for a, b, r in d["arc"]}, [sh.Notch(**n) for n in d["notches"]])


def blk_dump(blocks):
    return [{"kind": b.kind, "cx": b.cx, "cy": b.cy, "theta": b.theta, "p": list(b.p), "gain": b.gain} for b in blocks]


def ab_page(sid):
    """(a) and (b) on one page: scores (labelled pages), timings, (b)'s frame and start, tiles and crops."""
    path = PAGES / f"{key(sid)}_ab.json"
    if path.exists():
        return json.loads(path.read_text())
    lum = data.load_luminance(data.page_path(*sid))
    assert data.map_counts(data.PAGE_KINDS).get(sid, 1) == 1, sid
    t0 = time.perf_counter()
    p = layout_p(sid, lum.shape)
    t_up = time.perf_counter() - t0
    t0 = time.perf_counter()
    da = decide.decide(p, lum, 1).deleted
    t_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    K, removed, fr, kept, J, start, rect = shape_b(p, lum)
    db = ~K | removed
    t_b = time.perf_counter() - t0
    out = {"sid": list(sid), "seconds": {"upsample": t_up, "a": t_a, "b": t_b}, "start": start,
           "start_rect": {"phi": rect.phi, "beta": rect.beta}, "frame_b": fr_dump(fr), "blocks_b": blk_dump(kept),
           "J_b": J}
    if labelled(sid):
        out["a"] = score(da, sid, lum)
        out["b"] = score(db, sid, lum)
        P = np.asarray(max(data.load_label(*sid), key=len), float)
        out["side_err_b"] = side_errors(fr, P, *lum.shape)
    q = p[1] + p[2]
    for name, d in (("a", da), ("b", db)):
        tile(sid, lum, q, d, name)
        crops(sid, lum, d, name)
    PAGES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    log.info("ab %s/%s a %.1f s b %.1f s %s", *sid, t_a, t_b,
             f"cost a {out['a']['cost']:.0f} b {out['b']['cost']:.0f}" if "a" in out else "")
    return out


# ---------------------------------------------------------------------------------------------------------------
# W: (b)'s side-position errors against the label's side lines
# ---------------------------------------------------------------------------------------------------------------

def side_errors(fr, P, h, w):
    """Per side: the fitted boundary (tilt and bow included) against the label's side line (`issue71_shape.
    label_sides`, the middle 70%), signed px (+ = the fit lies inward of the label), over the middle 70% of the
    label's side; both clipped to the page (a side beyond the page edge cuts nothing there)."""
    sides = sh.label_sides(P, (h, w))
    cv = fr.canvas
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    out = {}
    for side in sh.SIDES:
        lo, hi = fr.ends(side)
        al = np.arange(lo, hi, 2.0)
        e = fr.bound(side, al)
        u, v = {"top": (al, e), "bottom": (al, -e), "left": (e, al), "right": (-e, al)}[side]
        x, y = cv.xy(u, v)
        a, b = sides[side]
        if side in ("top", "bottom"):
            sel = (x > x0 + 0.15 * (x1 - x0)) & (x < x1 - 0.15 * (x1 - x0))
            d = (np.clip(y, 0, h) - np.clip(a * x + b, 0, h)) / math.hypot(1, a)
            d = d if side == "top" else -d
        else:
            sel = (y > y0 + 0.15 * (y1 - y0)) & (y < y1 - 0.15 * (y1 - y0))
            d = (np.clip(x, 0, w) - np.clip(a * y + b, 0, w)) / math.hypot(1, a)
            d = d if side == "left" else -d
        d = d[sel]
        out[side] = {"max_abs": float(np.abs(d).max()) if len(d) else 0.0,
                     "min": float(d.min()) if len(d) else 0.0, "max": float(d.max()) if len(d) else 0.0}
    return out


def block_bounds():
    """The labelled `remove` polygons (all 21 labels): rectangles' largest bounding-rectangle sides (minimum-area
    rectangle) and round ones' largest axes, each x 1.5.  Round: aspect <= 1.2 and area / rectangle < 0.88 (the
    octagon badges and roses, fill ~0.83)."""
    rows, L, S, A, B = [], 0.0, 0.0, 0.0, 0.0
    for sid in data.labelled_sheets():
        for poly in data.load_remove(*sid):
            P = np.asarray(poly, np.float32)
            (_, _), (a, b), _ = cv2.minAreaRect(P)
            lo, hi = min(a, b), max(a, b)
            fill = cv2.contourArea(P) / max(a * b, 1.0)
            rnd = hi / max(lo, 1.0) <= 1.2 and fill < 0.88
            rows.append({"sid": list(sid), "long": hi, "short": lo, "fill": fill, "round": rnd})
            L, S = max(L, hi), max(S, lo)
            if rnd:
                A, B = max(A, hi), max(B, lo)
    return {"rect_long": 1.5 * L, "rect_short": 1.5 * S, "ell_major": 1.5 * A, "ell_minor": 1.5 * B,
            "measured": {"rect_long": L, "rect_short": S, "ell_major": A, "ell_minor": B}, "polygons": rows}


def cmd_wb():
    errs = {}
    for sid in held():
        r = json.loads((PAGES / f"{key(sid)}_ab.json").read_text())
        errs[key(sid)] = r["side_err_b"]
    worst = max(e["max_abs"] for s in errs.values() for e in s.values())
    W = 1.5 * worst
    out = {"W": W, "worst_side_err": worst, "side_err_b": errs, "bounds": block_bounds()}
    (OUT / "wb.json").write_text(json.dumps(out, indent=1, default=float))
    log.info("W %.1f (worst %.1f); bounds %s", W, worst,
             {k: round(v) for k, v in out["bounds"].items() if k != "polygons" and k != "measured"})


# ---------------------------------------------------------------------------------------------------------------
# (c): bounded blocks, blended refinements
# ---------------------------------------------------------------------------------------------------------------

def _rect_on_b(DKc, x0, y0, cx, cy, theta, init, lu, lv):
    """`issue71_shape._rect_on` with the extents bounded: u-extent <= lu, v-extent <= lv px."""
    hc, wc = DKc.shape
    half = math.hypot(hc, wc) / 2 + 4
    cv = sh.Canvas(theta, cx, cy, half + abs(cx - (x0 + wc / 2)), half + abs(cy - (y0 + hc / 2)))
    A = cv.warp(DKc, 1, 1, x_off=x0, y_off=y0)
    S = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
    S[1:, 1:] = A.cumsum(0, dtype=np.float64).cumsum(1)
    nj, ni = A.shape
    lu, lv = max(1, int(lu)), max(1, int(lv))
    il = int(np.clip(math.floor(init[0] + cv.U), 0, ni - 1)); ir = int(np.clip(math.ceil(init[1] + cv.U), il + 1, ni))
    jt = int(np.clip(math.floor(init[2] + cv.V), 0, nj - 1)); jb = int(np.clip(math.ceil(init[3] + cv.V), jt + 1, nj))
    if ir - il > lu:
        m = (il + ir) // 2
        il = max(0, m - lu // 2); ir = min(ni, il + lu)
    if jb - jt > lv:
        m = (jt + jb) // 2
        jt = max(0, m - lv // 2); jb = min(nj, jt + lv)
    for _ in range(20):
        old = (il, ir, jt, jb)
        f = S[:, ir] - S[:, il]
        lo = max(0, jb - lv); jt = lo + int(np.argmin(f[lo:jb]))
        hi = min(nj, jt + lv); jb = jt + 1 + int(np.argmax(f[jt + 1:hi + 1]))
        g = S[jb] - S[jt]
        lo = max(0, ir - lu); il = lo + int(np.argmin(g[lo:ir]))
        hi = min(ni, il + lu); ir = il + 1 + int(np.argmax(g[il + 1:hi + 1]))
        if (il, ir, jt, jb) == old:
            break
    gain = S[jb, ir] - S[jt, ir] - S[jb, il] + S[jt, il]
    return float(gain), sh.Block("rect", cx, cy, float(theta), (il - cv.U, ir - cv.U, jt - cv.V, jb - cv.V), float(gain))


def _ellipse_on_b(DKc, x0, y0, cx, cy, a, b, th, amax, bmax):
    """`issue71_shape._ellipse_on` with the axes bounded: the longer full axis <= amax, the shorter <= bmax."""
    hc, wc = DKc.shape
    f = max(1, int(math.ceil(max(hc, wc) / 300)))
    a, b = max(a, 2.0), max(b, 2.0)
    if a >= b:
        a, b = min(a, amax / 2), min(b, bmax / 2)
    else:
        a, b = min(a, bmax / 2), min(b, amax / 2)
    p = np.array([cx, cy, a, b, th])

    def ok(q):
        lo, hi = sorted((q[2], q[3]))
        return lo >= 2 and 2 * hi <= amax and 2 * lo <= bmax

    def run(arr, f, p, steps, minsteps):
        Y, X = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        X = x0 + f * X + f / 2.0
        Y = y0 + f * Y + f / 2.0
        cur = sh._ellipse_gain(arr, X, Y, *p)
        steps = np.array(steps, float)
        while np.any(steps >= minsteps):
            improved = False
            for k in range(5):
                if steps[k] < minsteps[k]:
                    continue
                for sgn in (1, -1):
                    q = p.copy()
                    q[k] += sgn * steps[k]
                    if not ok(q):
                        continue
                    g = sh._ellipse_gain(arr, X, Y, *q)
                    if g > cur:
                        cur, p, improved = g, q, True
            if not improved:
                steps = steps / 2
        return cur, p

    if f > 1:
        _, p = run(sh.pool(DKc, f), f, p, [8 * f, 8 * f, 8 * f, 8 * f, 8.0], [f, f, f, f, 1.0])
        st = [2 * f] * 4 + [1.0]
    else:
        st = [max(2.0, p[2] / 8)] * 2 + [max(2.0, p[2] / 8), max(2.0, p[3] / 8), 8.0]
    g, p = run(DKc, 1, p, st, [1, 1, 1, 1, 0.25])
    return g, sh.Block("ellipse", float(p[0]), float(p[1]), float(p[4]), (float(p[2]), float(p[3])), g)


def block_shapes_b(p, D, K, phi, bounds):
    """`issue71_shape.block_shapes` with bounded rectangles (both orientations) and bounded ellipses."""
    level = decide.cut_level()
    h, w = D.shape
    L, S_, A, B = bounds["rect_long"], bounds["rect_short"], bounds["ell_major"], bounds["ell_minor"]
    DK = np.where(K, D, 0).astype(np.float32)
    lab, n = ndimage.label((p[2] > level) & K, structure=sh._EIGHT)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or sizes[k] < sh.BLOCK_MIN_PX:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        bh, bw_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        mg = max(24, int(0.3 * max(bh, bw_)))
        y0, y1 = max(0, sl[0].start - mg), min(h, sl[0].stop + mg)
        x0, x1 = max(0, sl[1].start - mg), min(w, sl[1].stop + mg)
        DKc = DK[y0:y1, x0:x1]
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov)
        thm = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        best = None
        angles = sorted({round(sh._wrap90(thm) + d, 3) for d in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)} |
                        {round(phi + d, 3) for d in (-0.5, -0.25, 0, 0.25, 0.5)})
        for th in angles:
            t = math.radians(th)
            c, s = math.cos(t), math.sin(t)
            pu = c * (xs + 0.5 - cx) + s * (ys + 0.5 - cy)
            pv = -s * (xs + 0.5 - cx) + c * (ys + 0.5 - cy)
            init = (pu.min(), pu.max() + 1, pv.min(), pv.max() + 1)
            for lu, lv in ((L, S_), (S_, L)):
                g, blk = _rect_on_b(DKc, x0, y0, cx, cy, th, init, lu, lv)
                if best is None or g > best.gain:
                    best = blk
        a, b = 2 * math.sqrt(max(ev[1], 1.0)), 2 * math.sqrt(max(ev[0], 1.0))
        g, blk = _ellipse_on_b(DKc, x0, y0, cx, cy, a, b, thm, A, B)
        if g > best.gain:
            best = blk
        out.append(best)
    out.sort(key=lambda b: -b.gain)
    return out


def best_notch_c(DKl, DKb, IK, fr, bw, win):
    """`issue71_shape.best_notch` with the coarse search on the layout's D (DKl) and the full-resolution refinement
    (angle +-0.25 deg, ends and depth +-win px) on the blended D (DKb)."""
    h, w = DKl.shape
    s = 8
    DKp, IKp = sh.pool(DKl, s), sh.pool(IK, s)
    n_ang = int(math.ceil(2 * sh.NOTCH_DEG / 0.25)) + 1
    best = (-np.inf,)
    for dpsi in np.linspace(-sh.NOTCH_DEG, sh.NOTCH_DEG, n_ang):
        psi = fr.phi + float(dpsi)
        cv = sh.Canvas.page(psi, h, w)
        Dw, Iw = cv.warp(DKp, s, s), cv.warp(IKp, s, s)
        for side in sh.SIDES:
            E, _ = cv.EA(side)
            half = (h if side in ("top", "bottom") else w) / 2.0
            ne = int((E - half + sh.NOTCH_REACH * 2 * half) // s)
            r = sh.kadane(sh.side_view(Dw, side)[:ne], sh.side_view(Iw, side)[:ne], bw, s, 1)
            if r[0] > best[0]:
                best = (r[0], side, psi, r[1], r[2], r[3])
    if not np.isfinite(best[0]):
        return None
    _, side, psi_c, a0c, a1c, dc = best
    fine = (-np.inf,)
    for psi in psi_c + np.arange(-0.25, 0.2501, 0.05):
        cv = sh.Canvas.page(psi, h, w)
        E, A = cv.EA(side)
        n_e, n_a = (cv.cells(1) if side in ("top", "bottom") else cv.cells(1)[::-1])
        e1 = min(n_e, s * dc + win)
        al0, al1 = max(0, s * a0c - win), min(n_a, s * a1c + win)
        Dv = cv.side_window(side, DKb, 1, 1, 0, e1, al0, al1, binned=True)
        Iv = cv.side_window(side, IK, 1, 1, 0, e1, al0, al1, binned=True)
        r = sh.kadane(Dv, Iv, bw, 1, sh.STRIP, d_lo=max(1, s * dc - win), d_hi=e1)
        if r[0] > fine[0]:
            fine = (r[0], psi, r[1] + al0, r[2] + al0, r[3], E, A)
    if not np.isfinite(fine[0]):
        return None
    val, psi, a0, a1, d, E, A = fine
    return sh.Notch(side, float(psi), float(a0 - A), float(a1 - A), float(d - E), float(val))


def greedy_notches_c(Dl, Db, ink, K0, fr, bw, lam, win):
    K = K0.copy()
    out = []
    for _ in range(sh.NOTCH_CAP):
        IK = ink * K
        n = best_notch_c(Dl * K, Db * K, IK, fr, bw, win)
        if n is None:
            break
        m = sh.notch_mask(n, *K.shape) & K
        if not m.any():
            break
        n.ink = sh._side_ink(n, IK, K.shape)
        n.gain = float(Db[m].sum(dtype=np.float64)) + bw * n.ink
        if n.gain <= lam:
            break
        out.append(n)
        K &= ~m
    return out


def rect_given_w(centre, Dz, W):
    """`issue71_shape.rect_given` about (b)'s fitted rectangle with the offsets +-W (cells of 2), then +-0.02 deg and
    +-4 px at 1 px."""
    half = int(math.ceil(W / 2.0)) * 2
    pools = {2: sh.pool(Dz, 2)}
    b2 = sh._refine_rect(Dz, pools, centre.phi + np.arange(-0.5, 0.5 + 1e-9, 0.05), 2, half, centre.beta)
    b1 = sh._refine_rect(Dz, pools, b2[1] + np.arange(-0.02, 0.02 + 1e-9, 0.01), 1, 4, b2[2])
    return sh.Frame(centre.h, centre.w, b1[1], {k: float(v) for k, v in b1[2].items()})


def band_mask(fr_b, W):
    """Px within W of any of (b)'s fitted sides, fillets or notch sides: the boundary of (b)'s frame region (sides,
    arcs, notches; no blocks) inside the page, widened by W."""
    K = fr_b.kept()
    edge = np.zeros(K.shape, bool)
    dv = K[:-1] != K[1:]
    edge[:-1] |= dv
    edge[1:] |= dv
    dh = K[:, :-1] != K[:, 1:]
    edge[:, :-1] |= dh
    edge[:, 1:] |= dh
    if not edge.any():
        return np.zeros(K.shape, bool)
    dist = cv2.distanceTransform(np.where(edge, 0, 1).astype(np.uint8), cv2.DIST_L2, 5)
    return dist <= W


def edge_path(sid):
    c = sh.CACHE / f"{key(sid)}.npy"
    return c if c.exists() else TMP / "edge" / f"{key(sid)}.npy"


def c_page(sid, W, bounds, tag="c"):
    path = PAGES / f"{key(sid)}_{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    ab = json.loads((PAGES / f"{key(sid)}_ab.json").read_text())
    lum = data.load_luminance(data.page_path(*sid))
    h, w = lum.shape
    pl = layout_p(sid, lum.shape)
    t0 = time.perf_counter()
    pe = np.load(edge_path(sid)).astype(np.float32)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    fr_b = fr_load(ab["frame_b"])
    band = band_mask(fr_b, W)
    Dl, ink = sh.cost_map(pl, lum)
    De, _ = sh.cost_map(pe, lum)
    Db = np.where(band, De, Dl).astype(np.float32)
    del De
    t_band = time.perf_counter() - t0
    t0 = time.perf_counter()
    rect = sh.Frame(h, w, ab["start_rect"]["phi"], dict(ab["start_rect"]["beta"]))
    K0 = rect.kept()
    shapes = block_shapes_b(pl, Dl, K0, rect.phi, bounds)
    B0, _ = sh.accept_blocks(shapes, Dl, K0, LAM_B)
    Dzl = np.where(B0, 0, Dl).astype(np.float32)
    Dzb = np.where(B0, 0, Db).astype(np.float32)
    inkz = np.where(B0, 0, ink).astype(np.float32)
    t1 = time.perf_counter()
    notches = greedy_notches_c(Dzl, Dzb, inkz, K0, rect, BW, LAM_N, max(16, int(math.ceil(W))))
    del Dzl
    t2 = time.perf_counter()
    Nm = sh.notches_mask(notches, h, w)
    Dzzb = np.where(Nm, 0, Dzb).astype(np.float32)
    centre = sh.Frame(h, w, fr_b.phi, dict(fr_b.beta))
    fr = sh.fit_sides_arcs(rect_given_w(centre, Dzzb, W), Dzzb, notches)
    K = fr.kept()
    removed, kept = sh.accept_blocks(shapes, Db, K, LAM_B)
    removed &= K
    t3 = time.perf_counter()
    dc = ~K | removed
    out = {"sid": list(sid), "seconds": {"load_edge": t_load, "band": t_band, "blocks": t1 - t0, "notches": t2 - t1,
                                         "refine": t3 - t2, "c": t3 - t0 + t_band},
           "frame_c": fr_dump(fr), "blocks_c": blk_dump(kept), "band_px": int(band.sum()),
           "J_c": sh.objective(Db, K, removed, notches, kept, LAM_N, BW, LAM_B)}
    if labelled(sid):
        out["c"] = score(dc, sid, lum)
        P = np.asarray(max(data.load_label(*sid), key=len), float)
        out["side_err_c"] = side_errors(fr, P, h, w)
    q = np.where(band, pe[1] + pe[2], pl[1] + pl[2])
    del pe, pl
    tile(sid, lum, q, dc, tag)
    crops(sid, lum, dc, tag)
    path.write_text(json.dumps(out, indent=1, default=float))
    log.info("c %s/%s %.1f s %s", *sid, out["seconds"]["c"], f"cost {out['c']['cost']:.0f}" if "c" in out else "")
    return out


# ---------------------------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------------------------

def edge_chances(sids):
    """The full-resolution fold-0 net's chances (FOLD, LEVELS, WIDTH 16, 5, 32, BUCKET 256, T 2.086) for pages not in
    the shape spike's cache, one GPU turn, float16 to TMP/edge (deleted by `clean`)."""
    from sfzmaps import gpu
    todo = [s for s in sids if not edge_path(s).exists()]
    if not todo:
        return {}
    model = sh._setup_model()
    T, _ = sh._held()
    net = sh._net()
    (TMP / "edge").mkdir(parents=True, exist_ok=True)
    secs = {}
    with gpu.gpu_turn(True, sh.DEV):
        for sid in todo:
            lum = data.load_luminance(data.page_path(*sid))
            t0 = time.perf_counter()
            p = model.prob([net], lum, sh.DEV, T)
            secs[key(sid)] = time.perf_counter() - t0
            tmp = TMP / "edge" / f"{key(sid)}.tmp.npy"
            np.save(tmp, p.astype(np.float16))
            tmp.rename(TMP / "edge" / f"{key(sid)}.npy")
            log.info("edge chances %s/%s %.1f s (T %.3f)", *sid, secs[key(sid)], T)
    return secs


def cmd_ab():
    PAGES.mkdir(parents=True, exist_ok=True)
    with _pool() as ex:
        futs = [ex.submit(ab_page, sid) for sid in all_pages()]
        torch.set_num_threads(1)
        secs = edge_chances(all_pages())
        if secs:
            old = json.loads((OUT / "edge_seconds.json").read_text()) if (OUT / "edge_seconds.json").exists() else {}
            (OUT / "edge_seconds.json").write_text(json.dumps({**old, **secs}, indent=1))
        for f in futs:
            f.result()


def cmd_c():
    wb = json.loads((OUT / "wb.json").read_text())
    with _pool() as ex:
        for f in [ex.submit(c_page, sid, wb["W"], wb["bounds"]) for sid in all_pages()]:
            f.result()


def cmd_c_narrow(W="100"):
    """A diagnostic arm, not the design's W: (c) on the held sheets with the band's half-width W (default 100 px,
    the edge/layout split distance), to see how much of (c)'s result is the band's width."""
    wb = json.loads((OUT / "wb.json").read_text())
    with _pool() as ex:
        rows = [f.result() for f in [ex.submit(c_page, sid, float(W), wb["bounds"], f"c{W}") for sid in held()]]
    for r in rows:
        print(r["sid"], {k: round(v) for k, v in r["c"].items()}, {s: round(v["max_abs"]) for s, v in r["side_err_c"].items()})
    print("pooled", {k: round(sum(r["c"][k] for r in rows)) for k in rows[0]["c"]})


def _grid_picture(rows, path, caption_fmt):
    font = look._font()
    W = max(sum(t.width for t in ts) + (len(ts) + 1) * look.GAP_PX for _, ts, _ in rows)
    H = sum(max(t.height for t in ts) + look.CAPTION_PX + look.GAP_PX for _, ts, _ in rows) + look.GAP_PX
    canvas = Image.new("RGB", (W, H), look.BACKGROUND_RGB)
    d = ImageDraw.Draw(canvas)
    y = look.GAP_PX
    for title, ts, _ in rows:
        d.text((look.GAP_PX, y + 2), caption_fmt(title), fill=(0, 0, 0), font=font)
        y += look.CAPTION_PX
        x = look.GAP_PX
        for t in ts:
            canvas.paste(t, (x, y))
            x += t.width + look.GAP_PX
        y += max(t.height for t in ts) + look.GAP_PX
    canvas.save(path)
    log.info("%s %dx%d", path, W, H)


def cmd_pictures():
    def note(sid):
        if sid in IN_SAMPLE:
            return " (labelled, IN SAMPLE: in fold 0's training set; label in green)"
        if labelled(sid):
            return " (held out; label in green)"
        return " (unlabelled)"

    for name, sids in (("compare.png", PICS), ("compare_held.png", held())):
        rows = [(f"{sid[0]}/{sid[1]}{note(sid)}: (a) per-pixel | (b) shape | (c) shape + edge band, bounded blocks",
                 [Image.open(TILES / f"{key(sid)}_{m}.png") for m in METHODS], sid) for sid in sids]
        _grid_picture(rows, OUT / name, lambda t: t)
    rows = []
    for sid in [THIN] + [s for s in held() if s[0] == "1921"]:
        for k, (_, cap) in enumerate(crop_windows(sid, None)):
            rows.append((f"{cap}{note(sid)}: (a) | (b) | (c), 1:1; kept ink blue, cut ink red, cut edge magenta",
                         [Image.open(CROPS / f"{key(sid)}_{m}_{k}.png") for m in METHODS], sid))
    _grid_picture(rows, OUT / "thin_rule.png", lambda t: t)


def cmd_table():
    ours = json.loads((dino.OUT / "ours.json").read_text())
    wb = json.loads((OUT / "wb.json").read_text())
    rows = {}
    for sid in held() + [s for s in PICS if labelled(s)]:
        ab = json.loads((PAGES / f"{key(sid)}_ab.json").read_text())
        c = json.loads((PAGES / f"{key(sid)}_c.json").read_text())
        rows[sid] = {"a": ab["a"], "b": ab["b"], "c": c["c"], "region": ours.get(f"{sid[0]}/{sid[1]}")}
    fmt = lambda v: f"{v:,.0f}"
    print("| sheet | method | cost | frame_kept | map_cut | edge | layout |\n|---|---|---:|---:|---:|---:|---:|")
    for sid in held():
        for m in ("region", "a", "b", "c"):
            r = rows[sid][m]
            print(f"| {sid[0]}/{sid[1]} | {m} | {fmt(r['cost'])} | {fmt(r['frame_kept'])} | {fmt(r['map_cut'])} | "
                  f"{fmt(r['cost_edge'])} | {fmt(r['cost_layout'])} |")
    print("\n| pooled (6 held) | cost | frame_kept | map_cut | edge | layout |\n|---|---:|---:|---:|---:|---:|")
    for m in ("region", "a", "b", "c"):
        t = {k: sum(rows[s][m][k] for s in held()) for k in ("cost", "frame_kept", "map_cut", "cost_edge",
                                                            "cost_layout")}
        print(f"| {m} | {fmt(t['cost'])} | {fmt(t['frame_kept'])} | {fmt(t['map_cut'])} | {fmt(t['cost_edge'])} | "
              f"{fmt(t['cost_layout'])} |")
    print("\nin-sample (fold 0 training sheets):")
    for sid in [s for s in PICS if labelled(s)]:
        for m in ("a", "b", "c"):
            r = rows[sid][m]
            print(f"| {sid[0]}/{sid[1]} | {m} | {fmt(r['cost'])} | {fmt(r['frame_kept'])} | {fmt(r['map_cut'])} | "
                  f"{fmt(r['cost_edge'])} | {fmt(r['cost_layout'])} |")
    print("\nseconds per page:")
    edge_s = json.loads((OUT / "edge_seconds.json").read_text()) if (OUT / "edge_seconds.json").exists() else {}
    print("| page | Mpx | upsample | a | b | c | edge net |\n|---|---:|---:|---:|---:|---:|---:|")
    for sid in all_pages():
        ab = json.loads((PAGES / f"{key(sid)}_ab.json").read_text())
        c = json.loads((PAGES / f"{key(sid)}_c.json").read_text())
        mp = ab["frame_b"]["h"] * ab["frame_b"]["w"] / 1e6
        es = edge_s.get(key(sid))
        print(f"| {sid[0]}/{sid[1]} | {mp:.0f} | {ab['seconds']['upsample']:.1f} | {ab['seconds']['a']:.1f} | "
              f"{ab['seconds']['b']:.1f} | {c['seconds']['c']:.1f} | {'cached' if es is None else f'{es:.1f}'} |")
    print("\nside errors of (b) on held (max abs px):")
    for k, s in wb["side_err_b"].items():
        print(k, {sd: round(v["max_abs"], 1) for sd, v in s.items()})
    print("side errors of (c) on held (max abs px):")
    for sid in held():
        c = json.loads((PAGES / f"{key(sid)}_c.json").read_text())
        print(key(sid), {sd: round(v["max_abs"], 1) for sd, v in c["side_err_c"].items()})
    print("W", wb["W"], "bounds", {k: v for k, v in wb["bounds"].items() if k != "polygons"})
    print("\nframes / blocks / notches:")
    for sid in all_pages():
        ab = json.loads((PAGES / f"{key(sid)}_ab.json").read_text())
        c = json.loads((PAGES / f"{key(sid)}_c.json").read_text())
        for m, fr, bl in (("b", ab["frame_b"], ab["blocks_b"]), ("c", c["frame_c"], c["blocks_c"])):
            print(f"{key(sid)} {m} start={ab['start']} phi={fr['phi']:+.2f} notches={len(fr['notches'])} "
                  f"blocks={[(b['kind'], [round(x) for x in b['p']]) for b in bl]}")


def cmd_clean():
    shutil.rmtree(TMP, ignore_errors=True)
    log.info("removed %s", TMP)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"extract": cmd_extract, "layout": cmd_layout, "ab": cmd_ab, "wb": cmd_wb, "c": cmd_c, "pictures": cmd_pictures,
     "table": cmd_table, "clean": cmd_clean, "c_narrow": cmd_c_narrow}[sys.argv[1]](*sys.argv[2:])
