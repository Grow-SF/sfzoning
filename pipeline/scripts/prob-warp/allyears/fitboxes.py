"""Steps 3 and 4 on one window: fit the letters inside each of Terra's text boxes, then erase them.

The words come from records we trust: block numbers from the 2000 lot records (the block under
the box), codes from the code check (nearest label), street names from the street centreline
through the box (spelled as the map spells it); otherwise Terra's own reading.  The box fixes
the size and direction, so the search is small: cap height, slant, pen width, a few px of shift,
then each letter on its own.

usage: fitboxes.py SERIES SHEET x0,y0,x1,y1 OUTDIR
"""
import difflib
import faulthandler
faulthandler.enable(all_threads=True)      # on a crash, print every thread's Python stack (debugging the pool)
import json
import math
import os
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import shapely
import shapely.ops

sys.path.insert(0, str(Path(__file__).parent))
import textfit  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.districts import read  # noqa: E402
from sfzmaps.transform import SheetTransform  # noqa: E402

SP = Path(__file__).parent
series, sheet, box, outdir = sys.argv[1:5]
x0, y0, x1, y1 = map(int, box.split(","))
O = Path(outdir)
d = config.WORK_DIR / series / sheet
M = 80                       # work on the window plus a margin, so edge labels can be fitted
WH, WW = y1 - y0, x1 - x0
x0, y0, x1, y1 = x0 - M, y0 - M, x1 + M, y1 + M
def _load(p):
    return cv2.copyMakeBorder(cv2.imread(p, cv2.IMREAD_GRAYSCALE), M, M, M, M,
                              cv2.BORDER_CONSTANT, value=255)[y0 + M:y1 + M, x0 + M:x1 + M]   # paper past the edge


# the letters are fitted (and scored) on the scan as printed; they're erased on the scan with the block
# lines already deleted (step 4a, PAGE), so a letter as heavy as a block line isn't taken for line
gray0 = _load(str(d / "page.png"))
gray = _load(os.environ.get("PAGE") or str(d / "page.png"))
H, W = gray.shape
# the legend and title below the map frame's bottom rule are not map: painted white before any step looks
# at the sheet (frame_bottom.json: the rule's top edge, page px)
SF = Path(os.environ.get("SHEETFILES") or Path(__file__).parent)   # this sheet's walls.json, frame_bottom.json
_fb0 = json.load(open(SF / "frame_bottom.json"))
_yy, _xx = np.mgrid[0:H, 0:W]
_below = (_yy + y0) >= _fb0["k"] * (_xx + x0) + _fb0["b"]
gray0[_below] = 255
gray[_below] = 255
del _yy, _xx
ink0 = (gray0 < 128).astype(np.uint8)
dist = cv2.distanceTransform(1 - (gray < 128).astype(np.uint8), cv2.DIST_L2, 3)   # scored on the image the fit is on


def crop(img):
    return img[M:M + WH, M:M + WW]
ink = (gray < 128).astype(np.uint8)          # erase-side ink (block lines gone)
inkf = ink.astype(np.float32)                # the fit is found on the scan without block lines (none to pull it)
inner = cv2.distanceTransform(ink0, cv2.DIST_L2, 3)
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
# the sheet's local page distortion: the non-conformal part of page px -> ground at the window's
# centre (the paper's own drawing is taken as undistorted), applied to the drawn letters
cxw, cyw = (x0 + x1) / 2, (y0 + y1) / 2
gw = t.to_world(np.array([[cxw, cyw], [cxw + 50, cyw], [cxw, cyw + 50]], float))
J = np.c_[(gw[1] - gw[0]) / 50, -(gw[2] - gw[0]) / 50]          # ground per page px, y flipped
Uw, s, Vt = np.linalg.svd(J)
textfit.SKEW = Vt.T @ np.diag(math.sqrt(s[0] * s[1]) / s) @ Vt   # undo stretch/shear of the ground
labels = json.load(open(O / "textboxes_gpt-6.1-terra.json"))
for k in labels:
    k["box"] = [v + M for v in k["box"]]
lots = read.load_parcels(year=read.series_year(series))
streets = gpd.read_parquet(config.REF_DIR / "streets.parquet")
sid = sheet.rsplit("-", 1)[-1]
_allcodes = json.load(open(os.environ.get("CODES") or (SP / "codes_final.json")))
codes = [r for r in _allcodes if r["sid"] == sid and r.get("final")]
VALID = {r["final"] for r in _allcodes if r.get("final")}            # every code the map prints


def closest(cands, said):
    return max(cands, key=lambda c: difflib.SequenceMatcher(None, c.upper(), said.upper()).ratio())


OWNER_READS = {("2000-digital-zoning-p07", 2267, 1559, 2317, 1579): "RM-4"}   # read by the owner on the scan (Terra and the code check: RM-1)


def known_text(k):
    """(text to fit, where it came from)."""
    a, b, c, e = k["box"]
    kb = (sheet, *(int(v) - M for v in k["box"]))
    if kb in OWNER_READS:
        r_ = OWNER_READS[kb]
        return [(r_, "owner")] + ([(r_.replace("-", ""), "reading without its dash")] if "-" in r_ else [])
    cx, cy = (a + c) / 2 + x0, (b + e) / 2 + y0
    said = k["text"]
    if k["kind"] == "block number":
        w = t.to_world(np.array([[cx, cy]]))[0]
        hit = sorted(set(lots[lots.intersects(shapely.Point(*w).buffer(10))].block_num.astype(str)))
        if hit:
            bn = closest(hit, said)
            # the lot records write some block numbers with a leading zero ("0270") the map doesn't print
            # ("270"): the reading without it is tried too, and the one that fits the print better wins
            nz = bn.lstrip("0")
            return [(bn, "lot records")] + ([(nz, "lot records without the leading zero")] if nz and nz != bn else [])
    if k["kind"] == "zoning code" and codes:
        # every checked code near the box (their positions are from the half-resolution read),
        # and Terra's own reading: the one whose letters fit the print best wins
        near = {r["final"] for r in codes if (r["x"] - cx) ** 2 + (r["y"] - cy) ** 2 < 90 ** 2}
        # Terra's reading; confirmed when a checked code nearby says the same, else flagged
        # (the letter fit can't tell RH-2 from RH-3 reliably, so it doesn't choose)
        if said in VALID or not said:
            src_ = "code check" if said in near else "Terra, not in code check"
            return [(said, src_)] + ([(said.replace("-", ""), "reading without its dash")] if said and "-" in said else [])
        # Terra read part of a code ("RH" for RH-3): complete it from the checked codes nearby, else
        # from every code that starts with it; the letters that fit the print best win (unconfirmed)
        comp = sorted(c_ for c_ in near if c_.startswith(said)) or sorted(c_ for c_ in VALID if c_.startswith(said))
        if comp:
            return [(c_, "Terra, completed") for c_ in comp]
        return [(said, "Terra, not in code check")]
    if k["kind"] == "street name" and not said.rstrip(".").upper() in ("ST", "AVE", "BLVD", "WAY"):
        corners = t.to_world(np.array([[a + x0, b + y0], [c + x0, b + y0], [c + x0, e + y0], [a + x0, e + y0]], float))
        poly = shapely.Polygon(corners).buffer(15)
        names = [n for s in streets[streets.intersects(poly)].street.dropna().unique() for n in textfit.street_spellings(s)]
        if names:
            return closest(names, said), "street data"
    return said, "Terra"


# text height of each kind of label, measured on the ones printed level (their box's short side)
KIND_H = {}
for _kind in {k_["kind"] for k_ in labels}:
    _hs = [min(k_["box"][2] - k_["box"][0], k_["box"][3] - k_["box"][1]) for k_ in labels
           if k_["kind"] == _kind and k_.get("direction") != "slanted"]
    if _hs:
        KIND_H[_kind] = float(np.median(_hs))


import fitword  # noqa: E402


# the walls, for the score: the 2000 block outlines (lots merged by block, from the lot records), drawn at
# the printed block line's width; each wall pixel keeps its line's direction (deg, image coords)
def _known_lines():
    """Wall segments (x0, y0, x1, y1, this run's px): the 2000 block outlines with each side lined up with
    the print on its own (sidealign.py -> walls.json, page px)."""
    Wl = np.asarray(json.load(open(SF / "walls.json")), float)
    return Wl - [x0, y0, x0, y0]


LINES = _known_lines()
# the walls as a mask for the erase: each lined-up block-outline segment drawn 5 px wide (covers a thin or a
# heavy printed line centred on it)
WALLMASK = np.zeros((H, W), np.uint8)
for _s in LINES:
    cv2.line(WALLMASK, (int(round(_s[0])), int(round(_s[1]))), (int(round(_s[2])), int(round(_s[3]))), 1, 5)
WALLMASK = WALLMASK > 0
_WMID = (LINES[:, :2] + LINES[:, 2:]) / 2
_WANG = np.degrees(np.arctan2(LINES[:, 3] - LINES[:, 1], LINES[:, 2] - LINES[:, 0])) % 180
_WLEN = np.hypot(LINES[:, 2] - LINES[:, 0], LINES[:, 3] - LINES[:, 1])


def _grid_kernels(x0_, y0_, x1_, y1_, L_):
    """Line kernels (length L_) along the local street/block grid: the directions of the wall segments in
    this area, weighted by length, rounded to 5 deg; directions within 10 deg of horizontal or vertical are
    left out (those kernels are always used)."""
    near = (_WMID[:, 0] >= x0_) & (_WMID[:, 0] < x1_) & (_WMID[:, 1] >= y0_) & (_WMID[:, 1] < y1_) & (_WLEN >= 2)
    if not near.any():
        return ()
    h = np.bincount((np.round(_WANG[near] / 5).astype(int) % 36), weights=_WLEN[near], minlength=36)
    kerns = []
    for b in np.nonzero(h >= 0.15 * h.sum())[0]:
        a = b * 5.0
        if min(a % 90, 90 - a % 90) <= 10:
            continue
        k = np.zeros((L_ + 2, L_ + 2), np.uint8)
        cc_ = (L_ + 1) / 2
        dx, dy = math.cos(math.radians(a)) * (L_ - 1) / 2, math.sin(math.radians(a)) * (L_ - 1) / 2
        cv2.line(k, (int(round(cc_ - dx)), int(round(cc_ - dy))), (int(round(cc_ + dx)), int(round(cc_ + dy))), 1, 1)
        kerns.append(k)
    return tuple(kerns)
print("wall segments:", len(LINES))


def street_line(k):
    """The street centreline through a street-name box, in this run's (padded) px, or None: the
    centreline passing nearest the box's centre, cut to a stretch a few box-lengths long."""
    a, b, c, e = k["box"]
    corners = t.to_world(np.array([[a + x0, b + y0], [c + x0, b + y0], [c + x0, e + y0], [a + x0, e + y0]], float))
    poly = shapely.Polygon(corners)
    near = streets[streets.intersects(poly.buffer(15))]
    if not len(near):
        return None
    ctr = poly.centroid
    row = min(near.itertuples(), key=lambda r_: r_.geometry.distance(ctr))
    reach0 = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 1.5 + 50
    same = streets[(streets.street == row.street) & streets.intersects(ctr.buffer(reach0))] if row.street else near.iloc[:0]
    g = shapely.ops.unary_union(list(same.geometry)) if len(same) else row.geometry
    g = shapely.ops.linemerge(g) if g.geom_type == "MultiLineString" else g
    if g.geom_type == "MultiLineString":
        g = min(g.geoms, key=lambda g_: g_.distance(ctr))
    reach = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 1.5 + 50
    piece = g.intersection(ctr.buffer(reach))
    if piece.is_empty:
        return None
    if piece.geom_type != "LineString":
        piece = shapely.ops.linemerge(piece) if piece.geom_type == "MultiLineString" else piece
        if piece.geom_type != "LineString":
            piece = min(getattr(piece, "geoms", [piece]), key=lambda g_: g_.distance(ctr))
    return t.to_pixel(np.asarray(piece.coords)[:, :2]) - [x0, y0]


def fit_box(text, k):
    """The whole-word fit (fitword.py) on the scan as printed (the erase is on the 4a image): placed and judged by one score.  Street names
    follow their street's centreline."""
    kk = {**k, "_kind_h": KIND_H.get(k["kind"])}
    across = min(k["box"][2] - k["box"][0], k["box"][3] - k["box"][1])
    line = street_line(k) if k["kind"] == "street name" else None
    g = fitword.fit(text, kk, ink0, pw=max(2.0, 0.2 * across), line_px=line, lines=LINES)   # placed and scored on the scan as printed
    if g is None:
        return None
    q = np.round(g["pts"]).astype(int)
    _, keep = np.unique(q, axis=0, return_index=True)        # each stroke pixel once
    g["pts"], g["li"] = g["pts"][keep], g["li"][keep]
    on_l = ink0[np.clip(q[keep][:, 1], 0, H - 1), np.clip(q[keep][:, 0], 0, W - 1)] > 0
    g["letters"] = [round(float(on_l[g["li"] == i].mean()), 2) if (g["li"] == i).any() else 0.0 for i in range(len(text))]
    del g["li"]
    return g


def fit_box_old(text, k):
    """Fit `text` into its box: letters lie along the box's long side at Terra's angle.  Slanted text
    (Terra's direction 'slanted') is fitted at Terra's angle, a few degrees either way; its box is the
    upright rectangle round tilted text, so the letter height is this kind's measured level height and
    the length follows from the box and the angle."""
    a, b, c, e = k["box"]
    bw, bh = c - a, e - b
    cx, cy = (a + c) / 2, (b + e) / 2
    if k.get("direction") == "slanted" and 5 <= abs(k["angle_deg"]) <= 85 and k["kind"] in KIND_H:
        th = math.radians(abs(k["angle_deg"]))
        cs, sn = math.cos(th), math.sin(th)
        across = KIND_H[k["kind"]]
        along = (bw - across * sn) / cs if cs >= sn else (bh - across * cs) / sn
        along = max(along, across)
        angs = [-k["angle_deg"] + dd for dd in (-6, -3, 0, 3, 6)]       # image y runs down
    else:
        vertical = bh > bw * 1.3 and len(text) > 1 or abs(k["angle_deg"]) >= 45
        ang = -90.0 if k["angle_deg"] >= 45 or (vertical and k["angle_deg"] > -45) else 90.0 if vertical else 0.0
        along, across = (bh, bw) if vertical else (bw, bh)
        angs = [ang]
    best = None
    for ang in angs:
      for it in (0, 6, 10, 14, 18, 22):           # slant, degrees to the right
        for hf in (0.7, 0.8, 0.9, 1.0):
            h = max(5.0, across * hf)
            p, _ = textfit._points(text, it, h, 1.0, 0.0)
            if not len(p):
                continue
            length = p[:, 0].max() - p[:, 0].min()
            sx = along / max(1.0, length)         # the text spans Terra's box end to end
            if not 0.6 <= sx <= 1.6:           # these letters would have to be squashed or stretched
                continue
            pts, li = textfit._points(text, it, h, sx, ang)
            for pf in (0.1, 0.16, 0.22):
                pw = max(1.0, h * pf)
                v, x, y = textfit._best_ncc(inkf, pts, pw, cx, cy, max(2, round(across * 0.15)))
                if best is None or v > best["score"]:
                    best = {"score": v, "x": x, "y": y, "h": h, "sx": sx, "angle": ang, "italic": it, "pw": pw,
                            "across": across}
    if best is None:
        return None
    pts, li = textfit._points(text, best["italic"], best["h"], best["sx"], best["angle"])
    pts = pts + [best["x"], best["y"]]
    out, per = [], []
    for i in range(len(text)):
        m = li == i
        if not m.any():
            continue
        cc = pts[m].mean(0)
        v, x, y = textfit._best_ncc(inkf, pts[m] - cc, best["pw"], cc[0], cc[1], max(1, round(best["across"] * 0.1)))
        out.append(pts[m] - cc + [x, y] if v > -1 else pts[m])
        per.append(round(v, 2))
    best["pts"] = np.concatenate(out)
    best["letters"] = per
    return best


def fit_label(k):
    """One label's best fit over its candidate readings (or None)."""
    cands = known_text(k)
    if isinstance(cands, tuple):
        cands = [cands]
    f, text, src = None, None, None
    for tx_, sr_ in cands:
        kk = k
        if sr_ == "Terra, completed" and k["text"]:
            # Terra's box covers only the part it read: grow it in the reading direction for the rest
            a, b, c, e = k["box"]
            vert = k.get("direction") in ("bottom-to-top", "top-to-bottom")
            along = (e - b) if vert else (c - a)
            # by the pen font's own widths ("-" and "1" are far narrower than R and M), not letter count
            def _wid(s):
                p_, _ = textfit._points(s, 0, 10.0, 1.0, 0.0)
                return float(p_[:, 0].max() - p_[:, 0].min()) if len(p_) else 1.0
            extra = int(round(along * (_wid(tx_) / _wid(k["text"]) - 1)))
            dr = k.get("direction", "left-to-right")
            box = [a, b - extra, c, e] if dr == "bottom-to-top" else [a, b, c, e + extra] if dr == "top-to-bottom" \
                else [a - extra, b, c, e] if dr == "right-to-left" else [a, b, c + extra, e]
            kk = {**k, "box": box}
        g = fit_box(tx_, kk)
        if g is not None:
            g["fit_box"] = kk["box"]
        if g and (f is None or g["score"] > f["score"]):
            f, text, src = g, tx_, sr_
    if f:
        f["candidates"] = [c_ for c_, _ in cands]
        f.update(text=text, said=k["text"], source=src, kind=k["kind"], box=f.pop("fit_box", k["box"]))
    return f


# the legend and title below the map frame's bottom rule are not map: those labels are left out
_fb = json.load(open(SF / "frame_bottom.json"))
_n0 = len(labels)
labels = [k for k in labels if (k["box"][1] + k["box"][3]) / 2 - M < _fb["k"] * ((k["box"][0] + k["box"][2]) / 2 - M) + _fb["b"]]
print(f"legend cut: {_n0 - len(labels)} labels below the map frame left out", flush=True)
# CODES_ONLY: only the zoning codes are fitted and erased (they're printed as heavy as a district line; the
# thin street names and block numbers aren't, and the graph leaves out what lies inside their boxes).  Their
# boxes are still kept, so the despeckle never takes ink that may be theirs.
OTHER_BOXES = []
if os.environ.get("CODES_ONLY"):
    OTHER_BOXES = [k["box"] for k in labels if k["kind"] != "zoning code"]
    labels = [k for k in labels if k["kind"] == "zoning code"]
    print(f"codes only: {len(labels)} zoning codes fitted, {len(OTHER_BOXES)} other labels left as printed", flush=True)
import pickle  # noqa: E402
RESUME = bool(os.environ.get("RESUME"))
_labels_all = labels
if RESUME:
    labels = []                                   # fitting skipped: the fits come from the checkpoint

# quick runs: ONLY="x,y;x,y" (Terra boxes' top-left, page px) fits just those labels
if os.environ.get("ONLY"):
    _only = [tuple(map(int, v.split(","))) for v in os.environ["ONLY"].split(";")]
    labels = [k for k in labels if any(abs(k["box"][0] - M - x_) <= 2 and abs(k["box"][1] - M - y_) <= 2 for x_, y_ in _only)]
    print("quick run:", len(labels), "labels")

# every label is fitted on its own: one process per core (fork: the workers share this run's images)
import multiprocessing as _mp  # noqa: E402
NPROC = int(os.environ.get("NPROC", "8"))      # capped: memory, not cores, is the limit
def _fit_logged(k):
    """fit_label with a START / END line per label (worker pid, box, text, seconds)."""
    import time as _t
    t0 = _t.time()
    print(f"START pid={os.getpid()} box={[v - M for v in k['box']]} text={k['text']!r} kind={k['kind']}", flush=True)
    if os.environ.get("PROFILE_DIR"):        # profiling only: each worker's profile, rewritten after every label
        import cProfile
        global _PROF
        if "_PROF" not in globals():
            _PROF = cProfile.Profile()
        _PROF.enable()
        f = fit_label(k)
        _PROF.disable()
        _PROF.dump_stats(f"{os.environ['PROFILE_DIR']}/worker_{os.getpid()}.pstats")
    else:
        f = fit_label(k)
    print(f"END pid={os.getpid()} box={[v - M for v in k['box']]} {_t.time() - t0:.1f}s", flush=True)
    return f


MTPC = int(os.environ.get("MTPC", "0")) or None    # never retire workers: replacements are forked from the pool's handler thread and segfault (debugged 2026-11-06)
with _mp.get_context("fork").Pool(NPROC, maxtasksperchild=MTPC) as _pool:     # as in the run that stalled
    if os.environ.get("DEBUGPOOL"):
        import logging as _lg
        import threading as _th
        _mp.util.log_to_stderr(_lg.DEBUG)

        def _exits():
            seen = set()
            while True:
                for pr in list(_pool._pool):
                    if pr.exitcode is not None and pr.pid not in seen:
                        seen.add(pr.pid)
                        print(f"WORKER EXIT pid={pr.pid} exitcode={pr.exitcode}", flush=True)
                _time_.sleep(0.05)
        import time as _time_
        _th.Thread(target=_exits, daemon=True).start()
    import time as _time
    _t0, fits = _time.time(), []
    for _n, _f in enumerate(_pool.imap(_fit_logged, labels, chunksize=2), 1):    # progress as labels finish
        if _f:
            fits.append(_f)
        if _n % 25 == 0 or _n == len(labels):
            _el = _time.time() - _t0
            print(f"PROGRESS {_n}/{len(labels)} labels fitted | {_el / 60:.1f} min elapsed | "
                  f"about {_el / _n * (len(labels) - _n) / 60:.0f} min to go | {_time.strftime('%H:%M')}", flush=True)

# duplicates: two of Terra's boxes fitted with the same text whose letters mostly overlap (over half of the
# smaller fit's stroke pixels) are one printed label read twice; keep the better-scoring fit
_px = [set(map(tuple, np.round(np.asarray(f["pts"])).astype(int).tolist())) for f in fits]
_drop = set()
for i in range(len(fits)):
    for j in range(i + 1, len(fits)):
        if i in _drop or j in _drop or fits[i]["text"] != fits[j]["text"]:
            continue
        ov = len(_px[i] & _px[j]) / max(1, min(len(_px[i]), len(_px[j])))
        if ov > 0.5:
            lo = i if fits[i]["score"] < fits[j]["score"] else j
            _drop.add(lo)
DROPPED = [{"text": fits[i]["text"], "said": fits[i]["said"], "box": [v - M for v in fits[i]["box"]], "score": fits[i]["score"]} for i in sorted(_drop)]
fits = [f for i, f in enumerate(fits) if i not in _drop]
json.dump(DROPPED, open(O / "dropped_duplicates.json", "w"), default=float)
print("duplicates dropped:", [(r["text"], r["said"], r["box"][:2]) for r in DROPPED])
if RESUME:
    fits = pickle.load(open(O / "fits_raw.pkl", "rb"))
    print("resumed:", len(fits), "fits from the checkpoint", flush=True)
else:
    pickle.dump(fits, open(O / "fits_raw.pkl", "wb"))
    print("checkpoint saved:", len(fits), "fits", flush=True)

ON_INK, COVER = 0.40, 0.6     # PROVISIONAL: just under the lowest fit the owner called good (0.46); set from the score bands


scan = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
erase = np.zeros((H, W), bool)
img3 = cv2.addWeighted(scan, 0.55, np.full_like(scan, 255), 0.45, 0)
only = np.full_like(scan, 255)
lay = img3.copy()
# every label's box: ink inside another label is text, not evidence of a line
textbox = np.zeros((H, W), np.int32)
for i_, f in enumerate(fits):
    a, b, c, e = f["box"]
    textbox[max(0, b):e + 1, max(0, a):c + 1] = i_ + 1


def own_width(tp, letter, inner_r, S):
    """A heavy line keeps only its own width where a letter touches it.  The heavy-ink test takes whole
    connected regions, so a letter stroke lying against the line (a P's stem against its box) would be
    kept with it.  Find the line's centre where no letter is (thinning the line ink outside the letter's
    band), measure its width there, and inside the letter's band keep only pixels within that width of
    the centre; gaps where a letter crosses the line are bridged (closing over a couple of strokes)."""
    if not tp.any():
        return tp
    core = tp & ~letter
    sk = skeletonize(core)
    if not sk.any():
        return tp & ~letter
    w = 2 * float(np.median(inner_r[sk]))
    g = 2 * math.ceil(S) + 1
    sk = cv2.morphologyEx(sk.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    rr = 2 * math.ceil(w / 2) + 1
    band = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rr, rr))).astype(bool)
    return tp & (~letter | band)


def measured_stroke(f):
    """The ink's own stroke under the fitted letters: thin it to its centre lines, twice the median
    half-width there.  Where a label sits on a line this reads high, so it's only used pooled by kind."""
    q = np.round(f["pts"]).astype(int)
    a, b, c, e = f["box"]
    y0_, y1_ = max(0, b - 4), min(H, e + 5)
    x0_, x1_ = max(0, a - 4), min(W, c + 5)
    if y1_ <= y0_ or x1_ <= x0_:
        return None
    lm = np.zeros((y1_ - y0_, x1_ - x0_), np.uint8)
    for x, y in q:
        cv2.circle(lm, (int(x) - x0_, int(y) - y0_), max(2, int(round(f["pw"]))), 1, -1)
    under = (lm > 0) & (ink0[y0_:y1_, x0_:x1_] > 0)
    hw = inner[y0_:y1_, x0_:x1_][skeletonize(under)]
    hw = hw[hw > 0]
    return 2 * float(np.median(hw)) if len(hw) else None


# text comes in weights by kind (street names and block numbers light, zoning codes heavy): the median
# of measured stroke / height over every label of that kind on the sheet
_ratio = {}
for f in fits:
    try:
        sm = measured_stroke(f)
    except Exception as ex:
        P_ = np.asarray(f["pts"], float)
        print(f"STROKE MEASURE FAILED: {f['text']!r} box={f['box']} box types={[type(v).__name__ for v in f['box']]} "
              f"pts {P_.shape} nan={int(np.isnan(P_).sum())} min={np.nanmin(P_, 0) if P_.size else None} "
              f"max={np.nanmax(P_, 0) if P_.size else None} layout={f.get('layout')} h={f.get('h')} sx={f.get('sx')}: {ex}", flush=True)
        raise
    if sm and f["across"] > 0:
        _ratio.setdefault(f["kind"], []).append(sm / f["across"])
WEIGHT = {k: float(np.median(v)) for k, v in _ratio.items()}
_all = [r for v in _ratio.values() for r in v]
for f in fits:
    WEIGHT.setdefault(f["kind"], float(np.median(_all)) if _all else 0.15)
print("text weight (stroke / height):", {k: round(v, 3) for k, v in WEIGHT.items()})

for i_, f in enumerate(fits):
    q = np.round(f["pts"]).astype(int)
    q = q[(q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < W) & (q[:, 1] < H)]
    a, b, c, e = f["box"]
    # the label's measured stroke: twice the median ink half-width along its fitted centre line
    # (a median, so a line the letter crosses doesn't count).  Every threshold below is a multiple
    # of it, or of the text height.
    # the stroke: this kind of text's weight (measured over the sheet) at this label's height
    half = WEIGHT[f["kind"]] * f["across"] / 2
    S = 2 * half
    R = max(1, math.ceil(1.5 * half))                   # erase: the stroke, plus half of it for the fit's play
    em = np.zeros((H, W), np.uint8)
    for x, y in q:
        cv2.circle(em, (int(x), int(y)), R, 1, -1)
    m_ = math.ceil(S)
    boxm = np.zeros((H, W), np.uint8)
    cv2.rectangle(boxm, (a - m_, b - m_), (c + m_, e + m_), 1, -1)
    box_ink = (boxm > 0) & ink.astype(bool)
    # linework through or beside the box: straight runs longer than 1.5x the text height ...
    L = math.ceil(1.5 * f["across"])
    pa, pb, pc, pe = max(0, a - L), max(0, b - L), min(W, c + L), min(H, e + L)
    reg = ink[pb:pe, pa:pc]
    # "beyond the letter": half the text height away from the fitted letters.  A line through the
    # label runs on far past it; a letter printed a little wider or bolder than the fit only sticks
    # out by a stroke or two
    nr = math.ceil(0.5 * f["across"])
    near_letter = cv2.dilate(em[pb:pe, pa:pc], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * nr + 1, 2 * nr + 1))).astype(bool)
    other = textbox[pb:pe, pa:pc]
    other = (other > 0) & (other != i_ + 1)
    outside = ~near_letter & ~other                     # beyond the letter, and not another label

    def beyond(mask):
        """The parts of mask (connected pieces) that carry on outside the letter, into ink that
        isn't another label."""
        nl, ll = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        out = np.zeros(nl, bool)
        np.logical_or.at(out, ll[outside], True)
        out[0] = False
        return out[ll]

    straight = np.zeros(reg.shape, bool)
    for kern in (np.ones((1, L), np.uint8), np.ones((L, 1), np.uint8)) + _grid_kernels(pa, pb, pc, pe, L):
        straight |= cv2.morphologyEx(reg, cv2.MORPH_OPEN, kern).astype(bool)
    # ... and ink heavier than this text: it holds a disk 1.4x the text's measured stroke, which the
    # letters (even where strokes join) can't
    k = math.ceil(1.4 * S)
    thick = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    lines = np.zeros((H, W), bool)
    lines[pb:pe, pa:pc] = beyond(straight) | own_width(beyond(thick), em[pb:pe, pa:pc] > 0, inner[pb:pe, pa:pc], S)
    if os.environ.get("DBG") == f"{a - M},{b - M}":            # one label's masks, for checking by eye
        dbg = np.dstack([reg * 255] * 3).astype(np.uint8) // 3 + 170
        dbg[em[pb:pe, pa:pc] > 0] = (200, 200, 120)
        dbg[reg > 0] = (90, 90, 90)
        dbg[beyond(straight)] = (0, 0, 230)
        dbg[beyond(thick)] = (0, 160, 0)
        dbg[~outside & (reg > 0) & ~beyond(straight) & ~beyond(thick)] = (60, 60, 60)
        cv2.imwrite(str(O / "dbg_label.png"), cv2.resize(dbg, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        print("DBG", f["text"], "S", round(S, 2), "R", R, "k", k, "L", L)
    box_ink &= ~lines
    f["stroke"] = S
    # the one measure: the fit's own score (fitword.py), the same that placed it
    f["on_ink"], f["cover"] = f["on"], f["cov"]
    # cover: of the letters' own ink (ink joined to the fitted letters within half a text height,
    # not line), how much the fit covers; Terra's box can sit a little off, so it isn't the measure
    let = (ink[pb:pe, pa:pc] > 0) & near_letter & ~lines[pb:pe, pa:pc]
    nl_, ll_ = cv2.connectedComponents(let.astype(np.uint8), connectivity=8)
    hit = np.zeros(nl_, bool)
    hit[np.unique(ll_[(em[pb:pe, pa:pc] > 0) & let])] = True
    hit[0] = False
    own = hit[ll_]
    if os.environ.get("DBG") == f"{a - M},{b - M}":            # the cover sum, for checking by eye
        sh = own.shape
        P0 = cv2.cvtColor(gray0[pb:pe, pa:pc], cv2.COLOR_GRAY2BGR)
        P1 = np.full(sh + (3,), 255, np.uint8)
        P1[ink0[pb:pe, pa:pc] > 0] = (200, 200, 200)             # ink on the scan
        P1[ink[pb:pe, pa:pc] > 0] = (120, 120, 120)              # ink after 4a
        P1[lines[pb:pe, pa:pc] & (ink[pb:pe, pa:pc] > 0)] = (200, 120, 0)   # taken as line
        P1[own & (em[pb:pe, pa:pc] > 0)] = (0, 170, 0)            # letter ink the fit covers
        P1[own & ~(em[pb:pe, pa:pc] > 0)] = (0, 0, 230)           # letter ink the fit misses
        P2 = P0.copy()
        for x, y in q:
            if pa <= x < pc and pb <= y < pe:
                cv2.circle(P2, (int(x) - pa, int(y) - pb), 1, (0, 110, 255), -1)
        gap = np.full((sh[0], 3, 3), 255, np.uint8)
        big = cv2.resize(np.hstack([P0, gap, P2, gap, P1]), None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(O / "dbg_cover.png"), big)
        print("DBG cover", f["text"], "own", int(own.sum()), "covered", int(((em[pb:pe, pa:pc] > 0) & own).sum()), "cover", round(f["cover"], 3), "on_ink", round(f["on_ink"], 3))
    f["ok"] = f["on_ink"] >= ON_INK and f["cover"] >= COVER
    col = (0, 150, 0) if f["ok"] else (0, 0, 220)
    r = max(1, int(round(f["pw"] / 2)))
    m = np.zeros((H, W), np.uint8)
    for x, y in q:
        cv2.circle(m, (int(x), int(y)), r, 1, -1)
    lay[m > 0] = col
    only[m > 0] = col
    flag = f["source"] in ("Terra, not in code check", "Terra, completed")
    cv2.rectangle(lay, (a, b), (c, e), (0, 140, 255) if flag else (200, 0, 140) if f["ok"] else (0, 0, 220), 3 if flag else 1)
    cv2.putText(lay, f"{f['text']} {f['on_ink']:.2f}/{f['cover']:.2f}", (max(0, a), max(10, b - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
    if f["ok"]:
        # only the ink under the fitted strokes themselves, at their drawn width plus 1 px: nothing joined
        # to the letters and nothing further out is taken, so a line a letter touches stays
        sp = np.zeros((H, W), np.uint8)
        qq = np.unique(np.round(np.asarray(f["pts"])).astype(int), axis=0)
        qq = qq[(qq[:, 0] >= 0) & (qq[:, 1] >= 0) & (qq[:, 0] < W) & (qq[:, 1] < H)]
        sp[qq[:, 1], qq[:, 0]] = 1
        y0s, y1s, x0s, x1s = max(0, qq[:, 1].min() - 2), min(H, qq[:, 1].max() + 3), max(0, qq[:, 0].min() - 2), min(W, qq[:, 0].max() + 3)
        under = cv2.dilate(sp[y0s:y1s, x0s:x1s], np.ones((3, 3), np.uint8)) > 0
        # never erase line: ink the straight-run / thicker-than-letters tests marked as line, and ink on a
        # wall (the lined-up block outlines, which carry every curved zoning line)
        erase[y0s:y1s, x0s:x1s] |= under & (ink[y0s:y1s, x0s:x1s] > 0) & ~lines[y0s:y1s, x0s:x1s] & ~WALLMASK[y0s:y1s, x0s:x1s]
img3 = cv2.addWeighted(lay, 0.75, img3, 0.25, 0)
cv2.imwrite(str(O / "s3_detect.png"), crop(img3))
cv2.imwrite(str(O / "s3_after.png"), crop(only))
# despeckle: after erasing, leftover ink pieces lying entirely inside an erased label's box (+2 px) are scraps
# of its letters; a line runs on past the box and stays; a piece touching another label's box is kept
_left = (ink > 0) & ~erase
_n, _lab, _st, _ = cv2.connectedComponentsWithStats(_left.astype(np.uint8), connectivity=8)
_allbox = np.zeros((H, W), np.int32)
_owners = fits + [{"box": _bx, "ok": False} for _bx in OTHER_BOXES]     # labels not fitted: never erased
for _i, _f in enumerate(_owners):
    _a, _b, _c, _e = _f["box"]
    _sl = _allbox[max(0, _b - 2):_e + 3, max(0, _a - 2):_c + 3]
    _sl[_sl == 0] = _i + 1
speck = np.zeros((H, W), bool)
for _i, _f in enumerate(fits):
    if not _f["ok"]:
        continue
    _a, _b, _c, _e = _f["box"]
    xa, ya, xb, yb = max(0, _a - 2), max(0, _b - 2), min(W, _c + 3), min(H, _e + 3)
    for _k in np.unique(_lab[ya:yb, xa:xb]):
        if _k == 0:
            continue
        x_, y_, w_, h_ = _st[_k, :4]
        if x_ >= xa and y_ >= ya and x_ + w_ <= xb and y_ + h_ <= yb:      # wholly inside this box
            comp = _lab[y_:y_ + h_, x_:x_ + w_] == _k
            owners = np.unique(_allbox[y_:y_ + h_, x_:x_ + w_][comp])
            # inside this label's box only (another label's box overlapping it means it may be theirs)
            others = [o for o in owners if o not in (0, _i + 1)]
            other_ok = all(_owners[o - 1]["ok"] for o in others)
            if not others or other_ok:
                speck[y_:y_ + h_, x_:x_ + w_] |= comp
print("despeckle: px removed", int(speck.sum()), flush=True)
erase |= speck
img4 = scan.copy()
img4[erase] = (0, 110, 255)
cv2.imwrite(str(O / "s4_detect.png"), crop(img4))
clean = gray.copy()
clean[erase] = 255
cv2.imwrite(str(O / "s4_after.png"), crop(clean))
cv2.imwrite(str(O / "clean.png"), crop(clean))
src = {}
for f in fits:
    src[f["source"]] = src.get(f["source"], 0) + 1
st = json.load(open(O / "stats.json"))
st["step3"] = {"labels": len(fits), "fitted_well": sum(f["ok"] for f in fits),
               "fitted_poorly": sum(not f["ok"] for f in fits),
               **{f"text_from_{k.replace(' ', '_')}": v for k, v in src.items()}}
st["step4"] = {"pixels_erased": int(crop(erase).sum()), "ink_pixels": int(crop(ink).sum())}
json.dump(st, open(O / "stats.json", "w"), indent=1)
json.dump([{k: v for k, v in f.items() if k != "pts"} for f in fits], open(O / "fits3.json", "w"), indent=0, default=float)
# the fitted letters' points, page px, keyed like fits3.json's boxes (for comparison sheets)
json.dump({",".join(str(int(v)) for v in f["box"]): (np.unique(np.round(np.asarray(f["pts"])).astype(int), axis=0) - M).tolist() for f in fits},
          open(O / "fitpts.json", "w"))
print(st["step3"], st["step4"])
print("poor:", sorted([(f["text"], round(f["on_ink"], 2), round(f["cover"], 2)) for f in fits if not f["ok"]], key=lambda x: x[2]))
