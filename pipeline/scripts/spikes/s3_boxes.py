"""Spike S3: word boxes grouped from the text detector's per-px chance, measured against Terra's words.

A measurement, not shipped code (brief: .superpowers/sdd/2026-11-16-search-move-and-reader-spikes/task-S3-brief.md).

    uv run python scripts/spikes/s3_boxes.py predict     # the test drawings' chance (GPU turn), cached as PNGs
    uv run python scripts/spikes/s3_boxes.py measure     # every number: per_sheet.pkl (4 worker processes)
    uv run python scripts/spikes/s3_boxes.py ink         # the ink-gated variants I<level>: per_sheet_ink.pkl
    uv run python scripts/spikes/s3_boxes.py keep I0.8  # the boxes of one more level, for the crop pages
    uv run python scripts/spikes/s3_boxes.py report      # results.json, results.md, the plots and crop pages

What is scored (the detector's held-out scopes, as `textnet/scores.py` defines them):
  2000 map     the validation area (`data.VALIDATION_AREA`, through each sheet's placement; `scores.scored_px`)
  test maps    every px of every sheet of 1980, 1972, 1975 (`paths.TEXT_TEST_SERIES`, never trained on)
A Terra word counts when its centre px is in the scope, a grouped box when its centre is (as `scores._add_words`).

Grouping: px with chance >= LEVEL joined by 8-connectivity; a component's px weighted by chance give its centre and
second moments; the axis is the major axis (modulo 180 degrees, counter-clockwise from left-to-right, as Terra's
reading angle); the box is the extent of its px along and across that axis.  Components of fewer than MIN_PX px are
dropped (specks).  Variant I<level>: the same, counting only px within INK_R px of ink (the detector reads ink only,
so its chance on blank paper, such as the middle of a city block, is no evidence).  Variant H<level>: the same at that low level, keeping only components holding a px >= SEED
(hysteresis).  "self": Terra's own boxes through the same measuring code (the sanity check).
"""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import pickle  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import data, scores  # noqa: E402

cv2.setNumThreads(1)
OUT = config.DATA_DIR / "solver" / "experiments" / "spike-s3"
CHANCE_DIR = OUT / "chance"
LEVELS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)   # the brief asks 0.3-0.8; recall was still rising there
SEED = 0.8                    # hysteresis: a component must hold a px at least this
MIN_PX = 20                   # a component of fewer px is a speck
IOUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MAIN_IOU = 0.3                # the IoU the headline numbers are given at
MID = 0.5                     # the "mid" level the headline numbers and crops are given at
KINDS = ("zoning code", "block number", "street name", "other")
GROUPS = [("2000 val", "2000-digital-zoning")] + [(s, s) for s in paths.TEXT_TEST_SERIES]
SOL_RADIUS = (15, 30)         # px: a Sol point has a Terra word "nearby" when its box is within this
BOOT = 400
WORKERS = 4


INK_R = 4                     # variant I: the chance counts only within INK_R px of ink (the detector reads ink only)
INK_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)


def ink_variants() -> list[str]:
    return [f"I{lv:.2f}" for lv in INK_LEVELS]


SPECK_LEVELS = (0.5, 0.7, 0.8, 0.9)
SPECK_MIN = (0, 5, 20, 50)


def speck_variants() -> list[str]:
    """The speck-cutoff sensitivity: L<level>m<N> (components of at least N mask px) and I<level>m<N> (ink-gated;
    components holding at least N ink px)."""
    return [f"{k}{lv:.2f}m{m}" for k in "LI" for lv in SPECK_LEVELS for m in SPECK_MIN]


def variants() -> list[str]:
    return [f"L{lv:.2f}" for lv in LEVELS] + [f"H{lv:.2f}" for lv in LEVELS if lv < SEED] + ["self"]


# --- the detector's chance ---------------------------------------------------------------------------------------

def chance_file(series: str, sheet: str) -> Path:
    return paths.TEXT_CHANCE_DIR / f"{sheet}.png" if series == paths.SERIES else CHANCE_DIR / series / f"{sheet}.png"


def predict() -> None:
    """The test drawings' chance, with the stage's model, in one GPU turn (as `score_stage.build_scores`)."""
    from sfzmaps import gpu
    from sfzmaps.solver.textnet import predict as pr
    from sfzmaps.solver.textnet import train
    device = train.default_device()
    sheets = data.load_sheets(paths.TEXT_TEST_SERIES, hold_out_validation=False, placement=False)
    todo = [sd for sd in sheets if not chance_file(sd.series, sd.sheet).is_file()]
    print(f"{len(todo)} of {len(sheets)} sheets to predict on {device}", flush=True)
    with gpu.gpu_turn(True, device):
        train.settle_torch()
        net = pr.load_model(paths.TEXT_MODEL, device)
        for sd in todo:
            f = chance_file(sd.series, sd.sheet)
            f.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(f), pr.chance(net, data.page_ink(sd), device))
            print(sd.series, sd.sheet, flush=True)


def load_chance(series: str, sheet: str) -> np.ndarray:
    c = cv2.imread(str(chance_file(series, sheet)), cv2.IMREAD_GRAYSCALE)
    if c is None:
        raise FileNotFoundError(chance_file(series, sheet))
    return c


# --- grouping ----------------------------------------------------------------------------------------------------

def group(chance: np.ndarray, level: float, seed: float | None = None, gate: np.ndarray | None = None,
          min_px: int = MIN_PX, ink: np.ndarray | None = None) -> dict:
    """Boxes grouped from the chance (see the module): corners (n, 4, 2) page px, axis (degrees, modulo 180, as
    Terra's reading angle), length >= height (px), mean chance, px."""
    thr = int(np.ceil(level * 255 - 1e-9))
    mask = chance >= thr
    if gate is not None:
        mask &= gate
    _, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)
    empty = {"corners": np.zeros((0, 4, 2)), "axis": np.zeros(0), "length": np.zeros(0), "height": np.zeros(0),
             "mean": np.zeros(0), "npx": np.zeros(0, int)}
    if not len(ys):
        return empty
    lab_px = lab[ys, xs]
    order = np.argsort(lab_px, kind="stable")
    lab_px, xi, yi = lab_px[order], xs[order], ys[order]
    xs, ys = xi + 0.5, yi + 0.5
    w = chance[yi, xi] / 255.0
    start = np.flatnonzero(np.r_[True, lab_px[1:] != lab_px[:-1]])
    cnt = np.diff(np.r_[start, len(lab_px)])
    m = np.add.reduceat(w, start)
    cx, cy = np.add.reduceat(w * xs, start) / m, np.add.reduceat(w * ys, start) / m
    rx, ry = xs - np.repeat(cx, cnt), ys - np.repeat(cy, cnt)
    sxx, syy, sxy = (np.add.reduceat(w * a, start) / m for a in (rx * rx, ry * ry, rx * ry))
    th = 0.5 * np.arctan2(2 * sxy, sxx - syy)                    # major axis, image coordinates (y down)
    vx, vy = np.cos(th), np.sin(th)
    p = rx * np.repeat(vx, cnt) + ry * np.repeat(vy, cnt)
    q = -rx * np.repeat(vy, cnt) + ry * np.repeat(vx, cnt)
    p0, p1 = np.minimum.reduceat(p, start) - 0.5, np.maximum.reduceat(p, start) + 0.5
    q0, q1 = np.minimum.reduceat(q, start) - 0.5, np.maximum.reduceat(q, start) + 0.5
    size = cnt
    if ink is not None:                  # the speck cutoff counts the component's INK px, not its (dilated) mask px
        size = np.add.reduceat(ink[yi, xi].astype(np.int64), start)
    keep = size >= min_px
    if seed is not None:
        keep &= np.maximum.reduceat(w, start) >= seed
    v, n = np.stack([vx, vy], 1), np.stack([-vy, vx], 1)
    c = np.stack([cx, cy], 1)
    corners = np.stack([c + v * p0[:, None] + n * q0[:, None], c + v * p1[:, None] + n * q0[:, None],
                        c + v * p1[:, None] + n * q1[:, None], c + v * p0[:, None] + n * q1[:, None]], 1)
    axis = np.rad2deg(np.arctan2(-vy, vx)) % 180.0               # counter-clockwise from left-to-right
    return {"corners": corners[keep], "axis": axis[keep], "length": (p1 - p0)[keep], "height": (q1 - q0)[keep],
            "mean": (m / cnt)[keep], "npx": size[keep]}


def terra_polys(sd: data.SheetData) -> dict:
    """Terra's words as the same record as `group` gives (corners, axis, length, height)."""
    c = sd.words.astype(float)
    return {"corners": c, "axis": (sd.angles.astype(float) % 180.0),
            "length": np.linalg.norm(c[:, 1] - c[:, 0], axis=1), "height": np.linalg.norm(c[:, 3] - c[:, 0], axis=1),
            "mean": np.ones(len(c)), "npx": np.zeros(len(c), int)}


# --- the pair table and the measures ---------------------------------------------------------------------------

def poly_area(c: np.ndarray) -> np.ndarray:
    if not len(c):
        return np.zeros(0)
    x, y = c[..., 0], c[..., 1]
    return 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))


def pairs(T: np.ndarray, G: np.ndarray):
    """For every Terra polygon and grouped polygon whose bounding boxes meet and that overlap: (t, g, intersection
    area), the polygons (n, 4, 2) float."""
    if not len(T) or not len(G):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    tl, th_ = T.min(1), T.max(1)
    gl, gh = G.min(1), G.max(1)
    hit = ((tl[:, None, 0] < gh[None, :, 0]) & (th_[:, None, 0] > gl[None, :, 0])
           & (tl[:, None, 1] < gh[None, :, 1]) & (th_[:, None, 1] > gl[None, :, 1]))
    ts, gs = np.nonzero(hit)
    T32, G32 = T.astype(np.float32), G.astype(np.float32)
    inter = np.zeros(len(ts))
    for k, (t, g) in enumerate(zip(ts, gs)):
        inter[k] = cv2.intersectConvexConvex(T32[t], G32[g])[0]
    pos = inter > 0
    return ts[pos], gs[pos], inter[pos]


def axis_err(a, b):
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def measure_sheet(T: dict, tkind: np.ndarray, tin: np.ndarray, G: dict, gin: np.ndarray, gunk: np.ndarray):
    """The per-word and per-box results of one sheet at one variant.  `tin` / `gin`: in the scope; `gunk`: a grouped
    box whose centre is on an unknown px (text Terra may have missed)."""
    aT, aG = poly_area(T["corners"]), poly_area(G["corners"])
    t, g, inter = pairs(T["corners"], G["corners"])
    iou = inter / (aT[t] + aG[g] - inter)
    nT, nG = len(aT), len(aG)
    best = np.zeros(nT)
    bg = np.full(nT, -1)
    for k in np.argsort(iou):                       # ascending: the best pair is written last
        best[t[k]], bg[t[k]] = iou[k], g[k]
    gbest = np.zeros(nG)
    np.maximum.at(gbest, g, iou)
    err = np.full(nT, np.nan)
    has = bg >= 0
    err[has] = axis_err(T["axis"][has], G["axis"][bg[has]])
    # split: a Terra word holding two or more boxes, each mostly inside it (>= 50% of its own area) and each
    # covering >= 10% of the word.  merged: a box holding two or more Terra words, each mostly (>= 50%) inside it
    # and each making up >= 10% of the box.
    in_g = inter / aG[g]
    in_t = inter / aT[t]
    mine = (in_g >= 0.5) & (in_t >= 0.1)
    split = np.bincount(t[mine], minlength=nT) >= 2
    holds = (in_t >= 0.5) & (in_g >= 0.1)
    merged_g = np.bincount(g[holds], minlength=nG) >= 2
    merged_t = np.zeros(nT, bool)
    merged_t[t[holds & merged_g[g]]] = True
    out = {"tkind": tkind[tin].astype(np.uint8), "tbest": best[tin], "terr": err[tin], "tsplit": split[tin],
           "tmerged": merged_t[tin], "taspect": (T["length"] / np.maximum(T["height"], 1e-9))[tin],
           "gbest": gbest[gin], "gunk": gunk[gin], "gmerged": merged_g[gin],
           "gaspect": (G["length"] / np.maximum(G["height"], 1e-9))[gin], "gmean": G["mean"][gin],
           "gnpx": G["npx"][gin], "gheight": G["height"][gin]}
    return out, (best, bg, gbest, merged_g, merged_t, split)


def sol_miss(sd, T: dict, G: dict, scope: np.ndarray) -> dict:
    """Sol's label points in the scope, and how many have no Terra word (any kind; zoning-code words alone) within R px,
    and of those with none, how many have a grouped box within R px."""
    h, w = scope.shape
    pts = [p for p in sd.points if scope[min(max(p[1], 0), h - 1), min(max(p[0], 0), w - 1)]]
    out = {f"r{r}": {"points": len(pts), "no_terra": 0, "no_terra_zoning": 0, "no_terra_but_box": 0}
           for r in SOL_RADIUS}
    zoning = sd.kinds == data.KINDS["zoning code"]

    def near(polys, x, y, r):
        if not len(polys):
            return np.zeros(0, bool)
        lo, hi = polys.min(1) - r, polys.max(1) + r
        cand = np.flatnonzero((lo[:, 0] <= x) & (hi[:, 0] >= x) & (lo[:, 1] <= y) & (hi[:, 1] >= y))
        ok = np.zeros(len(polys), bool)
        for i in cand:
            ok[i] = cv2.pointPolygonTest(polys[i].astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)),
                                         True) >= -r
        return ok

    for x, y in pts:
        for r in SOL_RADIUS:
            o = out[f"r{r}"]
            nt = near(T["corners"], x, y, r)
            if not nt.any():
                o["no_terra"] += 1
                o["no_terra_but_box"] += int(near(G["corners"], x, y, r).any())
            if not nt[zoning].any():
                o["no_terra_zoning"] += 1
    return out


def one_sheet(job):
    """job: (series, sheet) for every variant, keeping the L<MID> boxes; or (series, sheet, variants, kept): those
    variants alone, keeping the boxes of `kept` (None: none)."""
    series, sheet = job[:2]
    vlist, kept = (job[2], job[3]) if len(job) > 2 else (variants(), f"L{MID:.2f}")
    gate = None
    sd = data.sheet_data(series, sheet, hold_out_validation=False, placement=(series == paths.SERIES))
    where = "validation" if series == paths.SERIES else "whole"
    scope = scores.scored_px(sd, where)
    _, unknown = data.page_text(sd)
    chance = load_chance(series, sheet)
    if chance.shape != (sd.height, sd.width):
        raise ValueError(f"{series} {sheet}: chance {chance.shape}, page {(sd.height, sd.width)}")
    h, w = scope.shape

    def centres(c):
        if not len(c):
            return np.zeros((0, 2), int), np.zeros(0, bool)
        ctr = np.rint(c.mean(1)).astype(int)
        return ctr, scope[np.clip(ctr[:, 1], 0, h - 1), np.clip(ctr[:, 0], 0, w - 1)]

    T = terra_polys(sd)
    _, tin = centres(T["corners"])
    res, keep = {}, {}
    for v in vlist:
        if v == "self":
            G = terra_polys(sd)
        elif v[0] in "LI" and "m" in v:
            lv, m = v[1:].split("m")
            if v[0] == "I" and gate is None:
                k = 2 * INK_R + 1
                ink_page = data.page_ink(sd)
                gate = cv2.dilate(ink_page.view(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                       (k, k))).view(bool)
            if v[0] == "I":
                G = group(chance, float(lv), gate=gate, min_px=int(m), ink=ink_page)
            else:
                G = group(chance, float(lv), min_px=int(m))
        elif v[0] == "I":
            if gate is None:
                k = 2 * INK_R + 1
                gate = cv2.dilate(data.page_ink(sd).view(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                                  (k, k))).view(bool)
            G = group(chance, float(v[1:]), gate=gate)
        elif v[0] == "L":
            G = group(chance, float(v[1:]))
        else:
            G = group(chance, float(v[1:]), seed=SEED)
        gc, gin = centres(G["corners"])
        gunk = (unknown[np.clip(gc[:, 1], 0, h - 1), np.clip(gc[:, 0], 0, w - 1)] if len(gc)
                else np.zeros(0, bool))
        res[v], full = measure_sheet(T, sd.kinds, tin, G, gin, gunk)
        if v == kept:
            best, bg, gbest, merged_g, merged_t, split = full
            keep = {"T": T["corners"], "tin": tin, "tkind": sd.kinds, "G": G["corners"], "gin": gin, "gunk": gunk,
                    "tbest": best, "terr": axis_err(T["axis"], G["axis"][np.maximum(bg, 0)]) if len(G["axis"])
                    else np.full(len(T["axis"]), np.nan), "tbg": bg, "gbest": gbest, "gmerged": merged_g,
                    "tmerged": merged_t, "tsplit": split, "gaspect": G["length"] / np.maximum(G["height"], 1e-9),
                    "gmean": G["mean"]}
            res["sol"] = sol_miss(sd, T, G, scope)
    res["n_scope_px"] = int(scope.sum())
    return job[:2], res, keep


# --- the run -----------------------------------------------------------------------------------------------------

def jobs() -> list[tuple[str, str]]:
    return [(s, n) for _, s in GROUPS for n in paths.TEXT_SHEETS[s]]


def measure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    with Pool(WORKERS) as pool:
        for job, res, keep in pool.imap(one_sheet, jobs()):
            done.append((job, res, keep))
            print(*job, flush=True)
    with open(OUT / "per_sheet.pkl", "wb") as f:
        pickle.dump({f"{s}/{n}": r for (s, n), r, _ in done}, f)
    with open(OUT / "keep_L050.pkl", "wb") as f:
        pickle.dump({f"{s}/{n}": k for (s, n), _, k in done}, f)


def keep_level(arg: str) -> None:
    """Only the boxes of one variant (a level "0.8" is L0.80; "I0.8" the ink-gated one), kept for the crop pages
    (keep_<variant>.pkl)."""
    v = f"{arg[0]}{float(arg[1:]):.2f}" if arg[0] in "LI" else f"L{float(arg):.2f}"
    done = {}
    with Pool(WORKERS) as pool:
        for (s, n), _, keep in pool.imap(one_sheet, [(s, n, [v], v) for s, n in jobs()]):
            done[f"{s}/{n}"] = keep
            print(s, n, flush=True)
    with open(OUT / f"keep_{v.replace('.', '')}.pkl", "wb") as f:
        pickle.dump(done, f)


def measure_speck() -> None:
    """The speck-cutoff sensitivity variants: per_sheet_speck.pkl."""
    done = {}
    vs = speck_variants()
    with Pool(WORKERS) as pool:
        for (s, n), res, _ in pool.imap(one_sheet, [(s, n, vs, None) for s, n in jobs()]):
            done[f"{s}/{n}"] = {v: res[v] for v in vs}
            print(s, n, flush=True)
    with open(OUT / "per_sheet_speck.pkl", "wb") as f:
        pickle.dump(done, f)


def measure_ink() -> None:
    """The ink-gated variants (I<level>): per_sheet_ink.pkl, merged with per_sheet.pkl by the report."""
    done = {}
    with Pool(WORKERS) as pool:
        for (s, n), res, _ in pool.imap(one_sheet, [(s, n, ink_variants(), None) for s, n in jobs()]):
            done[f"{s}/{n}"] = {v: res[v] for v in ink_variants()}
            print(s, n, flush=True)
    with open(OUT / "per_sheet_ink.pkl", "wb") as f:
        pickle.dump(done, f)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "predict":
        predict()
    elif cmd == "measure":
        measure()
    elif cmd == "speck":
        measure_speck()
    elif cmd == "ink":
        measure_ink()
    elif cmd == "keep":
        keep_level(sys.argv[2])
    elif cmd == "report":
        sys.path.insert(0, str(Path(__file__).parent))
        import s3_report
        s3_report.report()
    else:
        sys.exit(__doc__)
