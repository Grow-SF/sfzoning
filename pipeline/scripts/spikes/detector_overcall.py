"""Spike: why the held-out text detectors (stage `solver-text-reader`) over-call ink as text.  A read-only measurement:
nothing here trains, writes under pipeline/src or touches a stage's output; it writes only
data/solver/experiments/detector-overcall/ (OUT).  CPU only, one thread.

    uv run python scripts/spikes/detector_overcall.py sheets [SHEET ...]    # the 2000 sheets' counts (part 1)
    uv run python scripts/spikes/detector_overcall.py pages                 # every training sheet's lines and shares
    uv run python scripts/spikes/detector_overcall.py training CONFIG ...   # what a training's loss read (all, nw..se)
    uv run python scripts/spikes/detector_overcall.py pictures              # crops to look at, and radius-word
    uv run python scripts/spikes/detector_overcall.py radius-word           # the word that set the 328-px radius
    uv run python scripts/spikes/detector_overcall.py report                # tables.json and tables.md

The detectors compared on every 2000 sheet's ink (chance >= 0.5 "called text", and >= 0.9):
  all     the all-sheets detector (`solver-text`): text/chance/<sheet>.png, as built (in-sample on these sheets)
  held    the held-out detectors (`solver-text-reader`): text-reader/<sheet>/chance.png, as built (each px from the
          detector that held its quarter out)
  old     the old quarter models (text-<q>/chance/<sheet>.png, left on disk from the removed stages
          solver-text-<q>, commit 487e633): each px from the model that held its quarter out
  trial   the recipe trial (experiments/text-half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1/model.pt): the
          stage's recipe with only the validation area held out, 30-px discs; scored here on the CPU
  r_<q>   each held-out detector (text-reader/models/<q>.pt) scored on the whole page here on the CPU, so that on the
          other three quarters it is in-sample
Sets of ink px (Terra's words drawn as the detector learns them, `data._words`; the other passes' boxes `data._paint`):
  text      ink in a Terra word
  nonbox    ink in no Terra word;  nonbox_other: of it, in another pass's box (unknown for every detector)
  lines     ink on long lines (`scores.long_lines`' rule: strokes >= 200 ft long, median width <= 48 ft) farther than
            MARGIN px from every Terra word and other pass's box: the clean not-text set
  heavy     of `lines`, the strokes of median width >= HEAVY_FT (district-line weight; ordinary lines are ~16 ft)
  lines_m0 / lines_m25   `lines` with no margin, or a 25-px margin
  heavy_nb  ink on heavy long lines in no Terra word (no margin; other passes' boxes kept); heavy_nb_other: of it,
            in another pass's box
Zones by the distance to the sheet's nearest Sol point (page px; Sol's points are the training's disc centres):
  ZONE_EDGES; the held-out detectors' unknown disc is the first four (<= RADIUS), the all-sheets one the first.
Regions: the px's quarter (by its map point), whether it is in the validation area, in an answered block (held out of
all four held-out detectors) and on the sheet's warped footprint.
"""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(1)

from sfzmaps.solver import paths, runs, strokes  # noqa: E402
from sfzmaps.solver.textnet import data, predict, reader_stage, scores, stage, train  # noqa: E402

data.LOAD_THREADS = 1          # one sheet read at a time (the load rule: one thread a process)
OUT = runs.EXPERIMENTS_DIR / "detector-overcall"
TRIAL_MODEL = runs.EXPERIMENTS_DIR / "text-half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1" / "model.pt"
HELD = json.loads((paths.READER_TEXT_DIR / reader_stage.HELD_FILE).read_text())
RADIUS = int(HELD["unknown_radius"]["px"])          # 328: the held-out detectors' disc
SMALL = data.READ_POINT_RADIUS                      # 30: the all-sheets detector's disc
ZONE_EDGES = (SMALL, 100, 200, RADIUS, RADIUS + RADIUS // 2, 2 * RADIUS, 1000)
ZONES = ("<=30", "30-100", "100-200", f"200-{RADIUS}", f"{RADIUS}-{RADIUS + RADIUS // 2}",
         f"{RADIUS + RADIUS // 2}-{2 * RADIUS}", f"{2 * RADIUS}-1000", ">1000")
MARGIN = 10
HEAVY_FT = 22.0
SETS = ("text", "nonbox", "nonbox_other", "lines", "heavy", "lines_m0", "lines_m25", "heavy_nb", "heavy_nb_other")
DETECTORS = ("all", "held", "old", "trial", "r_nw", "r_ne", "r_sw", "r_se")
N_REGIONS = 32                 # quarter (4) x validation (2) x answered (2) x footprint (2)
SEED = 0


# --- masks of a page ------------------------------------------------------------------------------------------------

def line_masks(ink: np.ndarray, ft_per_px: float) -> tuple[np.ndarray, np.ndarray]:
    """The ink px on long lines (exactly `scores.long_lines`) and, of them, those whose stroke is heavy (median width
    >= HEAVY_FT): the same assignment of each ink px to its nearest centre-line px in its own piece of ink."""
    s = strokes.strokes(ink, ft_per_px, scores._page_px)
    off = s["off"]
    long_out, heavy_out = np.zeros(ink.shape, bool), np.zeros(ink.shape, bool)
    if len(off) < 2:
        return long_out, heavy_out
    stroke = np.repeat(np.arange(len(off) - 1), np.diff(off))
    on_path = np.ones(len(stroke), bool)
    on_path[off[:-1]] = on_path[off[1:] - 1] = False
    col = np.floor(s["xy"][on_path, 0]).astype(np.intp)
    row = np.floor(s["xy"][on_path, 1]).astype(np.intp)
    is_long_stroke = (s["length"] >= scores.LONG_LINE_FT) & (s["wmed"] <= scores.THIN_LINE_FT)
    long = is_long_stroke[stroke[on_path]]
    heavy = (is_long_stroke & (s["wmed"] >= HEAVY_FT))[stroke[on_path]]
    centre = np.zeros(ink.shape, bool)
    centre[row, col] = True
    on_long = np.zeros(ink.shape, bool)
    on_long[row[long], col[long]] = True
    on_heavy = np.zeros(ink.shape, bool)
    on_heavy[row[heavy], col[heavy]] = True
    _, nearest = cv2.distanceTransformWithLabels((~centre).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    _, piece = cv2.connectedComponents(ink.view(np.uint8), connectivity=8)
    n = int(nearest.max()) + 1
    is_long, is_heavy = np.zeros(n, bool), np.zeros(n, bool)
    is_long[nearest[centre]] = on_long[centre]
    is_heavy[nearest[centre]] = on_heavy[centre]
    piece_of = np.zeros(n, np.int32)
    piece_of[nearest[centre]] = piece[centre]
    same = piece_of[nearest] == piece
    return ink & is_long[nearest] & same, ink & is_heavy[nearest] & same


def lines_cached(sd: data.SheetData) -> tuple[np.ndarray, np.ndarray]:
    """`line_masks` of a sheet, kept in OUT/lines (packed bits) under its `scores.line_key`."""
    f = OUT / "lines" / sd.series / f"{sd.sheet}.npz"
    key = repr(scores.line_key(sd) + (HEAVY_FT,))
    if f.is_file():
        with np.load(f) as z:
            if str(z["key"]) == key:
                return (np.unpackbits(z["long"], axis=1, count=sd.width).view(bool),
                        np.unpackbits(z["heavy"], axis=1, count=sd.width).view(bool))
    long, heavy = line_masks(data.page_ink(sd), sd.ft_per_px)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, long=np.packbits(long, axis=1), heavy=np.packbits(heavy, axis=1), key=np.array(key))
    return long, heavy


def disc_mask(sd: data.SheetData, radius: int) -> np.ndarray:
    """The px within `radius` of one of Sol's points, painted as the training paints them (`data._paint`)."""
    return data._paint(np.zeros((0, 4, 2), np.int32), np.zeros(0, np.int32), 0, 0, sd.height, sd.width,
                       sd.points, radius) > 0


def other_mask(sd: data.SheetData) -> np.ndarray:
    return data._paint(sd.other, np.ones(len(sd.other), np.int32), 0, 0, sd.height, sd.width) > 0


def sol_distance(sd: data.SheetData) -> np.ndarray:
    """Each page px's distance to the sheet's nearest Sol point (float32; inf without any on the page)."""
    h, w = sd.height, sd.width
    seed = np.ones((h, w), np.uint8)
    p = sd.points
    p = p[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)]
    if not len(p):
        return np.full((h, w), np.inf, np.float32)
    seed[p[:, 1], p[:, 0]] = 0
    return cv2.distanceTransform(seed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def banded(sd: data.SheetData, area) -> np.ndarray:
    return np.concatenate([data.in_area(sd, area, y0, 0, min(data.BAND, sd.height - y0), sd.width)
                           for y0 in range(0, sd.height, data.BAND)])


def footprint_mask(sd: data.SheetData) -> np.ndarray:
    """The px whose map point (nearest placement sample, every PLACEMENT_STEP px) is in the sheet's warped
    footprint."""
    poly = shapely.from_geojson((data.warp_dir(sd.sheet, sd.series) / "warped.footprint.geojson").read_text())
    inside = shapely.contains_xy(poly, sd.world[..., 0], sd.world[..., 1])
    step = data.PLACEMENT_STEP
    r = np.minimum(np.rint(np.arange(sd.height) / step).astype(int), inside.shape[0] - 1)
    c = np.minimum(np.rint(np.arange(sd.width) / step).astype(int), inside.shape[1] - 1)
    return inside[r][:, c]


_AREA = None


def answered_area():
    global _AREA
    if _AREA is None:
        _AREA = reader_stage.load_block_area(HELD["blocks"])
    return _AREA


# --- part 1: the 2000 sheets ----------------------------------------------------------------------------------------

def load_nets() -> dict:
    nets = {"trial": predict.load_model(TRIAL_MODEL, "cpu")}
    for q in paths.QUARTERS:
        nets[f"r_{q}"] = predict.load_model(paths.READER_TEXT_DIR / reader_stage.MODELS_DIR / f"{q}.pt", "cpu")
    return nets


def chances(sd: data.SheetData, ink: np.ndarray, quarter: np.ndarray, nets: dict) -> dict[str, np.ndarray]:
    name = sd.sheet
    got = {"all": cv2.imread(str(paths.TEXT_CHANCE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE),
           "held": cv2.imread(str(paths.READER_TEXT_DIR / name / reader_stage.CHANCE_FILE), cv2.IMREAD_GRAYSCALE)}
    old = np.zeros_like(got["all"])
    for q, qn in enumerate(paths.QUARTERS):
        c = cv2.imread(str(paths.SOLVER_DIR / f"text-{qn}" / "chance" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        old[quarter == q] = c[quarter == q]
    got["old"] = old
    cpu_dir = OUT / "chance"
    for k, net in nets.items():
        f = cpu_dir / k / f"{name}.png"
        if f.is_file():
            got[k] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        else:
            got[k] = predict.chance(net, ink, "cpu")
            f.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(f), got[k])
    for k, v in got.items():
        if v is None or v.shape != ink.shape:
            raise ValueError(f"{name}: chance {k} missing or of another shape")
    return got


def sheet_counts(name: str, nets: dict) -> None:
    t0 = time.perf_counter()
    sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
    h, w = sd.height, sd.width
    ink = data.page_ink(sd)
    word = data._words(sd, 0, 0, h, w) > 0
    other = other_mask(sd)
    dist = sol_distance(sd)
    disc_r, disc_s = disc_mask(sd, RADIUS), disc_mask(sd, SMALL)
    quarter = reader_stage.page_quarters(sd)
    val = scores.scored_px(sd, "validation")
    ans = banded(sd, answered_area())
    foot = footprint_mask(sd)
    long, heavy = lines_cached(sd)
    near = cv2.distanceTransform((~(word | other)).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sets = {
        "text": ink & word,
        "nonbox": ink & ~word,
        "nonbox_other": ink & ~word & other,
        "lines": long & (near > MARGIN),
        "heavy": long & heavy & (near > MARGIN),
        "lines_m0": long & ~word & ~other,
        "lines_m25": long & (near > 25),
        "heavy_nb": long & heavy & ~word,
        "heavy_nb_other": long & heavy & ~word & other,
    }
    zone = np.digitize(dist, ZONE_EDGES, right=True).astype(np.int32)
    region = (quarter.astype(np.int32) * 8 + val * 4 + ans * 2 + foot).astype(np.int32)
    ch = chances(sd, ink, quarter, nets)
    n_keys = N_REGIONS * len(SETS) * len(ZONES) * 256
    hist = {k: np.zeros(n_keys, np.int64) for k in DETECTORS}
    for s, (sname, mask) in enumerate(sets.items()):
        idx = np.flatnonzero(mask)
        base = ((region.ravel()[idx] * len(SETS) + s) * len(ZONES) + zone.ravel()[idx]) * 256
        for k in DETECTORS:
            hist[k] += np.bincount(base + ch[k].ravel()[idx], minlength=n_keys)
    # sanity: the CPU scoring of each held-out model against the built chance on its own quarter's ink
    sanity = {}
    for q, qn in enumerate(paths.QUARTERS):
        m = ink & (quarter == q)
        if m.any():
            d = np.abs(ch[f"r_{qn}"][m].astype(int) - ch["held"][m].astype(int))
            sanity[qn] = {"ink_px": int(m.sum()), "mean_abs": float(d.mean()), "max_abs": int(d.max()),
                          "call_differs": float(((ch[f"r_{qn}"][m] >= 128) != (ch["held"][m] >= 128)).mean())}
    painted = {"radius_disagree_px": int(((dist <= RADIUS) != disc_r).sum()),
               "small_disagree_px": int(((dist <= SMALL) != disc_s).sum()),
               "radius_px": int(disc_r.sum()), "small_px": int(disc_s.sum())}
    out = OUT / "counts"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **{k: v.astype(np.int32) for k, v in hist.items()})
    meta = {"sheet": name, "shape": [h, w], "ft_per_px": sd.ft_per_px, "points": int(len(sd.points)),
            "words": int(len(sd.words)), "other_boxes": int(len(sd.other)), "sanity": sanity, "painted": painted,
            "page_px": h * w, "ink_px": int(ink.sum()), "disc_page_share": float(disc_r.mean()),
            "seconds": time.perf_counter() - t0}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(name, f"{meta['seconds']:.0f} s", json.dumps(sanity), flush=True)


def run_sheets(names: list[str]) -> None:
    nets = load_nets()
    for name in names or list(paths.SHEETS):
        sheet_counts(name, nets)


# --- part 2: every training sheet's shares, and what each training's loss read ---------------------------------

def load_training_sheets() -> list[data.SheetData]:
    return data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)     # placement read, 30-px discs


def run_pages() -> None:
    """Every training sheet: its long and heavy lines (cached) and the ink not in a Terra word by where it is."""
    sheets = load_training_sheets()
    rows = []
    for sd in sheets:
        t0 = time.perf_counter()
        ink = data.page_ink(sd)
        word = data._words(sd, 0, 0, sd.height, sd.width) > 0
        other = other_mask(sd)
        disc_r, disc_s = disc_mask(sd, RADIUS), disc_mask(sd, SMALL)
        long, heavy = lines_cached(sd)
        nb = ink & ~word
        row = {"series": sd.series, "sheet": sd.sheet, "ink": int(ink.sum()), "text_ink": int((ink & word).sum()),
               "points": int(len(sd.points)), "page_px": sd.height * sd.width,
               "disc_r_page": int(disc_r.sum()), "disc_s_page": int(disc_s.sum())}
        for sname, m in (("nonbox", nb), ("long_nb", long & ~word), ("heavy_nb", long & heavy & ~word),
                         ("text_long", long & word)):
            row[sname] = int(m.sum())
            row[f"{sname}_disc_r"] = int((m & disc_r).sum())
            row[f"{sname}_disc_s"] = int((m & disc_s).sum())
            row[f"{sname}_other"] = int((m & other).sum())
            row[f"{sname}_unknown_r"] = int((m & (disc_r | other)).sum())
            row[f"{sname}_unknown_s"] = int((m & (disc_s | other)).sum())
        rows.append(row)
        print(sd.series, sd.sheet, f"{time.perf_counter() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pages.json").write_text(json.dumps(rows, indent=1) + "\n")


def config_sheets(config: str, sheets: list[data.SheetData]) -> list[data.SheetData]:
    """The sheets as a training read them: `all` the stage's (30-px discs, nothing held out); a quarter's name the
    held-out detector's (RADIUS discs, that quarter and the answered blocks held out)."""
    if config == "all":
        return sheets
    q = paths.QUARTERS.index(config)
    sheets = [dataclasses.replace(sd, point_radius=RADIUS) for sd in sheets]
    return data.hold_out(sheets, (data.Quarter(q), answered_area()))


def draws(sheets, pools, recipe, steps: int) -> list[tuple]:
    """The training's crops, as `train.draw` draws them from the seeded generator: (step, k, from_text, sheet index,
    row, col, scale, turns)."""
    rng = np.random.default_rng(SEED)
    out = []
    for step in range(steps):
        for k in range(recipe.batch):
            from_text = rng.random() < train.TEXT_SHARE
            pool = pools[0] if from_text else pools[1]
            i, y, x = pool[rng.integers(len(pool))]
            scale = float(np.exp(rng.uniform(np.log(train.SCALE[0]), np.log(train.SCALE[1]))))
            out.append((step, k, bool(from_text), i, y, x, scale, int(rng.integers(4))))
    return out


TALLY = ("crops", "pos", "neg", "neg_ink", "neg_paper", "lost_unknown", "lost_unknown_ink", "lost_disc",
         "lost_disc_ink", "lost_other_only", "lost_held", "pos_ink", "neg_long", "neg_heavy", "lost_long",
         "lost_heavy", "lost_disc_heavy", "nb_px", "nb_ink")


def crop_tally(sd: data.SheetData, y: int, x: int, size: int, scale: float, long, heavy) -> dict:
    """What the loss of one crop reads (`train.loss_weight` of "ink+1px" on `data.crop_targets`), split."""
    src = int(round(size / scale))
    y0 = int(np.clip(y - src // 2, 0, sd.height - src))
    x0 = int(np.clip(x - src // 2, 0, sd.width - src))
    word = data._words(sd, y0, x0, src, src)
    disc = data._paint(np.zeros((0, 4, 2), np.int32), np.zeros(0, np.int32), y0, x0, src, src, sd.points,
                       sd.point_radius) > 0
    other = data._paint(sd.other, np.ones(len(sd.other), np.int32), y0, x0, src, src) > 0
    parts = {"ink": data._ink_window(sd, y0, x0, src, src), "text": word > 0, "disc": disc, "other": other,
             "held": data._held(sd, y0, x0, src, src), "long": long[y0:y0 + src, x0:x0 + src],
             "heavy": heavy[y0:y0 + src, x0:x0 + src]}
    if src != size:
        parts = {k: cv2.resize(v.view(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).view(bool)
                 for k, v in parts.items()}
    ink, text = parts["ink"], parts["text"]
    unknown = (parts["disc"] | parts["other"]) & ~text
    held = parts["held"]
    known = ~unknown & ~held
    nb = train._next_to(ink[None])[0]
    wt = known & nb
    lost = nb & unknown & ~held
    return {"crops": 1, "pos": int((wt & text).sum()), "neg": int((wt & ~text).sum()),
            "pos_ink": int((wt & text & ink).sum()),
            "neg_ink": int((wt & ~text & ink).sum()), "neg_paper": int((wt & ~text & ~ink).sum()),
            "lost_unknown": int(lost.sum()), "lost_unknown_ink": int((lost & ink).sum()),
            "lost_disc": int((nb & parts["disc"] & ~text & ~held).sum()),
            "lost_disc_ink": int((ink & parts["disc"] & ~text & ~held).sum()),
            "lost_other_only": int((nb & parts["other"] & ~parts["disc"] & ~text & ~held).sum()),
            "lost_held": int((nb & held).sum()),
            "neg_long": int((wt & ~text & parts["long"]).sum()), "neg_heavy": int((wt & ~text & parts["heavy"]).sum()),
            "lost_long": int((lost & parts["long"]).sum()), "lost_heavy": int((lost & parts["heavy"]).sum()),
            "lost_disc_heavy": int((parts["heavy"] & parts["disc"] & ~text & ~held).sum()),
            "nb_px": int((nb & ~held).sum()), "nb_ink": int((ink & ~held).sum())}, known


def run_training(config: str, steps: int, base: list[data.SheetData]) -> None:
    """Re-draw a training's crops (the first `steps` steps, from its seed: the very crops it trained on) and tally
    what its loss read, by crop kind (text-centred or ink-centred), by map, and in all."""
    t0 = time.perf_counter()
    recipe = stage.DEFAULT_RECIPE
    sheets = config_sheets(config, base)
    pools = train.centre_pools(sheets)
    print(config, "pools", len(pools[0]), len(pools[1]), f"{time.perf_counter() - t0:.0f} s", flush=True)
    crops = draws(sheets, pools, recipe, steps)
    # check: the first batch as `train.draw` draws it from the same generator, crop for crop
    first = train.draw(sheets, pools, recipe, np.random.default_rng(SEED))
    by_sheet: dict[int, list] = {}
    for c in crops:
        by_sheet.setdefault(c[3], []).append(c)
    tallies: dict[tuple, dict] = {}
    checked = 0
    for i in sorted(by_sheet):
        sd = sheets[i]
        long, heavy = lines_cached(sd)
        for (step, k, from_text, _, y, x, scale, turns) in by_sheet[i]:
            t, known = crop_tally(sd, y, x, recipe.crop, scale, long, heavy)
            if step == 0:
                ref = first[k]
                if not np.array_equal(np.rot90(known, turns), ref["known"]):
                    raise AssertionError(f"{config}: crop {k} of step 0 differs from train.draw's")
                checked += 1
            for key in (("all",), ("kind", "text" if from_text else "ink"), ("series", sd.series),
                        ("series_kind", sd.series, "text" if from_text else "ink")):
                acc = tallies.setdefault(key, dict.fromkeys(TALLY, 0))
                for f, v in t.items():
                    acc[f] += v
    if checked != recipe.batch:
        raise AssertionError(f"{config}: checked {checked} crops of step 0")
    out = {"config": config, "steps": steps, "of_steps": recipe.iterations, "batch": recipe.batch,
           "pools": [len(pools[0]), len(pools[1])], "radius": sheets[0].point_radius,
           "tallies": [{"key": list(k), **v} for k, v in tallies.items()], "seconds": time.perf_counter() - t0}
    (OUT / "training").mkdir(parents=True, exist_ok=True)
    (OUT / "training" / f"{config}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(config, f"{out['seconds']:.0f} s", flush=True)


# --- pictures ------------------------------------------------------------------------------------------------------

WIN = 640


def _panel(page_ink, chance, word_rects, other_rects, pts, title) -> np.ndarray:
    """One window: ink called text (>= 0.5) red, other ink grey; Terra's words green, other passes' boxes cyan; Sol's
    points blue with their 30-px circle; paper farther than RADIUS from every Sol point pale yellow."""
    img = np.full(page_ink.shape + (3,), 255, np.uint8)
    inside = np.zeros(page_ink.shape, np.uint8)
    for x, y in pts:
        cv2.circle(inside, (int(x), int(y)), RADIUS, 1, -1)
    img[inside == 0] = (190, 245, 255)            # outside every RADIUS disc: pale yellow
    img[page_ink] = (150, 150, 150)
    img[page_ink & (chance >= 128)] = (0, 0, 230)
    for r in other_rects:
        cv2.polylines(img, [r.reshape(-1, 1, 2)], True, (220, 200, 0), 1)
    for r in word_rects:
        cv2.polylines(img, [r.reshape(-1, 1, 2)], True, (0, 170, 0), 1)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 6, (200, 0, 0), -1)
        cv2.circle(img, (int(x), int(y)), SMALL, (200, 0, 0), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_pictures() -> None:
    """Windows of WIN px: the three where the held-out detectors call the most clean not-text ink text inside the
    discs, the one where they call the most far from them, and two drawn at random (seeded) from windows holding
    clean not-text ink inside the discs; each shown by the held-out detector, the held-out models in-sample (the model
    of the opposite quarter), the trial and the all-sheets detector."""
    cand = []
    per = {}
    for name in paths.SHEETS:
        sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
        ink = data.page_ink(sd)
        word = data._words(sd, 0, 0, sd.height, sd.width) > 0
        other = other_mask(sd)
        near = cv2.distanceTransform((~(word | other)).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        long, _ = lines_cached(sd)
        clean = long & (near > MARGIN)
        dist = sol_distance(sd)
        held = cv2.imread(str(paths.READER_TEXT_DIR / name / reader_stage.CHANCE_FILE), cv2.IMREAD_GRAYSCALE)
        inside_called = (clean & (dist <= RADIUS) & (held >= 128)).astype(np.int64)
        far_called = (clean & (dist > 2 * RADIUS) & (held >= 128)).astype(np.int64)
        inside_any = (clean & (dist <= RADIUS)).astype(np.int64)
        for y in range(0, sd.height - WIN + 1, WIN // 2):
            for x in range(0, sd.width - WIN + 1, WIN // 2):
                s = np.s_[y:y + WIN, x:x + WIN]
                cand.append((name, y, x, int(inside_called[s].sum()), int(far_called[s].sum()),
                             int(inside_any[s].sum())))
        per[name] = sd
    rng = np.random.default_rng(SEED)
    picks = []

    def free(c):
        return all(c[0] != p[0] or abs(c[1] - p[1]) >= WIN or abs(c[2] - p[2]) >= WIN for p, _ in picks)

    for c in sorted(cand, key=lambda c: -c[3]):
        if len([p for p in picks if p[1] == "most inside"]) == 3:
            break
        if free(c):
            picks.append((c, "most inside"))
    for c in sorted(cand, key=lambda c: -c[4]):
        if free(c):
            picks.append((c, "most far"))
            break
    pool = [c for c in cand if c[5] > 2000 and free(c)]
    for j in rng.choice(len(pool), 2, replace=False):
        picks.append((pool[j], "random inside"))
    out = OUT / "pictures"
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for n, ((name, y, x, ic, fc, ia), why) in enumerate(picks):
        sd = per[name]
        ink = data.page_ink(sd)[y:y + WIN, x:x + WIN]
        q = int(reader_stage.page_quarters(sd)[y + WIN // 2, x + WIN // 2])
        opp = paths.QUARTERS[3 - q]
        shift = np.array([x, y], np.int32)
        words = [r - shift for r in sd.words]
        others = [r - shift for r in sd.other]
        pts = [p - shift for p in sd.points]
        ch = {"held": cv2.imread(str(paths.READER_TEXT_DIR / name / reader_stage.CHANCE_FILE), cv2.IMREAD_GRAYSCALE),
              "in": cv2.imread(str(OUT / "chance" / f"r_{opp}" / f"{name}.png"), cv2.IMREAD_GRAYSCALE),
              "trial": cv2.imread(str(OUT / "chance" / "trial" / f"{name}.png"), cv2.IMREAD_GRAYSCALE),
              "all": cv2.imread(str(paths.TEXT_CHANCE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)}
        val = bool(scores.scored_px(sd, "validation")[y + WIN // 2, x + WIN // 2])
        titles = {"held": f"held-out ({paths.QUARTERS[q]}, r{RADIUS}, out)", "in": f"model {opp} (r{RADIUS}, in)",
                  "trial": f"trial (r{SMALL}, {'out' if val else 'in'})", "all": f"all-sheets (r{SMALL}, in)"}
        panels = [_panel(ink, ch[k][y:y + WIN, x:x + WIN], words, others, pts, titles[k]) for k in ch]
        gap = np.full((WIN, 8, 3), 255, np.uint8)
        row = np.concatenate([panels[0], gap, panels[1], gap, panels[2], gap, panels[3]], axis=1)
        f = out / f"{n}-{why.replace(' ', '-')}-{name[-3:]}-{y}-{x}.png"
        cv2.imwrite(str(f), row)
        listing.append({"file": f.name, "why": why, "sheet": name, "y": y, "x": x, "quarter": paths.QUARTERS[q],
                        "validation": val, "clean_inside_called_by_held": ic, "clean_far_called_by_held": fc,
                        "clean_inside": ia})
    (out / "pictures.json").write_text(json.dumps(listing, indent=1) + "\n")
    print(json.dumps(listing, indent=1))
    radius_word()


def radius_word() -> None:
    """The word that set the held-out detectors' disc radius (held.json `longest`): its rectangle (red) and the other
    Terra words (green) drawn on its page, 300 px around its centre."""
    lw = HELD["unknown_radius"]["longest"]
    sd = data.sheet_data(lw["series"], lw["sheet"], hold_out_validation=False, placement=False)
    found = data.boxes.terra_boxes(sd.series, sd.sheet)
    code = data.KINDS["zoning code"]
    cx, cy = lw["centre"]
    j = min((j for j in np.flatnonzero(sd.kinds == code)),
            key=lambda j: float(np.hypot(*(sd.words[j].mean(axis=0) - [cx, cy]))))
    y0, x0 = max(cy - 300, 0), max(cx - 300, 0)
    gray = data.load_page(sd.sheet, sd.series)[y0:cy + 300, x0:cx + 300]
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for k in range(len(sd.words)):
        colour = (0, 0, 230) if k == j else (0, 170, 0)
        rect = (sd.words[k] - [x0, y0]).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [rect], True, colour, 2 if k == j else 1)
    cv2.imwrite(str(OUT / "pictures" / "radius-word.png"), img)
    print("radius word", found[j]["text"], found[j]["kind"], found[j]["box"], found[j].get("angle_deg"),
          "first edge", float(np.hypot(*(sd.words[j][1] - sd.words[j][0]).astype(float))))


# --- the tables -----------------------------------------------------------------------------------------------------

ZONE_GROUPS = {"inside <=30": [0], f"inside 30-{RADIUS}": [1, 2, 3], f"ring {RADIUS}-{2 * RADIUS}": [4, 5],
               f"far >{2 * RADIUS}": [6, 7]}
CALLS = {"0.5": 128, "0.9": 230}


def load_counts() -> dict[str, np.ndarray]:
    """Every 2000 sheet's histograms summed: detector -> (quarter, val, ans, foot, set, zone, chance)."""
    total = None
    for name in paths.SHEETS:
        with np.load(OUT / "counts" / f"{name}.npz") as z:
            got = {k: z[k].astype(np.int64) for k in DETECTORS}
        total = got if total is None else {k: total[k] + got[k] for k in DETECTORS}
    return {k: v.reshape(4, 2, 2, 2, len(SETS), len(ZONES), 256) for k, v in total.items()}


def cell(h: dict, det: str, s: str, quarters, zones, val=(0, 1), ans=(0, 1), foot=(1,)) -> tuple[int, dict]:
    """(px, {call level: share called}) of one detector on one set, summed over the given regions and zones.  `det`
    "insample" is the held-out detectors scored on the quarters they trained on: for each px, the three models whose
    held-out quarter is not the px's, pooled."""
    si = SETS.index(s)
    n = 0
    called = dict.fromkeys(CALLS, 0)
    mean = 0
    for q in quarters:
        dets = [f"r_{qn}" for j, qn in enumerate(paths.QUARTERS) if j != q] if det == "insample" else [det]
        for d in dets:
            sub = h[d][q][np.ix_(val, ans, foot)][..., si, :, :][..., zones, :].sum(axis=(0, 1, 2, 3))
            n += int(sub.sum()) / len(dets)
            for lvl, c in CALLS.items():
                called[lvl] += int(sub[c:].sum()) / len(dets)
            mean += float((sub * np.arange(256)).sum()) / 255 / len(dets)
    n = int(round(n))
    return n, {**{lvl: (c / n if n else None) for lvl, c in called.items()}, "mean": mean / n if n else None}


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def run_report() -> None:
    h = load_counts()
    tables = {}
    md = []
    q_all = list(range(4))
    # 1. the spatial test, per held-out quarter, on the footprint
    det_cols = (("held", "held-out (r328, out)"), ("insample", "held-out models in-sample (r328, in)"),
                ("all", "all-sheets (r30, in)"), ("old", "old quarter models (r30, out, old recipe)"))
    for s in ("lines", "heavy", "text", "nonbox"):
        rows = []
        for qi, qn in [(i, n) for i, n in enumerate(paths.QUARTERS)] + [(None, "all")]:
            qs = q_all if qi is None else [qi]
            for zname, zs in ZONE_GROUPS.items():
                row = {"quarter": qn, "zone": zname}
                for d, _ in det_cols:
                    n, r = cell(h, d, s, qs, zs)
                    row["px"] = n if d == "held" else row.get("px")
                    row[d] = r
                rows.append(row)
        tables[f"spatial_{s}"] = rows
    # 1b. the in-sample comparison without the answered blocks (held out of all four)
    for s in ("lines", "heavy", "text"):
        rows = []
        for zname, zs in ZONE_GROUPS.items():
            row = {"zone": zname}
            for d, _ in det_cols:
                n, r = cell(h, d, s, q_all, zs, ans=(0,))
                row["px"] = n if d == "held" else row.get("px")
                row[d] = r
            rows.append(row)
        tables[f"spatial_noanswered_{s}"] = rows
    # 2. the 2 x 2 on the validation area (no answered blocks): disc radius x held out or not, one recipe
    for s in ("lines", "heavy", "text", "nonbox"):
        rows = []
        for zname, zs in list(ZONE_GROUPS.items()) + [("every zone", list(range(len(ZONES))))]:
            row = {"zone": zname}
            for d in ("held", "insample", "trial", "all", "old"):
                n, r = cell(h, d, s, q_all, zs, val=(1,), ans=(0,))
                row["px"] = n if d == "held" else row.get("px")
                row[d] = r
            rows.append(row)
        tables[f"validation_{s}"] = rows
    # 3. the fine curve by distance, all quarters
    for s in ("lines", "heavy", "text"):
        rows = []
        for z, zname in enumerate(ZONES):
            row = {"zone": zname}
            for d in ("held", "insample", "all", "old", "trial"):
                n, r = cell(h, d, s, q_all, [z])
                row["px"] = n if d == "held" else row.get("px")
                row[d] = r
            rows.append(row)
        tables[f"curve_{s}"] = rows
    # 4. how the 2000 sheets' ink splits by zone (footprint), for the shares
    shares = {}
    for s in SETS:
        si = SETS.index(s)
        by_zone = h["held"][..., 1, si, :, :].sum(axis=(0, 1, 2, 4))
        shares[s] = {zname: int(by_zone[zs].sum()) for zname, zs in ZONE_GROUPS.items()}
    tables["px_by_zone"] = shares
    (OUT / "tables.json").write_text(json.dumps(tables, indent=1) + "\n")

    def emit(title, key, cols, zone_key="zone", lead=("quarter",)):
        md.append(f"\n#### {title}\n")
        heads = [c for c in lead if c in tables[key][0]] + ["zone", "ink px"] + \
            [f"{lab} >=0.5 / >=0.9" for _, lab in cols]
        md.append("| " + " | ".join(heads) + " |")
        md.append("|" + "---|" * len(heads))
        for row in tables[key]:
            vals = [str(row[c]) for c in lead if c in row] + [row[zone_key], f"{row['px']:,}"]
            vals += [f"{pct(row[d]['0.5'])} / {pct(row[d]['0.9'])}" for d, _ in cols]
            md.append("| " + " | ".join(vals) + " |")

    for s, label in (("lines", "clean not-text ink (long lines, >10 px from any box)"),
                     ("heavy", "heavy long lines (district weight), >10 px from any box"),
                     ("text", "true text (ink in Terra words)"), ("nonbox", "all ink outside Terra words")):
        emit(f"Spatial test, {label}: % called text", f"spatial_{s}", det_cols)
    for s in ("lines", "heavy", "text"):
        emit(f"Without the answered blocks, {s}", f"spatial_noanswered_{s}", det_cols, lead=())
    vcols = (("held", "r328 out (held)"), ("insample", "r328 in (held-out models)"), ("trial", "r30 out (trial)"),
             ("all", "r30 in (all-sheets)"), ("old", "r30 out, old recipe"))
    for s in ("lines", "heavy", "text", "nonbox"):
        emit(f"Validation area 2 x 2, {s}", f"validation_{s}", vcols, lead=())
    ccols = (("held", "held"), ("insample", "r328 in"), ("all", "all"), ("old", "old"), ("trial", "trial"))
    for s in ("lines", "heavy", "text"):
        emit(f"By distance to the nearest Sol point, {s}", f"curve_{s}", ccols, lead=())
    md += pooled_tables(h)
    md += training_tables()
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def pooled_tables(h: dict) -> list[str]:
    """Every quarter pooled (each px from the detector of its kind, see the module), on the footprint: the five
    detectors on each set by zone, with the disc's inside whole."""
    md = []
    zones = [("inside <=30", [0]), (f"inside 30-{RADIUS}", [1, 2, 3]),
             (f"inside <={RADIUS} (whole disc)", [0, 1, 2, 3]), (f"ring {RADIUS}-{2 * RADIUS}", [4, 5]),
             (f"far >{2 * RADIUS}", [6, 7])]
    dets = (("held", "held-out (r328, out)"), ("insample", "same models in-sample (r328, in)"),
            ("all", "all-sheets (r30, in)"), ("trial", "trial (r30)"), ("old", "old quarter models (r30, out)"))
    for s, label in (("lines", "clean not-text ink"), ("heavy", "district-weight lines"),
                     ("text", "true text"), ("nonbox", "all ink outside Terra words")):
        md.append(f"\n#### Pooled, {label} ({s}): % called text at >=0.5 / >=0.9\n")
        md.append("| zone | ink px | " + " | ".join(lab for _, lab in dets) + " |")
        md.append("|" + "---|" * (len(dets) + 2))
        for zname, zs in zones:
            vals = []
            px = 0
            for d, _ in dets:
                n, r = cell(h, d, s, range(4), zs)
                vals.append(f"{pct(r['0.5'])} / {pct(r['0.9'])}")
                if d == "held":
                    px = n
            md.append(f"| {zname} | {px:,} | " + " | ".join(vals) + " |")
    return md


def training_tables() -> list[str]:
    """What each training's loss read (training/<config>.json, scaled to all its steps) and where every training
    sheet's not-text ink lies (pages.json)."""
    md = []
    configs = [c for c in ("all", *paths.QUARTERS) if (OUT / "training" / f"{c}.json").is_file()]
    if configs:
        md.append("\n#### What each training's loss read (px of the 320 x 320 crops, all steps)\n")
        heads = ["training", "crops", "radius", "positives (text)", "negatives (not text)", "of them ink",
                 "of them paper next to ink", "neg : pos", "negatives on long lines", "on heavy lines",
                 "not-text ink+1px lost to unknown", "of it in discs", "heavy-line px lost to unknown",
                 "ink+1px lost to held-out areas"]
        md.append("| " + " | ".join(heads) + " |")
        md.append("|" + "---|" * len(heads))
        for c in configs:
            got = json.loads((OUT / "training" / f"{c}.json").read_text())
            f = got["of_steps"] / got["steps"]
            for t in got["tallies"]:
                if t["key"] == ["all"]:
                    a = {k: v * f for k, v in t.items() if k != "key"}
            nt = a["neg"] + a["lost_unknown"]
            hv = a["neg_heavy"] + a["lost_heavy"]
            md.append("| " + " | ".join([
                "all-sheets" if c == "all" else f"held-out {c}", f"{a['crops']:,.0f}", str(got["radius"]),
                f"{a['pos'] / 1e6:,.0f} M", f"{a['neg'] / 1e6:,.0f} M", f"{a['neg_ink'] / 1e6:,.0f} M",
                f"{a['neg_paper'] / 1e6:,.0f} M", f"{a['neg'] / a['pos']:.2f}",
                f"{a['neg_long'] / 1e6:,.1f} M", f"{a['neg_heavy'] / 1e6:,.1f} M",
                f"{pct(a['lost_unknown'] / nt)}%", f"{pct(a['lost_disc'] / nt)}%",
                f"{pct(a['lost_heavy'] / hv) if hv else '-'}%",
                f"{a['lost_held'] / 1e6:,.0f} M"]) + " |")
        md.append("\n#### The same by crop kind (text-centred / ink-centred)\n")
        md.append("| training | crop kind | crops | positives | negatives | neg : pos | "
                  "not-text ink+1px lost to unknown |")
        md.append("|---|---|---|---|---|---|---|")
        for c in configs:
            got = json.loads((OUT / "training" / f"{c}.json").read_text())
            f = got["of_steps"] / got["steps"]
            for t in got["tallies"]:
                if t["key"][0] == "kind":
                    nt = t["neg"] + t["lost_unknown"]
                    md.append(f"| {c} | {t['key'][1]} | {t['crops'] * f:,.0f} | {t['pos'] * f / 1e6:,.1f} M | "
                              f"{t['neg'] * f / 1e6:,.1f} M | {t['neg'] / max(t['pos'], 1):.2f} | "
                              f"{pct(t['lost_unknown'] / nt)}% |")
        md.append("\n#### The same, 2000 sheets' crops only\n")
        md.append("| training | crops | positives | negatives | neg : pos | not-text ink+1px lost to unknown | "
                  "heavy-line px lost to unknown |")
        md.append("|---|---|---|---|---|---|---|")
        for c in configs:
            got = json.loads((OUT / "training" / f"{c}.json").read_text())
            f = got["of_steps"] / got["steps"]
            for t in got["tallies"]:
                if t["key"] == ["series", paths.SERIES]:
                    nt = t["neg"] + t["lost_unknown"]
                    hv = t["neg_heavy"] + t["lost_heavy"]
                    md.append(f"| {c} | {t['crops'] * f:,.0f} | {t['pos'] * f / 1e6:,.1f} M | "
                              f"{t['neg'] * f / 1e6:,.1f} M | {t['neg'] / max(t['pos'], 1):.2f} | "
                              f"{pct(t['lost_unknown'] / nt)}% | {pct(t['lost_heavy'] / hv) if hv else '-'}% |")
    if (OUT / "pages.json").is_file():
        rows = json.loads((OUT / "pages.json").read_text())
        md.append("\n#### Where every training sheet's not-text ink lies (whole pages, ink only)\n")
        md.append(f"| map | sheets | ink outside Terra words | in r{SMALL} discs | in r{RADIUS} discs | "
                  f"in other passes' boxes | unknown at r{SMALL} | unknown at r{RADIUS} | heavy-line ink outside "
                  f"words | unknown at r{SMALL} | unknown at r{RADIUS} | Terra-word ink on long lines |")
        md.append("|" + "---|" * 12)
        for series in (*paths.TEXT_TRAIN_SERIES, "all"):
            rs = [r for r in rows if series in ("all", r["series"])]
            s = {k: sum(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k], int)}
            md.append(f"| {series} | {len(rs)} | {s['nonbox'] / 1e6:,.1f} M | "
                      f"{pct(s['nonbox_disc_s'] / s['nonbox'])}% | "
                      f"{pct(s['nonbox_disc_r'] / s['nonbox'])}% | {pct(s['nonbox_other'] / s['nonbox'])}% | "
                      f"{pct(s['nonbox_unknown_s'] / s['nonbox'])}% | {pct(s['nonbox_unknown_r'] / s['nonbox'])}% | "
                      f"{s['heavy_nb'] / 1e6:,.2f} M | "
                      f"{pct(s['heavy_nb_unknown_s'] / max(s['heavy_nb'], 1))}% | "
                      f"{pct(s['heavy_nb_unknown_r'] / max(s['heavy_nb'], 1))}% | "
                      f"{pct(s['text_long'] / max(s['text_ink'], 1))}% |")
    return md


# --- main ----------------------------------------------------------------------------------------------------------

def main(argv: list[str]) -> None:
    cmd, rest = argv[0], argv[1:]
    if cmd == "sheets":
        run_sheets(rest)
    elif cmd == "pages":
        run_pages()
    elif cmd == "training":
        steps = int(os.environ.get("STEPS", stage.DEFAULT_RECIPE.iterations))
        base = load_training_sheets()
        for config in rest:
            run_training(config, steps, base)
    elif cmd == "pictures":
        run_pictures()
    elif cmd == "radius-word":
        radius_word()
    elif cmd == "report":
        run_report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
