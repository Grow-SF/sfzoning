"""The learned reader's training data: each sheet's scan as printed, today's blocks placed on it, the solver lot under
every page px, and the lot table (the old reader's reads and the owner's answers).

Per sheet (`sheet_rasters`), in the scan's own px, turned upright:

  ink        the scan's luminance in [0, 1], 0 = ink (`load_luminance`: a grey scan's value / its full scale, a colour
             scan's Rec. 601 luma), every px exactly as scanned: nothing erased, cleaned, masked or thresholded
  block      the target of the learned street output: 1 where the px lies in today's blocks (the solver's lots with
             their enclosed holes filled: `binary_fill_holes(lot >= 0)` on the whole grid), 0 where it doesn't (a
             street, or no lot), 255 (unknown) outside the sheet's neatline and on the area a sheet drawn above it
             owns, so each place is labelled once
  lot        the solver lot index of the px (`lots.npz` `lot`, 4 ft grid px), -1 where none and wherever `block` is
             255 (each lot's px are pooled on one sheet only)
  crop       the neatline interior less the notes, as the merged page crops it (`hatching.load_crop`)
  ft_per_px  the placement's median scale (`SheetTransform.median_scale_ft_per_px`): measured on the placement, never
             on the reads

The placement is `data/work/<series>/<sheet>/transform.json` (the one the reads were made through; the warp's correction
in data/warp bends straight print), read as the merged page reads a sheet: `hatching.PageWorld` (the placement every
LATTICE px, bilinear between) and `hatching.page_values`.  Which sheet owns a px: the one whose neatline holds it and
that no neatline of a sheet drawn above it (the published draw order, `stitch.draw_order`) holds, as `hatching.owned`
says for points; a px is carried to an above sheet's page by that sheet's inverse placement at the lattice nodes,
bilinear between (the same lattice).  A sheet scanned on its side (`hatching.paper_turn`) is turned back by its quarter
turn, which is lossless (`turn_of`); every raster is computed in the scan's own px and turned with it.

The lot table (`lot_table`), one row per solver lot:

  lot, blklot, block, quarter (nw / ne / sw / se, `paths.QUARTERS`), centre_sheet (the sheet that owns the lot's
  centre: its representative point, `shapely.point_on_surface`, by `hatching.owned`; "" none)
  read_code, read_conf, read_via, read_sheet   the pattern reader's read (`district_pattern.json`; its off-sheet and
            no-tile marks and empty codes are no reads), one per lot: the read of `centre_sheet`, the others dropped
  n_reads   how many sheets read the lot (before keeping one)
  answer, answer_page, answer_case, stratum, n_answers   the owner's answer (truth by `answers1921.owner_answers`:
            ok the code shown, fix the fix, unzoned 2R, canttell / unsure dropped) on the lot of its blklot, else of the
            first lot of its footprint that is a solver lot (`answers1921`'s footprint matching); stratum random or
            disagree (the random page's cases), likely or lowconf (the pages').  A lot answered twice keeps the random
            page's answer; two answers that disagree stop the run.

Writes, under data/solver/experiments/hatch-cnn/data/<series>/ (`run`):
  <sheet>.npz     ink, block, lot, crop, ft_per_px, turn (np.rot90 k), scanned_shape, sheet, series
  lots.parquet    the lot table, with px: the lot's owned page px over every sheet
  summary.json    the sheets used and not, the draw order, per-sheet and per-quarter counts, reads per code, the
                  lots read on several sheets, answers per page and stratum, the legend swatches

$0: no API calls.  Deterministic: the same inputs give the same bytes (the npz members are dated, as `textread.store`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from scipy import ndimage

from sfzmaps import config
from sfzmaps.solver import answers1921, hatching, paths
from sfzmaps.solver.codes import canon
from sfzmaps.solver.textread.store import save_npz
from sfzmaps.transform import SheetTransform

EXPERIMENT = config.DATA_DIR / "solver" / "experiments" / "hatch-cnn"
LOOK = EXPERIMENT / "look"
SKIP_VIA = hatching.SKIP_VIA            # the reader's marks for a lot not on the sheet: no reads
LATTICE = hatching.LATTICE              # page px between the nodes where a placement is evaluated
UNKNOWN = 255                           # `block` where the place is not this sheet's to label
LUMA = (0.299, 0.587, 0.114)            # Rec. 601 luma (R, G, B): a colour scan's grey, as PIL and OpenCV take it
ROWS = 512                              # page rows laid at once (memory only: the result doesn't depend on it)
ANSWER_PAGES = {"1921": answers1921.PAGES}     # the owner's answer pages per series (none yet for the others)


@dataclass(frozen=True)
class Sources:
    """Where a series is read: the pages, placements and reads (work), the neatlines and the solver's lots (solver),
    the legend's swatch boxes, the owner's answers and their case files; `order`: the draw order, bottom first (None:
    the published map's)."""
    series: str
    work: Path
    solver: Path
    legends: Path
    answers: Path
    cases: Path
    order: tuple[str, ...] | None = None


def sources(series: str) -> Sources:
    return Sources(series, config.WORK_DIR / series, config.DATA_DIR / "solver" / series, hatching.LEGENDS_FILE,
                   answers1921.ANSWERS_DIR, answers1921.EXPERIMENT / "cases")


def data_dir(series: str) -> Path:
    return EXPERIMENT / "data" / series


# --- the scan -------------------------------------------------------------------------------------------------------

def load_luminance(path: Path) -> np.ndarray:
    """A page as scanned: float32 luminance in [0, 1], 0 = ink.  A grey scan's value over its full scale (1-bit and
    8-bit 1921 scans alike); a colour scan's (1948) Rec. 601 luma (`LUMA`, in double precision, banded for memory);
    an alpha channel is not part of the print."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(f"can't read {path}")
    if a.dtype.kind != "u":
        raise ValueError(f"{path}: {a.dtype} px, expected unsigned integers")
    full = float(np.iinfo(a.dtype).max)
    if a.ndim == 2:
        return a.astype(np.float32) / np.float32(full)
    if a.shape[2] <= 2:                                   # grey (+ alpha)
        return a[..., 0].astype(np.float32) / np.float32(full)
    out = np.empty(a.shape[:2], np.float32)
    for y0 in range(0, a.shape[0], ROWS):
        b, g, r = (a[y0:y0 + ROWS, :, k].astype(np.float64) for k in range(3))     # OpenCV reads BGR
        out[y0:y0 + ROWS] = (LUMA[0] * r + LUMA[1] * g + LUMA[2] * b) / full
    return out


def load_transform(src: Sources, sheet: str) -> SheetTransform:
    """The sheet's placement in data/work (not data/warp)."""
    folder = src.work / sheet
    return SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)


def frames(src: Sources) -> dict:
    return json.loads((src.solver / "page" / "frames.json").read_text())


def page_shape(src: Sources, sheet: str) -> tuple[int, int]:
    """(rows, cols) of the sheet's page.png as scanned, from its header."""
    with Image.open(src.work / sheet / "page.png") as im:
        w, h = im.size
    return h, w


def turn_of(quad, transform, shape: tuple[int, int]) -> int:
    """The `np.rot90` k that turns a scan upright: 0 for a sheet scanned upright; for one scanned on its side
    (`hatching.paper_turn`: its neatline wider than tall), the quarter turn, anticlockwise (1: its right edge comes to
    the top) or clockwise (3: its left edge), that brings the sheet's north nearer the top, as on the upright sheets
    (from its placement at the page's centre)."""
    if hatching.paper_turn(quad) == 0:
        return 0
    h, w = shape
    c = np.array([[(w - 1) / 2.0, (h - 1) / 2.0]])
    right = (transform.to_world(c + [LATTICE, 0.0]) - transform.to_world(c))[0]

    def off_north(v):
        return abs(float(np.arctan2(v[0], v[1])))

    return 1 if off_north(right) <= off_north(-right) else 3


def turn_box(box, shape: tuple[int, int], k: int) -> tuple[int, int, int, int]:
    """A box (x0, y0, x1, y1, ends exclusive) on a page of `shape` (rows, cols) as scanned, on the page turned by
    `np.rot90(page, k)`."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = shape
    k %= 4
    if k == 0:
        return x0, y0, x1, y1
    if k == 1:
        return y0, w - x1, y1, w - x0
    if k == 2:
        return w - x1, h - y1, w - x0, h - y0
    return h - y1, x0, h - y0, x1


def load_scan(series: str, sheet: str, src: Sources | None = None) -> np.ndarray:
    """The sheet's page as scanned (`load_luminance`), turned upright (`turn_of`)."""
    src = src or sources(series)
    lum = load_luminance(src.work / sheet / "page.png")
    k = turn_of(frames(src)[sheet], load_transform(src, sheet), lum.shape)
    return np.ascontiguousarray(np.rot90(lum, k))


# --- the sheets -----------------------------------------------------------------------------------------------------

@dataclass
class Sheet:
    """A sheet's geometry in its scanned px: the neatline quad, the page shape, the placement, the neatline crop and
    the crop's world bounds (ft)."""
    name: str
    quad: list
    shape: tuple[int, int]
    transform: SheetTransform
    crop: np.ndarray
    bbox: tuple[float, float, float, float]

    def above(self) -> hatching.Above:
        return hatching.Above(self.transform, self.crop, self.bbox)


def load_sheet(src: Sources, name: str, quad) -> Sheet:
    shape = page_shape(src, name)
    tr = load_transform(src, name)
    crop = hatching.load_crop(src, name, shape, quad)
    return Sheet(name, quad, shape, tr, crop, hatching.crop_bbox(tr, crop))


def sheet_names(src: Sources) -> list[str]:
    """The map sheets: a page, a placement (data/work) and a neatline (frames.json)."""
    fr = frames(src)
    return [d.name for d in sorted(p for p in src.work.iterdir() if p.is_dir())
            if (d / "page.png").is_file() and (d / "transform.json").is_file() and d.name in fr]


def not_used(src: Sources) -> dict[str, list[str]]:
    """The series' folders that are not map sheets, with what they lack."""
    fr = frames(src)
    out = {}
    for d in sorted(p for p in src.work.iterdir() if p.is_dir()):
        miss = [f for f in ("page.png", "transform.json") if not (d / f).is_file()]
        if d.name not in fr:
            miss.append("frame")
        if miss:
            out[d.name] = miss
    return out


def draw_order(src: Sources, sheets: list[str]) -> list[str]:
    """The sheets bottom first, as the published map draws them (`stitch.draw_order`, or `src.order`); any it doesn't
    rank at the bottom, by name."""
    if src.order is not None:
        ranked = [s for s in src.order if s in sheets]
    else:
        from sfzmaps import stitch
        ranked = [s for s in stitch.draw_order(src.series, stitch.load_scores(src.series)) if s in sheets]
    return sorted(set(sheets) - set(ranked)) + ranked


def overlaps(a: Sheet, b: Sheet) -> bool:
    return a.bbox[0] <= b.bbox[2] and a.bbox[2] >= b.bbox[0] and a.bbox[1] <= b.bbox[3] and a.bbox[3] >= b.bbox[1]


def drawn_above(name: str, order: list[str], sheets: dict[str, Sheet]) -> list[Sheet]:
    """The sheets drawn above `name` whose neatlines' bounds meet its own."""
    me = sheets[name]
    return [sheets[a] for a in order[order.index(name) + 1:] if overlaps(me, sheets[a])]


# --- the rasters ----------------------------------------------------------------------------------------------------

@dataclass
class SheetRasters:
    """A sheet's training rasters, turned upright (see the module)."""
    sheet: str
    ink: np.ndarray
    block: np.ndarray
    lot: np.ndarray
    crop: np.ndarray
    ft_per_px: float
    turn: int = 0
    scanned_shape: tuple[int, int] = (0, 0)


def solver_grid(src: Sources) -> tuple[np.ndarray, np.ndarray]:
    """The solver's lot raster and today's blocks (the lots with their enclosed holes, filled on the whole grid)."""
    with np.load(src.solver / "lots" / "lots.npz") as z:
        lot = z["lot"]
    return lot, ndimage.binary_fill_holes(lot >= 0)


def _bilinear(nodes: np.ndarray, pw: hatching.PageWorld, rows: slice, w: int) -> np.ndarray:
    """Values (rows x w x 2) at page px of the lattice `nodes` (at `pw`'s node px), bilinear; NaN where any of a px's
    four nodes is."""
    yy, xx = np.mgrid[rows, 0:w].astype(float)
    fx, fy = (xx - pw.nx0) / LATTICE, (yy - pw.ny0) / LATTICE
    j = np.clip(np.floor(fx).astype(np.int64), 0, nodes.shape[1] - 2)
    i = np.clip(np.floor(fy).astype(np.int64), 0, nodes.shape[0] - 2)
    ax, ay = (fx - j)[..., None], (fy - i)[..., None]
    return ((nodes[i, j] * (1 - ax) + nodes[i, j + 1] * ax) * (1 - ay)
            + (nodes[i + 1, j] * (1 - ax) + nodes[i + 1, j + 1] * ax) * ay)


def owned_px(pw: hatching.PageWorld, shape: tuple[int, int], crop: np.ndarray, above: list[Sheet]) -> tuple[np.ndarray, int]:
    """The page px a sheet labels: inside its neatline (`crop`) and inside no neatline of a sheet drawn above it
    (`hatching.owned`'s rule, every px): each px is carried to an above sheet's page by that sheet's inverse placement
    (`grid_to_pixel`) at this page's lattice nodes, bilinear between.  Nodes beyond reach of an above sheet's neatline
    bounds (twice the lattice's longest cell diagonal) aren't carried: no px of a cell they bound can lie in it.  Also
    returns the nodes the inverse placement could not place (their cells stay this sheet's)."""
    h, w = shape
    ok = np.asarray(crop, bool).copy()
    nodes = pw.nodes
    if nodes.shape[0] < 2 or nodes.shape[1] < 2:
        raise ValueError(f"a page of {shape} px has too few lattice nodes")
    diag = max(float(np.linalg.norm(nodes[1:, 1:] - nodes[:-1, :-1], axis=-1).max()),
               float(np.linalg.norm(nodes[1:, :-1] - nodes[:-1, 1:], axis=-1).max()))
    reach = 2.0 * diag
    flat = nodes.reshape(-1, 2)
    failed = 0
    for t in above:
        x0, y0, x1, y1 = t.bbox
        cand = ((flat[:, 0] >= x0 - reach) & (flat[:, 0] <= x1 + reach) &
                (flat[:, 1] >= y0 - reach) & (flat[:, 1] <= y1 + reach))
        if not cand.any():
            continue
        P = np.full(flat.shape, np.nan)
        P[cand] = t.transform.grid_to_pixel(flat[cand], t.shape)
        failed += int((cand & ~np.isfinite(P).all(axis=1)).sum())
        P = P.reshape(nodes.shape)
        for y0b in range(0, h, ROWS):
            rows = slice(y0b, min(h, y0b + ROWS))
            xy = _bilinear(P, pw, rows, w).reshape(-1, 2)
            ok[rows] &= ~hatching._in_crop(t.crop, xy).reshape(rows.stop - rows.start, w)
    return ok, failed


def _rasters(src: Sources, sh: Sheet, lot_cut: np.ndarray, block_cut: np.ndarray, offset, above: list[Sheet]):
    """A sheet's rasters (`SheetRasters`) and counts, from its cut of the solver's grid (`lot_cut`, `block_cut` from
    grid row and column `offset`)."""
    lum = load_luminance(src.work / sh.name / "page.png")
    if lum.shape != tuple(sh.shape):
        raise ValueError(f"{sh.name}: page {lum.shape}, header {sh.shape}")
    pw = hatching.PageWorld(sh.transform, sh.shape)
    lot_px, blk_px = hatching.page_values(pw, sh.shape, [(lot_cut, -1), (block_cut.astype(np.uint8), 0)], offset)
    mine, failed = owned_px(pw, sh.shape, sh.crop, above)
    block = np.where(mine, blk_px, UNKNOWN).astype(np.uint8)
    lot = np.where(mine, lot_px, -1).astype(np.int32)
    k = turn_of(sh.quad, sh.transform, sh.shape)

    def up(a):
        return np.ascontiguousarray(np.rot90(a, k))

    r = SheetRasters(sh.name, up(lum), up(block), up(lot), up(np.asarray(sh.crop, bool)),
                     float(sh.transform.median_scale_ft_per_px(sh.shape)), int(k), tuple(int(v) for v in sh.shape))
    stats = {"sheet": sh.name, "scanned_shape": list(r.scanned_shape), "shape": list(r.ink.shape), "turn": r.turn,
             "ft_per_px": r.ft_per_px, "above": [a.name for a in above], "px": int(lum.size),
             "crop_px": int(np.count_nonzero(sh.crop)), "owned_px": int(np.count_nonzero(mine)),
             "block_px": int(np.count_nonzero(block == 1)), "street_px": int(np.count_nonzero(block == 0)),
             "unknown_px": int(np.count_nonzero(block == UNKNOWN)), "lot_px": int(np.count_nonzero(lot >= 0)),
             "lots": int(len(np.unique(lot[lot >= 0]))), "unplaced_nodes": failed}
    return r, stats


def sheet_rasters(series: str, sheet: str, src: Sources | None = None) -> SheetRasters:
    """One sheet's training rasters (see the module)."""
    src = src or sources(series)
    names = sheet_names(src)
    if sheet not in names:
        raise KeyError(f"{sheet} is not a {series} map sheet (a page, a data/work placement and a frame)")
    order = draw_order(src, names)
    fr = frames(src)
    sheets = {s: load_sheet(src, s, fr[s]) for s in order[order.index(sheet):]}
    lot, blocks = solver_grid(src)
    me = sheets[sheet]
    r0, r1, c0, c1 = hatching._slice(me.transform, me.quad, lot.shape)
    return _rasters(src, me, lot[r0:r1, c0:c1], blocks[r0:r1, c0:c1], (r0, c0), drawn_above(sheet, order, sheets))[0]


def write_rasters(f: Path, r: SheetRasters, series: str) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    save_npz(f, ink=r.ink, block=r.block, lot=r.lot, crop=r.crop, ft_per_px=np.float64(r.ft_per_px),
             turn=np.int64(r.turn), scanned_shape=np.array(r.scanned_shape, np.int64), sheet=np.array(r.sheet),
             series=np.array(series))


def read_rasters(f: Path) -> SheetRasters:
    with np.load(f) as z:
        return SheetRasters(str(z["sheet"]), z["ink"], z["block"], z["lot"], z["crop"], float(z["ft_per_px"]),
                            int(z["turn"]), tuple(int(v) for v in z["scanned_shape"]))


# --- the legend -----------------------------------------------------------------------------------------------------

def swatches(series: str, src: Sources | None = None) -> list[tuple[str, str, tuple[int, int, int, int]]]:
    """The legend's swatches (`districts/legends.json`: the outer printed rectangle of each, ends exclusive, as
    `hatching` reads them), (sheet, code, box) on the sheet's page turned upright."""
    src = src or sources(series)
    leg = json.loads(Path(src.legends).read_text()).get(series)
    if not leg:
        return []
    sheet = leg["sheet"]
    shape = page_shape(src, sheet)
    fr = frames(src)
    k = turn_of(fr[sheet], load_transform(src, sheet), shape) if sheet in fr else 0
    return [(sheet, code, turn_box(box, shape, k)) for code, box in leg["swatches"].items()]


# --- the lot table --------------------------------------------------------------------------------------------------

def centre_owner(points: np.ndarray, sheets: list[Sheet]) -> tuple[np.ndarray, int]:
    """The sheet that owns each map point (n, 2), `sheets` bottom first: the one whose neatline holds it and no
    neatline of a sheet drawn above it does (`hatching.owned`); "" none.  Also the points two sheets both claimed (at a
    neatline's edge `owned` reads only within the bounds of a sheet above; the upper one keeps them)."""
    pts = np.asarray(points, float)
    owner = np.full(len(pts), "", dtype=object)
    finite = np.isfinite(pts).all(axis=1)
    twice = 0
    for i, sh in enumerate(sheets):
        x0, y0, x1, y1 = sh.bbox
        margin = 2.0 * sh.transform.median_scale_ft_per_px(sh.shape)       # the crop's nearest px reaches 1/2 px out
        cand = np.flatnonzero(finite & (pts[:, 0] >= x0 - margin) & (pts[:, 0] <= x1 + margin)
                              & (pts[:, 1] >= y0 - margin) & (pts[:, 1] <= y1 + margin))
        if not len(cand):
            continue
        px = sh.transform.grid_to_pixel(pts[cand], sh.shape)
        above = [s.above() for s in sheets[i + 1:] if overlaps(sh, s)]
        mine = cand[hatching.owned(px, pts[cand], sh.crop, above)]
        twice += int((owner[mine] != "").sum())
        owner[mine] = sh.name
    return owner, twice


def reads(src: Sources, sheets: list[str]) -> pd.DataFrame:
    """The pattern reader's reads on the sheets (sheet, blklot, code in the one spelling, confidence, via), less its
    off-sheet and no-tile marks and those with no code (as `hatchfit.reads`)."""
    rows = []
    for s in sheets:
        f = src.work / s / "district_pattern.json"
        if not f.is_file():
            continue
        for p in json.loads(f.read_text())["parcels"]:
            code = p.get("code")
            if not isinstance(code, str) or not code.strip() or p.get("via") in SKIP_VIA:
                continue
            rows.append({"sheet": s, "blklot": str(p["blklot"]), "code": canon(code),
                         "confidence": float(p["confidence"]), "via": str(p.get("via", ""))})
    return pd.DataFrame(rows, columns=["sheet", "blklot", "code", "confidence", "via"])


def _stratum(page: str, case: dict, series: str) -> str:
    """random or disagree (the random page's cases); otherwise the page's kind (likely, lowconf)."""
    s = case.get("stratum")
    if s and s != "all":
        return str(s)
    return page.removeprefix(f"{series}-") if page != series else page


def answers(src: Sources) -> pd.DataFrame:
    """The owner's answers kept (`answers1921.owner_answers`: blklot, block, truth, page, case, blklots) with each
    one's stratum; attrs as owner_answers'."""
    pages = ANSWER_PAGES.get(src.series, ())
    dirs = [src.answers / f"answers-{p}" for p in pages]
    files = [f for p in pages for f in (src.cases / f"{p}.json", src.cases / f"{p}-retired.json") if f.is_file()]
    a = answers1921.owner_answers(dirs, files)
    cases: dict[str, dict[str, dict]] = {}
    for f in files:
        page = cases.setdefault(answers1921._page_of(f), {})
        for c in json.loads(f.read_text())["cases"]:
            page.setdefault(c["id"], c)
    a["stratum"] = [_stratum(p, cases[p][c], src.series) for p, c in zip(a["page"], a["case"])]
    return a


def lot_table(series: str, src: Sources | None = None) -> pd.DataFrame:
    """One row per solver lot (see the module); `attrs` holds the counts of how the reads and answers were kept."""
    src = src or sources(series)
    table = pd.read_parquet(src.solver / "lots" / "table.parquet", columns=["blklot", "block_num", "geometry"])
    with np.load(src.solver / "lots" / "lots.npz") as z:
        quarter = z["quarter"].astype(np.int64)
    n = len(table)
    if len(quarter) != n:
        raise ValueError(f"lots.npz has {len(quarter)} quarters, the table {n} lots")
    blklot = table["blklot"].astype(str).to_numpy()
    names = np.array(paths.QUARTERS + ("",), dtype=object)
    out = pd.DataFrame({"lot": np.arange(n, dtype=np.int64), "blklot": blklot,
                        "block": table["block_num"].astype(str).to_numpy(),
                        "quarter": names[np.where(quarter >= 0, quarter, len(paths.QUARTERS))]})
    lots_of: dict[str, list[int]] = {}
    for i, b in enumerate(blklot):
        lots_of.setdefault(b, []).append(i)

    # the sheet owning each lot's centre
    used = sheet_names(src)
    order = draw_order(src, used)
    fr = frames(src)
    sheets = [load_sheet(src, s, fr[s]) for s in order]
    geoms = shapely.from_wkb(table["geometry"].to_numpy()) if n and isinstance(
        table["geometry"].iloc[0], (bytes, bytearray)) else np.asarray(table["geometry"].to_numpy())
    pts = shapely.point_on_surface(geoms)
    centres = np.c_[shapely.get_x(pts), shapely.get_y(pts)]
    owner, twice = centre_owner(centres, sheets)
    out["centre_sheet"] = owner.astype(str)

    # one read per lot: the centre's sheet's
    rd = reads(src, used)
    code = np.full(n, "", dtype=object)
    conf = np.full(n, np.nan)
    via = np.full(n, "", dtype=object)
    rsheet = np.full(n, "", dtype=object)
    sheets_read: list[set] = [set() for _ in range(n)]
    unmatched = 0
    repeated = 0
    for r in rd.itertuples(index=False):
        lots = lots_of.get(r.blklot)
        if not lots:
            unmatched += 1
            continue
        for i in lots:
            if r.sheet in sheets_read[i]:
                repeated += 1                         # the same sheet reading a lot twice: its first read stands
                continue
            sheets_read[i].add(r.sheet)
            if r.sheet == owner[i]:
                code[i], conf[i], via[i], rsheet[i] = r.code, r.confidence, r.via, r.sheet
    n_reads = np.array([len(s) for s in sheets_read], np.int64)
    out["read_code"], out["read_conf"], out["read_via"], out["read_sheet"] = (
        code.astype(str), conf, via.astype(str), rsheet.astype(str))
    out["n_reads"] = n_reads

    # the owner's answers
    ans = answers(src)
    priority = {p: k for k, p in enumerate([series] + [p for p in ANSWER_PAGES.get(series, ()) if p != series])}
    ans = ans.assign(_p=ans["page"].map(priority)).sort_values(["_p", "case"], kind="stable")
    a_code = np.full(n, "", dtype=object)
    a_page = np.full(n, "", dtype=object)
    a_case = np.full(n, "", dtype=object)
    a_strat = np.full(n, "", dtype=object)
    n_ans = np.zeros(n, np.int64)
    unattached = []
    for r in ans.itertuples(index=False):
        tried = [r.blklot] + [b for b in (r.blklots or ()) if b != r.blklot]
        hit = next((lots_of[b] for b in tried if b in lots_of), None)
        if hit is None:
            unattached.append(f"{r.page}/{r.case}")
            continue
        for i in hit:
            n_ans[i] += 1
            if a_code[i] == "":
                a_code[i], a_page[i], a_case[i], a_strat[i] = r.truth, r.page, r.case, r.stratum
            elif a_code[i] != r.truth:
                raise ValueError(f"lot {i} ({blklot[i]}): the owner answered {a_code[i]} ({a_page[i]}/{a_case[i]}) "
                                 f"and {r.truth} ({r.page}/{r.case})")
    out["answer"], out["answer_page"], out["answer_case"], out["stratum"] = (
        a_code.astype(str), a_page.astype(str), a_case.astype(str), a_strat.astype(str))
    out["n_answers"] = n_ans
    out.attrs = {
        "sheets": used, "draw_order": order, "centres_claimed_twice": twice,
        "lots_without_centre_sheet": int((owner == "").sum()),
        "reads": {"kept": int((out["read_code"] != "").sum()), "total": int(len(rd)), "unmatched_blklot": unmatched,
                  "repeated_on_one_sheet": repeated, "lots_read": int((n_reads > 0).sum()),
                  "lots_read_on_several_sheets": int((n_reads > 1).sum()),
                  "lots_read_but_not_by_centre_sheet": int(((n_reads > 0) & (out["read_code"] == "")).sum()),
                  "by_sheet": {s: int(v) for s, v in rd["sheet"].value_counts().sort_index().items()}},
        "answers": {"kept": int(len(ans)), "dropped": int(ans.attrs.get("dropped", 0)),
                    "unattached": unattached, "lots_answered_twice": int((n_ans > 1).sum()),
                    "by_page": answers1921_pages(ans)},
    }
    return out


def answers1921_pages(ans: pd.DataFrame) -> dict:
    return {p: {"dir_exists": bool(i["dir_exists"]), "read": int(i["read"]), "kept": int(i["kept"]),
                "dropped": int(i["dropped"])} for p, i in sorted(ans.attrs.get("pages", {}).items())}


# --- the run --------------------------------------------------------------------------------------------------------

def _init_worker():
    cv2.setNumThreads(1)


def _job(job) -> dict:
    """One sheet: its rasters written to `out`/<sheet>.npz; returns its counts and its lots' px."""
    src, name, order, lot_cut, block_cut, offset, out, n_lots = job
    fr = frames(src)
    sheets = {s: load_sheet(src, s, fr[s]) for s in order[order.index(name):]}
    r, stats = _rasters(src, sheets[name], lot_cut, block_cut, offset, drawn_above(name, order, sheets))
    write_rasters(out / f"{name}.npz", r, src.series)
    stats["lot_px_per_lot"] = np.bincount(r.lot[r.lot >= 0], minlength=n_lots).astype(np.int64)
    return stats


def _map(fn, jobs, workers: int):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
        return list(pool.imap_unordered(fn, jobs))


def run(series: str, out: Path | None = None, workers: int = 4, src: Sources | None = None,
        only: list[str] | None = None) -> dict:
    """Every map sheet's rasters (or `only` those; the draw order still takes every sheet), the lot table and the
    summary, into `out` (`data_dir(series)`)."""
    src = src or sources(series)
    out = Path(out or data_dir(series))
    names = sheet_names(src)
    if not names:
        raise FileNotFoundError(f"no {series} sheets with a page, a placement and a frame under {src.work}")
    order = draw_order(src, names)
    fr = frames(src)
    lot, blocks = solver_grid(src)
    table = lot_table(series, src)
    n_lots = len(table)
    jobs = []
    for s in order:
        if only and s not in only:
            continue
        tr = load_transform(src, s)
        r0, r1, c0, c1 = hatching._slice(tr, fr[s], lot.shape)
        jobs.append((src, s, order, lot[r0:r1, c0:c1], blocks[r0:r1, c0:c1], (r0, c0), out, n_lots))
    jobs.sort(key=lambda j: -int(np.prod(page_shape(src, j[1]))))          # the largest first: the pool ends sooner
    out.mkdir(parents=True, exist_ok=True)
    done = sorted(_map(_job, jobs, workers), key=lambda d: d["sheet"])
    px = np.zeros(n_lots, np.int64)
    for d in done:
        px += d.pop("lot_px_per_lot")
    table["px"] = px
    attrs, table.attrs = table.attrs, {}
    table.to_parquet(out / "lots.parquet", index=False)
    summary = {"series": series, "sheets_used": names, "not_used": not_used(src), "draw_order": order,
               "sheets": {d["sheet"]: {k: v for k, v in d.items() if k != "sheet"} for d in done},
               "quarters": quarter_counts(table), "reads_per_code": reads_per_code(table),
               "answers_per_page_stratum": answers_per_page(table),
               "lot_table": attrs, "swatches": [[s, c, list(b)] for s, c, b in swatches(series, src)]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def quarter_counts(t: pd.DataFrame) -> dict:
    out = {}
    for q in paths.QUARTERS:
        m = (t["quarter"] == q).to_numpy()
        has = m & (t["px"].to_numpy() > 0) if "px" in t else m
        out[q] = {"lots": int(m.sum()), "lots_with_px": int(has.sum()),
                  "px": int(t["px"].to_numpy()[m].sum()) if "px" in t else 0,
                  "lots_read": int((m & (t["read_code"] != "").to_numpy()).sum()),
                  "lots_read_with_px": int((has & (t["read_code"] != "").to_numpy()).sum()),
                  "lots_answered": int((m & (t["answer"] != "").to_numpy()).sum()),
                  "lots_answered_random": int((m & (t["stratum"] == "random").to_numpy()).sum())}
    return out


def reads_per_code(t: pd.DataFrame) -> dict:
    r = t[t["read_code"] != ""]
    return {c: int(v) for c, v in r["read_code"].value_counts().sort_index().items()}


def answers_per_page(t: pd.DataFrame) -> dict:
    a = t[t["answer"] != ""]
    out: dict = {}
    for (p, s), v in a.groupby(["answer_page", "stratum"]).size().sort_index().items():
        out.setdefault(p, {})[s] = int(v)
    return out


# --- the look picture -----------------------------------------------------------------------------------------------

TINT = {1: (255, 110, 0), 0: (0, 140, 255), UNKNOWN: (128, 128, 128)}    # BGR: blocks blue, not a block orange
ALPHA = {1: 0.40, 0: 0.40, UNKNOWN: 0.55}
ZOOM = 3                       # the zoom's px per scan px (nearest: each scan px a square)
ZOOM_SIDE = 520                # the zoom's side in scan px


def _tinted(r: SheetRasters) -> np.ndarray:
    grey = np.round(r.ink * 255).astype(np.uint8)
    img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for v, bgr in TINT.items():
        m = r.block == v
        img[m] = img[m] * (1 - ALPHA[v]) + np.array(bgr, np.float32) * ALPHA[v]
    return np.round(img).astype(np.uint8)


def _banner(img: np.ndarray, lines: list[str], scale: float) -> np.ndarray:
    h = int(round(34 * scale)) * len(lines) + int(round(16 * scale))
    top = np.full((h, img.shape[1], 3), 255, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(top, line, (int(12 * scale), int(round(34 * scale)) * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (0, 0, 0), max(1, int(round(2 * scale))), cv2.LINE_AA)
    return np.vstack([top, img])


def _zoom_window(r: SheetRasters, side: int) -> tuple[int, int]:
    """The top-left of the `side` square with the most px of both blocks and not-blocks (the most to compare)."""
    h, w = r.block.shape
    side = min(side, h, w)

    def integral(m):
        return np.pad(np.cumsum(np.cumsum(m.astype(np.int64), 0), 1), ((1, 0), (1, 0)))

    a, b = integral(r.block == 1), integral(r.block == 0)
    ys = np.arange(0, h - side + 1, max(1, side // 8))
    xs = np.arange(0, w - side + 1, max(1, side // 8))
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    def count(s):
        return s[Y + side, X + side] - s[Y, X + side] - s[Y + side, X] + s[Y, X]

    score = np.minimum(count(a), count(b))
    k = np.unravel_index(np.argmax(score), score.shape)
    return int(Y[k]), int(X[k])


def look_targets(series: str, sheets: list[str], data: Path | None = None, dest: Path = LOOK) -> list[Path]:
    """For the owner: each sheet's street target (today's blocks placed on the scan: blue 1, orange 0, grey 255
    unknown) tinted over the scan, whole at the scan's own px, and a zoom (ZOOM x, nearest) where blocks and streets
    meet most, the target's block edges drawn thin."""
    data = Path(data or data_dir(series))
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    legend = ["blue: today's blocks placed on the scan (target 1)   orange: not a block / street (0)",
              "grey: unknown (outside the neatline or another sheet's area)"]
    for sheet in sheets:
        r = read_rasters(data / f"{sheet}.npz")
        img = _tinted(r)
        head = f"{series} {sheet}  turn {r.turn}  {r.ft_per_px:.2f} ft/px  street target, data/work placement"
        whole = _banner(img, [head] + legend, 1.6)
        f = dest / f"targets_{sheet}_whole.png"
        cv2.imwrite(str(f), whole)
        written.append(f)
        y0, x0 = _zoom_window(r, ZOOM_SIDE)
        s = min(ZOOM_SIDE, *r.block.shape)
        crop = np.repeat(np.repeat(img[y0:y0 + s, x0:x0 + s], ZOOM, 0), ZOOM, 1)
        blk = np.repeat(np.repeat(r.block[y0:y0 + s, x0:x0 + s] == 1, ZOOM, 0), ZOOM, 1)
        edge = blk & (cv2.erode(blk.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
        crop[edge] = TINT[1]
        zoom = _banner(crop, [head, f"zoom x{ZOOM}: scan px x {x0}-{x0 + s}, y {y0}-{y0 + s} (upright)",
                              legend[0], legend[1], "thin blue: the target's block edges"], 1.2)
        f = dest / f"targets_{sheet}_zoom.png"
        cv2.imwrite(str(f), zoom)
        written.append(f)
    return written
