"""Reader A: zoning districts from pattern maps (1921, 1929, 1935, 1948).

The district is the fill pattern inside each block or lot (white, diagonal hatch, solid
black, cross-hatch, square grid, checkerboard), keyed by each era's legend.  The vision
model classifies batched mosaics: the legend strip on top, then up to MAX_TILES numbered
tiles.  Each tile shows a parcel (or a whole block) at a fixed square window around it, the
parcel outlined in red and everything outside it tinted blue as context only.

Per sheet:
  1. legend swatches, read once per series (vision on the page.json mask boxes, saved as
     work/<series>/<sheet>/district_legend.json; districts/legends.json is the hand-entered
     fallback);
  2. per unique parcel geometry (condo stacks share one), robust features: tone (median
     blurred luminance, sampling.tone) and sampling.texture_stats over the parcel inset 3 ft;
  3. a block whose parcels all sit within TOL of the block median is one tile (the block's
     union), confidence x BLOCK_FACTOR; a mixed block sends one tile per parcel;
  4. results mapped back by tile number; written to work/<series>/<sheet>/district_pattern.json.

Calibration (1948/1, 2.1 ft/px): fill tone separates white ~255, hatch ~150-215, cross-hatch
~70-110, solid ~20-40.  Printed labels and block strokes push a few lots of an otherwise
uniform block out by up to ~90 (white lots with a label reach tone ~165), and real mixed
blocks (a commercial frontage on a residential block) differ by ~200, so TONE_TOL = 45/255
splits the clear cases and sends ambiguous ones per parcel.  The ink-fraction check is
deliberately loose (labels move it by 0.4).
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from sfzmaps import config, llm
from sfzmaps.districts import sampling

log = logging.getLogger(__name__)

TILE_PX = 96          # tile edge in the mosaic (brief: <= 96)
MAX_TILES = 48        # tiles per mosaic
COLS = 8
GAP = 4
NUM_H = 14            # number band above each tile
SWATCH_H = 40
CODE_H = 16
TINT = np.array([190, 215, 245], np.float32)     # RGB light blue: outside parcel / off sheet
TINT_MIX = 0.55                                  # outside-parcel context: 45% scan, 55% tint
OUTLINE = (220, 0, 0)                            # RGB red
MARGIN_FT = 20.0      # a geometry this close to the tile edge is centred on an inside point
INSET_FT = 3.0
STATS_MAX_PX = 512
TONE_TOL = 45 / 255
INK_TOL = 0.6
TOL = np.array([TONE_TOL, INK_TOL, np.inf, np.inf, np.inf, np.inf, np.inf])
BLOCK_FACTOR = 0.9
LEGEND_MAX_CALLS = 8
WORKERS = 20          # concurrent mosaic calls (latency-bound: ~25 s per mosaic)
LEGEND_EDGE = 1024
LEGENDS_JSON = Path(__file__).with_name("legends.json")


class PatternItem(BaseModel):
    index: int
    code: str
    confidence: float


class PatternResult(BaseModel):
    items: list[PatternItem]


class LegendSwatch(BaseModel):
    code: str
    label: str
    x0: int
    y0: int
    x1: int
    y1: int


class LegendRead(BaseModel):
    is_legend: bool
    swatches: list[LegendSwatch]


# ---------------------------------------------------------------- legend

def era_codes(series: str) -> dict[str, str]:
    """District code -> name for the series' research-CSV era (non-'special' rows)."""
    from sfzmaps.districts.rules import _default_rules
    from sfzmaps.districts.series import SERIES

    year = SERIES[series].year if series in SERIES else int(series)
    r = _default_rules()
    r = r[(r["start"] <= year) & (r["end"] >= year) & (r["category"] != "special")]
    out: dict[str, str] = {}
    for _, row in r.sort_values("start").iterrows():
        out[row["code"]] = row["name"]
    return out


def _canon(code: str) -> str:
    return re.sub(r"[\s\-_.]", "", str(code)).upper()


def _match_code(code: str, codes) -> str | None:
    want = _canon(code)
    for c in codes:
        if _canon(c) == want:
            return c
    return None


def _legend_config() -> dict:
    try:
        return json.loads(LEGENDS_JSON.read_text())
    except (OSError, ValueError):
        return {}


def _page_rgb(series: str, sheet: str) -> np.ndarray:
    img = cv2.imread(str(config.WORK_DIR / series / sheet / "page.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"no page.png for {series}/{sheet}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _swatches_from_boxes(series: str, sheet: str, boxes: dict) -> dict[str, np.ndarray]:
    """Crop each swatch's outer box from page.png, inset past its printed border."""
    page = _page_rgb(series, sheet)
    out = {}
    for code, (x0, y0, x1, y1) in boxes.items():
        k = max(3, round(0.1 * min(x1 - x0, y1 - y0)))
        crop = page[y0 + k:y1 - k, x0 + k:x1 - k]
        if crop.size:
            out[code] = crop.copy()
    return out


def _refine_box(gray: np.ndarray, box) -> tuple[int, int, int, int]:
    """Snap a model-estimated swatch box to the printed rectangle: the bounding rect of the
    dark contour near it with the best overlap (falls back to the estimate)."""
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    ex, ey = round(0.3 * bw) + 4, round(0.3 * bh) + 4
    X0, Y0 = max(0, x0 - ex), max(0, y0 - ey)
    X1, Y1 = min(gray.shape[1], x1 + ex), min(gray.shape[0], y1 + ey)
    region = (gray[Y0:Y1, X0:X1] < 128).astype(np.uint8)
    if not region.size:
        return box
    contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_iou = None, 0.4
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.5 * bw or h < 0.5 * bh:
            continue
        r = (x + X0, y + Y0, x + X0 + w, y + Y0 + h)
        ix = max(0, min(r[2], x1) - max(r[0], x0))
        iy = max(0, min(r[3], y1) - max(r[1], y0))
        inter = ix * iy
        iou = inter / (bw * bh + w * h - inter)
        if iou > best_iou:
            best, best_iou = r, iou
    return tuple(int(v) for v in best) if best else (int(x0), int(y0), int(x1), int(y1))


def legend_prompt(codes: dict[str, str]) -> str:
    lst = "; ".join(f"{c} = {n}" for c, n in codes.items())
    return (
        "This is a crop from a scanned San Francisco zoning map page. If it contains the map "
        "LEGEND (small rectangles, each filled with a pattern such as white, diagonal hatching, "
        "solid black, cross-hatching, a square grid or a checkerboard, next to a district name), "
        "set is_legend true and list every legend rectangle. For each: code = the district code "
        f"it stands for, chosen from this list ({lst}); label = the text printed beside it; "
        "x0, y0, x1, y1 = the pixel box of the patterned rectangle only (not its text), in this "
        "image's pixel coordinates (origin top-left). If there is no legend, set is_legend false "
        "and return no swatches.")


def legend_inputs_sha256() -> str:
    """sha256 of the legend prompt template (codes as a placeholder) and response schema."""
    import hashlib

    schema = json.dumps(LegendRead.model_json_schema(), sort_keys=True)
    text = legend_prompt({"<CODE>": "<NAME>"}) + "\0" + schema
    return hashlib.sha256(text.encode()).hexdigest()


# district_legend.json files written before the read's inputs were stored (through 2d1bf49)
# lack these fields; they were read with this template + schema fingerprint and the current
# LEGEND_EDGE, era codes taken as unchanged.  A missing field means exactly that.
LEGACY_LEGEND_SHA256 = "058496117906e9850b5d2b6842fd3cdc17e057fe21a01e2d9831fd4cb33e8570"


def legend_key(series: str) -> dict:
    return {"inputs_sha256": legend_inputs_sha256(), "era_codes": sorted(era_codes(series)),
            "edge": LEGEND_EDGE}


def legend_fresh(saved: Path, series: str) -> bool:
    """A saved district_legend.json was read with the current prompt, schema, codes and edge."""
    try:
        d = json.loads(Path(saved).read_text())
    except (OSError, ValueError):
        return False
    want = legend_key(series)
    have = {"inputs_sha256": d.get("inputs_sha256", LEGACY_LEGEND_SHA256),
            "era_codes": sorted(d.get("era_codes", want["era_codes"])),
            "edge": int(d.get("edge", want["edge"]))}
    return have == want


def legend_swatches(series: str, *, client=None) -> dict[str, np.ndarray]:
    """Era legend code -> RGB swatch crop.

    Order: a hand-entered districts/legends.json entry marked "force"; a saved
    district_legend.json in any of the series' sheet dirs; a vision read of the page.json
    mask boxes (saved as district_legend.json); the unforced legends.json entry."""
    cfg = _legend_config().get(series)
    if cfg and cfg.get("force"):
        return _swatches_from_boxes(series, cfg["sheet"], cfg["swatches"])
    root = config.WORK_DIR / series
    for saved in sorted(root.glob("*/district_legend.json")):
        if not legend_fresh(saved, series):
            log.info("%s: %s is stale (prompt, schema, codes or edge changed); rereading",
                     series, saved.relative_to(root))
            continue
        d = json.loads(saved.read_text())
        return _swatches_from_boxes(series, d["sheet"], d["swatches"])
    codes = era_codes(series)
    prompt = legend_prompt(codes)
    calls = 0
    for sheet_dir in sorted(p.parent for p in root.glob("*/page.json")):
        info = json.loads((sheet_dir / "page.json").read_text())
        boxes = info.get("mask_boxes") or []
        if not boxes:
            continue
        page = _page_rgb(series, sheet_dir.name)
        H, W = page.shape[:2]
        gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
        for mb in boxes:
            w, h = mb["x1"] - mb["x0"], mb["y1"] - mb["y0"]
            if w * h > 0.2 * W * H or min(w, h) < 80 or max(w, h) > 4 * min(w, h):
                continue
            crop = page[mb["y0"]:mb["y1"], mb["x0"]:mb["x1"]]
            s = min(1.0, LEGEND_EDGE / max(crop.shape[:2]))
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if calls >= LEGEND_MAX_CALLS:
                break
            calls += 1
            res = llm.ask_json(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), prompt, LegendRead, client=client)
            if not res.is_legend:
                continue
            found, labels = {}, {}
            for sw in res.swatches:
                code = _match_code(sw.code, codes)
                if code is None or code in found:
                    continue
                est = (round(sw.x0 / s) + mb["x0"], round(sw.y0 / s) + mb["y0"],
                       round(sw.x1 / s) + mb["x0"], round(sw.y1 / s) + mb["y0"])
                if est[2] - est[0] < 6 or est[3] - est[1] < 6:
                    continue
                found[code] = list(_refine_box(gray, est))
                labels[code] = sw.label
            if len(found) >= 3:
                art = {"sheet": sheet_dir.name, "mask_box": mb, "swatches": found, "labels": labels,
                       "source": "vision", **legend_key(series)}
                (sheet_dir / "district_legend.json").write_text(json.dumps(art, indent=1))
                return _swatches_from_boxes(series, sheet_dir.name, found)
    if cfg:
        return _swatches_from_boxes(series, cfg["sheet"], cfg["swatches"])
    raise RuntimeError(f"no legend found for series {series}; add it to {LEGENDS_JSON.name}")


# ---------------------------------------------------------------- crops and mosaic

@lru_cache(maxsize=64)
def _res(warped_tif: str) -> float:
    import rasterio

    with rasterio.open(warped_tif) as ds:
        return float(ds.res[0])


def tile_crop(warped_tif: Path, geom, *, px: int = TILE_PX,
              ft_per_px: float | None = None) -> np.ndarray | None:
    """RGB px x px tile of the scan at its native scale (ft_per_px: default the raster's), so
    the fill looks as it does in the legend swatches (upsampled lots turn fine cross-hatch
    into something like the checkerboard; measured on 1948/1).  The window is centred on the
    geometry (on a point inside it when it is bigger than the window); the geometry is
    outlined in red, outside it tinted, off-sheet solid tint.  None when under half the
    geometry inside the window is on the sheet."""
    from rasterio import features

    half = px * (ft_per_px or _res(str(warped_tif))) / 2
    x0, y0, x1, y1 = geom.bounds
    if max(x1 - x0, y1 - y0) <= 2 * half - 2 * MARGIN_FT:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        pt = geom.representative_point()
        cx, cy = pt.x, pt.y
    rgb, on, tr = sampling.read_window(warped_tif, (cx - half, cy - half, cx + half, cy + half))
    mask = features.rasterize([(geom, 1)], out_shape=on.shape, transform=tr, fill=0, dtype="uint8") > 0
    if not mask.any() or on[mask].mean() < sampling.MIN_COVER:
        return None
    out = rgb.astype(np.float32)
    out[~mask] = (1 - TINT_MIX) * out[~mask] + TINT_MIX * TINT
    out[~on] = TINT
    rows, cols = on.shape
    tile = cv2.resize(out.clip(0, 255).astype(np.uint8), (px, px), interpolation=cv2.INTER_AREA)
    inv = ~tr
    sx, sy = px / cols, px / rows
    for p in getattr(geom, "geoms", [geom]):
        xy = np.asarray(p.exterior.coords)[:, :2]
        pts = np.column_stack([inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c,
                               inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f]) * [sx, sy]
        cv2.polylines(tile, [np.round(pts).astype(np.int32)], True, OUTLINE, 1, cv2.LINE_AA)
    return tile


def _as_tile(crop: np.ndarray, tile: int) -> np.ndarray:
    """Any RGB/RGBA crop -> RGB tile x tile: aspect kept (transparent padding), alpha 0 tinted."""
    if crop.ndim == 2:
        crop = np.repeat(crop[..., None], 3, axis=2)
    if crop.shape[2] == 3:
        crop = np.dstack([crop, np.full(crop.shape[:2], 255, np.uint8)])
    if crop.shape[:2] != (tile, tile):
        crop = sampling.fit_square(crop, tile)
    out = np.where(crop[..., 3:4] > 0, crop[..., :3], TINT.astype(np.uint8))
    return out.astype(np.uint8)


def _legend_layout(swatches: dict, tile: int):
    """[(code, x, y, w)] for each legend entry and the strip height."""
    width = COLS * (tile + GAP) + GAP
    x, y, out = GAP, GAP, []
    for code, sw in swatches.items():
        w = int(min(2.5 * SWATCH_H, max(SWATCH_H, SWATCH_H * sw.shape[1] / max(1, sw.shape[0]))))
        cell = max(w, 10 * len(code)) + 3 * GAP
        if x + cell > width and x > GAP:
            x, y = GAP, y + SWATCH_H + CODE_H + GAP
        out.append((code, x, y, w))
        x += cell
    return out, y + SWATCH_H + CODE_H + 2 * GAP + 2


def legend_height(swatches: dict, tile: int = TILE_PX) -> int:
    return _legend_layout(swatches, tile)[1]


def tile_boxes(n: int, legend_h: int, tile: int = TILE_PX) -> list[tuple[int, int, int, int]]:
    """Pixel box (x0, y0, x1, y1) of each of n tiles in a mosaic whose legend strip is legend_h."""
    out = []
    for i in range(n):
        r, c = divmod(i, COLS)
        x0 = GAP + c * (tile + GAP)
        y0 = legend_h + r * (tile + NUM_H + GAP) + NUM_H
        out.append((x0, y0, x0 + tile, y0 + tile))
    return out


def build_mosaic(crops: list[np.ndarray], swatches: dict[str, np.ndarray], *,
                 tile: int = TILE_PX) -> tuple[np.ndarray, list[int]]:
    """RGB mosaic: the legend strip (each swatch with its code under it), then crops as
    numbered tiles, COLS per row.  Returns (image, numbers): crops[i] is labelled numbers[i]."""
    if len(crops) > MAX_TILES:
        raise ValueError(f"{len(crops)} crops > {MAX_TILES} per mosaic")
    layout, lh = _legend_layout(swatches, tile)
    rows = max(1, -(-len(crops) // COLS))
    width = COLS * (tile + GAP) + GAP
    img = np.full((lh + rows * (tile + NUM_H + GAP) + GAP, width, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for code, x, y, w in layout:
        sw = swatches[code]
        sw = sw[..., :3] if sw.ndim == 3 else np.repeat(sw[..., None], 3, axis=2)
        if sw.shape[0] >= SWATCH_H and sw.shape[1] >= w:     # native scale, like the tiles
            oy, ox = (sw.shape[0] - SWATCH_H) // 2, (sw.shape[1] - w) // 2
            img[y:y + SWATCH_H, x:x + w] = sw[oy:oy + SWATCH_H, ox:ox + w]
        else:
            img[y:y + SWATCH_H, x:x + w] = cv2.resize(np.ascontiguousarray(sw), (w, SWATCH_H),
                                                      interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (x - 1, y - 1), (x + w, y + SWATCH_H), (0, 0, 0), 1)
        cv2.putText(img, code, (x, y + SWATCH_H + CODE_H - 3), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    img[lh - 2:lh, :] = 0
    numbers = []
    for i, (crop, (x0, y0, x1, y1)) in enumerate(zip(crops, tile_boxes(len(crops), lh, tile))):
        img[y0:y1, x0:x1] = _as_tile(crop, tile)
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x1, y1), (128, 128, 128), 1)
        cv2.putText(img, str(i + 1), (x0, y0 - 3), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        numbers.append(i + 1)
    return img, numbers


# ---------------------------------------------------------------- classify

def pattern_prompt(n: int, codes: list[str], names: dict[str, str] | None = None,
                   year: int | None = None) -> str:
    key = "; ".join(f"{c} = {names[c]}" if names and c in names else c for c in codes)
    era = f"{year} " if year else ""
    return (
        f"The top strip of this image is the legend of a {era}San Francisco zoning map: each "
        f"fill pattern is labelled with its district code ({key}). Below it are tiles "
        f"numbered 1 to {n}, each cut from the same scanned map. In each tile a thin red outline "
        "marks one parcel or block; everything outside the outline is tinted blue and is "
        "context only; solid light blue is off the map. For each tile, decide which legend "
        "pattern fills the area inside the red outline. Compare pattern type (empty white, "
        "diagonal hatching, solid black, fine cross-hatching, square grid, checkerboard) and "
        "darkness with the legend. Ignore printed text, numbers, street names and the thin "
        "black block boundary lines. If the outline covers two patterns, choose the one "
        "covering most of the area inside it. Return one item per tile: index = the tile "
        f"number, code = one of {', '.join(codes)} (or 'none' if no pattern can be seen), "
        "confidence = 0 to 1.")


def _uniform(stats: np.ndarray) -> bool:
    med = np.median(stats, axis=0)
    tol = TOL[:stats.shape[1]] if stats.shape[1] <= len(TOL) else \
        np.concatenate([TOL, np.full(stats.shape[1] - len(TOL), np.inf)])
    return bool((np.abs(stats - med) <= tol).all())


def _get(x):
    return x() if callable(x) else x


def classify(parcels_with_crops: pd.DataFrame, swatches: dict[str, np.ndarray], *, client=None,
             block_crops: dict | None = None, year: int | None = None,
             names: dict[str, str] | None = None, workers: int = WORKERS) -> pd.DataFrame:
    """Columns blklot, code, confidence, via ('block' | 'parcel' | 'no_tile').

    parcels_with_crops: blklot, block, crop (a tile or a callable returning one), stats (a
    feature vector compared against TOL: [tone/255, *texture_stats]).  A block whose
    parcels' stats are all within TOL of the block median is classified once, from
    block_crops[block] (the block's own tile) or else its first parcel's crop, confidence x
    BLOCK_FACTOR.  Codes not in the legend come back as code None, confidence 0."""
    df = parcels_with_crops.reset_index(drop=True)
    items: list[tuple[np.ndarray, list[int], str]] = []

    def per_parcel(g):
        for i, c in zip(g.index, g["crop"]):
            c = _get(c)
            if c is not None:              # no tile (window mostly off the sheet): unread
                items.append((c, [i], "parcel"))

    for block, g in df.groupby("block", sort=False):
        stats = np.stack([np.asarray(s, float) for s in g["stats"]])
        if len(g) > 1 and _uniform(stats):
            crop = _get((block_crops or {}).get(block))
            if crop is None:
                crop = _get(g["crop"].iloc[0])
            if crop is not None:
                items.append((crop, list(g.index), "block"))
                continue
        per_parcel(g)
    codes = list(swatches)
    code = [None] * len(df)
    conf = [0.0] * len(df)
    via = ["no_tile"] * len(df)
    batches = [items[b0:b0 + MAX_TILES] for b0 in range(0, len(items), MAX_TILES)]

    def ask(batch):
        img, numbers = build_mosaic([c for c, _, _ in batch], swatches)
        prompt = pattern_prompt(len(batch), codes, names, year)
        res = llm.ask_json(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), prompt, PatternResult, client=client)
        return numbers, res

    for batch, (numbers, res) in zip(batches, llm.pmap(ask, batches, workers)):
        by_index = {it.index: it for it in res.items}
        for num, (_, rows, kind) in zip(numbers, batch):
            it = by_index.get(num)
            c = _match_code(it.code, codes) if it else None
            f = float(np.clip(it.confidence, 0, 1)) * (BLOCK_FACTOR if kind == "block" else 1.0) if c else 0.0
            for r in rows:
                code[r], conf[r], via[r] = c, f, kind
    return pd.DataFrame({"blklot": df["blklot"], "code": pd.Series(code, dtype=object),
                         "confidence": conf, "via": via})


# ---------------------------------------------------------------- per sheet / series

def parcel_stats(warped_tif: Path, geom) -> np.ndarray | None:
    """[tone/255, *texture_stats] over the parcel inset INSET_FT, or None if off the sheet."""
    c = sampling.parcel_crop(warped_tif, geom, inset_ft=INSET_FT, px=None)
    if c is None:
        return None
    if max(c.shape[:2]) > STATS_MAX_PX:
        c = sampling.fit_square(c, STATS_MAX_PX)
    return np.array([sampling.tone(c) / 255, *sampling.texture_stats(c)])


def read_sheet(series: str, sheet_id: str, parcels, *, swatches: dict | None = None,
               client=None) -> pd.DataFrame:
    """Classify the parcels assigned to one sheet (GeoDataFrame: blklot, block_num, geometry).
    Parcels sharing one geometry (condo stacks) are read once.  Writes
    work/<series>/<sheet>/district_pattern.json and returns blklot, code, confidence, via."""
    from shapely.ops import unary_union

    from sfzmaps.districts.series import SERIES

    tif = config.WORK_DIR / series / sheet_id / "warped.tif"
    if swatches is None:
        swatches = legend_swatches(series, client=client)
    live0 = llm.CallCounter.live
    p = parcels.reset_index(drop=True)
    keys = [g.wkb for g in p.geometry]
    first: dict[bytes, int] = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    uniq = sorted(first.values())
    rows, off = [], set()
    for i in uniq:
        geom = p.geometry.iloc[i]
        st = parcel_stats(tif, geom)
        if st is None:
            off.add(i)
            continue
        rows.append({"blklot": p["blklot"].iloc[i], "block": p["block_num"].iloc[i],
                     "crop": (lambda g=geom: tile_crop(tif, g)), "stats": st})
    df = pd.DataFrame(rows, columns=["blklot", "block", "crop", "stats"])
    geoms_by_block = {}
    for i in uniq:
        if i not in off:
            geoms_by_block.setdefault(p["block_num"].iloc[i], []).append(p.geometry.iloc[i])
    block_crops = {b: (lambda gs=gs: tile_crop(tif, unary_union(gs).buffer(0.5).buffer(-0.5)))
                   for b, gs in geoms_by_block.items() if len(gs) > 1}
    info = SERIES.get(series)
    year = info.year if info else None
    names = era_codes(series) if info else None
    res = classify(df, swatches, client=client, block_crops=block_crops, year=year, names=names) \
        if len(df) else pd.DataFrame(columns=["blklot", "code", "confidence", "via"])
    by_blklot = {r.blklot: r for r in res.itertuples(index=False)}
    out = []
    for i, k in enumerate(keys):
        src = p["blklot"].iloc[first[k]]
        r = by_blklot.get(src)
        out.append({"blklot": p["blklot"].iloc[i], "code": r.code if r else None,
                    "confidence": float(r.confidence) if r else 0.0,
                    "via": (r.via if r else "off_sheet")})
    outdf = pd.DataFrame(out, columns=["blklot", "code", "confidence", "via"])
    art = {"series": series, "sheet": sheet_id, "codes": list(swatches),
           "live_calls": llm.CallCounter.live - live0,
           "parcels": outdf.to_dict("records")}
    (tif.parent / "district_pattern.json").write_text(json.dumps(art))
    return outdf


def read_series(series: str, *, sheets: list[str] | None = None, client=None) -> pd.DataFrame:
    """Read every (or the listed) sheet of a pattern series over the active modern parcels,
    each parcel on its highest-priority covering sheet (sampling.assign_sheets)."""
    import geopandas as gpd

    parcels = gpd.read_parquet(config.REF_DIR / "parcels.parquet")
    parcels = parcels[parcels["active"]].reset_index(drop=True)
    assigned = sampling.assign_sheets(parcels, series)
    parcels["sheet"] = assigned.to_numpy()
    swatches = legend_swatches(series, client=client)
    out = []
    for sid in sheets or sampling.sheet_order(series):
        sub = parcels[parcels["sheet"] == sid]
        if len(sub):
            out.append(read_sheet(series, sid, sub, swatches=swatches, client=client))
    return pd.concat(out, ignore_index=True) if out else \
        pd.DataFrame(columns=["blklot", "code", "confidence", "via"])
