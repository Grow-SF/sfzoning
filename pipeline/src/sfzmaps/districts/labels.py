"""Reader B: zoning districts from label-and-boundary maps (1960 onward).

Districts are regions bounded by heavy lines, each carrying a printed code (R-1, C-2, RH-2;
height maps print 40-X, 65-A...).  Steps, all in page pixels until the parcel join:

1. ``read_district_labels``: a vision prompt reads the codes with positions from page tiles
   (``reading.select_tiles``: same tiling, skip mask and cache as step one's label reading).
   Cached as ``work/<series>/<sheet>/district_labels.json`` keyed by the page.png sha256.
2. ``heavy_line_mask``: heavy district lines vs thin block / lot lines by local stroke width
   (distance transform), relative to the sheet's measured thin-stroke width.
3. ``regions``: connected areas inside the neatline between heavy lines (``walls``: their gaps
   bridged where the bridge runs over printed ink), as polygons.
4. ``assign_regions``: one code per region; conflicting codes split the region by a geodesic
   Voronoi of its labels, distance measured around heavy lines (minority codes capped to a
   disk round their labels); an unlabelled region
   takes the code a per-region vision read names for it (``regionread``, at READ_CONF), else
   inherits only when all its coded neighbours agree.
5. ``assign_parcels``: regions to EPSG:2227 through the sheet transform; each parcel takes
   the code covering the majority of its area, and records how far it lies from the printed
   label backing that code (``supporting_labels``; read.pick_readings uses it to choose
   between sheets).  Fewer than MIN_REGIONS regions (boundary extraction failed): each parcel
   takes the nearest label in its own block at FALLBACK_CONF.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel
from shapely import STRtree, voronoi_polygons
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from sfzmaps import answers, config, linework, llm, reading
from sfzmaps.transform import SheetTransform

log = logging.getLogger(__name__)

# --- tuning (page px unless noted) ------------------------------------------------------
HEAVY_FACTOR = 1.6       # heavy line: estimated width >= this x the thin stroke width ...
HEAVY_MIN_EXTRA = 2.0    # ... and at least this many px wider than it
MIN_THIN_PX = 2          # ridges narrower than this are hairlines, not the thin stroke
GAP_CLOSE_PX = 9         # closing disk that bridges small breaks in heavy lines
BRIDGE_PX = 120         # a heavy line's loose end joins heavy ink up to this far ahead ...
BRIDGE_CONE_DEG = 35.0  # ... within this angle of the line's direction
BRIDGE_BACK_PX = 30     # the direction is taken over this many skeleton px ...
BRIDGE_STRAIGHT = 0.85  # ... which must run this straight (chord / path length)
BRIDGE_MIN_LINE_PX = 200  # only skeleton pieces this long bridge or are bridged to (not letters)
BLOB_FACTOR = 3.0       # heavy ink surviving an opening this x the heavy width is a blob
MIN_REGION_PX = 150      # smaller free areas are letter counters / specks, not districts
MIN_HOLE_PX = 4000       # holes smaller than this (printed codes, specks) are filled
SIMPLIFY_PX = 1.0
SNAP_RADIUS_PX = 30      # a label belongs to the region covering most of this disk around it
SNAP_PX = 80             # else (all ink around it) to the nearest region this close
TOUCH_PX = 15            # regions this close are neighbours (they're split by a heavy line)
SPLIT_CONF = 0.8         # confidence factor for pieces of a region split between labels
READ_CONF = 0.8         # confidence factor for a region named by the per-region read
READ_PARCEL_SHARE = 0.3  # only regions at least this much covered by parcels are read
INHERIT_CONF = 0.6       # confidence factor for a region that inherits its neighbours' code
PLURALITY_CONF = 0.3     # confidence of a region whose neighbours disagree: longest contact wins ...
PLURALITY_MARGIN = 1.25  # ... by at least this factor over the runner-up (else unknown)
# Parks / public sites and the small commercial clusters are islands labelled inside their own
# boxes: never spread by plurality, and capped in splits (2000-digital: P and NC-1 flooded
# residential blocks; capping them raised agreement on unchanged districts 66.8% -> 69.2%).
NO_PLURALITY = {"P", "OS", "C-1", "NC-1", "NC-2", "NC-3", "NC-S", "RC-1", "RC-2"}
SPLIT_P_RADIUS_PX = 150  # ... and in a split region a P label takes at most this disk (~1 block)
# A region that no heavy line closes (the page background, a district whose line stops short)
# can hold many labels of one code and a stray one of another.  Split evenly, a lone label took
# as much as its neighbours (1960/1964: an R-3 printed by a small C-1 box at Restani made ~300
# Mt Vernon lots R-3; 1964: an R-4 made 490 Silver Terrace lots R-4).  So when one code has at
# least SPLIT_MAJORITY x the labels of any other (P and the NO_PLURALITY codes aside, capped
# above), a code printed only SPLIT_LONE time keeps only this disk round its label and the
# main code fills the rest.  Chosen on the seam reviews' cases (wrong lots 1718 -> 1204; 150
# and 200 px did worse) and agreement between maps drawn on the same base (1960/1964
# 0.939 -> 0.947, 1996/2000 0.689 -> 0.692).
SPLIT_MINOR_RADIUS_PX = 100
SPLIT_MAJORITY = 2
# Only a code printed once is capped: in a huge unclosed region (1980 sheet 12) a code printed
# several times nearby is a real district even when another code outnumbers it overall;
# capping those made ~1,100 Forest Hill lots RH-2 from a label 3,000 ft away.
SPLIT_LONE = 1
# ... and a lone code is capped whenever another code is printed at least LONE_MAIN times in the
# region, however many the other codes have (1964 sheet 13, after INK_SHARE opened the Mission
# C-2 strip: R-1 x7, R-2 x4, C-2 x4 and the Restani R-3 x1 again made ~300 Mt Vernon lots R-3).
# 2 and 3 also capped the real, singly printed RH-1(D) of Bayview Hill (1980 sheet 19: M-1 x3).
LONE_MAIN: int | None = 4
# A bridge is kept only if at least INK_SHARE of it lies on printed ink (dilated 1 px): a
# district line that breaks where it runs along a block outline has a thin line under its gap,
# while bridges across a street or a strip of blank paper cut districts in different places on
# two scans of one drawing (#34: 1996 and 2000 agreed on 69% of lots).  Measured with
# `sfzmaps districts --check`, with GEO_SPLIT and LONE_MAIN: seam-case wrong lots 1002 -> 634,
# harm cases 202 -> 72, same-base agreement 1996/2000 0.685 -> 0.746, 1960/1964 0.947 ->
# 0.960, and every adjacent-era pair up (0.4 and 0.6 kept more stray bridges).
INK_SHARE: float | None = 0.8
INK_NEAR_PX = 1                  # ink this close to a bridge pixel counts as under it
# Split a multi-code region by distance travelled around heavy lines (geodesic), not in a
# straight line: a district line with a gap at one end still keeps each side with its own label
# (1980 Forest Hill: 119 -> 0 lots wrong).  Computed on a grid of GEO_PX page px per cell.
GEO_SPLIT = True
GEO_PX = 4
# Parks are drawn as closed outlines, and their gaps are mostly where a street crosses (blank
# paper), which INK_SHARE no longer bridges: McLaren Park (1980 sheet 23) then joined the
# residential page around it and kept only its SPLIT_P_RADIUS_PX disk.  So a P label also takes
# the region it would have with every bridge (ink or not), when no other code is printed in
# it and it is at most PARK_MAX_PX.
PARK_CLOSED = True
PARK_MAX_PX = 5_000_000
MIN_REGIONS = 5          # fewer closed regions: boundary extraction failed, use the fallback
FALLBACK_CONF = 0.3
FALLBACK_BLOCK_FT = 20.0  # labels within this distance of a block count as in it
DEDUPE_PX = 60           # same code closer than this (tile overlaps) is one label
SEGMENT_PX = 10.0        # densify outlines before the (non-linear) page -> world transform

# District codes are printed larger than block numbers, so tiles are read at half the
# block-number scale (config.label_scale) unless a series is calibrated here.
DISTRICT_SCALE_FACTOR = 0.5
DISTRICT_SCALE: dict[str, float] = {}

DISTRICT_PROMPT = """This image is a crop of a scanned historic San Francisco zoning district map. It is {w} pixels wide and {h} pixels high. Return x and y as integer pixel coordinates within this image (x from 0 to {w}, y from 0 to {h}, origin at the top-left).

The map divides the city into zoning districts: areas outlined by heavy lines, each with a district code printed inside it, often in large letters. Find every printed zoning district code:
- kind "use": use-district codes such as R-1, R-1-D, R-2, R-3, R-4, R-5, C-1, C-2, C-3-O, C-M, M-1, M-2, P, RH-1, RH-1(D), RH-2, RH-3, RM-1, RM-4, RC-3, NC-1, NC-2, NC-3, NC-S, CVR, SSO, RED, or neighborhood district names abbreviated as codes (for example "UPPER MARKET NCD").
- kind "height": height-and-bulk district codes such as 40-X, 65-A, 80-B, 105-E, 160-F, or OS (open space).

For each code give the text exactly as printed, its kind, the pixel center (x, y) of the printed code, and a confidence from 0 to 1.

Ignore block numbers (plain 1-4 digit numbers), street names, lot numbers and dimensions, legend text, titles and notes. If a code is cut off by the image edge, skip it. Return an empty list rather than guessing."""


def district_prompt(img: np.ndarray) -> str:
    return DISTRICT_PROMPT.format(h=img.shape[0], w=img.shape[1])


class _RawLabel(BaseModel):
    code: str
    kind: Literal["use", "height"]
    x: int
    y: int
    confidence: float


class _DistrictLabels(BaseModel):
    labels: list[_RawLabel]


class DistrictLabel(BaseModel):
    code: str
    kind: Literal["use", "height"]
    x: float        # page px
    y: float
    confidence: float


def normalise(code: str) -> str:
    """Upper case, one kind of hyphen, no spaces around hyphens."""
    c = code.strip().upper().replace("–", "-").replace("—", "-").replace("‐", "-")
    return " ".join("-".join(p.strip() for p in c.split("-")).split())


def hyphenate(code: str | None) -> str | None:
    """A use code printed without its hyphens ("R1", "R1D", "RH1(D)": 1968 and 1975 print
    some) in its usual form ("R-1"), so one district isn't two codes.  Applied to the codes
    parcels take, not to the labels: those also go into the region-read prompts, whose cached
    answers are keyed by their text."""
    from sfzmaps.districts.rules import normalise_code

    if code and re.fullmatch(r"[A-Z]{1,3}\d.*", code):
        return normalise_code(code, 0) or code
    return code


def district_scale(series: str) -> float:
    return DISTRICT_SCALE.get(series, config.label_scale(series) * DISTRICT_SCALE_FACTOR)


# --- artifact ---------------------------------------------------------------------------

def page_sha(sheet_dir: Path) -> str:
    return hashlib.sha256((Path(sheet_dir) / "page.png").read_bytes()).hexdigest()


def inputs_sha256() -> str:
    """sha256 of the read's prompt template and response schema."""
    schema = json.dumps(_DistrictLabels.model_json_schema(), sort_keys=True)
    return hashlib.sha256((DISTRICT_PROMPT + "\0" + schema).encode()).hexdigest()


# Artifacts written before the read's inputs were stored (through 2d1bf49) carry only
# page_sha256.  They were read with this prompt + schema fingerprint and the default scale,
# tile and overlap, so a missing field is taken as exactly that: they stay fresh until the
# prompt, schema or a parameter really changes.
LEGACY_INPUTS_SHA256 = "502dab5a88665982308590f15a9cec48e8732b64495d97c5494f6d972b52f56b"


def read_key(series: str, *, scale: float | None = None, tile: int = config.VISION_TILE,
             overlap: int = config.VISION_OVERLAP) -> dict:
    """What district_labels.json must match besides the page: prompt + schema, scale, tiling."""
    return {"inputs_sha256": inputs_sha256(), "scale": round(float(scale or district_scale(series)), 6),
            "tile": int(tile), "overlap": int(overlap)}


def stored_key(path: Path, series: str) -> dict | None:
    """The read key stored in a district_labels.json; missing fields (older artifacts) are the
    legacy fingerprint and the current defaults.  None if unreadable."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    default = read_key(series)
    return {"inputs_sha256": d.get("inputs_sha256", LEGACY_INPUTS_SHA256),
            "scale": round(float(d.get("scale", default["scale"])), 6),
            "tile": int(d.get("tile", default["tile"])),
            "overlap": int(d.get("overlap", default["overlap"]))}


def labels_fresh(sheet_dir: Path, *, scale: float | None = None, tile: int = config.VISION_TILE,
                 overlap: int = config.VISION_OVERLAP) -> bool:
    """district_labels.json matches the page and every input of the read."""
    d = Path(sheet_dir)
    art, series = d / "district_labels.json", d.parent.name
    if reading.stored_page_sha256(art) != page_sha(d):
        return False
    return stored_key(art, series) == read_key(series, scale=scale, tile=tile, overlap=overlap)


def save_labels(path: Path, labs: list[DistrictLabel], *, page_sha256: str,
                key: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"page_sha256": page_sha256, **(key or {}),
                               "labels": [l.model_dump() for l in labs]}, indent=1))
    tmp.replace(path)


def load_labels(path: Path) -> list[DistrictLabel]:
    return [DistrictLabel.model_validate(x) for x in json.loads(Path(path).read_text())["labels"]]


def _page_info(sheet_dir: Path) -> reading.PageInfo | None:
    p = Path(sheet_dir) / "page.json"
    return reading.load_page(p) if p.exists() else None


def _dedupe(labs: list[DistrictLabel]) -> list[DistrictLabel]:
    kept: list[DistrictLabel] = []
    for lb in sorted(labs, key=lambda l: -l.confidence):
        if any(k.kind == lb.kind and k.code == lb.code and abs(k.x - lb.x) <= DEDUPE_PX
               and abs(k.y - lb.y) <= DEDUPE_PX for k in kept):
            continue
        kept.append(lb)
    return sorted(kept, key=lambda l: (l.y, l.x))


def read_district_labels(series: str, sheet_id: str, *, client: llm.LLMClient | None = None,
                         scale: float | None = None, tile: int = config.VISION_TILE,
                         overlap: int = config.VISION_OVERLAP, force: bool = False
                         ) -> list[DistrictLabel]:
    """District codes with page-px positions; cached in district_labels.json, reused while the
    page sha, the prompt + schema fingerprint, scale, tile and overlap all match."""
    d = config.WORK_DIR / series / sheet_id
    art, sha = d / "district_labels.json", page_sha(d)
    key = read_key(series, scale=scale, tile=tile, overlap=overlap)
    fresh = not force and reading.stored_page_sha256(art) == sha and stored_key(art, series) == key
    if fresh and answers.covered(art):
        answers.consume(art)
        return load_labels(art)
    if fresh:     # made before answers were recorded: replay it from the cache, never paying
        try:
            with llm.cache_only(True):
                again = _read(d, series, sha, key, scale, tile, overlap, client, write=False)
        except llm.LLMCacheMiss:
            log.info("%s/%s: district labels kept as is (answers not all cached under today's "
                     "prompt; `sfzmaps answers --baseline` lists it)", series, sheet_id)
            return load_labels(art)
        if again[0] == load_labels(art):
            answers.write(art, again[1])
            answers.consume(art)
            return load_labels(art)
    return _read(d, series, sha, key, scale, tile, overlap, client)[0]


def _read(d, series, sha, key, scale, tile, overlap, client, write: bool = True):
    """Read a sheet's district labels under recording: (labels, answer keys); with write, save
    them (sidecar first) and fold their answers into the current recording."""
    art = d / "district_labels.json"
    img = cv2.imread(str(d / "page.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read {d / 'page.png'}")
    skip = reading.exclusion_mask(img, _page_info(d))
    scale = scale or district_scale(series)
    _, tiles = reading.select_tiles(img, scale=scale, tile=tile, overlap=overlap, skip=skip)
    if client is None and tiles and not llm.CACHE_ONLY and not all(
            llm.is_cached(t[2], district_prompt(t[2]), _DistrictLabels) for t in tiles):
        client = llm.OpenAIClient()

    def one(t):
        return llm.ask_json(t[2], district_prompt(t[2]), _DistrictLabels, client=client)

    with llm.recording() as keys:
        results = llm.pmap(one, tiles, config.VISION_CONCURRENCY)
    h, w = img.shape
    out: list[DistrictLabel] = []
    for (x0, y0, t), res in zip(tiles, results):
        th, tw = t.shape[:2]
        for lb in res.labels:
            if not (0 <= lb.x <= tw and 0 <= lb.y <= th) or not lb.code.strip():
                continue
            x, y = (lb.x + x0) / scale, (lb.y + y0) / scale
            if skip[min(h - 1, int(y)), min(w - 1, int(x))]:
                continue
            out.append(DistrictLabel(code=normalise(lb.code), kind=lb.kind, x=x, y=y,
                                     confidence=lb.confidence))
    out = _dedupe(out)
    if write:
        answers.write(art, keys)          # sidecar first: an artifact never exists without one
        save_labels(art, out, page_sha256=sha, key=key)
        answers.consume(art)
    return out, keys


# --- boundaries -------------------------------------------------------------------------

def heavy_line_mask(page_img: np.ndarray) -> np.ndarray:
    """Bool mask of heavy-line ink: strokes whose local width (2 x distance-to-paper on the
    stroke's ridge, minus 1) is well above the sheet's typical thin-stroke width."""
    return _heavy(page_img)[0]


def thin_stroke_width(ink: np.ndarray) -> float:
    """The sheet's thin-stroke (block outline) width: the most common stroke-ridge width
    (2 x distance-to-paper - 1, at local maxima) of at least MIN_THIN_PX.  Hairlines are left
    out: the digital-era sheets draw lot lines 1 px wide, more ink than anything else, and
    taking them as the thin stroke made every 3 px block outline read heavy (2000-digital p11
    and others: each block its own region)."""
    d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = (cv2.dilate(d, np.ones((3, 3), np.uint8)) == d) & (d > 0)
    w = np.round(2 * d[ridge] - 1).astype(int)
    w = w[(w >= MIN_THIN_PX) & (w < 4 * MIN_THIN_PX + 8)]
    if not w.size:
        return float(linework.stroke_width(ink) or 1.0)
    return float(np.bincount(w).argmax())


def _heavy(page_img: np.ndarray) -> tuple[np.ndarray, float]:
    """(heavy-line mask, the width threshold in px)."""
    ink = linework.binarize(page_img)
    sw = thin_stroke_width(ink)
    width_thr = max(HEAVY_FACTOR * sw, sw + HEAVY_MIN_EXTRA)
    d_thr = (width_thr + 1) / 2
    d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    core = (d >= d_thr).astype(np.uint8)
    r = int(np.ceil(d_thr)) + 1
    heavy = (cv2.dilate(core, linework._disk(2 * r + 1)) > 0) & ink
    if GAP_CLOSE_PX > 1:
        heavy = cv2.morphologyEx(heavy.astype(np.uint8), cv2.MORPH_CLOSE,
                                 linework._disk(GAP_CLOSE_PX)) > 0
    return heavy, width_thr


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def bridge_gaps(heavy: np.ndarray, max_gap: float = BRIDGE_PX,
                ink: np.ndarray | None = None) -> np.ndarray:
    """Close gaps in heavy lines: each loose end of the heavy-line skeleton is joined to
    the nearest heavy skeleton pixel ahead of it (within BRIDGE_CONE_DEG of its direction,
    at most max_gap away) that isn't just the same line bending back.  Real district lines
    are closed; on the scans they break where the pen thinned or ran along a block outline.
    A wrong bridge only cuts off a sliver that then inherits its neighbours' code, while a
    gap merges two districts."""
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize

    sk = skeletonize(heavy)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    if n < 2:
        return heavy
    sk &= (stats[:, cv2.CC_STAT_AREA] >= BRIDGE_MIN_LINE_PX)[comp] & (comp > 0)
    pts = np.argwhere(sk)
    if len(pts) < 2:
        return heavy
    nb = sum(np.roll(np.roll(sk, dy, 0), dx, 1) for dy, dx in _NB) * sk
    ends = np.argwhere(sk & (nb == 1))
    tree = cKDTree(pts)
    h, w = sk.shape
    out = heavy.astype(np.uint8).copy()
    cos_lim = np.cos(np.radians(BRIDGE_CONE_DEG))
    reach = int(3 * max_gap)
    for ey, ex in ends:
        # walk the skeleton from the end (BFS) to find the line's own pixels and direction
        seen = {(ey, ex): 0}
        frontier, back = [(ey, ex)], None
        for step in range(1, reach + 1):
            nxt = []
            for y, x in frontier:
                for dy, dx in _NB:
                    q = (y + dy, x + dx)
                    if 0 <= q[0] < h and 0 <= q[1] < w and sk[q] and q not in seen:
                        seen[q] = step
                        nxt.append(q)
            if step == BRIDGE_BACK_PX and nxt:
                back = nxt[0]
            if not nxt:
                break
            frontier = nxt
        if back is None:
            continue        # a short spur: no direction
        v = np.array([ey - back[0], ex - back[1]], float)
        if np.linalg.norm(v) < BRIDGE_STRAIGHT * BRIDGE_BACK_PX:
            continue        # curls (lettering touching a line), no clear direction
        v /= np.linalg.norm(v)
        best, best_d = None, np.inf
        for i in tree.query_ball_point((ey, ex), max_gap):
            q = tuple(pts[i])
            if q in seen:
                continue
            u = np.array([q[0] - ey, q[1] - ex], float)
            dist = float(np.linalg.norm(u))
            if dist < 2 or u @ v < cos_lim * dist:
                continue
            if dist < best_d:
                best, best_d = q, dist
        if best is not None:
            cv2.line(out, (int(ex), int(ey)), (int(best[1]), int(best[0])), 1, 3)
    out = out > 0
    if ink is not None and INK_SHARE is not None:
        out = _ink_bridges(heavy, out, ink)
    return out


def _ink_bridges(heavy: np.ndarray, out: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Drop bridges that mostly cross blank paper: a district line that broke where it ran
    along a block outline has a thin printed line under its gap; one across a street has none."""
    new = (out & ~heavy).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
    if n < 2:
        return out
    k = 2 * INK_NEAR_PX + 1
    ink1 = cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    on = np.bincount(comp[ink1 & (comp > 0)], minlength=n)
    bad = on / np.maximum(stats[:, cv2.CC_STAT_AREA], 1) < INK_SHARE
    bad[0] = False
    return out & ~bad[comp]


def walls(page_img: np.ndarray) -> tuple[np.ndarray, float]:
    """(heavy lines with their gaps bridged, the heavy width threshold): what bounds regions."""
    heavy, width_thr = _heavy(page_img)
    ink = linework.binarize(page_img) if INK_SHARE is not None else None
    return bridge_gaps(heavy, BRIDGE_PX, ink=ink), width_thr


def _component_polygon(mask: np.ndarray, ox: int, oy: int) -> Polygon | None:
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    hier = hier[0]
    shells = [i for i in range(len(cs)) if hier[i][3] < 0]
    outer = max(shells, key=lambda i: cv2.contourArea(cs[i]))
    if len(cs[outer]) < 3:
        return None
    holes = [cs[i][:, 0, :] + (ox, oy) for i in range(len(cs))
             if hier[i][3] == outer and len(cs[i]) >= 3 and cv2.contourArea(cs[i]) >= MIN_HOLE_PX]
    poly = Polygon(cs[outer][:, 0, :] + (ox, oy), holes).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.simplify(SIMPLIFY_PX)
    return poly if poly.is_valid and not poly.is_empty else None


def regions(page_img: np.ndarray, exclude: np.ndarray | None = None,
            wall: tuple[np.ndarray, float] | None = None) -> list[Polygon]:
    """Polygons (page px) of the connected areas between heavy lines, inside the neatline
    (exclude: True outside it / in legends).  Solid heavy blobs much thicker than a line (a
    tiny district box filled by its own bold code) are regions too."""
    heavy, width_thr = _heavy(page_img)
    bridged = (wall or walls(page_img))[0]
    inside = ~exclude if exclude is not None else np.ones(heavy.shape, bool)
    k = int(BLOB_FACTOR * width_thr) | 1
    blobs = cv2.morphologyEx(heavy.astype(np.uint8), cv2.MORPH_OPEN, linework._disk(k)) > 0
    free = _polygons(~bridged & inside, 4, MIN_REGION_PX)
    solid = _polygons(blobs & inside, 8, max(MIN_REGION_PX, 4 * k * k))   # not line corners
    if solid:   # small holes are filled in free regions; a blob's hole must stay open
        tree = STRtree(solid)
        for i, f in enumerate(free):
            hit = [solid[int(j)] for j in tree.query(f, predicate="intersects")]
            if hit:
                g = f.difference(unary_union(hit))
                free[i] = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
    return free + solid


def _polygons(mask: np.ndarray, connectivity: int, min_area: float) -> list[Polygon]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        p = _component_polygon(lab[y:y + h, x:x + w] == i, x, y)
        if p is not None and p.area >= min_area:
            out.append(p)
    return out


# --- assignment -------------------------------------------------------------------------

def _region_of(pt: Point, regs: list[Polygon], tree: STRtree) -> int | None:
    """The (smallest) region containing the label; else, as the label's own letters are heavy
    ink, the region covering most of a disk around it."""
    inside = [int(i) for i in tree.query(pt, predicate="intersects")]
    if inside:      # e.g. a boxed code whose bold letters fill the box (a blob region)
        return min(inside, key=lambda i: regs[i].area)
    disk = pt.buffer(SNAP_RADIUS_PX)
    cand = [(regs[int(i)].intersection(disk).area, int(i))
            for i in tree.query(disk, predicate="intersects")]
    if cand:
        return max(cand)[1]
    i = tree.query_nearest(pt, max_distance=SNAP_PX)
    return int(i[0]) if len(i) else None


def _voronoi_pieces(region, pts: list[DistrictLabel]) -> dict[str, list]:
    cells = voronoi_polygons(MultiPoint([(l.x, l.y) for l in pts]), extend_to=region)
    pieces: dict[str, list] = {}
    for cell in cells.geoms:
        l = min(pts, key=lambda l: cell.distance(Point(l.x, l.y)))
        piece = cell.intersection(region)
        if not piece.is_empty:
            pieces.setdefault(l.code, []).append(piece)
    return pieces


def _raster(geom, ox: int, oy: int, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for g in getattr(geom, "geoms", [geom]):
        if g.is_empty or g.geom_type != "Polygon":
            continue
        cv2.fillPoly(m, [np.int32(np.array(g.exterior.coords) - [ox, oy])], 1)
        for r in g.interiors:
            cv2.fillPoly(m, [np.int32(np.array(r.coords) - [ox, oy])], 0)
    return m > 0


def _geo_pieces(region, pts: list[DistrictLabel], wall: np.ndarray) -> dict[str, list] | None:
    """Like _voronoi_pieces, but by distance travelled through the region around heavy lines
    (geodesic Voronoi, i.e. seeded region growing from the labels): a district line with a gap
    still keeps each side with the label printed on it, and only the area by the gap is
    contested.  Computed on a GEO_PX grid; None when the region is too small for it."""
    from scipy.ndimage import distance_transform_edt
    from skimage.graph import MCP_Geometric

    f = GEO_PX
    x0, y0, x1, y1 = (int(v) for v in region.bounds)
    x0, y0 = max(0, x0 - f), max(0, y0 - f)
    x1, y1 = min(wall.shape[1], x1 + f), min(wall.shape[0], y1 + f)
    hh, ww = (y1 - y0) // f, (x1 - x0) // f
    if hh < 2 or ww < 2:
        return None
    fine = _raster(region, x0, y0, x1 - x0, y1 - y0)[:hh * f, :ww * f]
    inreg = fine.reshape(hh, f, ww, f).any(axis=(1, 3))
    free = (fine & ~wall[y0:y0 + hh * f, x0:x0 + ww * f]).reshape(hh, f, ww, f).all(axis=(1, 3))
    if not free.any():
        return None
    _, (iy, ix) = distance_transform_edt(~free, return_indices=True)
    codes = sorted({l.code for l in pts})
    cost = np.where(free, 1.0, np.inf)
    dist = np.full((len(codes), hh, ww), np.inf)
    for k, c in enumerate(codes):
        seeds = set()
        for l in pts:
            if l.code == c:
                cy = min(hh - 1, max(0, int((l.y - y0) / f)))
                cx = min(ww - 1, max(0, int((l.x - x0) / f)))
                seeds.add((int(iy[cy, cx]), int(ix[cy, cx])))
        dist[k] = MCP_Geometric(cost).find_costs(list(seeds))[0]
    lab = np.where(np.isfinite(dist.min(axis=0)), dist.argmin(axis=0) + 1, 0)
    # cells no label reaches (walled off) and the walls themselves: the nearest labelled cell
    if (lab == 0).all():
        return None
    _, (ny, nx) = distance_transform_edt(lab == 0, return_indices=True)
    lab = np.where(inreg, lab[ny, nx], 0)
    from affine import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape

    cells: dict[int, list] = {}
    for gj, v in shapes(lab.astype(np.int32), mask=lab > 0, transform=Affine(f, 0, x0, 0, f, y0)):
        cells.setdefault(int(v), []).append(shape(gj))
    pieces: dict[str, list] = {}
    for v, gs in cells.items():
        g = _areal(unary_union(gs).intersection(region))
        if g is not None:
            pieces.setdefault(codes[v - 1], []).append(g)
    return pieces


def _areal(g):
    """The polygonal part of a geometry (intersections can add slivers of line), or None."""
    parts = [p for p in getattr(g, "geoms", [g])
             if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]
    return unary_union(parts) if parts else None


def _pieces(region, pts: list[DistrictLabel], wall: np.ndarray | None) -> dict[str, list]:
    if GEO_SPLIT and wall is not None and len({l.code for l in pts}) > 1:
        got = _geo_pieces(region, pts, wall)
        if got is not None:
            return got
    return _voronoi_pieces(region, pts)


def _split(region: Polygon, labs: list[DistrictLabel],
           wall: np.ndarray | None = None) -> list[tuple[Polygon, str, float]]:
    """A region carrying conflicting codes, split by a Voronoi of its labels.  Cells of
    NO_PLURALITY (P) labels are capped to a SPLIT_P_RADIUS_PX disk round the label: parks and
    public sites are small and labelled inside themselves, and an uncapped cell handed whole
    neighbourhoods to a park whose outline didn't close the region.  Likewise, when one code
    dominates the region's labels, the others are capped to SPLIT_MINOR_RADIUS_PX."""
    best: dict[tuple[float, float], DistrictLabel] = {}
    for l in labs:      # one label per position (highest confidence)
        k = (l.x, l.y)
        if k not in best or l.confidence > best[k].confidence:
            best[k] = l
    pts = list(best.values())
    conf: dict[str, float] = {}
    for l in pts:
        conf[l.code] = max(conf.get(l.code, 0.0), l.confidence)
    pieces = _pieces(region, pts, wall)
    radius = {l.code: SPLIT_P_RADIUS_PX for l in pts if l.code in NO_PLURALITY}
    count: dict[str, int] = {}
    for l in pts:
        if l.code not in radius:
            count[l.code] = count.get(l.code, 0) + 1
    top = sorted(count.values(), reverse=True) + [0]
    lone_cap = LONE_MAIN is not None and top[0] >= LONE_MAIN
    if len(count) > 1 and (top[0] >= SPLIT_MAJORITY * top[1] or lone_cap):
        main = max(count, key=count.get)
        for c in count:
            if c != main and count[c] <= SPLIT_LONE:
                radius[c] = SPLIT_MINOR_RADIUS_PX
    capped = [l for l in pts if l.code in radius]
    others = [l for l in pts if l.code not in radius]
    if capped and others:
        disks = {c: unary_union([Point(l.x, l.y).buffer(radius[c]) for l in capped if l.code == c])
                 for c in radius}
        public = {c: unary_union(ps).intersection(disks[c]) for c, ps in pieces.items()
                  if c in radius}
        taken = unary_union([g for g in public.values() if not g.is_empty])
        rest = region.difference(taken) if not taken.is_empty else region
        pieces = {c: [g] for c, g in public.items() if not g.is_empty}
        if not rest.is_empty:
            for c, ps in _pieces(rest, others, wall).items():
                pieces.setdefault(c, []).extend(ps)
    return [(unary_union(ps), c, conf[c] * SPLIT_CONF) for c, ps in pieces.items()]


def labels_by_region(regs: list[Polygon], labs: list[DistrictLabel]) -> dict[int, list[DistrictLabel]]:
    """Region index -> the labels that sit in it."""
    tree = STRtree(regs)
    per: dict[int, list[DistrictLabel]] = {}
    for l in labs:
        i = _region_of(Point(l.x, l.y), regs, tree)
        if i is not None:
            per.setdefault(i, []).append(l)
    return per


def assign_regions(regs: list[Polygon], labs: list[DistrictLabel], *, touch_px: float = TOUCH_PX,
                   reads: dict[int, tuple[str, float]] | None = None,
                   wall: np.ndarray | None = None) -> list[tuple[Polygon, str | None, float]]:
    """Each region with its code and confidence (see module docstring).  Pass labels of
    one kind (use or height).  reads: {region index: (code, confidence)} from the per-region
    read (regionread.read_regions) for unlabelled regions; they count as coded neighbours."""
    if not regs:
        return []
    tree = STRtree(regs)
    per = labels_by_region(regs, labs)
    out: list[tuple[Polygon, str | None, float]] = []
    labelled: dict[int, list[tuple[Polygon, str, float]]] = {}
    for i, ls in per.items():
        codes = {l.code for l in ls}
        if len(codes) == 1:
            labelled[i] = [(regs[i], ls[0].code, max(l.confidence for l in ls))]
        else:
            labelled[i] = _split(regs[i], ls, wall)
    code_of: dict[int, tuple[str, float]] = {}
    for i, pieces in labelled.items():
        if len(pieces) == 1:
            code_of[i] = (pieces[0][1], pieces[0][2])
    for i, (c, cf) in (reads or {}).items():
        if i not in labelled:
            code_of[i] = (c, cf * READ_CONF)
    near = [r.buffer(touch_px) for r in regs]

    def neighbours(i):
        """(code, conf, contact) of each coded neighbour; contact = overlap area of the two
        regions' touch buffers, proportional to their shared boundary length."""
        out_ = []
        for j in tree.query(near[i], predicate="intersects"):
            j = int(j)
            if j == i:
                continue
            if j in code_of:
                c, cf = code_of[j]
                out_.append((c, cf, near[i].intersection(near[j]).area))
            elif j in labelled:                       # split region: each piece separately
                for pc, c, cf in labelled[j]:
                    if pc.intersects(near[i]):
                        out_.append((c, cf, near[i].intersection(pc.buffer(touch_px)).area))
        return out_

    todo = [i for i in range(len(regs)) if i not in labelled and i not in code_of]
    while todo:
        # 1. unanimous inheritance, repeated until nothing changes (chains of unlabelled regions)
        changed = True
        while changed:
            changed, rest = False, []
            for i in todo:
                nb = neighbours(i)
                codes = {c for c, _, _ in nb}
                if len(codes) == 1:
                    code_of[i] = (codes.pop(), min(cf for _, cf, _ in nb) * INHERIT_CONF)
                    changed = True
                else:
                    rest.append(i)
            todo = rest
        # 2. disagreeing neighbours: the code with the longest contact, at low confidence (a
        #    tie stays unknown); then go back to unanimous inheritance from there
        picked = {}
        for i in todo:
            contact: dict[str, float] = {}
            for c, _, a in neighbours(i):
                if c not in NO_PLURALITY:
                    contact[c] = contact.get(c, 0.0) + a
            if not contact:
                continue
            ranked = sorted(contact.items(), key=lambda kv: -kv[1]) + [("", 0.0)]
            if ranked[0][1] > ranked[1][1] * PLURALITY_MARGIN:
                picked[i] = (ranked[0][0], PLURALITY_CONF)
        if not picked:
            break
        code_of.update(picked)
        todo = [i for i in todo if i not in picked]
    for i, r in enumerate(regs):
        if i in labelled:
            out.extend(labelled[i])
        elif i in code_of:
            out.append((r, *code_of[i]))
        else:
            out.append((r, None, 0.0))
    return out


# --- parcels ----------------------------------------------------------------------------

def _poly_to_world(poly, transform: SheetTransform):
    import shapely

    g = shapely.segmentize(poly, SEGMENT_PX)
    coords = shapely.get_coordinates(g)
    if not len(coords):
        return None
    w = transform.to_world(coords)
    out = shapely.set_coordinates(g, w)
    return out if out.is_valid else out.buffer(0)


def closed_parks(assigned: list[tuple[Polygon, str | None, float]], closed: list[Polygon],
                 labs: list[DistrictLabel]) -> list[tuple[Polygon, str | None, float]]:
    """assigned, with each P label's region among `closed` (regions with every gap bridged) made
    P, when only P is printed in it and it is at most PARK_MAX_PX (see PARK_CLOSED); the other
    pieces give way to it."""
    parks = []
    for i, ls in labels_by_region(closed, labs).items():
        if {l.code for l in ls} == {"P"} and closed[i].area <= PARK_MAX_PX:
            parks.append((closed[i], max(l.confidence for l in ls)))
    if not parks:
        return assigned
    taken = unary_union([p for p, _ in parks])
    out = []
    for poly, code, cf in assigned:
        rest = _areal(poly.difference(taken)) if code != "P" else poly.difference(taken)
        if rest is not None and not rest.is_empty:
            out.append((rest, code, cf))
    return out + [(p, "P", cf) for p, cf in parks]


def _block_of(parcels) -> "np.ndarray":
    if "block_num" in parcels.columns:
        return parcels["block_num"].astype(str).to_numpy()
    return parcels["blklot"].astype(str).str[:-3].to_numpy()


def supporting_labels(regs: list[Polygon], labs: list[DistrictLabel],
                      assigned: list[tuple[Polygon, str | None, float]]) -> list[np.ndarray]:
    """Per assigned piece, the page px (n x 2) of the printed labels it takes its code from:
    the labels of that code sitting in the piece's own region.  Empty for a region that
    inherited its code or was named by the per-region read."""
    per = labels_by_region(regs, labs)
    tree = STRtree(regs)
    out = []
    for poly, code, _ in assigned:
        pts = np.zeros((0, 2))
        if code is not None and not poly.is_empty:
            rp = poly.representative_point()
            home = [int(i) for i in tree.query(rp, predicate="intersects")]
            if home:
                i = min(home, key=lambda i: regs[i].area)
                src = [(l.x, l.y) for l in per.get(i, []) if l.code == code]
                if src:
                    pts = np.array(src, float)
        out.append(pts)
    return out


def _majority(parcels, assigned, transform: SheetTransform, support: list[np.ndarray] | None = None):
    """(code, confidence, label distance) per parcel row: the code covering the largest share
    of its area, and the distance (ft) from the parcel to the nearest printed label backing
    that code (support: page px per assigned piece, supporting_labels); inf when none does."""
    import geopandas as gpd

    n = len(parcels)
    codes, confs, dists = [None] * n, [0.0] * n, [np.inf] * n
    support = support if support is not None else [np.zeros((0, 2))] * len(assigned)
    rows = [(_poly_to_world(p, transform), c, cf,
             transform.to_world(s) if len(s) else s) for (p, c, cf), s in zip(assigned, support)]
    rows = [r for r in rows if r[0] is not None and not r[0].is_empty]
    if not rows or not n:
        return codes, confs, dists
    rg = gpd.GeoDataFrame({"code": [r[1] if r[1] is not None else "" for r in rows],
                           "rconf": [r[2] for r in rows], "rid": np.arange(len(rows))},
                          geometry=[r[0] for r in rows], crs=config.WORK_CRS)
    pg = gpd.GeoDataFrame({"pid": np.arange(n)}, geometry=list(parcels.geometry), crs=config.WORK_CRS)
    ov = gpd.overlay(pg, rg, how="intersection", keep_geom_type=False)
    if ov.empty:
        return codes, confs, dists
    ov["area"] = ov.geometry.area
    ov["wconf"] = ov["area"] * ov["rconf"]
    g = ov.groupby(["pid", "code"])[["area", "wconf"]].sum().reset_index()
    g = g.sort_values("area", ascending=False).drop_duplicates("pid")
    rids = ov.groupby(["pid", "code"])["rid"].agg(list)
    parea = pg.geometry.area.to_numpy()
    cent = pg.geometry.centroid
    cx, cy = cent.x.to_numpy(), cent.y.to_numpy()
    for pid, code, area, wconf in g[["pid", "code", "area", "wconf"]].itertuples(index=False):
        if code == "":
            continue
        share = min(1.0, area / parea[pid]) if parea[pid] > 0 else 0.0
        codes[pid], confs[pid] = code, float(wconf / area * share)
        pts = [rows[r][3] for r in rids[(pid, code)] if len(rows[r][3])]
        if pts:
            p = np.vstack(pts)
            dists[pid] = float(np.hypot(p[:, 0] - cx[pid], p[:, 1] - cy[pid]).min())
    return codes, confs, dists


def _fallback(parcels, labs: list[DistrictLabel], transform: SheetTransform):
    """Nearest label inside each parcel's own block (FALLBACK_CONF), else None."""
    n = len(parcels)
    codes, confs, dists = [None] * n, [0.0] * n, [np.inf] * n
    if not labs or not n:
        return codes, confs, dists
    w = transform.to_world(np.array([[l.x, l.y] for l in labs]))
    pts = [Point(x, y) for x, y in w]
    tree = STRtree(pts)
    blocks = _block_of(parcels)
    geoms = list(parcels.geometry)
    by_block: dict[str, list[int]] = {}
    for i, b in enumerate(blocks):
        by_block.setdefault(b, []).append(i)
    for b, idx in by_block.items():
        bg = unary_union([geoms[i] for i in idx]).buffer(FALLBACK_BLOCK_FT)
        cand = [int(j) for j in tree.query(bg, predicate="intersects")]
        if not cand:
            continue
        for i in idx:
            g = geoms[i]
            j = min(cand, key=lambda j: (g.distance(pts[j]), g.centroid.distance(pts[j])))
            codes[i], confs[i] = labs[j].code, FALLBACK_CONF
            dists[i] = float(g.centroid.distance(pts[j]))
    return codes, confs, dists


ROLE_MIN_LABELS = 8     # fewer district codes than this: not a use or height sheet (SU, SSD)
ROLE_SHARE = 0.25       # a sheet reads a kind when at least this share of its codes are that kind


def sheet_role(labs: list[DistrictLabel]) -> str:
    """"use", "height" or "skip" for one sheet from its read labels.  Several series bind use,
    height-and-bulk, special-use and sign sheets together (1980: sheets 1..13, 1H..13H, SU,
    SSD), so each parcel must be read on a sheet of the right kind.  A sheet with enough use
    codes is a use sheet (even if it prints heights too); else enough height codes make it a
    height sheet; else it is skipped."""
    n_use = len(use_code_labels(labs))
    n_h = sum(lb.kind == "height" for lb in labs)
    n = n_use + n_h
    if n < ROLE_MIN_LABELS:
        return "skip"
    if n_use >= ROLE_MIN_LABELS and n_use >= ROLE_SHARE * n:
        return "use"
    if n_h >= ROLE_MIN_LABELS and n_h >= ROLE_SHARE * n:
        return "height"
    return "skip"


def sheet_roles(series: str, sheets: list[str]) -> dict[str, str]:
    return {sid: sheet_role(read_district_labels(series, sid)) for sid in sheets}


_OVERLAY = re.compile(r"S\.?\s*U\.?\s*D\b|\bRUSD\b|SPECIAL\s+USE")


def use_code_labels(labs: list[DistrictLabel]) -> list[DistrictLabel]:
    """Use labels that name a use district.  Special-use-district overlays ("NORTH OF MARKET
    RES. S.U.D.", "RINCON HILL RES. SUD") are printed inside use districts on the 1990s-2000s
    maps and would otherwise split them; "NC-2 & FINANCIAL RUSD" keeps its district, NC-2."""
    out = []
    for lb in labs:
        if lb.kind != "use":
            continue
        code = lb.code.split("&")[0].strip()
        if not code or _OVERLAY.search(code):
            continue
        out.append(lb if code == lb.code else lb.model_copy(update={"code": code}))
    return out


def assign_parcels(parcels, sheet, transform: SheetTransform | None = None, *,
                   client: llm.LLMClient | None = None, all_parcels=None):
    """DataFrame (blklot, code, height_code, confidence, use/height confidence and label_ft)
    for `parcels` (EPSG:2227) from one sheet.  `sheet` is the sheet's work dir
    (work/<series>/<sheet_id>); transform defaults to its transform.json.  Confidence is for
    the series' role (height series: height_code).  Unlabelled regions are read per region
    (regionread; client: the vision client) unless regionread.ENABLED is off; which regions
    are worth a read is judged against all_parcels (default `parcels`), so a sheet reads a
    lot the same whichever other lots it is asked about."""
    from sfzmaps.districts import regionread

    import pandas as pd

    from sfzmaps.districts.series import SERIES

    d = Path(sheet)
    series, sid = d.parent.name, d.name
    if transform is None:
        transform = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
    labs = read_district_labels(series, sid)
    img = cv2.imread(str(d / "page.png"), cv2.IMREAD_GRAYSCALE)
    wall = walls(img)
    exclude = reading.exclusion_mask(img, _page_info(d))
    regs = regions(img, exclude, wall)
    closed = None
    if PARK_CLOSED and INK_SHARE is not None:
        heavy, thr = _heavy(img)
        closed = regions(img, exclude, (bridge_gaps(heavy, BRIDGE_PX), thr))
    every = parcels if all_parcels is None else all_parcels
    ptree = STRtree(list({g.wkb: g for g in every.geometry if g is not None and not g.is_empty}.values()))
    share: dict[int, float] = {}

    def parcel_share(i):
        """Share of region i (in world ft) covered by parcels: streets, water and margins
        aren't worth a read."""
        if i not in share:
            w = _poly_to_world(regs[i], transform)
            if w is None or w.is_empty or w.area <= 0:
                share[i] = 0.0
            else:
                hit = ptree.geometries.take(ptree.query(w, predicate="intersects"))
                share[i] = sum(w.intersection(g).area for g in hit) / w.area
        return share[i]

    res = {}
    for kind in ("use", "height"):
        ls = use_code_labels(labs) if kind == "use" else [lb for lb in labs if lb.kind == kind]
        if not ls:
            res[kind] = ([None] * len(parcels), [0.0] * len(parcels), [np.inf] * len(parcels))
        elif len(regs) >= MIN_REGIONS:
            reads = {}
            if regionread.ENABLED:
                per = labels_by_region(regs, ls)
                todo = [i for i in range(len(regs)) if i not in per and parcel_share(i) >= READ_PARCEL_SHARE]
                reads = regionread.read_regions(img, regs, todo, ls, kind, client=client)
            assigned = assign_regions(regs, ls, reads=reads, wall=wall[0])
            if closed is not None and kind == "use":
                assigned = closed_parks(assigned, closed, ls)
            res[kind] = _majority(parcels, assigned, transform, supporting_labels(regs, ls, assigned))
        else:
            res[kind] = _fallback(parcels, ls, transform)
    info = SERIES.get(series)
    role = "height" if info is not None and info.role == "height" else "use"
    return pd.DataFrame({"blklot": parcels["blklot"].to_numpy(),
                         "code": pd.Series([hyphenate(c) for c in res["use"][0]], dtype=object),
                         "height_code": pd.Series(res["height"][0], dtype=object),
                         "confidence": res[role][1],
                         "use_confidence": res["use"][1], "height_confidence": res["height"][1],
                         "use_label_ft": res["use"][2], "height_label_ft": res["height"][2]})
    return out
