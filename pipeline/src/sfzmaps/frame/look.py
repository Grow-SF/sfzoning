"""Pictures of the frame finder's outputs, for the owner's eye.

A sheet picture is the scan with its paper white and each ink px coloured by q, the chance it is deleted, from blue
(kept) to red (deleted), as strongly as the px is inked; the boundary of the kept region (removed blocks' too) is drawn
on top in magenta.  Pages are shrunk (Pillow BOX) before they are coloured and never enlarged.

Under `out/look/`: one picture per map page at `LONG_SIDE` (`<series>/<sheet>.png`), one mosaic per series
(`<series>.png`; a labelled sheet's caption shows its held-out `frame_kept` and `map_cut`) and `closeups.png`, crops of
`CLOSEUPS` at full scan resolution (never shrunk).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from sfzmaps.frame import data, run
from sfzmaps.frame.data import OUT, SheetId

log = logging.getLogger(__name__)

LONG_SIDE = 1400          # px, a sheet picture's longest side
MOSAIC_SIDE = 320         # px, a series mosaic tile's longest side
MOSAIC_COLUMNS = 8
CAPTION_PX = 40           # px under each tile for its two caption lines
GAP_PX = 6
FONT_PX = 14
KEPT_RGB = (0, 0, 255)
DELETED_RGB = (255, 0, 0)
CONTOUR_RGB = (255, 0, 255)
PAPER_RGB = (255, 255, 255)
BACKGROUND_RGB = (200, 200, 200)   # around the tiles, so a page's white paper shows where it ends

# (series, sheet, (x0, y0, x1, y1) in page px, caption): places that have gone wrong before.
CLOSEUPS: list[tuple[str, str, tuple[int, int, int, int], str]] = [
    ("1921", "20180102094204401", (54, 82, 354, 382), "1921 thin rule and arc, top-left"),
    ("1921", "20180102094204401", (54, 1506, 354, 1806), "1921 thin rule, mid-left"),
    ("1929", "20180102100804566-p09", (1562, 740, 2179, 840), "1929 p09 Ocean Beach bar"),
    ("1948", "3", (563, 5702, 1263, 6402), "1948/3 commercial bands"),
    ("1948", "4", (680, 5000, 1160, 5800), "1948/4 Great Highway"),
    ("1948", "4", (4334, 5905, 4834, 6305), "1948/4 blocks 1724-1726 and frame bar"),
    ("1948", "8", (0, 1500, 600, 4500), "1948/8 left edge"),
]


def _shrink(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`a` resized (Pillow BOX) to `size` = (width, height), float32."""
    a = np.ascontiguousarray(a, np.float32)
    if (a.shape[1], a.shape[0]) == size:
        return a
    return np.asarray(Image.fromarray(a).resize(size, Image.BOX), np.float32)


def sheet_picture(lum: np.ndarray, q: np.ndarray, deleted: np.ndarray, side: int = LONG_SIDE) -> Image.Image:
    """The page coloured by `q` (see the module), its longest side `side` px (a smaller page keeps its own size:
    scans are never enlarged); `deleted` is the cut's mask, whose complement's boundary is drawn."""
    if q.shape != lum.shape or deleted.shape != lum.shape:
        raise ValueError(f"q {q.shape} and deleted {deleted.shape} are not the page's shape {lum.shape}")
    h, w = lum.shape
    s = min(1.0, side / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    qs = np.clip(_shrink(q, size), 0.0, 1.0)[..., None]
    ink = np.clip(data.ink_weight(_shrink(lum, size)), 0.0, 1.0)[..., None]
    colour = qs * np.array(DELETED_RGB, np.float32) + (1.0 - qs) * np.array(KEPT_RGB, np.float32)
    rgb = (1.0 - ink) * np.array(PAPER_RGB, np.float32) + ink * colour
    kept = _shrink(deleted.astype(np.float32), size) < 0.5
    edge = ndimage.binary_dilation(kept & ~ndimage.binary_erosion(kept, border_value=1))
    rgb[edge] = CONTOUR_RGB
    return Image.fromarray(np.round(rgb).astype(np.uint8), "RGB")


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default(size=FONT_PX)


def _load_outputs(out: Path, sid: SheetId) -> tuple[np.ndarray, np.ndarray] | None:
    """(q in [0, 1], deleted) of a page, or None (logged) when it has no outputs."""
    d = run.sheet_dir(out, sid)
    if not (d / run.PROB_FILE).exists() or not (d / run.OUTSIDE_PNG).exists():
        log.info("no outputs for %s/%s; left out of the pictures", *sid)
        return None
    with np.load(d / run.PROB_FILE) as z:
        q = z["q"].astype(np.float32) / 255.0
    with Image.open(d / run.OUTSIDE_PNG) as im:
        deleted = np.asarray(im.convert("L")) > 0
    return q, deleted


def _held_out(out: Path) -> dict[SheetId, dict]:
    path = out / run.FOLDS_FILE
    if not path.exists():
        return {}
    return {(r["series"], r["sheet"]): r["scores"] for r in json.loads(path.read_text())["sheets"]}


def _mosaic(tiles: list[tuple[Image.Image, tuple[str, str]]], side: int, columns: int, path: Path) -> Path:
    """The tiles on a grid of `columns`, each cell side x side plus its caption."""
    font = _font()
    cols = max(1, min(columns, len(tiles)))
    rows = max(1, math.ceil(len(tiles) / cols))
    cell_w, cell_h = side + GAP_PX, side + CAPTION_PX + GAP_PX
    canvas = Image.new("RGB", (cols * cell_w + GAP_PX, rows * cell_h + GAP_PX), BACKGROUND_RGB)
    draw = ImageDraw.Draw(canvas)
    for i, (img, (line1, line2)) in enumerate(tiles):
        x, y = GAP_PX + (i % cols) * cell_w, GAP_PX + (i // cols) * cell_h
        canvas.paste(img, (x, y))
        draw.text((x, y + side + 2), line1, fill=(0, 0, 0), font=font)
        draw.text((x, y + side + 2 + FONT_PX + 4), line2, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    canvas.save(tmp)
    tmp.replace(path)
    return path


def series_mosaics(out: Path = OUT) -> list[Path]:
    """One mosaic per series under `out/look/` of every map page that has outputs (see the module); each page's own
    picture is saved beside it."""
    held = _held_out(out)
    by_series: dict[str, list[SheetId]] = {}
    for sid in data.map_pages(data.PAGE_KINDS, data.WORK):
        by_series.setdefault(sid[0], []).append(sid)
    paths = []
    for series, sids in sorted(by_series.items()):
        tiles = []
        for sid in sids:
            loaded = _load_outputs(out, sid)
            if loaded is None:
                continue
            lum = data.load_luminance(data.page_path(*sid, work=data.WORK))
            full = out / "look" / series / f"{sid[1]}.png"
            full.parent.mkdir(parents=True, exist_ok=True)
            sheet_picture(lum, *loaded, LONG_SIDE).save(full)
            sc = held.get(sid)
            line2 = "" if sc is None else f"held-out kept {sc['frame_kept']:.1f} cut {sc['map_cut']:.2f}"
            tiles.append((sheet_picture(lum, *loaded, MOSAIC_SIDE), (sid[1], line2)))
        if not tiles:
            log.info("series %s has no outputs; no mosaic", series)
            continue
        paths.append(_mosaic(tiles, MOSAIC_SIDE, MOSAIC_COLUMNS, out / "look" / f"{series}.png"))
        log.info("mosaic %s: %d pages", paths[-1], len(tiles))
    return paths


def closeups(out: Path = OUT, items=CLOSEUPS) -> Path:
    """`out/look/closeups.png`: each crop of `items` at the scan's own resolution, outside ink tinted red, the kept
    region's outline drawn, captioned, stacked.  A page with no outputs is skipped (logged)."""
    font = _font()
    panels = []
    for series, sheet, (x0, y0, x1, y1), caption in items:
        sid = (series, sheet)
        loaded = _load_outputs(out, sid)
        if loaded is None:
            continue
        deleted = loaded[1]
        lum = data.load_luminance(data.page_path(*sid, work=data.WORK))
        h, w = lum.shape
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            log.info("%s lies outside %s/%s; skipped", caption, *sid)
            continue
        ink = np.clip(data.ink_weight(lum[y0:y1, x0:x1]), 0.0, 1.0)[..., None]
        tint = np.where(deleted[y0:y1, x0:x1, None], np.array(DELETED_RGB, np.float32),
                        np.zeros(3, np.float32))
        img = Image.fromarray(np.round((1.0 - ink) * 255.0 + ink * tint).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img)
        outline = json.loads((run.sheet_dir(out, sid) / run.OUTLINE_FILE).read_text())["polygons"]
        for poly in outline:
            for ring in [poly["exterior"], *poly["holes"]]:
                draw.line([(x - x0, y - y0) for x, y in ring], fill=CONTOUR_RGB, width=1)
        panels.append((img, f"{caption} ({series}/{sheet} x {x0}-{x1}, y {y0}-{y1})"))
    width = max([img.width for img, _ in panels], default=GAP_PX + 200)
    height = sum(img.height + CAPTION_PX + GAP_PX for img, _ in panels) + GAP_PX
    canvas = Image.new("RGB", (width + 2 * GAP_PX, max(height, CAPTION_PX)), BACKGROUND_RGB)
    draw = ImageDraw.Draw(canvas)
    if not panels:
        draw.text((GAP_PX, GAP_PX), "no close-up page has outputs", fill=(0, 0, 0), font=font)
    y = GAP_PX
    for img, caption in panels:
        draw.text((GAP_PX, y + 2), caption, fill=(0, 0, 0), font=font)
        y += CAPTION_PX
        canvas.paste(img, (GAP_PX, y))
        y += img.height + GAP_PX
    path = out / "look" / "closeups.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    canvas.save(tmp)
    tmp.replace(path)
    return path
