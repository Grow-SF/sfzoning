"""Pictures of the frame finder's predictions, for the owner's eye.

Each picture is the scan with its paper left white and each ink px coloured by its map probability, from blue (map,
p = 1) to red (frame, p = 0), as strongly as the px is inked (`data.ink_weight`); the probability-1/2 contour is drawn
on top in magenta.  Pages are shrunk (Pillow BOX) before they are coloured, so a whole page is never coloured at full
resolution.

Under `out/look/`: one picture per map page at `LONG_SIDE` (`<series>/<sheet>.png`), one mosaic per series
(`<series>.png`, every map page with a prediction, each tile captioned with its sheet and, for a labelled sheet, its
held-out log-loss over all ink) and `odd_cases.png`, larger tiles of the pages that are unusual (`ODD_CASES`).
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
ODD_SIDE = 700            # px, an odd-case tile's longest side
ODD_COLUMNS = 3
CAPTION_PX = 40           # px under each tile for its two caption lines
GAP_PX = 6
FONT_PX = 14
MAP_RGB = (0, 0, 255)
FRAME_RGB = (255, 0, 0)
CONTOUR_RGB = (255, 0, 255)
PAPER_RGB = (255, 255, 255)
BACKGROUND_RGB = (200, 200, 200)   # around the tiles, so a page's white paper shows where it ends

# Pages called out for a closer look (spec, "Pictures").  The 1948 page with several framed maps on one photo is
# added once the mosaics exist and it can be found in them.
ODD_CASES: list[SheetId] = [
    ("1948", "7"),                                     # no frame
    ("1935", "20180102102853571-p15"),                 # half-page scan
    ("1980", "20180102115319417-p18"),                 # cropped 1980s scans
    ("1980-1990-zoning", "1980-1990-zoning-p07"),
    ("1921", "20180102094727628"),                     # the Section 12 duplicates: the same scan twice
    ("1921", "20180102100525362"),
]


def _shrink(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`a` resized (Pillow BOX) to `size` = (width, height), float32."""
    a = np.ascontiguousarray(a, np.float32)
    if (a.shape[1], a.shape[0]) == size:
        return a
    return np.asarray(Image.fromarray(a).resize(size, Image.BOX), np.float32)


def sheet_picture(lum: np.ndarray, map_prob: np.ndarray, side: int = LONG_SIDE) -> Image.Image:
    """The page coloured by its map probability (see the module), its longest side `side` px (a smaller page keeps
    its own size: scans are never enlarged)."""
    if map_prob.shape != lum.shape:
        raise ValueError(f"map_prob {map_prob.shape} is not the page's shape {lum.shape}")
    h, w = lum.shape
    s = min(1.0, side / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    p = np.clip(_shrink(map_prob, size), 0.0, 1.0)[..., None]
    ink = np.clip(data.ink_weight(_shrink(lum, size)), 0.0, 1.0)[..., None]
    colour = p * np.array(MAP_RGB, np.float32) + (1.0 - p) * np.array(FRAME_RGB, np.float32)
    rgb = (1.0 - ink) * np.array(PAPER_RGB, np.float32) + ink * colour
    region = p[..., 0] >= 0.5
    edge = ndimage.binary_dilation(region & ~ndimage.binary_erosion(region, border_value=1))
    rgb[edge] = CONTOUR_RGB
    return Image.fromarray(np.round(rgb).astype(np.uint8), "RGB")


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default(size=FONT_PX)


def _held_out_loss(out: Path, sid: SheetId) -> float | None:
    path = run.sheet_dir(out, sid) / run.SCORES_FILE
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["log_loss_all_ink"])


def _picture(out: Path, sid: SheetId, side: int, save_full: bool) -> Image.Image | None:
    """The page's picture at `side`, or None (logged) when it has no prediction; with `save_full`, also saved at
    LONG_SIDE to `out/look/<series>/<sheet>.png`."""
    if not (run.sheet_dir(out, sid) / run.PROB_FILE).exists():
        log.info("no prediction for %s/%s; left out of the pictures", *sid)
        return None
    lum = data.load_luminance(data.page_path(*sid, work=data.WORK))
    prob = run.load_prob(out, sid)["map"]
    if save_full:
        full = out / "look" / sid[0] / f"{sid[1]}.png"
        full.parent.mkdir(parents=True, exist_ok=True)
        sheet_picture(lum, prob, LONG_SIDE).save(full)
    return sheet_picture(lum, prob, side)


def _caption(out: Path, sid: SheetId, with_series: bool) -> tuple[str, str]:
    loss = _held_out_loss(out, sid)
    name = f"{sid[0]}/{sid[1]}" if with_series else sid[1]
    return name, "" if loss is None else f"held-out log-loss {loss:.4f}"


def _mosaic(tiles: list[tuple[Image.Image, tuple[str, str]]], side: int, columns: int, path: Path) -> Path:
    """The tiles on a grid of `columns`, each cell side x side plus its caption; an empty grid says so."""
    font = _font()
    cols = max(1, min(columns, len(tiles)))
    rows = max(1, math.ceil(len(tiles) / cols))
    cell_w, cell_h = side + GAP_PX, side + CAPTION_PX + GAP_PX
    canvas = Image.new("RGB", (cols * cell_w + GAP_PX, rows * cell_h + GAP_PX), BACKGROUND_RGB)
    draw = ImageDraw.Draw(canvas)
    if not tiles:
        draw.text((GAP_PX, GAP_PX), "no page here has a prediction", fill=(0, 0, 0), font=font)
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
    """One mosaic per series under `out/look/` of every map page that has a prediction (see the module); each page's
    own picture is saved beside it."""
    by_series: dict[str, list[SheetId]] = {}
    for sid in data.map_pages(data.PAGE_KINDS, data.WORK):
        by_series.setdefault(sid[0], []).append(sid)
    paths = []
    for series, sids in sorted(by_series.items()):
        tiles = []
        for sid in sids:
            img = _picture(out, sid, MOSAIC_SIDE, save_full=True)
            if img is not None:
                tiles.append((img, _caption(out, sid, with_series=False)))
        if not tiles:
            log.info("series %s has no predictions; no mosaic", series)
            continue
        paths.append(_mosaic(tiles, MOSAIC_SIDE, MOSAIC_COLUMNS, out / "look" / f"{series}.png"))
        log.info("mosaic %s: %d pages", paths[-1], len(tiles))
    return paths


def odd_cases(out: Path = OUT) -> Path:
    """`out/look/odd_cases.png`: larger tiles of `ODD_CASES`; a listed page with no prediction is left out (logged)."""
    tiles = []
    for sid in ODD_CASES:
        img = _picture(out, sid, ODD_SIDE, save_full=False)
        if img is not None:
            tiles.append((img, _caption(out, sid, with_series=True)))
    return _mosaic(tiles, ODD_SIDE, ODD_COLUMNS, out / "look" / "odd_cases.png")
