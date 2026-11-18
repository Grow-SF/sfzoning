"""The text detector's chance of text on the solver's grid, from one of two sources (`SOURCES`), each a chance on every
2000 sheet's page (uint8, chance x 255 in its page px):

  all-sheets  (ALL_SHEETS, the default, what the stage's walls read) the one detector of stage `solver-text`, trained on
              every px of every map Terra read, these sheets too: `paths.TEXT_CHANCE_DIR/<sheet>.png`
  held-out    (HELD_OUT, an experiment's) the four held-out detectors of stage `solver-text-reader`, each page px from
              the detector that never saw its quarter of the city or the answered blocks:
              `paths.READER_TEXT_DIR/<sheet>/chance.png`

Each sheet's chance is read through the sheet's corrected placement exactly as the merged page reads its ink
(`page.sheet_px`: the same crop, the frame of `page/frames.json` minus the cut boxes, and the same page px for every
grid px).  A grid px takes the chance only from the sheets that inked it (page < INK_BELOW at the page px it reads, as
the merge reads ink); where several did, the largest chance wins, and a px no sheet inked is 0.  The sheets are those
the warp placed, once they are seen to be the detector's (`paths.check_sheets`).

`text_chance` computes it afresh at every call and writes nothing; the walls stage writes what it used with
`write_text_chance` (WALLS_DIR/text_chance.png: uint8, chance x 255, HEIGHT x WIDTH of the grid).
"""
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config
from sfzmaps.solver import grid, page, paths
from sfzmaps.transform import SheetTransform

log = logging.getLogger(__name__)

TEXT_CHANCE_FILE = "text_chance.png"
CHANCE_FILE = "chance.png"        # a sheet's chance in the reader stage's folder (`textnet.reader_stage.CHANCE_FILE`)
ALL_SHEETS = "all-sheets"         # the one detector of `solver-text`, trained on every px (the stage's walls read it)
HELD_OUT = "held-out"             # the four held-out detectors of `solver-text-reader` (2026-11-16's T1; an option)
SOURCES = (ALL_SHEETS, HELD_OUT)
DEFAULT_SOURCE = ALL_SHEETS


def check_source(source: str | None) -> str:
    """The chance's source: `source`, or DEFAULT_SOURCE for None; an unknown name is an error."""
    source = DEFAULT_SOURCE if source is None else source
    if source not in SOURCES:
        raise ValueError(f"unknown text chance {source!r}: one of {', '.join(SOURCES)}")
    return source


def _chance_files(names, source: str = DEFAULT_SOURCE) -> dict[str, Path]:
    """The chance file of each sheet from `source` (`SOURCES`); a missing one is an error naming its stage."""
    if check_source(source) == ALL_SHEETS:
        files = {name: paths.TEXT_CHANCE_DIR / f"{name}.png" for name in names}
        stage = "the detector first (`sfzmaps solver text`, stage solver-text)"
    else:
        files = {name: paths.READER_TEXT_DIR / name / CHANCE_FILE for name in names}
        stage = "the detectors first (`sfzmaps solver text-reader`, stage solver-text-reader)"
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"can't read {missing[0]} ({len(missing)} of {len(names)} sheets missing): train "
                                + stage)
    return files


def _read(path, shape) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"can't read {path}")
    if a.shape != tuple(shape):
        raise ValueError(f"{path} is {a.shape[1]} x {a.shape[0]} px, its page {shape[1]} x {shape[0]}")
    return a


def text_chance(source: str | None = None) -> np.ndarray:
    """The chance of text at every grid px (uint8, chance x 255, HEIGHT x WIDTH; see the module), from `source`
    (`SOURCES`; None for DEFAULT_SOURCE).  A missing chance file is an error.  Nothing is written."""
    t0 = time.perf_counter()
    names = paths.check_sheets()
    files = _chance_files(names, check_source(source))
    frames = json.loads((paths.PAGE_DIR / "frames.json").read_text())
    cuts = page.load_cuts()
    out = np.zeros((grid.HEIGHT, grid.WIDTH), np.uint8)
    for name in names:
        if name not in frames:
            raise KeyError(f"{paths.PAGE_DIR / 'frames.json'} has no frame for {name}: run `sfzmaps solver page`")
        folder = config.DATA_DIR / "warp" / paths.SERIES / name
        tr = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
        pg = page.load_page(name)
        dark = pg < grid.INK_BELOW
        quad = np.asarray(frames[name], float)
        crop = page.crop_mask(pg.shape, quad, cuts.get(name, []))
        chance = _read(files[name], pg.shape)
        del pg
        for rows, cols, on, yi, xi in page.sheet_px(tr, crop, out.shape, page.grid_box(tr, quad, out.shape)):
            r, c = np.nonzero(on)
            y, x = yi[r, c], xi[r, c]
            inked = dark[y, x]
            r, c, y, x = r[inked], c[inked], y[inked], x[inked]
            if not len(r):
                continue
            band = out[rows, cols]
            band[r, c] = np.maximum(band[r, c], chance[y, x])
        log.info("text chance: %s laid (%.0f s)", name, time.perf_counter() - t0)
    return out


def write_text_chance(folder: Path, chance: np.ndarray) -> Path:
    """Write `chance` (from `text_chance`) to folder/text_chance.png: to a temporary file in the folder first, then
    renamed into place, so a reader never sees a part-written file.  Returns the file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / TEXT_CHANCE_FILE
    part = folder / f".{TEXT_CHANCE_FILE}.{os.getpid()}.part.png"     # cv2 picks the format by the extension
    try:
        if not cv2.imwrite(str(part), chance):
            raise OSError(f"can't write {part}")
        os.replace(part, final)
    finally:
        part.unlink(missing_ok=True)
    return final
