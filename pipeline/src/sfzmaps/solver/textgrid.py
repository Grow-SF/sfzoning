"""The text detector's chance of text on the solver's grid, from the one detector (stage `solver-text`) and its chance
on each 2000 sheet's page (`paths.TEXT_CHANCE_DIR/<sheet>.png`, uint8 chance x 255 in its page px).

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


def _chance_files(names) -> dict[str, Path]:
    """The detector's chance file of each sheet; a missing one is an error naming its stage."""
    files = {name: paths.TEXT_CHANCE_DIR / f"{name}.png" for name in names}
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"can't read {missing[0]} ({len(missing)} of {len(names)} sheets missing): train the "
                                "detector first (`sfzmaps solver text`, stage solver-text)")
    return files


def _read(path, shape) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"can't read {path}")
    if a.shape != tuple(shape):
        raise ValueError(f"{path} is {a.shape[1]} x {a.shape[0]} px, its page {shape[1]} x {shape[0]}")
    return a


def text_chance() -> np.ndarray:
    """The chance of text at every grid px (uint8, chance x 255, HEIGHT x WIDTH; see the module).  A missing chance
    file is an error.  Nothing is written."""
    t0 = time.perf_counter()
    names = paths.check_sheets()
    files = _chance_files(names)
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
