"""The text detector's chance of text on the solver's grid, out of fold: every grid px takes the chance of the model
that held out its own quarter (`grid.quarter_of` of the px's map point), so no px is scored by a model that learned
from it.

Each sheet's chance (`text-<q>/chance/<sheet>.png`, uint8 chance x 255 in its page px) is read through the sheet's
corrected placement exactly as the merged page reads its ink (`page.sheet_px`: the same crop, the frame of
`page/frames.json` minus the cut boxes, and the same page px for every grid px).  A grid px takes the chance only from
the sheets that inked it (page < INK_BELOW there, as the merge reads ink); where several did, the largest chance
wins, and a px no sheet inked is 0.

Writes WALLS_DIR/text_chance.png (uint8, chance x 255, HEIGHT x WIDTH of the grid) when made from all four quarters.
"""
import json
import logging
import time

import cv2
import numpy as np

from sfzmaps import config
from sfzmaps.solver import grid, page, paths
from sfzmaps.transform import SheetTransform

log = logging.getLogger(__name__)

TEXT_CHANCE_FILE = "text_chance.png"


def _chance_files(quarters, names) -> dict[str, dict]:
    """Each quarter's chance file per sheet; a missing one is an error naming the quarter's stage."""
    out = {}
    for q in quarters:
        files = {name: paths.chance_dir(q) / f"{name}.png" for name in names}
        missing = [str(f) for f in files.values() if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"can't read {missing[0]} ({len(missing)} of {len(names)} sheets missing): "
                                    f"train the detector first (`sfzmaps solver text {q}`, stage solver-text-{q})")
        out[q] = files
    return out


def _read(path, shape) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"can't read {path}")
    if a.shape != tuple(shape):
        raise ValueError(f"{path} is {a.shape[1]} x {a.shape[0]} px, its page {shape[1]} x {shape[0]}")
    return a


def text_chance(quarters=None, cache: bool = True) -> np.ndarray:
    """The chance of text at every grid px (uint8, chance x 255, HEIGHT x WIDTH), out of fold (see the module).

    `quarters`: the quarters whose models are read (default all four); a px in another quarter is 0, so a subset is
    only a partial look.  Made from all four, the result is written to WALLS_DIR/text_chance.png; with `cache` that
    file is read back instead of remade when it is there (it holds the chances as they were when it was written: a
    retrained detector needs `cache=False`).  A missing chance file of a quarter asked for is an error."""
    t0 = time.perf_counter()
    asked = set(paths.QUARTERS if quarters is None else quarters)
    unknown = sorted(asked - set(paths.QUARTERS))
    if unknown:
        raise ValueError(f"no quarter {unknown[0]!r}: one of {', '.join(paths.QUARTERS)}")
    quarters = [q for q in paths.QUARTERS if q in asked]
    every = len(quarters) == len(paths.QUARTERS)
    cached = paths.WALLS_DIR / TEXT_CHANCE_FILE
    if every and cache and cached.is_file():
        return _read(cached, (grid.HEIGHT, grid.WIDTH))
    from sfzmaps.solver.textnet import data
    names = paths.sheet_names()
    files = _chance_files(quarters, names)
    frames = json.loads((paths.PAGE_DIR / "frames.json").read_text())
    cuts = page.load_cuts()
    model_of = np.full(len(paths.QUARTERS), -1)                    # quarter index -> its model's row in `chance`
    model_of[[paths.QUARTERS.index(q) for q in quarters]] = np.arange(len(quarters))
    out = np.zeros((grid.HEIGHT, grid.WIDTH), np.uint8)
    for name in names:
        if name not in frames:
            raise KeyError(f"{paths.PAGE_DIR / 'frames.json'} has no frame for {name}: run `sfzmaps solver page`")
        folder = config.DATA_DIR / "warp" / paths.SERIES / name
        tr = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
        pg = data.load_page(name)
        dark = pg < grid.INK_BELOW
        quad = np.asarray(frames[name], float)
        crop = page.crop_mask(pg.shape, quad, cuts.get(name, []))
        chance = np.stack([_read(files[q][name], pg.shape) for q in quarters])      # (quarters, h, w)
        del pg
        for rows, cols, on, yi, xi in page.sheet_px(tr, crop, out.shape, page.grid_box(tr, quad, out.shape)):
            r, c = np.nonzero(on)
            y, x = yi[r, c], xi[r, c]
            inked = dark[y, x]
            r, c, y, x = r[inked], c[inked], y[inked], x[inked]
            if not len(r):
                continue
            model = model_of[grid.quarter_of(grid.to_map(np.c_[c + cols.start + 0.5, r + rows.start + 0.5]))]
            value = np.where(model >= 0, chance[np.maximum(model, 0), y, x], 0).astype(np.uint8)
            band = out[rows, cols]
            band[r, c] = np.maximum(band[r, c], value)
        log.info("text chance: %s laid (%.0f s)", name, time.perf_counter() - t0)
    if every:
        paths.WALLS_DIR.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(cached), out):
            raise OSError(f"can't write {cached}")
    return out
