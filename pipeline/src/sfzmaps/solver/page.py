"""The merged page: every warped sheet of the 2000 series laid onto the solver's grid, as one picture of the printed
ink.  Each sheet is cropped to its printed frame (`frame.find_frame`, 4 px inside) minus the cut boxes of
`inputs/merged-page/` (notes and the north arrow printed inside the frame), and read through its corrected
placement: `to_pixel` at every STEP-th grid px, bilinear between, then the nearest page px.  That is the only
resampling, so the 1-bit scans stay 1-bit.  A grid px is ink where any sheet shows ink there (a sheet that is blank
there never cancels another's ink).

Writes, under data/solver/page/:
  ink.png       uint8, 0 = ink, 255 = paper, HEIGHT x WIDTH of the grid
  sheets.png    uint8, the number of sheets whose crop covers each grid px (0 = no sheet: not mapped)
  frames.json   each sheet's crop quad, page px (x, y): top-left, top-right, bottom-right, bottom-left
"""
import json
import logging
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps import config
from sfzmaps.solver import frame, grid, paths
from sfzmaps.transform import SheetTransform

log = logging.getLogger(__name__)

STEP = 8                    # grid px between the points where the placement itself is evaluated
BAND_ROWS = 512             # grid rows laid at once (memory only: the result doesn't depend on it)
INK_BELOW = grid.INK_BELOW     # a page px darker than this (128) is ink, as for the frame
OUTLINE_POINTS = 80         # points along each side of the crop when finding where on the grid it lands


def load_page(sheet: str) -> np.ndarray:
    """A sheet's page (data/work/<series>/<sheet>/page.png), grayscale uint8: the bytes `textnet.data.load_page` reads
    (kept apart, so the walls' code deps hold none of the detector's training code)."""
    f = config.WORK_DIR / paths.SERIES / sheet / "page.png"
    gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"can't read {f}")
    return gray


def load_cuts() -> dict[str, list[list[int]]]:
    """Each sheet's cut boxes: [x0, y0, x1, y1] page px, both ends inside the box.  From the checked-in file
    (`paths.CUTS_FILE`, picked by eye: the 2000 map's); for a series with none, the mask boxes of each sheet's
    page.json with its hand-made patch (`reading.load_page`: the notes, legend and north arrow the pipeline's page
    read found, the boxes the warp leaves out of the published sheets)."""
    if paths.CUTS_FILE.is_file():
        found = json.loads(paths.CUTS_FILE.read_text())
        return {name: [c["box"] for c in cuts] for name, cuts in found.items()}
    from sfzmaps import reading
    out = {}
    for name in paths.sheet_names():
        boxes = reading.load_page(config.WORK_DIR / paths.SERIES / name / "page.json").mask_boxes
        out[name] = [[b.x0, b.y0, b.x1 - 1, b.y1 - 1] for b in boxes]      # page.json's ends are past the box
    return out


def frame_series() -> bool:
    """The series is printed on the 2000 drawing's sheets (`paths.FRAME_SERIES`): its crop is the printed frame."""
    return paths.SERIES in paths.FRAME_SERIES


def crop_file(name: str):
    """Where the page stage keeps a sheet's crop when it is no frame quad (a series not in FRAME_SERIES)."""
    return paths.PAGE_DIR / "crops" / f"{name}.png"


def _frame_or_none(pg: np.ndarray):
    """The printed frame (`frame.find_frame`) when it is found and is the 2000 drawing's (FRAME_WIDTH_PX by
    FRAME_HEIGHT_PX, as on every 2000 sheet); else None."""
    try:
        with np.errstate(all="ignore"):
            quad = frame.find_frame(pg)
    except (ValueError, np.linalg.LinAlgError, IndexError):
        return None
    w = np.hypot(*(quad[1] - quad[0])), np.hypot(*(quad[2] - quad[3]))
    h = np.hypot(*(quad[3] - quad[0])), np.hypot(*(quad[2] - quad[1]))
    ok = all(FRAME_WIDTH_PX[0] <= v <= FRAME_WIDTH_PX[1] for v in w) and \
        all(FRAME_HEIGHT_PX[0] <= v <= FRAME_HEIGHT_PX[1] for v in h)
    return quad if ok and np.isfinite(quad).all() else None


# The 2000 drawing's printed frame, page px: on the 13 2000 sheets 3,080-3,095 wide and 3,445-3,525 high.
FRAME_WIDTH_PX = (3000.0, 3150.0)
FRAME_HEIGHT_PX = (3350.0, 3600.0)


def sheet_crop(name: str, pg: np.ndarray, cuts: dict) -> tuple[np.ndarray, np.ndarray]:
    """A sheet's crop: (quad, mask).  For a series on the 2000 drawing's sheets, the printed frame (`frame.find_frame`)
    minus the cut boxes, where the frame is found and has the 2000 drawing's size; otherwise (any other series, or a
    sheet whose frame isn't found) the neatline interior less the page.json mask boxes, exactly the area the warp
    shows of the sheet (`warp.inset_exclusion` on the page as the warp reads it), with the quad its bounding box."""
    if frame_series():
        quad = frame.find_frame(pg) if paths.is_2000() else _frame_or_none(pg)
        if quad is not None:
            return quad, crop_mask(pg.shape, quad, cuts.get(name, []))
        log.warning("%s: no printed frame of the 2000 drawing's size found; cropped to its neatline", name)
    from sfzmaps import reading, warp
    folder = config.WORK_DIR / paths.SERIES / name
    img = cv2.imread(str(folder / "page.png"), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    crop = ~warp.inset_exclusion(img, reading.load_page(folder / "page.json").mask_boxes)
    ys, xs = np.nonzero(crop)
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), crop


def load_crop(name: str, shape, quad, cuts: dict) -> np.ndarray:
    """A sheet's crop as the page stage made it: for the 2000 map, the frame quad minus the cut boxes; for any other
    series, the mask the page stage saved (`crop_file`)."""
    if paths.is_2000():
        return crop_mask(shape, np.asarray(quad, float), cuts.get(name, []))
    crop = _read(crop_file(name)) > 0
    if crop.shape != tuple(shape):
        raise ValueError(f"{crop_file(name)} is {crop.shape}, its page {tuple(shape)}")
    return crop


def crop_mask(shape: tuple[int, int], quad: np.ndarray, cuts: list) -> np.ndarray:
    """The page px (bool, `shape`) a sheet contributes: inside the quad (its sides included) and outside every cut
    box.  `quad`: (4, 2) page px (x, y), top-left, top-right, bottom-right, bottom-left."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    q = np.asarray(quad, float)
    inside = np.ones(shape, bool)
    for e in range(4):
        (ax, ay), (bx, by) = q[e], q[(e + 1) % 4]
        inside &= (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0
    for x0, y0, x1, y1 in cuts:
        inside[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = False
    return inside


def grid_box(transform, quad: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """The grid rows and columns (r0, r1, c0, c1, ends exclusive) a sheet's crop can reach: its outline mapped to the
    grid by `transform.to_world`, STEP px wider each way, inside a grid of `shape`."""
    q = np.asarray(quad, float)
    t = np.linspace(0, 1, OUTLINE_POINTS)[:, None]
    outline = np.concatenate([q[e] + t * (q[(e + 1) % 4] - q[e]) for e in range(4)])
    cr = grid.to_grid(transform.to_world(outline))
    rows, cols = shape
    r0 = max(0, int(np.floor(cr[:, 1].min())) - STEP)
    r1 = min(rows, int(np.ceil(cr[:, 1].max())) + STEP)
    c0 = max(0, int(np.floor(cr[:, 0].min())) - STEP)
    c1 = min(cols, int(np.ceil(cr[:, 0].max())) + STEP)
    return r0, r1, c0, c1


def sheet_px(transform, crop: np.ndarray, shape: tuple[int, int], box: tuple[int, int, int, int] | None = None):
    """The page px each grid px of a sheet reads, band of grid rows by band: yields (rows, cols, on, yi, xi), where
    `rows` and `cols` are the band's slices of the grid, `on` (bool) marks the grid px whose page px is in the crop
    and `yi`, `xi` (int64) are that page px (meaningful where `on`).

    `transform`: anything with `to_pixel(map points (n, 2)) -> page px (n, 2)`.  It is evaluated at the centres of
    the grid px whose row and column are multiples of STEP, read bilinearly between, and rounded to the nearest page
    px; `crop`: the page px the sheet contributes (`crop_mask`, the page's shape); `shape`: the grid's.  `box`: the
    grid rows and columns to read (`grid_box`), the whole grid when None; the points are the same either way, so a
    box that holds the crop gives the same result."""
    rows, cols = shape
    r0, r1, c0, c1 = box if box is not None else (0, rows, 0, cols)
    if r1 <= r0 or c1 <= c0:
        return
    h, w = crop.shape
    nr0, nc0 = r0 // STEP * STEP, c0 // STEP * STEP
    node_rows = np.arange(nr0, r1 - 1 + STEP, STEP)          # the last one at or past r1 - 1: no px past the nodes
    node_cols = np.arange(nc0, c1 - 1 + STEP, STEP)
    nc, nr = np.meshgrid(node_cols, node_rows)
    centres = grid.to_map(np.c_[nc.ravel() + 0.5, nr.ravel() + 0.5])
    at_nodes = np.asarray(transform.to_pixel(centres), float).reshape(nr.shape + (2,))
    for a0 in range(r0, r1, BAND_ROWS):
        a1 = min(r1, a0 + BAND_ROWS)
        rr, cc = np.mgrid[a0:a1, c0:c1]
        between = [(rr - nr0) / STEP, (cc - nc0) / STEP]
        x, y = (map_coordinates(at_nodes[..., k], between, order=1, mode="nearest") for k in (0, 1))
        on = np.isfinite(x) & np.isfinite(y)
        xi = np.round(np.where(on, x, -1)).astype(np.int64)
        yi = np.round(np.where(on, y, -1)).astype(np.int64)
        on &= (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        on[on] = crop[yi[on], xi[on]]
        yield slice(a0, a1), slice(c0, c1), on, yi, xi


def lay_sheet(transform, page: np.ndarray, crop: np.ndarray, ink: np.ndarray, sheets: np.ndarray,
              box: tuple[int, int, int, int] | None = None) -> None:
    """Lay one sheet onto the grid: set `ink` where the sheet shows ink inside its crop and add 1 to `sheets` where
    its crop covers.  `ink` and `sheets` are grid arrays from the grid's top-left corner, changed in place.
    `transform`, `crop` and `box` as `sheet_px` reads them; `page`: the sheet's grayscale page."""
    dark = page < INK_BELOW
    for rows, cols, on, yi, xi in sheet_px(transform, crop, ink.shape, box):
        seen = np.zeros(on.shape, bool)
        seen[on] = dark[yi[on], xi[on]]
        ink[rows, cols] |= seen
        sheets[rows, cols] += on.astype(np.uint8)


def build_page() -> dict:
    """Lay every warped sheet of the series onto the grid and write ink.png, sheets.png and frames.json into
    PAGE_DIR.  Returns a summary: sheets laid, ink px, covered px, seconds."""
    t0 = time.perf_counter()
    names = paths.sheet_names()
    if not names:
        raise FileNotFoundError(f"no warped sheets under {config.DATA_DIR / 'warp' / paths.SERIES}: "
                                f"run the warp-{paths.SERIES} stage first")
    cuts = load_cuts()
    ink = np.zeros((grid.HEIGHT, grid.WIDTH), bool)
    sheets = np.zeros((grid.HEIGHT, grid.WIDTH), np.uint8)
    frames = {}
    for name in names:
        folder = config.DATA_DIR / "warp" / paths.SERIES / name
        tr = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
        page_file = config.WORK_DIR / paths.SERIES / name / "page.png"
        pg = cv2.imread(str(page_file), cv2.IMREAD_GRAYSCALE)
        if pg is None:
            raise FileNotFoundError(f"can't read {page_file}")
        quad, crop = sheet_crop(name, pg, cuts)
        lay_sheet(tr, pg, crop, ink, sheets, grid_box(tr, quad, ink.shape))
        frames[name] = quad.tolist()
        if not paths.is_2000():
            crop_file(name).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_file(name)), crop.astype(np.uint8) * 255)
        log.info("%s laid (%.0f s)", name, time.perf_counter() - t0)
    paths.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paths.PAGE_DIR / "ink.png"), np.where(ink, 0, 255).astype(np.uint8))
    cv2.imwrite(str(paths.PAGE_DIR / "sheets.png"), sheets)
    (paths.PAGE_DIR / "frames.json").write_text(json.dumps(frames, indent=1) + "\n")
    return {"sheets": len(names), "ink_px": int(ink.sum()), "covered_px": int((sheets > 0).sum()),
            "seconds": round(time.perf_counter() - t0, 1)}


def load_ink() -> np.ndarray:
    """The merged page's ink (bool, True = ink), HEIGHT x WIDTH of the grid."""
    return _read(paths.PAGE_DIR / "ink.png") < INK_BELOW


def load_sheets() -> np.ndarray:
    """How many sheets' crops cover each grid px (uint8); 0 is outside every sheet."""
    return _read(paths.PAGE_DIR / "sheets.png")


def _read(path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"can't read {path}: run `sfzmaps solver page`")
    return a
