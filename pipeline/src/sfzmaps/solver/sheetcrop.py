"""A sheet's crop where it has no printed frame of the 2000 drawing (another series' sheets, `page.sheet_crop`), and
its cut boxes where no file of them is checked in (`page.load_cuts`): the area the warp shows of the sheet, its
neatline interior less the mask boxes of its page.json with its hand-made patch (`reading.load_page`).

Kept apart from `page`, which loads it by name (`page.other_series`), so the pipeline's page reading and warp code
(and the model client they import) are no code deps of the 2000 map's stages; the stages of another series' page
name it as their code beside `page`."""
import cv2
import numpy as np

from sfzmaps import config, reading, warp
from sfzmaps.solver import paths


def page_json_cuts(names) -> dict[str, list[list[int]]]:
    """Each sheet's cut boxes, [x0, y0, x1, y1] page px with both ends inside the box: its page.json's mask boxes (the
    notes, legend and north arrow the pipeline's page read found, which the warp leaves out of the published sheet)."""
    out = {}
    for name in names:
        boxes = reading.load_page(config.WORK_DIR / paths.SERIES / name / "page.json").mask_boxes
        out[name] = [[b.x0, b.y0, b.x1 - 1, b.y1 - 1] for b in boxes]      # page.json's ends are past the box
    return out


def neatline_crop(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(quad, mask): the sheet's neatline interior less its mask boxes, exactly the area the warp shows of it
    (`warp.inset_exclusion` on the page as the warp reads it), the quad its bounding box."""
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
