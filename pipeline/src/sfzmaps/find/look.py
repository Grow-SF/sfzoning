"""Pictures of a sheet's labels (`lotlabels.sheet_labels`) on its scan, to check by eye that they sit on the print.

The scan in grey with street tinted orange and block blue, as strongly as the label's weight; the cut statements drawn
over it, each px as strong as its prior times its cut's weight (so labels of blocks the sheet does not draw fade out):
"no zoning border crosses" faint green, then "a zoning border crosses" red, then block-edge cuts cyan; a key and the
sheet's name at the top.  `busiest` finds the close-up: the window holding the most cut px.

Reads nothing itself; `write` writes PNGs to data/find/look/labels/<placement>/."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sfzmaps.find import cuts, pages
from sfzmaps.find.pages import SheetId
from sfzmaps.solver.hatchnet.data import NOT_STREET, STREET

# BGR
ORANGE, BLUE, CYAN, RED, GREEN = (0, 140, 255), (255, 90, 0), (255, 255, 0), (0, 0, 255), (0, 170, 0)
TINT = 0.25                 # picture setting: how strongly street and block tint the scan
STRENGTH = 0.85             # picture setting: a cut px's colour at prior 1
FAINT = 0.35                # picture setting: the same for "no zoning border crosses", which covers whole streets and
                            # lot lines and would hide the print and the other cuts
CLOSE = 640                 # picture setting: the close-up's side, page px
ZOOM = 2                    # picture setting: the close-up's enlargement (nearest neighbour)
KEY = (("orange = street", ORANGE), ("blue = block", BLUE), ("cyan ticks = block-edge cuts", CYAN),
       ("red = a zoning border crosses", RED), ("green = no zoning border crosses", GREEN))


def _paint(img: np.ndarray, where: np.ndarray, colour, alpha) -> None:
    a = np.asarray(alpha, np.float32)[..., None] if np.ndim(alpha) else np.float32(alpha)
    img[where] = img[where] * (1 - a) + np.asarray(colour, np.float32) * a


def busiest(lab: dict, shape: tuple[int, int], size: int = CLOSE) -> tuple[int, int, int, int]:
    """The `size` px window (x0, y0, x1, y1) holding the most cut px (the first such, row by row)."""
    h, w = shape
    sh, sw = min(size, h), min(size, w)
    count = np.bincount(lab["cuts"].px, minlength=h * w).reshape(shape).astype(np.int64)
    s = np.zeros((h + 1, w + 1), np.int64)
    s[1:, 1:] = count.cumsum(0).cumsum(1)
    win = s[sh:, sw:] - s[:-sh, sw:] - s[sh:, :-sw] + s[:-sh, :-sw]
    y0, x0 = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(x0), int(y0), int(x0) + sw, int(y0) + sh


def labels(sid: SheetId, lum: np.ndarray, lab: dict, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The picture (BGR uint8) of the labels `lab` on the scan `lum` (1 paper, 0 ink); `box` (x0, y0, x1, y1): only
    that window, enlarged ZOOM times."""
    h, w = lum.shape
    x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
    img = np.repeat((lum[y0:y1, x0:x1] * 255)[..., None], 3, axis=2).astype(np.float32)
    street, weight = lab["street"][y0:y1, x0:x1], lab["weight"][y0:y1, x0:x1]
    for colour, where in ((ORANGE, street == STREET), (BLUE, street == NOT_STREET)):
        _paint(img, where, colour, TINT * weight[where])
    c: cuts.Cuts = lab["cuts"]
    for colour, strong, which in ((GREEN, FAINT, (c.answer == cuts.BORDER) & ~c.says),       # drawn under the rest
                                  (RED, STRENGTH, (c.answer == cuts.BORDER) & c.says),
                                  (CYAN, STRENGTH, c.answer == cuts.EDGE)):
        strength = np.zeros(h * w, np.float32)
        on = which[c.cut]
        np.maximum.at(strength, c.px[on], (c.prior * c.weight[c.cut])[on])
        strength = strength.reshape(h, w)[y0:y1, x0:x1]
        _paint(img, strength > 0, colour, strong * strength[strength > 0])
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if box is not None:
        img = np.repeat(np.repeat(img, ZOOM, axis=0), ZOOM, axis=1)
    return _keyed(img, f"{sid[0]} / {sid[1]}  ({lab['placement']} placement, sigma {lab['sigma']:.2f} px)"
                       + ("" if box is None else f"  page x {x0}-{x1}, y {y0}-{y1}, x{ZOOM}"))


def _keyed(img: np.ndarray, title: str) -> np.ndarray:
    """`img` under a white band holding the title and the key."""
    scale = max(0.5, img.shape[1] / 1800)
    line = int(round(30 * scale))
    band = np.full((line * (len(KEY) + 1) + line // 2, img.shape[1], 3), 255, np.uint8)
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 1.5)))
    cv2.putText(band, title, (line // 2, line), font, 0.7 * scale, (0, 0, 0), thick, cv2.LINE_AA)
    for i, (text, colour) in enumerate(KEY):
        y = line * (i + 2)
        cv2.rectangle(band, (line // 2, y - line // 2), (line, y), colour, -1)
        cv2.putText(band, text, (int(line * 1.3), y), font, 0.6 * scale, (0, 0, 0), thick, cv2.LINE_AA)
    return np.vstack([band, img])


def write(sid: SheetId, lum: np.ndarray, lab: dict) -> list[Path]:
    """The whole sheet and the close-up (`busiest`) as data/find/look/labels/<placement>/<series>__<sheet>__whole.png
    and __close.png."""
    folder = pages.OUT / "look" / "labels" / lab["placement"]
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, box in (("whole", None), ("close", busiest(lab, lum.shape))):
        f = folder / f"{sid[0]}__{sid[1]}__{name}.png"
        cv2.imwrite(str(f), labels(sid, lum, lab, box))
        out.append(f)
    return out
