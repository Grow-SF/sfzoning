"""Pictures of a sheet's labels (`lotlabels.sheet_labels`) on its scan, to check by eye that they sit on the print.

The scan in grey with street tinted orange and block blue, as strongly as the label's weight; the block-edge cuts drawn
cyan over it, each px as strong as its prior, a cut not used (weight 0: a block the sheet does not draw) not at all; a
key and the sheet's name at the top.  `busiest` finds the close-up: the window holding the most cut px.

`beside`: a window of a sheet's page, the scan on the left and on the right the scan with every label that counts
(weight above 0) tinted at one strength, street orange and block blue, px with no label untinted: where the labels
are, not how sure they are.

`crop`: a hand crop (`handlabels`) and its labels side by side, both enlarged ZOOM times, with page-px rulers to read
coordinates off for the crop file.  `ruled`: a close-up of the scan with rulers and a grid, to draw crops from.

`read_picture`: a crop as the finder read it (`find.counts`): the scan, the chances of street and of a block edge, and
the areas (`find.areas`: the connected pieces of ground the model calls not street), the answer key's blocks outlined
by what became of them (the answer key: block outlines marked by hand on the scan, never shown to the model).

Reads nothing itself; `write` writes PNGs to data/find/look/labels/<placement>/, `write_crop` to
data/find/look/crops/<set>/, `write_ruled` to data/find/look/ruled/<series>__<sheet>__<x0>_<y0>_<w>_<h>/."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from sfzmaps.find import cuts, handlabels, model, pages
from sfzmaps.find.pages import SheetId
from sfzmaps.solver.hatchnet.data import NOT_STREET, STREET

# BGR
ORANGE, BLUE, CYAN = (0, 140, 255), (255, 90, 0), (255, 255, 0)
TINT = 0.25                 # picture setting: how strongly street and block tint the scan
STRENGTH = 0.85             # picture setting: a cut px's colour at prior 1
CLOSE = 640                 # picture setting: the close-up's side, page px
ZOOM = 2                    # picture setting: the close-up's enlargement (nearest neighbour)
KEY = (("orange = street", ORANGE), ("blue = block", BLUE), ("cyan ticks = block-edge cuts", CYAN))


def _paint(img: np.ndarray, where: np.ndarray, colour, alpha) -> None:
    a = np.asarray(alpha, np.float32)[..., None] if np.ndim(alpha) else np.float32(alpha)
    img[where] = img[where] * (1 - a) + np.asarray(colour, np.float32) * a


def busiest(lab: dict, shape: tuple[int, int], size: int = CLOSE) -> tuple[int, int, int, int]:
    """The `size` px window (x0, y0, x1, y1) holding the most px of used cuts (weight > 0; the first such, row by
    row)."""
    h, w = shape
    sh, sw = min(size, h), min(size, w)
    c = lab["cuts"]
    used = c.weight[c.cut] > 0
    count = np.bincount(c.px[used], minlength=h * w).reshape(shape).astype(np.int64)
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
    strength = np.zeros(h * w, np.float32)
    np.maximum.at(strength, c.px, c.prior * c.weight[c.cut])
    strength = strength.reshape(h, w)[y0:y1, x0:x1]
    _paint(img, strength > 0, CYAN, STRENGTH * strength[strength > 0])
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if box is not None:
        img = np.repeat(np.repeat(img, ZOOM, axis=0), ZOOM, axis=1)
    return _keyed(img, f"{sid[0]} / {sid[1]}  ({lab['placement']} placement, sigma {lab['sigma']:.2f} px)"
                       + ("" if box is None else f"  page x {x0}-{x1}, y {y0}-{y1}, x{ZOOM}"))


def _keyed(img: np.ndarray, title: str, key=KEY) -> np.ndarray:
    """`img` under a white band holding the title and the key: (text, colour) pairs, a colour a BGR swatch or a
    function drawing one into the box (x0, y0, x1, y1) of the band."""
    scale = max(0.5, img.shape[1] / 1800)
    line = int(round(30 * scale))
    band = np.full((line * (len(key) + 1) + line // 2, img.shape[1], 3), 255, np.uint8)
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 1.5)))
    cv2.putText(band, title, (line // 2, line), font, 0.7 * scale, (0, 0, 0), thick, cv2.LINE_AA)
    for i, (text, colour) in enumerate(key):
        y = line * (i + 2)
        if callable(colour):
            colour(band, (line // 2, y - line // 2, line, y))
        else:
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


BESIDE_TINT = 0.45          # picture setting: how strongly `beside` tints a labelled px
BESIDE_KEY = (("left: the scan; right: the labels over it", (255, 255, 255)), ("orange = street", ORANGE),
              ("blue = block", BLUE), ("untinted = no label", (255, 255, 255)))


def beside(lum: np.ndarray, street: np.ndarray, weight: np.ndarray, box: tuple[int, int, int, int],
           title: str) -> np.ndarray:
    """The picture (BGR uint8) of the page window `box` (x0, y0, x1, y1) at the scan's own size: the scan `lum` (1
    paper, 0 ink) on the left; on the right the scan with the px whose label counts (`weight` above 0) tinted
    BESIDE_TINT, street (`street` 1) orange and block (0) blue, every other px untinted; a key and `title` above."""
    x0, y0, x1, y1 = box
    scan = np.repeat((lum[y0:y1, x0:x1] * 255)[..., None], 3, axis=2).astype(np.float32)
    over = scan.copy()
    s, w = street[y0:y1, x0:x1], weight[y0:y1, x0:x1] > 0
    for colour, where in ((ORANGE, w & (s == STREET)), (BLUE, w & (s == NOT_STREET))):
        _paint(over, where, colour, BESIDE_TINT)
    gap = np.full((y1 - y0, GAP, 3), 255, np.float32)
    row = np.clip(np.rint(np.hstack([scan, gap, over])), 0, 255).astype(np.uint8)
    return _keyed(row, title, BESIDE_KEY)


# The crop picture.  BGR; no area colour near the street's orange, the edges' black, the no-label grey or the
# marks' magenta.
AREAS = ((255, 90, 0), (60, 170, 60), (140, 200, 230), (230, 200, 60), (90, 60, 160), (40, 160, 160),
         (170, 120, 255), (120, 200, 120), (180, 140, 90), (60, 90, 230))
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (160, 160, 160)
MAGENTA, YELLOW = (255, 0, 255), (0, 230, 255)
STRIPE = 3                  # picture setting: open land's stripe width, crop px
DOT = 4                     # picture setting: a seed's radius, picture px
RULER = (30, 56)            # picture setting: the rulers' height (top) and width (left), picture px
GAP = 16                    # picture setting: the white gap between the two panels, picture px


def _stripes(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((xx + yy) // STRIPE) % 2 == 1


def _seed(img: np.ndarray, xy: tuple[int, int], kind: str) -> None:
    fill, ring, centre = {"street": (WHITE, BLACK, None), "block": (BLACK, BLACK, None),
                          "open": (BLACK, BLACK, WHITE)}[kind]
    cv2.circle(img, xy, DOT, fill, -1, cv2.LINE_AA)
    cv2.circle(img, xy, DOT, ring, 1, cv2.LINE_AA)
    if centre is not None:
        cv2.circle(img, xy, max(1, DOT // 2), centre, -1, cv2.LINE_AA)


def _marks(img: np.ndarray, c: handlabels.Crop) -> None:
    """The crop's strokes, limits and seeds, or its outlines with a dot at each corner, drawn on a panel
    enlarged ZOOM times."""
    x0, y0 = c.box[:2]

    def at(x, y):
        return (x - x0) * ZOOM + ZOOM // 2, (y - y0) * ZOOM + ZOOM // 2

    for lines, colour in ((c.strokes, MAGENTA), (c.limits, YELLOW)):
        for pts in lines:
            cv2.polylines(img, [np.array([at(*p) for p in pts], np.int32)], False, colour, 1, cv2.LINE_AA)
    for poly in c.polygons + c.open_polygons:
        corners = np.array([at(*p) for p in poly], np.int32)
        cv2.polylines(img, [corners], True, MAGENTA, 1, cv2.LINE_AA)
        for xy in corners:
            cv2.circle(img, (int(xy[0]), int(xy[1])), 2, MAGENTA, -1, cv2.LINE_AA)
    for kind, pts in (("street", c.street), ("block", c.blocks), ("open", c.open)):
        for p in pts:
            _seed(img, at(*p), kind)


def _ruled(panel: np.ndarray, x0: int, y0: int, zoom: int = ZOOM, number: int = 100) -> np.ndarray:
    """`panel` (enlarged `zoom` times, its corner page px (x0, y0)) with page-px rulers above and to its left: ticks
    every 10 px, longer every 50, numbered every `number`."""
    top, left = RULER
    h, w = panel.shape[:2]
    out = np.full((h + top, w + left, 3), 255, np.uint8)
    out[top:, left:] = panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    for page0, n, along in ((x0, w // zoom, "x"), (y0, h // zoom, "y")):
        for v in range(-(-page0 // 10) * 10, page0 + n, 10):
            p = (v - page0) * zoom + zoom // 2
            size = 10 if v % 50 == 0 else 5
            if along == "x":
                cv2.line(out, (left + p, top - 1), (left + p, top - size), BLACK, 1)
                if v % number == 0:
                    cv2.putText(out, str(v), (left + p - 14, top - 13), font, 0.4, BLACK, 1, cv2.LINE_AA)
            else:
                cv2.line(out, (left - 1, top + p), (left - size, top + p), BLACK, 1)
                if v % number == 0:
                    cv2.putText(out, str(v), (2, top + p + 4), font, 0.4, BLACK, 1, cv2.LINE_AA)
    return out


def _key_stripes(band, box):
    x0, y0, x1, y1 = box
    band[y0:y1, x0:x1] = AREAS[0]
    band[y0:y1, x0:x1][_stripes((y1 - y0, x1 - x0))] = WHITE


def _key_seeds(band, box):
    x0, y0, x1, y1 = box
    r = (x1 - x0) // 2
    for i, kind in enumerate(("street", "block", "open")):
        _seed(band, (x0 + r + i * 3 * DOT, (y0 + y1) // 2), kind)


def _key_lines(band, box):
    x0, y0, x1, y1 = box
    cv2.line(band, (x0, y0 + 2), (x1, y0 + 2), MAGENTA, 2)
    cv2.line(band, (x0, y1 - 2), (x1, y1 - 2), YELLOW, 2)


CROP_KEY = (("orange = street", ORANGE), ("each area its own colour; open land striped", _key_stripes),
            ("black = block edge", BLACK), ("grey = no label (a stroke or limit, beside a limit, the crop's outer px)",
                                            GREY),
            ("     seeds: street white ringed black, block black, open black with a white centre", _key_seeds),
            ("magenta = stroke, yellow = limit (thin lines)", _key_lines))
OUTLINE_KEY = (("orange = street", ORANGE), ("each area its own colour; open land striped", _key_stripes),
               ("black = block edge", BLACK), ("grey = no label (the crop's outer px)", GREY),
               ("magenta = the answer key's outlines as typed (a dot at each corner)", MAGENTA),
               ("answer key: block outlines marked by hand on the scan, never shown to the model", WHITE))


def crop(lum_crop: np.ndarray, labels: handlabels.CropLabels, c: handlabels.Crop) -> np.ndarray:
    """The picture (BGR uint8) of the hand crop `c`: its scan `lum_crop` (1 paper, 0 ink) and its labels side by side,
    both enlarged ZOOM times (nearest neighbour), the crop's marks (seeds and strokes, or outlines) drawn on both,
    page-px rulers and a key."""
    scan = np.repeat((np.asarray(lum_crop, np.float32) * 255)[..., None], 3, axis=2)
    scan = np.clip(np.rint(scan), 0, 255).astype(np.uint8)
    lab = np.full(scan.shape, 255, np.uint8)
    lab[labels.street == 1] = ORANGE
    stripes = _stripes(lum_crop.shape)
    for i, kind in labels.kinds.items():
        on = labels.area == i
        lab[on] = AREAS[(i - 1) % len(AREAS)]
        if kind == "open":
            lab[on & stripes] = WHITE
    lab[((labels.edge == 255) | (labels.street == 255)) & (labels.edge != 1)] = GREY
    lab[labels.edge == 1] = BLACK
    x0, y0, w, h = c.box
    panels = []
    for p in (scan, lab):
        p = np.ascontiguousarray(np.repeat(np.repeat(p, ZOOM, axis=0), ZOOM, axis=1))
        _marks(p, c)
        panels.append(_ruled(p, x0, y0))
    gap = np.full((panels[0].shape[0], GAP, 3), 255, np.uint8)
    counts = ", ".join(f"{k} {list(labels.kinds.values()).count(k)}" for k in ("block", "open"))
    return _keyed(np.hstack([panels[0], gap, panels[1]]),
                  f"{c.set} / {c.series} / {c.sheet}  page x {x0}-{x0 + w}, y {y0}-{y0 + h}, x{ZOOM}; areas: {counts}",
                  OUTLINE_KEY if c.outlined else CROP_KEY)


def write_crop(c: handlabels.Crop, lum_crop: np.ndarray, labels: handlabels.CropLabels,
               folder: Path | None = None) -> Path:
    """`crop`'s picture as <folder>/<series>__<file stem>.png (None: data/find/look/crops/<set>/)."""
    folder = pages.OUT / "look" / "crops" / c.set if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{c.series}__{c.path.stem}.png"
    cv2.imwrite(str(f), crop(lum_crop, labels, c))
    return f


GRID = 50                   # picture setting: the close-up's grid and ruler numbers, every this many page px
GRID_GREY = (205, 205, 205)


def ruled(lum_crop: np.ndarray, box: tuple[int, int, int, int], zoom: int) -> np.ndarray:
    """The scan `lum_crop` of the page box (x0, y0, w, h), enlarged `zoom` times (nearest neighbour), thin light-grey
    lines every GRID page px over it, and page-px rulers along its top and left (a tick every 10 px, a number every
    GRID)."""
    x0, y0, w, h = box
    if lum_crop.shape != (h, w):
        raise ValueError(f"the crop is {lum_crop.shape}, its box {w} x {h}")
    img = np.repeat((np.clip(np.asarray(lum_crop, np.float32), 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    img = np.ascontiguousarray(np.repeat(np.repeat(img, zoom, axis=0), zoom, axis=1))
    half = zoom // 2
    for v in range(-(-x0 // GRID) * GRID, x0 + w, GRID):
        col = img[:, (v - x0) * zoom + half]
        col[:] = np.minimum(col, GRID_GREY)
    for v in range(-(-y0 // GRID) * GRID, y0 + h, GRID):
        row = img[(v - y0) * zoom + half]
        row[:] = np.minimum(row, GRID_GREY)
    return _ruled(img, x0, y0, zoom, GRID)


def ruled_tiles(lum_crop: np.ndarray, box: tuple[int, int, int, int], zoom: int,
                tiles: int) -> dict[tuple[int, int], np.ndarray]:
    """The box cut into `tiles` x `tiles` equal tiles, each as `ruled` draws it, by (row, column)."""
    x0, y0, w, h = box
    xs = [w * i // tiles for i in range(tiles + 1)]
    ys = [h * i // tiles for i in range(tiles + 1)]
    return {(r, c): ruled(lum_crop[ys[r]:ys[r + 1], xs[c]:xs[c + 1]],
                          (x0 + xs[c], y0 + ys[r], xs[c + 1] - xs[c], ys[r + 1] - ys[r]), zoom)
            for r in range(tiles) for c in range(tiles)}


def write_ruled(sid: SheetId, lum_crop: np.ndarray, box: tuple[int, int, int, int], zoom: int,
                tiles: int) -> list[Path]:
    """`ruled_tiles` as r<row>_c<col>.png and the whole box at ZOOM as whole.png, in
    data/find/look/ruled/<series>__<sheet>__<x0>_<y0>_<w>_<h>/."""
    folder = pages.OUT / "look" / "ruled" / f"{sid[0]}__{sid[1]}__{'_'.join(str(v) for v in box)}"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for (r, c), img in ruled_tiles(lum_crop, box, zoom, tiles).items():
        out.append(folder / f"r{r}_c{c}.png")
        cv2.imwrite(str(out[-1]), img)
    out.append(folder / "whole.png")
    cv2.imwrite(str(out[-1]), ruled(lum_crop, box, ZOOM))
    return out


# The picture of a read crop (`read_picture`).  BGR.  The found areas' tints hold no green, red, orange or purple,
# the colours of the hand blocks' outcomes.
OUTCOME_COLOUR = {"found": (0, 170, 0), "joined": (0, 0, 255), "cut_up": (0, 140, 255), "lost": (200, 0, 170)}
FOUND_TINTS = ((255, 90, 0), (230, 200, 60), (40, 160, 160), (0, 220, 255), (160, 120, 40))
FOUND_TINT = 0.35           # picture setting: how strongly a found area tints the scan
SIDE = (120, 40, 0)         # picture setting: the found areas' sides, dark blue
OUTLINE = 2                 # picture setting: a hand block's outline width, px
LABEL = 30                  # picture setting: the height of the white strip naming a 1,000 px panel, px


def _grey(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(np.rint(np.asarray(a, np.float32) * 255), 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def _named(panel: np.ndarray, name: str) -> np.ndarray:
    scale = max(0.5, panel.shape[1] / 1000)
    line = int(round(LABEL * 1.6 * scale))
    strip = np.full((line, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, name, (6, line - int(round(10 * scale))), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, BLACK,
                max(1, int(round(2 * scale))), cv2.LINE_AA)
    return np.vstack([strip, panel])


def _key_tints(band, box):
    x0, y0, x1, y1 = box
    step = max(1, (x1 - x0) // len(FOUND_TINTS))
    for i, colour in enumerate(FOUND_TINTS):
        band[y0:y1, x0 + i * step:x0 + (i + 1) * step] = colour


READ_KEY = (("panels: the scan; the chance of street and of a block edge (white 0, black 1); the found areas over "
             "the scan", WHITE),
            ("found areas: the connected pieces of ground the model calls not street, each tinted its own colour, "
             "their sides dark blue", _key_tints),
            ("answer key: block outlines marked by hand on the scan, never shown to the model", WHITE),
            ("the answer key's blocks outlined by outcome: green = found", OUTCOME_COLOUR["found"]),
            ("red = joined (the found area holding most of it lies mostly outside it)", OUTCOME_COLOUR["joined"]),
            ("orange = cut up among found areas", OUTCOME_COLOUR["cut_up"]),
            ("purple = lost (most of it in no found area)", OUTCOME_COLOUR["lost"]))


def read_picture(lum: np.ndarray, chances: np.ndarray, found: np.ndarray, truth: handlabels.CropLabels,
                 status: dict[int, str], title: str) -> np.ndarray:
    """The picture (BGR uint8) of a crop's read, four panels side by side at the scan's own size: the scan `lum` (1
    paper, 0 ink); the chances (2, h, w) of street and of a block edge, each as grey from white 0 to black 1; and the
    found areas `found` (`find.areas`: the connected pieces of ground the model calls not street) tinted over the scan
    with their sides, each hand block outlined in its outcome's colour (`status`, by hand area number:
    `OUTCOME_COLOUR`); a key and `title` above."""
    scan = _grey(lum)
    panels = [_named(scan, "the scan")]
    for i, name in ((model.STREET, "street"), (model.EDGE, "a block edge")):
        panels.append(_named(_grey(1 - np.clip(chances[i], 0, 1)), f"the chance of {name}"))
    over = scan.astype(np.float32)
    for k in range(1, int(found.max(initial=0)) + 1):
        _paint(over, found == k, FOUND_TINTS[(k - 1) % len(FOUND_TINTS)], FOUND_TINT)
    over = np.clip(np.rint(over), 0, 255).astype(np.uint8)
    h, w = found.shape
    padded = np.pad(found, 1, mode="edge")
    other = np.zeros(found.shape, bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != found
    over[(found > 0) & other] = SIDE
    for i, outcome in sorted(status.items()):
        block = truth.area == i
        inner = ndimage.binary_erosion(block, iterations=OUTLINE, border_value=1)
        over[block & ~inner] = OUTCOME_COLOUR[outcome]
    panels.append(_named(over, "the pieces called not street; the answer key's blocks by outcome"))
    gap = np.full((panels[0].shape[0], GAP, 3), 255, np.uint8)
    row = [panels[0]]
    for p in panels[1:]:
        row += [gap, p]
    return _keyed(np.hstack(row), title, READ_KEY)
