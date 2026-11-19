"""Pictures for the owner: what the network learned, painted over the scan as printed (upright, the scan's own px).

  streets   the learned streets and blocks: the scan in grey; each px tinted orange by its street probability (the
            target pictures' colour for not-a-block, so the two read alike); each block (`train.blocks`) filled faintly
            and outlined in its own colour, so two printed blocks that came out as one show one colour; outside the
            neatline grey (no blocks there).  Whole, at the scan's own px, and a zoom (data.ZOOM x, nearest: each scan
            px a square)
  compare   the zoom side by side: the target (today's blocks placed on the scan, as `data.look_targets` draws it)
            and what was learned
  district_lots  the learned districts by lot: every lot in its pooled district's colour (DISTRICT_BGR), at opacity =
            its pooled probability, multiplied into the scan so the print stays readable; whole, and a zoom
  districts the learned per-px evidence: every px inside the neatline tinted in its own district argmax's colour, the
            tint's strength the px's learned pooling weight (`weight_strength`: strong where the px counts most in its
            lot, pale where it is discounted, so its argmax hardly counts); the banner gives the lots' weight-mass
            shares; whole, and a zoom

Nothing here is used by the reading: the pictures only show it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from sfzmaps.solver.hatchnet import data, model

STREET_BGR = data.TINT[0]          # orange: the target pictures' not-a-block
STREET_ALPHA = 0.6                 # the street tint's opacity at probability 1 (it scales with the probability)
FILL_ALPHA = 0.18                  # each block's faint fill in its own colour
WHOLE_EDGE = 2                     # the block outline's width inside the block, scan px (whole picture)
ZOOM_EDGE = 2                      # ... zoom px (the zoom)
GOLDEN = 0.6180339887498949        # block hues step round the circle by this: neighbours rarely look alike
HUES = (0.20, 0.90)                # block hues from yellow-green to magenta: never orange, the street tint


def palette(n: int) -> np.ndarray:
    """(n + 1, 3) uint8 BGR: a colour for each block 1..n (row 0 unused), saturated, never orange."""
    k = np.arange(n + 1, dtype=np.float64)
    hue = HUES[0] + (HUES[1] - HUES[0]) * ((k * GOLDEN) % 1.0)
    hsv = np.stack([np.round(hue * 180) % 180, np.full_like(k, 235), np.full_like(k, 230)], -1).astype(np.uint8)
    return cv2.cvtColor(hsv[:, None], cv2.COLOR_HSV2BGR)[:, 0]


def _grey(ink: np.ndarray) -> np.ndarray:
    g = np.round(np.clip(ink, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)


def learned(sheet: data.SheetRasters, street_prob: np.ndarray, blocks: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """The scan with the street probability tinted and each block filled faintly (float32 BGR, no outlines)."""
    img = _grey(sheet.ink)
    out = ~np.asarray(sheet.crop, bool)
    a = (STREET_ALPHA * np.where(out, 0, np.clip(street_prob, 0, 1)))[..., None]
    img = img * (1 - a) + np.array(STREET_BGR, np.float32) * a
    inb = blocks > 0
    img[inb] = img[inb] * (1 - FILL_ALPHA) + colours[blocks[inb]].astype(np.float32) * FILL_ALPHA
    ga = data.ALPHA[data.UNKNOWN]
    img[out] = img[out] * (1 - ga) + np.array(data.TINT[data.UNKNOWN], np.float32) * ga
    return img


def target(sheet: data.SheetRasters) -> np.ndarray:
    """The street target tinted over the scan (float32 BGR), as `data.look_targets` draws it."""
    img = _grey(sheet.ink)
    for v, bgr in data.TINT.items():
        m = sheet.block == v
        img[m] = img[m] * (1 - data.ALPHA[v]) + np.array(bgr, np.float32) * data.ALPHA[v]
    return img


def outline(img: np.ndarray, labels: np.ndarray, colours: np.ndarray, width: int) -> np.ndarray:
    """`img` (uint8 BGR) with each labelled area's inner band `width` px wide (px within `width` of another label,
    a side or corner away) in its colour."""
    size = 2 * width + 1
    band = (labels > 0) & ((ndimage.maximum_filter(labels, size, mode="nearest") != labels) |
                           (ndimage.minimum_filter(labels, size, mode="nearest") != labels))
    img = img.copy()
    img[band] = colours[labels[band]]
    return img


def _banner(img: np.ndarray, lines: list[str], scale: float) -> np.ndarray:
    h = int(round(34 * scale)) * len(lines) + int(round(16 * scale))
    top = np.full((h, img.shape[1], 3), 255, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(top, line, (int(12 * scale), int(round(34 * scale)) * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (0, 0, 0), max(1, int(round(2 * scale))), cv2.LINE_AA)
    return np.vstack([top, img])


def _zoom(a: np.ndarray, window) -> np.ndarray:
    y0, x0, s = window
    return np.repeat(np.repeat(a[y0:y0 + s, x0:x0 + s], data.ZOOM, 0), data.ZOOM, 1)


def _window(sheet: data.SheetRasters, at) -> tuple[int, int, int]:
    """(y0, x0, side): the zoom's square, data.ZOOM_SIDE px from `at` (top-left), or the page's middle."""
    h, w = sheet.ink.shape
    s = min(data.ZOOM_SIDE, h, w)
    y0, x0 = at if at is not None else ((h - s) // 2, (w - s) // 2)
    return int(np.clip(y0, 0, h - s)), int(np.clip(x0, 0, w - s)), s


def _held_share(window, cell: int | None) -> float | None:
    if cell is None:
        return None
    from sfzmaps.solver.hatchnet.train import held_out
    y0, x0, s = window
    return float(held_out(np.arange(y0, y0 + s), np.arange(x0, x0 + s), cell).mean())


LEGEND = ["orange: learned street probability (stronger = surer)   each colour: one learned block, outlined",
          "grey: outside the neatline (no blocks there)"]


def _head(sheet: data.SheetRasters, blocks: np.ndarray, short: bool = False) -> str:
    if short:
        return f"{sheet.sheet}  {sheet.ft_per_px:.2f} ft/px  learned streets and blocks ({int(blocks.max())} blocks)"
    return (f"{sheet.sheet}  {sheet.ft_per_px:.2f} ft/px  learned streets and blocks ({int(blocks.max())} blocks: "
            f"street probability <= 1/2 inside the neatline, 4-connected)")


def held_window(sheet: data.SheetRasters, cell: int) -> tuple[int, int]:
    """The top-left (y, x) of a data.ZOOM_SIDE square in the middle of the sheet's held-out cell with the most
    labelled px (for a picture of what the network does where it never saw a label)."""
    from sfzmaps.solver.hatchnet.train import held_out
    h, w = sheet.block.shape
    known = (sheet.block != data.UNKNOWN) & held_out(np.arange(h), np.arange(w), cell)
    counts = {}
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            n = int(known[i:i + cell, j:j + cell].sum())
            if n:
                counts[(i, j)] = n
    if not counts:
        raise ValueError(f"{sheet.sheet} has no labelled held-out px")
    i, j = max(sorted(counts), key=lambda k: counts[k])
    s = min(data.ZOOM_SIDE, cell)
    return i + (min(cell, h - i) - s) // 2, j + (min(cell, w - j) - s) // 2     # `_window` keeps it on the page


def streets(sheet: data.SheetRasters, street_prob: np.ndarray, blocks: np.ndarray, out: Path, at=None,
            cell: int | None = None) -> Path:
    """Write the learned streets and blocks over the scan into the folder `out`: streets_<sheet>_whole.png (the scan's
    own px) and streets_<sheet>_zoom.png (data.ZOOM_SIDE px from `at` = (y0, x0), or the page's middle, at data.ZOOM
    x); `cell`: the held-out cell size, to say how much of the zoom was held out.  Returns the whole picture's path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    colours = palette(int(blocks.max()))
    base = np.round(learned(sheet, street_prob, blocks, colours)).astype(np.uint8)
    whole = _banner(outline(base, blocks, colours, WHOLE_EDGE), [_head(sheet, blocks)] + LEGEND, 1.6)
    f = out / f"streets_{sheet.sheet}_whole.png"
    cv2.imwrite(str(f), whole)
    w = _window(sheet, at)
    zoom = outline(_zoom(base, w), _zoom(blocks, w), colours, ZOOM_EDGE)
    share = _held_share(w, cell)
    where = f"zoom x{data.ZOOM}: scan px x {w[1]}-{w[1] + w[2]}, y {w[0]}-{w[0] + w[2]} (upright)"
    if share is not None:
        where += f";  {100 * share:.0f}% of it in held-out areas (labels never trained on)"
    cv2.imwrite(str(out / f"streets_{sheet.sheet}_zoom.png"),
                _banner(zoom, [_head(sheet, blocks, short=True), where] + LEGEND, 1.2))
    return f


def compare(sheet: data.SheetRasters, street_prob: np.ndarray, blocks: np.ndarray, out: Path, at=None,
            cell: int | None = None, name: str = "compare") -> Path:
    """Write streets_<sheet>_<name>.png into the folder `out`: the zoom (as `streets`) of the target, left, and of
    what was learned, right.  Returns its path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    colours = palette(int(blocks.max()))
    w = _window(sheet, at)
    tz = np.round(_zoom(target(sheet), w)).astype(np.uint8)
    blk = _zoom(sheet.block, w) == 1
    edge = blk & (cv2.erode(blk.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    tz[edge] = data.TINT[1]
    lz = outline(np.round(_zoom(learned(sheet, street_prob, blocks, colours), w)).astype(np.uint8), _zoom(blocks, w),
                 colours, ZOOM_EDGE)
    tz = _banner(tz, ["TARGET: today's blocks placed on the scan", "blue block (1), orange not a block (0), "
                      "grey unknown"], 1.2)
    lz = _banner(lz, ["LEARNED: street probability (orange), blocks", "each colour one block, outlined"], 1.2)
    gap = np.full((tz.shape[0], 16, 3), 255, np.uint8)
    share = _held_share(w, cell)
    head = [f"{sheet.sheet}  {sheet.ft_per_px:.2f} ft/px  target vs learned, zoom x{data.ZOOM}: scan px x {w[1]}-"
            f"{w[1] + w[2]}, y {w[0]}-{w[0] + w[2]} (upright)"]
    if share is not None:
        head.append(f"{100 * share:.0f}% of this window is in held-out areas (its labels never trained on)")
    f = out / f"streets_{sheet.sheet}_{name}.png"
    cv2.imwrite(str(f), _banner(np.hstack([tz, gap, lz]), head, 1.2))
    return f


OVERVIEW_SIDE = 1400               # px, the longer side of each half of an overview


def overview(sheet: data.SheetRasters, street_prob: np.ndarray, blocks: np.ndarray, out: Path,
             note: str = "") -> Path:
    """Write overview_<sheet>.png into the folder `out`: the whole sheet small (area-averaged down to OVERVIEW_SIDE
    px), the target left and what was learned right (block fills, no outlines: they vanish when shrunk).  Returns its
    path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    k = OVERVIEW_SIDE / max(sheet.ink.shape)
    small = [cv2.resize(np.round(a).astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
             for a in (target(sheet), learned(sheet, street_prob, blocks, palette(int(blocks.max()))))]
    gap = np.full((small[0].shape[0], 16, 3), 255, np.uint8)
    head = [f"{sheet.sheet}  {sheet.ft_per_px:.2f} ft/px  {int(blocks.max())} learned blocks  {note}",
            "left TARGET: blue today's blocks, orange not a block, grey unknown (another sheet's or off the neatline)",
            "right LEARNED: orange street probability, each colour one block, grey off the neatline"]
    f = out / f"overview_{sheet.sheet}.png"
    cv2.imwrite(str(f), _banner(np.hstack([small[0], gap, small[1]]), head, 0.9))
    return f


def all_pictures(sheet: data.SheetRasters, street_prob: np.ndarray, blocks: np.ndarray, out: Path, at=None,
                 cell: int | None = None) -> list[Path]:
    """`streets` and `compare`: the whole picture, the zoom and the side by side; with `cell`, also the side by side
    in the sheet's largest held-out area (`held_window`, streets_<sheet>_compare_heldout.png)."""
    whole = streets(sheet, street_prob, blocks, out, at, cell)
    got = [whole, whole.with_name(f"streets_{sheet.sheet}_zoom.png"),
           compare(sheet, street_prob, blocks, out, at, cell)]
    if cell is not None:
        got.append(compare(sheet, street_prob, blocks, out, held_window(sheet, cell), cell, "compare_heldout"))
    return got


# --- the districts --------------------------------------------------------------------------------------------------

# Each district's colour (BGR), model.NAMES order: the published map's colours (sfzmaps.districts.assemble.COLORS by
# 1921 category), with LI yellow where the published map gives LI and HI one industrial brown.
DISTRICT_BGR = {"1R": (0, 68, 254), "2R": (0, 150, 0), "C": (175, 73, 17), "HI": (11, 66, 133), "LI": (0, 190, 230),
                "U": (175, 98, 133)}
DISTRICT_WORD = {"1R": "orange-red", "2R": "green", "C": "blue", "HI": "brown", "LI": "yellow", "U": "purple"}
TINT_LOW, TINT_HIGH = 0.12, 0.62   # the district tint's opacity at the least and most weight
INK_GREY = 0.3                     # ink drawn this grey (paper white) under the district tint, so its colour shows on
                                   # solid ink (C) and hatch lines too (a display choice: the network read the scan)


def weight_strength(sheet: data.SheetRasters, weight: np.ndarray) -> np.ndarray:
    """Each px's weight shown as a strength in (0, 1): sigmoid(weight - the median weight of the sheet's lot px in the
    neatline).  A lot pools its px in proportion to exp(weight), so a px 3 above the median counts 20 times a median
    px (strength 0.95) and one 3 below a twentieth (0.05)."""
    ref = np.asarray(sheet.crop, bool) & (sheet.lot >= 0)
    if not ref.any():
        ref = np.asarray(sheet.crop, bool)
    med = float(np.median(weight[ref])) if ref.any() else 0.0
    return (1.0 / (1.0 + np.exp(-(weight.astype(np.float64) - med)))).astype(np.float32)


def district_tint(sheet: data.SheetRasters, argmax: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The scan (ink drawn INK_GREY) with every px inside the neatline tinted in its district's colour, the tint's
    opacity from TINT_LOW (least weight) to TINT_HIGH (most), and grey outside the neatline (float32 BGR)."""
    img = _grey(INK_GREY + (1.0 - INK_GREY) * sheet.ink)
    crop = np.asarray(sheet.crop, bool)
    colours = np.array([DISTRICT_BGR[n] for n in model.NAMES], np.float32)
    a = (TINT_LOW + (TINT_HIGH - TINT_LOW) * weight_strength(sheet, weight))[..., None]
    img = np.where(crop[..., None], img * (1 - a) + colours[argmax] * a, img)
    ga = data.ALPHA[data.UNKNOWN]
    img[~crop] = img[~crop] * (1 - ga) + np.array(data.TINT[data.UNKNOWN], np.float32) * ga
    return img


def _key(width: int, scale: float) -> np.ndarray:
    """A banner row naming each district beside a swatch of its colour."""
    h = int(round(40 * scale))
    row = np.full((h, width, 3), 255, np.uint8)
    x, side = int(12 * scale), int(round(26 * scale))
    for n in model.NAMES:
        cv2.rectangle(row, (x, (h - side) // 2), (x + side, (h + side) // 2), DISTRICT_BGR[n], -1)
        x += side + int(8 * scale)
        label = f"{n} {DISTRICT_WORD[n]}"
        cv2.putText(row, label, (x, h // 2 + int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 0),
                    max(1, int(round(2 * scale))), cv2.LINE_AA)
        x += int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, max(1, int(round(2 * scale))))[0][0]
                 + 28 * scale)
    return row


def _shares(shares) -> str:
    return ", ".join(f"{n} {100 * float(v):.1f}%" for n, v in zip(model.NAMES, shares))


def _write(base: np.ndarray, sheet: data.SheetRasters, out: Path, name: str, at, head: list[str],
           legend: list[str]) -> list[Path]:
    """Write districts_<sheet>_<name>_whole.png (the scan's own px) and, with `at` = (y0, x0), ..._zoom.png
    (data.ZOOM_SIDE px from there, at data.ZOOM x), each under the colour key and a banner."""
    out.mkdir(parents=True, exist_ok=True)
    whole = _banner(base, head + legend, 1.6)
    f = out / f"districts_{sheet.sheet}_{name}_whole.png"
    cv2.imwrite(str(f), np.vstack([_key(whole.shape[1], 1.6), whole]))
    got = [f]
    if at is not None:
        w = _window(sheet, at)
        where = f"zoom x{data.ZOOM}: scan px x {w[1]}-{w[1] + w[2]}, y {w[0]}-{w[0] + w[2]} (upright)"
        zoom = _banner(_zoom(base, w), head + [where] + legend, 1.0)
        g = out / f"districts_{sheet.sheet}_{name}_zoom.png"
        cv2.imwrite(str(g), np.vstack([_key(zoom.shape[1], 1.0), zoom]))
        got.append(g)
    return got


DISTRICT_LEGEND = ["colour: each px's own district argmax; strength: its learned pooling weight",
                   "strong = counts most in its lot; pale = discounted: its colour hardly counts",
                   "px in no lot (streets; areas another sheet labels): never trained; grey: off the neatline"]


def districts(sheet: data.SheetRasters, argmax: np.ndarray, weight: np.ndarray, out: Path, at=None,
              note: str = "", mass=None) -> list[Path]:
    """The per-px district evidence over the scan (`district_tint`), into the folder `out`:
    districts_<sheet>_px_whole.png and, with `at`, districts_<sheet>_px_zoom.png.  `mass`: the weight-mass shares of the
    sheet's lots (sum over lots of sum_px softmax_lot(weight)_px p_px, over the lots), for the banner.  Returns the
    paths."""
    base = np.round(district_tint(sheet, argmax, weight)).astype(np.uint8)
    head = [f"{sheet.sheet}  {sheet.ft_per_px:.2f} ft/px  per-px district evidence  {note}"]
    if mass is not None:
        head.append(f"weight-mass shares of the sheet's lots: {_shares(mass)}")
    return _write(base, sheet, Path(out), "px", at, head, DISTRICT_LEGEND)


LOT_INK = 0.45                     # ink drawn this grey under the lots' multiply blend, so solid C ink shows its colour
                                   # (dark) and hatch lines stay readable (a display choice)


def district_lots_tint(sheet: data.SheetRasters, code: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """The scan (ink drawn LOT_INK grey) with every lot px in its lot's pooled district colour at opacity = the lot's
    pooled probability, in a multiply blend (colour x scan, so ink stays darker than paper and the print readable); px
    of no lot as scanned; grey outside the neatline (float32 BGR).  `code` (h, w): the lot's district index, -1 none;
    `prob` (h, w)."""
    img = _grey(LOT_INK + (1.0 - LOT_INK) * sheet.ink)
    colours = np.array([DISTRICT_BGR[n] for n in model.NAMES], np.float32)
    on = code >= 0
    a = prob[on].astype(np.float32)[:, None]
    img[on] = img[on] * (1 - a) + (colours[code[on]] * img[on] / 255.0) * a
    crop = np.asarray(sheet.crop, bool)
    ga = data.ALPHA[data.UNKNOWN]
    img[~crop] = img[~crop] * (1 - ga) + np.array(data.TINT[data.UNKNOWN], np.float32) * ga
    return img


LOTS_LEGEND = ["each lot: its pooled district (its px on this sheet, the learned weights softmaxed over the lot)",
               "opacity = the pooled probability (multiply blend over the scan: ink stays darker)",
               "px in no lot (streets; areas another sheet labels): as scanned; grey: off the neatline"]


def district_lots(sheet: data.SheetRasters, code: np.ndarray, prob: np.ndarray, out: Path, at=None,
                  note: str = "", shares=None, lots: int = 0) -> list[Path]:
    """Every lot in its pooled district (`district_lots_tint`), into the folder `out`: districts_<sheet>_lots_whole.png
    and, with `at`, districts_<sheet>_lots_zoom.png.  `shares`: the share of the sheet's `lots` pooled to each district,
    for the banner.  Returns the paths."""
    base = np.round(district_lots_tint(sheet, code, prob)).astype(np.uint8)
    head = [f"{sheet.sheet}  {sheet.ft_per_px:.2f} ft/px  pooled lot districts  {note}"]
    if shares is not None:
        head.append(f"the sheet's {lots} lots pooled to: {_shares(shares)}")
    return _write(base, sheet, Path(out), "lots", at, head, LOTS_LEGEND)
