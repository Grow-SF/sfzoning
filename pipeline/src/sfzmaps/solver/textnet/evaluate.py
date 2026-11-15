"""The text detectors scored, each on the quarter it never trained on (`sfzmaps solver text-scores`, stage
`solver-text-scores`, after the four `solver-text-<Q>` stages): from the chances they wrote, never their weights.

Only the held-out quarter's px count, and of those only px on the map: px whose map point is inside the sheet's warped
footprint (the legend and margins, alike on every sheet, are left out; a map area two sheets both show counts on
each).  A px's map point is `data.page_world`'s, and its grid px `floor(grid.to_grid)`.  A chance of CALLED (one half)
or more is "called text".

  ink px          text against not-text (Terra's boxes; unknown areas left out): AUC, log loss, the share of text
                  ink called text, and the share of ink called text that is in a box
  words           Terra's words (connected regions of box px) found: the mean chance over their ink is CALLED or
                  more, by kind
  new text        pieces of ink called text (px within PIECE_JOIN_PX joined) with MIN_PIECE_INK or more ink px, near
                  no Terra box and no unknown area: text Terra missed, or false alarms
  district lines  ink on the table's district lines away from every box and unknown area: the share called text.  A
                  district line is a border between two lots of the table (`grown` on the grid, so lines down the
                  middle of streets too) whose known codes differ; each border crack marks the grid px after it
                  (right of it or below it), widened to LINE_WIDTH_PX on the grid.

Writes data/solver/text-scores/evaluation_<q>.json, one per quarter.
"""
import json
import logging

import cv2
import numpy as np
import shapely

from sfzmaps.solver import grid, lots, paths
from sfzmaps.solver.textnet import data

log = logging.getLogger(__name__)

CALLED = 128            # a chance of at least this (of 255) is "called text"
NEAR_BOX_PX = 7         # the side of the square that widens boxes and unknown areas, for "near" one
PIECE_JOIN_PX = 9       # the side of the square that joins px called text into one piece
MIN_PIECE_INK = 20      # ink px a piece of new text has at least
LINE_WIDTH_PX = 5       # the side of the square that widens each district-line px on the grid


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """From histograms of scores 0..255 of positives and negatives: the chance a random positive scores above a random
    negative (ties count half)."""
    pos, neg = pos.astype(float), neg.astype(float)
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / max(pos.sum() * neg.sum(), 1))


def log_loss(pos: np.ndarray, neg: np.ndarray) -> float:
    """From the same histograms: the mean log loss of chance score / 255 (clipped to 1e-4 .. 1 - 1e-4)."""
    p = np.clip(np.arange(len(pos)) / 255.0, 1e-4, 1 - 1e-4)
    return float(-(pos * np.log(p) + neg * np.log(1 - p)).sum() / max(pos.sum() + neg.sum(), 1))


def district_line_px(grown: np.ndarray, codes: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The grid px (bool) just after each crack between `grown` lots whose codes are both known and differ: right
    of a crack between columns, below one between rows."""
    _, code_id = np.unique(codes, return_inverse=True)
    lot_code = np.where(known, code_id, -1).astype(np.int32)
    g = lot_code[grown]
    lines = np.zeros(grown.shape, bool)
    lines[:, 1:] |= (g[:, :-1] >= 0) & (g[:, 1:] >= 0) & (g[:, :-1] != g[:, 1:])
    lines[1:] |= (g[:-1] >= 0) & (g[1:] >= 0) & (g[:-1] != g[1:])
    return lines


def _square(side: int) -> np.ndarray:
    return np.ones((side, side), np.uint8)


def footprint_on_grid(sheet: str) -> np.ndarray:
    """The grid px (bool) whose centres are inside the sheet's warped footprint (its polygons' outer rings)."""
    geom = shapely.from_geojson((data.warp_dir(sheet) / "warped.footprint.geojson").read_text())
    mask = np.zeros((grid.HEIGHT, grid.WIDTH), np.uint8)
    scale = 1 << lots.SUBPIXEL_BITS
    for poly in shapely.get_parts(shapely.get_parts(geom)):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        xy = grid.to_grid(shapely.get_coordinates(poly.exterior)) - 0.5
        cv2.fillPoly(mask, [np.rint(xy * scale).astype(np.int32)], 1, shift=lots.SUBPIXEL_BITS)
    return mask > 0


def _on_grid(raster: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A grid raster (bool) read at each page px's map point (x, y rasters): False off the grid."""
    cr = np.floor(grid.to_grid(np.stack([x, y], -1))).astype(np.int64)
    c, r = cr[..., 0], cr[..., 1]
    inside = (r >= 0) & (r < raster.shape[0]) & (c >= 0) & (c < raster.shape[1])
    out = np.zeros(x.shape, bool)
    out[inside] = raster[r[inside], c[inside]]
    return out


class _Tally:
    """The counts the scores are made of, added up sheet by sheet."""

    def __init__(self):
        self.pos = np.zeros(256, np.int64)
        self.neg = np.zeros(256, np.int64)
        self.words = {k: [0, 0] for k in data.KINDS}       # found, of
        self.pieces: list[dict] = []
        self.line_px = self.line_called = 0
        self.line_chance = 0.0

    def add_sheet(self, sheet: str, t: data.SheetTargets, chance: np.ndarray, on_map: np.ndarray,
                  on_line: np.ndarray) -> None:
        inq = t.heldout & on_map
        m = t.ink & inq & ~t.unknown
        self.pos += np.bincount(chance[m & (t.text > 0)], minlength=256)
        self.neg += np.bincount(chance[m & (t.text == 0)], minlength=256)
        n, lab = cv2.connectedComponents(((t.text > 0) & inq).astype(np.uint8), connectivity=8)
        sel = t.ink & (lab > 0)
        total = np.bincount(lab[sel], chance[sel].astype(float), minlength=n)
        count = np.bincount(lab[sel], minlength=n)
        kind = np.full(n, data.NO_KIND, np.int64)
        kind[lab[lab > 0]] = t.kind[lab > 0]
        found = (count > 0) & (total >= CALLED * np.maximum(count, 1))
        for name, k in data.KINDS.items():
            word = (count > 0) & (kind == k)
            self.words[name][0] += int((found & word).sum())
            self.words[name][1] += int(word.sum())
        called = (chance >= CALLED) & t.ink & inq
        near = cv2.dilate(((t.text > 0) | t.unknown).astype(np.uint8), _square(NEAR_BOX_PX)) > 0
        pieces = cv2.dilate(called.astype(np.uint8), _square(PIECE_JOIN_PX))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pieces, connectivity=8)
        touch = np.zeros(n, bool)
        touch[np.unique(lab[near & (pieces > 0)])] = True
        ink_called = np.bincount(lab[called], minlength=n)
        for i in range(1, n):
            if not touch[i] and ink_called[i] >= MIN_PIECE_INK:
                x, y, w, h = (int(v) for v in stats[i, :4])
                self.pieces.append({"sheet": sheet, "box": [x, y, x + w, y + h], "ink_px": int(ink_called[i])})
        lm = on_line & t.ink & inq & ~near
        self.line_px += int(lm.sum())
        self.line_called += int((chance[lm] >= CALLED).sum())
        self.line_chance += float(chance[lm].sum()) / 255

    def scores(self, q: str) -> dict:
        called_text, called = int(self.pos[CALLED:].sum()), int(self.pos[CALLED:].sum() + self.neg[CALLED:].sum())
        found, of = sum(f for f, _ in self.words.values()), sum(n for _, n in self.words.values())
        return {
            "quarter": q,
            "ink_px": int(self.pos.sum() + self.neg.sum()), "text_ink_px": int(self.pos.sum()),
            "auc": auc(self.pos, self.neg), "log_loss": log_loss(self.pos, self.neg),
            "text_ink_found": called_text / max(int(self.pos.sum()), 1),
            "called_text_in_a_box": called_text / max(called, 1),
            "words": {**{k: {"found": f, "of": n} for k, (f, n) in self.words.items()},
                      "all": {"found": found, "of": of}},
            "words_found": found / max(of, 1),
            "new_text_pieces": len(self.pieces),
            "district_lines": {"ink_px": self.line_px, "called_text": self.line_called / max(self.line_px, 1),
                               "mean_chance": self.line_chance / max(self.line_px, 1)},
            "new_text": self.pieces,
        }


def district_lines() -> np.ndarray:
    """The table's district lines on the grid (bool), each line px widened to LINE_WIDTH_PX (see the module)."""
    built = lots.load_lots()
    return cv2.dilate(district_line_px(built.grown, built.codes, built.known).astype(np.uint8),
                      _square(LINE_WIDTH_PX)) > 0


def score_quarter(q: str, lines: np.ndarray) -> dict:
    """The scores of the detector that held out `q`, on that quarter (see the module), from its chances
    (`paths.chance_dir(q)`) and the district lines (`district_lines`)."""
    data.quarter_index(q)
    tally = _Tally()
    for name in paths.sheet_names():
        f = paths.chance_dir(q) / f"{name}.png"
        chance = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if chance is None:
            raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver text {q}` first")
        t = data.sheet_targets(name, q)
        if chance.shape != t.ink.shape:
            raise ValueError(f"{f} is {chance.shape}, its page {t.ink.shape}: run `sfzmaps solver text {q}` again")
        x, y = data.page_world(data.load_transform(name), chance.shape)
        tally.add_sheet(name, t, chance, _on_grid(footprint_on_grid(name), x, y), _on_grid(lines, x, y))
        log.info("%s: %s evaluated", q, name)
    return tally.scores(q)


def build_scores() -> dict:
    """Score each quarter's detector on its held-out quarter and write TEXT_SCORES_DIR/evaluation_<q>.json.  Returns
    the scores by quarter."""
    lines = district_lines()
    paths.TEXT_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for q in paths.QUARTERS:
        out[q] = score_quarter(q, lines)
        (paths.TEXT_SCORES_DIR / f"evaluation_{q}.json").write_text(json.dumps(out[q], indent=1) + "\n")
    return out
