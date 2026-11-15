"""The text detector's scores, the same measure for every comparison (`score_sheets`): from its chance of text on each
sheet's page (as `predict.chance` gives it: uint8, the chance x 255; a float chance from 0 to 1 is rounded to that),
on the sheets' compact data (`data.SheetData`).

Where:
  validation  the px whose map point is in VALIDATION_AREA (`data.in_validation`): on the training maps, trained with
              the area held out, exactly the px that carried no loss.  A sheet with none is left out
  whole       every px of every sheet (the test drawings, never trained on)

What is scored there, on the ink px (text: the px of Terra's words, as the detector learns them, `data.page_text`; the
unknown px, near Sol's points and in the other passes' boxes, left out).  A chance of 0.5 or more (CALLED of 255) is
"called text".

  ink px      text against not-text: the AUC and the log loss (from histograms of the 256 chances); text ink found
              (the share of the text ink called text); ink outside every box called text (the share of the ink in no
              word); and the share of the ink called text that is in a word
  words       Terra's words found, by kind: a word is found when the mean chance over its ink is CALLED or more (on
              average half its ink called text: the rule of the quarter models' scores).  Each word is its own
              rectangle, as drawn for training, and its ink the ink px inside it: where two overlap, that ink is each
              one's.  A word with no ink is not counted.  In validation a word is counted when its centre px (its
              rectangle's corners' mean, rounded) is in the area, and is judged on its ink in the area alone, so no px
              the detector trained on counts; a word whose centre is outside is not counted, though its ink inside
              still counts as text ink
  long lines  ink on long thin strokes (`long_lines`): the share called text and the mean chance; the share of it in a
              word (which a detector calling exactly the words would call), and the share of the rest called text.
              It stands in for district-line ink called text, which needs the zoning table

Summed over the sheets of each map (series), and over all of them.  The counts are whole numbers, divided only at the
end, so the scores don't depend on the order of the sheets; a share of nothing is None; nothing in them is a time or a
path, so the same chances give the same bytes.
"""
import dataclasses
from collections.abc import Mapping

import cv2
import numpy as np

from sfzmaps.solver import paths, strokes
from sfzmaps.solver.textnet import data

CALLED = 128              # a chance of at least this (of 255) is "called text": a chance of 0.5 or more
LONG_LINE_FT = 200.0      # a long line: a stroke at least this long ...
THIN_LINE_FT = 12.0       # ... and at most this wide (its median width); both in the sheet's own feet (`ft_per_px`)
WHERE = ("validation", "whole")


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """From histograms of scores 0..255 of positives and negatives: the chance a random positive scores above a random
    negative (ties count half).  None without both."""
    if not pos.sum() or not neg.sum():
        return None
    pos, neg = pos.astype(float), neg.astype(float)
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def log_loss(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """From the same histograms: the mean log loss of chance score / 255 (clipped to 1e-4 .. 1 - 1e-4).  None
    without any."""
    n = pos.sum() + neg.sum()
    if not n:
        return None
    p = np.clip(np.arange(len(pos)) / 255.0, 1e-4, 1 - 1e-4)
    return float(-(pos * np.log(p) + neg * np.log(1 - p)).sum() / n)


def as_chance(chance) -> np.ndarray:
    """A chance of text as `predict.chance` gives it: uint8, the chance x 255.  A float chance (0 to 1) is rounded to
    that, as `predict.chance` rounds."""
    chance = np.asarray(chance)
    if chance.dtype == np.uint8:
        return chance
    if np.issubdtype(chance.dtype, np.floating):
        return np.rint(np.clip(chance, 0.0, 1.0) * 255).astype(np.uint8)
    raise TypeError(f"a chance is uint8 (x 255) or float (0 to 1), not {chance.dtype}")


def _page_px(colrow: np.ndarray) -> np.ndarray:
    """Stroke points kept as page coordinates (`strokes.strokes` gives each px's centre, (col + 0.5, row + 0.5))."""
    return np.asarray(colrow, float)


def long_lines(ink: np.ndarray, ft_per_px: float) -> np.ndarray:
    """The ink px (bool) on long thin strokes: of the strokes of the ink (`strokes.strokes`: its centre lines, split
    at their ends and junctions), those at least LONG_LINE_FT long and at most THIN_LINE_FT wide (median), measured at
    `ft_per_px` ft per px.  Each ink px is its nearest centre-line px's, in its own piece of ink (a speck whose centre
    line is no stroke is no line's); a centre-line px where strokes meet is on a long line when one of them is."""
    ink = np.asarray(ink, bool)
    s = strokes.strokes(ink, ft_per_px, _page_px)
    off = s["off"]
    out = np.zeros(ink.shape, bool)
    if len(off) < 2:
        return out
    stroke = np.repeat(np.arange(len(off) - 1), np.diff(off))
    on_path = np.ones(len(stroke), bool)
    on_path[off[:-1]] = on_path[off[1:] - 1] = False          # the first and last points are its end nodes' centres
    col = np.floor(s["xy"][on_path, 0]).astype(np.intp)
    row = np.floor(s["xy"][on_path, 1]).astype(np.intp)
    long = ((s["length"] >= LONG_LINE_FT) & (s["wmed"] <= THIN_LINE_FT))[stroke[on_path]]
    centre = np.zeros(ink.shape, bool)
    centre[row, col] = True
    on_long = np.zeros(ink.shape, bool)
    on_long[row[long], col[long]] = True
    _, nearest = cv2.distanceTransformWithLabels((~centre).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    _, piece = cv2.connectedComponents(ink.view(np.uint8), connectivity=8)
    is_long = np.zeros(int(nearest.max()) + 1, bool)
    is_long[nearest[centre]] = on_long[centre]
    piece_of = np.zeros(len(is_long), np.int32)
    piece_of[nearest[centre]] = piece[centre]
    return ink & is_long[nearest] & (piece_of[nearest] == piece)


@dataclasses.dataclass
class _Tally:
    """The counts the scores are made of, added up sheet by sheet."""
    sheets: int = 0
    pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(256, np.int64))   # text ink, by chance
    neg: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(256, np.int64))   # other ink, by chance
    words: dict = dataclasses.field(default_factory=lambda: {k: [0, 0] for k in data.KINDS})   # found, of
    line_px: int = 0
    line_called: int = 0
    line_chance: int = 0      # the sum of the chances (x 255)
    line_in_word: int = 0
    line_rest_called: int = 0

    def add(self, other: "_Tally") -> None:
        self.sheets += other.sheets
        self.pos += other.pos
        self.neg += other.neg
        for k, (f, n) in other.words.items():
            self.words[k][0] += f
            self.words[k][1] += n
        for f in ("line_px", "line_called", "line_chance", "line_in_word", "line_rest_called"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def scores(self) -> dict:
        def share(a, b):
            return a / b if b else None

        text, other = int(self.pos.sum()), int(self.neg.sum())
        text_called, other_called = int(self.pos[CALLED:].sum()), int(self.neg[CALLED:].sum())
        words = {k: {"found": f, "of": n} for k, (f, n) in self.words.items()}
        found, of = sum(f for f, _ in self.words.values()), sum(n for _, n in self.words.values())
        words["all"] = {"found": found, "of": of}
        return {
            "sheets": self.sheets, "ink_px": text + other, "text_ink_px": text,
            "auc": auc(self.pos, self.neg), "log_loss": log_loss(self.pos, self.neg),
            "text_ink_found": share(text_called, text),
            "ink_outside_boxes_called_text": share(other_called, other),
            "called_text_in_a_box": share(text_called, text_called + other_called),
            "words": words, "words_found": share(found, of),
            "long_lines": {"ink_px": self.line_px, "called_text": share(self.line_called, self.line_px),
                           "mean_chance": share(self.line_chance, 255 * self.line_px),
                           "in_a_box": share(self.line_in_word, self.line_px),
                           "rest_called_text": share(self.line_rest_called, self.line_px - self.line_in_word)},
        }


def scored_px(sd: data.SheetData, where: str) -> np.ndarray:
    """The px of a sheet's page that are scored `where` (bool): every px, or those in VALIDATION_AREA (found
    data.BAND rows at a time: a whole page's map points would be 16 bytes a px)."""
    if where == "whole":
        return np.ones((sd.height, sd.width), bool)
    return np.concatenate([data.in_validation(sd, y0, 0, min(data.BAND, sd.height - y0), sd.width)
                           for y0 in range(0, sd.height, data.BAND)])


def _add_words(t: _Tally, sd: data.SheetData, chance: np.ndarray, ink: np.ndarray, scope: np.ndarray) -> None:
    """Count the words whose centre px is in `scope`, each found or not on its ink in `ink` (see the module)."""
    h, w = ink.shape
    kind_name = {k: name for name, k in data.KINDS.items()}
    centre = np.rint(sd.words.mean(axis=1)).astype(np.intp)
    counted = scope[np.clip(centre[:, 1], 0, h - 1), np.clip(centre[:, 0], 0, w - 1)]
    lo, hi = sd.words.min(axis=1), sd.words.max(axis=1) + 1
    for j in np.flatnonzero(counted):
        (x0, y0), (x1, y1) = lo[j], hi[j]
        rect = np.zeros((y1 - y0, x1 - x0), np.uint8)       # filled whole, as on a page large enough (`data._paint`)
        cv2.fillPoly(rect, [(sd.words[j] - lo[j]).reshape(-1, 1, 2)], 1)
        cy0, cx0, cy1, cx1 = max(y0, 0), max(x0, 0), min(y1, h), min(x1, w)
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        mine = rect[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].view(bool) & ink[cy0:cy1, cx0:cx1]
        n = int(mine.sum())
        if n:
            tally = t.words[kind_name[int(sd.kinds[j])]]
            tally[0] += int(chance[cy0:cy1, cx0:cx1][mine].sum(dtype=np.int64)) >= CALLED * n
            tally[1] += 1


def score_sheet(sd: data.SheetData, chance: np.ndarray, scope: np.ndarray) -> _Tally:
    """One sheet's counts: its chance (uint8, the page's shape) on the px of `scope` (`scored_px`)."""
    ink = data.page_ink(sd)
    text, unknown = data.page_text(sd)
    scored = ink & ~unknown & scope
    t = _Tally(sheets=1)
    t.pos += np.bincount(chance[scored & text], minlength=256)
    t.neg += np.bincount(chance[scored & ~text], minlength=256)
    _add_words(t, sd, chance, ink & scope, scope)
    line = long_lines(ink, sd.ft_per_px) & scored
    t.line_px = int(line.sum())
    t.line_called = int((chance[line] >= CALLED).sum())
    t.line_chance = int(chance[line].sum(dtype=np.int64))
    t.line_in_word = int((line & text).sum())
    t.line_rest_called = int((chance[line & ~text] >= CALLED).sum())
    return t


def score_sheets(chances: Mapping[tuple[str, str], np.ndarray], where: str,
                 sheets: list[data.SheetData] | None = None) -> dict:
    """The scores (see the module) of the chances of text `chances` ((series, sheet) -> its page's chance) `where`
    ("validation" or "whole") on `sheets` (by default every sheet Terra read of the maps in `chances`,
    `data.load_sheets`): per map, in the sheets' order, and over all; JSON-able.  Every sheet with px to score must
    have its chance, of its page's shape, and every chance its sheet (ValueError)."""
    if where not in WHERE:
        raise ValueError(f"where is one of {', '.join(WHERE)}, not {where!r}")
    if sheets is None:
        present = {s for s, _ in chances}
        series = [s for s in paths.TEXT_SHEETS if s in present] + sorted(present - set(paths.TEXT_SHEETS))
        sheets = data.load_sheets(series, hold_out_validation=False)
    keys = [(sd.series, sd.sheet) for sd in sheets]
    extra = [k for k in chances if k not in set(keys)]
    if extra:
        raise ValueError(f"chances of sheets not among those scored: {', '.join(f'{s} {n}' for s, n in extra)}")
    if where == "whole":
        missing = [k for k in keys if k not in chances]
        if missing:
            raise ValueError(f"no chance for {', '.join(f'{s} {n}' for s, n in missing)}: whole sheets are scored")
    tallies: dict[str, _Tally] = {}
    for sd, key in zip(sheets, keys):
        scope = scored_px(sd, where)
        if not scope.any():
            continue
        if key not in chances:
            raise ValueError(f"no chance for {key[0]} {key[1]}, which has px in the validation area")
        chance = as_chance(chances[key])
        if chance.shape != (sd.height, sd.width):
            raise ValueError(f"the chance of {key[0]} {key[1]} is {chance.shape}, not its page's shape "
                             f"{(sd.height, sd.width)}")
        tallies.setdefault(sd.series, _Tally()).add(score_sheet(sd, chance, scope))
    total = _Tally()
    for t in tallies.values():
        total.add(t)
    return {"where": where, "called": CALLED, "long_line_ft": LONG_LINE_FT, "thin_line_ft": THIN_LINE_FT,
            "series": {s: t.scores() for s, t in tallies.items()}, "all": total.scores()}
