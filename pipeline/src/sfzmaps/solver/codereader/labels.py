"""The labels the code reader reads, the code list, and the long-line masks over each label's crop.

A label is one print of the 2000 sheets (`prints.parquet`, kinds 0-3: see `sfzmaps.solver.prints`), or one of the
owner's answered code-check cases that no print holds (kind -1).

The owner's answers (`inputs/code-check-2000/`: 1,648 cases, one per Sol "use" point; 743 answered) are the reader's
test.  A case joins the print whose Terra box holds its point (the box's edges included; of several, the one whose
point is nearest), or else the Sol-only print (kind 2) at its very point.  A label's `answers` are its joined cases the
owner answered; its `answer` is their code when they agree ("" where the owner said no code is printed there), and
None when it has none, only "unsure", or answers that disagree (two codes inside one Terra box).  An answered case
that joins no print is a label of its own at its point (kind -1, a point crop), read by Sol alone (the case's code).

Held out (`held_out`): every label an answer joined, whatever the answer.  `training` leaves out those and every label
in a held-out block: the table's block number of a held-out label's lot (`lab_lot`), and of the lot under each
answered case's own point (through the sheet's placement, its nearest lot: `grown`), so the same code printed again on
a neighbouring sheet is never trained on.  A label with no block is left out too.  `make_labels` marks both kinds
(`held_block`), so a list of labels to learn from can be checked without the rest.  The table's code at a label's lot
is never read; only the table's list of code names is (`table_codes`).

`fold` splits the labels with an owner's code (or "no label") into two halves for the cross-fitted calibration: in
each stratum (flagged on the code-check page or not, x a "no label" answer or not; the strata in that order), the
labels in a fixed order (sheet, print, point) are permuted by one generator seeded with FOLD_SEED, and given 0 and 1
alternately.  Every other label's fold is -1.

The crop (`place_crop`): the page around the label, turned so its text reads left to right.
  - With Terra's box: the word's rectangle (`textnet.data.word_rect`, from the box and Terra's reading angle) and a
    margin of REACH letter heights on every side, a letter height being the median height of the training labels'
    word rectangles (`measure_sizes`).  A quarter turn (0, 90, 180 or 270 degrees) copies the page's px exactly; any
    other angle samples the page bilinearly (OpenCV's `warpAffine`, INTER_LINEAR; white beyond the page), the same
    numbers every time.
  - Without one (Sol's point alone, the turned pass's alone, or a case on no print): an upright square around the
    point, reaching half the training words' LENGTH_QUANTILE length and the margin each way.  The reading direction
    isn't known there, so the window serves upright and sideways codes alike.
Points on the page and in the crop are in px-edge coordinates (px (col, row) covers [col, col + 1) x [row, row + 1));
`to_page` maps a crop px's index (its centre) to the page px index it samples.

The line mask (`line_mask`): the sheet's line ink by one or more rules, united, dilated by LINE_DILATE_PX and carried
into the crop as the crop was made (exactly for a quarter turn; otherwise a crop px is on a line when any page px its
bilinear sample reads is).  The reader treats those px as unobserved.  The rules (LINE_RULES):
  long   the text detector's long lines (`textnet.scores.line_ink`): strokes at least 200 ft long between their
         junctions, no wider than a heavy line.  Junctions cut a line into short strokes: where lot lines meet, or
         where a letter touches a line, much line ink near a label is missed, and large letters' strokes pass 200 ft
  width  the long lines, and every stroke at least HEAVY_FT wide whatever its length (`width_lines`).  Letters' and
         lines' strokes share widths on the 2000 map (4, 6 and 8 px both), so heavy letters go with them
  run    ink on a straight run of ink at least 200 ft long (`run_lines`): junctions don't cut a run
Which the reader uses is chosen by its likelihood on the training labels.
"""
import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from sfzmaps import config
from sfzmaps.solver import codes, grid, paths, saved, strokes
from sfzmaps.solver import prints as prints_stage
from sfzmaps.solver.textnet import data, scores

CODE_CHECK = "code-check-2000"           # the owner's answers, under config.INPUTS_DIR
CASES_FILE = "codecheck_cases_published.json"
ANSWERS_FILE = "codecheck_answers.json"
FOLD_SEED = 20261115
REACH = 3.0              # a crop's margin, in letter heights (the reading point may sit off the code)
LENGTH_QUANTILE = 0.95   # a point window holds a word this long or shorter (of the training words), turned either way
LINE_DILATE_PX = 1       # the line ink is grown by this many px (8-neighbours) before it is unobserved
LINE_RULES = ("long", "width", "run")   # the line rules `line_mask` can unite (see the module)
HEAVY_FT = 24.0          # "width": a stroke this wide is a line at any length (heavy district lines: 24-32 ft)
RUN_ANGLES = 32          # "run": the directions a straight run is looked for in, over a half turn
READERS = ("terra", "sol", "turned", "third", "recheck")    # the saved reads of a print (prints.parquet's <r>_code)
# Printed names of districts that `codes.canon` doesn't take to the table's code: "NOE VALLEY NCD" is printed for the
# 24th Street-Noe Valley NCD, whose table code is "24TH-NOE VALLEY".
ALIAS = {"NOE VALLEY": "24TH-NOE VALLEY"}


@dataclass(eq=False)
class Label:
    """One printed zoning code the reader reads (see the module)."""
    index: int                         # its row in prints.parquet; -1 for an answered case on no print
    sheet: str
    kind: int                          # the print's kind (0-3), or -1
    point: tuple[float, float]         # the print's point on the page (Terra's box centre, or Sol's point)
    box: tuple[float, float, float, float] | None = None    # Terra's box, page px
    direction: str = ""                # Terra's reading direction ("" without a box)
    angle: float = 0.0                 # Terra's reading angle, degrees counter-clockwise from left-to-right
    terra_text: str = ""               # Terra's read as written (upper case, single spaces)
    sol_text: str = ""                 # Sol's read as written
    reads: dict[str, str] = field(default_factory=dict)     # every saved read (READERS), `codes.canon` spelling
    lab_lot: int = -1                  # the lot under the point (the nearest lot in a street)
    block: str = ""                    # that lot's block number in the table
    cases: tuple[str, ...] = ()        # the code-check cases joined to it, answered or not
    answers: tuple[tuple[str, str | None, bool], ...] = ()  # the answered ones: (id, the owner's code, flagged)
    answer: str | None = None          # the owner's code ("" no code there), when the answers agree
    held_out: bool = False             # an owner's answer joined it
    held_block: bool = False           # in a held-out block, or in none (`mark_held_blocks`): never learned from
    flagged: bool = False              # one of its answered cases was flagged on the code-check page
    case_blocks: tuple[str, ...] = ()  # the blocks under its answered cases' own points
    fold: int = -1                     # 0 or 1 for the cross-fitting (labels with an owner's code), else -1
    crop: np.ndarray | None = None     # uint8, the page around it, reading left to right (`place_crop`)
    to_page: np.ndarray | None = None  # (2, 3): crop px index (u, v, 1) -> page px index (x, y)
    centre: tuple[float, float] | None = None              # its point in the crop
    code_rect: tuple[float, float, float, float] | None = None   # Terra's word in the crop (u0, v0, u1, v1)

    @property
    def ink(self) -> np.ndarray:
        """The crop's ink (bool)."""
        return self.crop < grid.INK_BELOW

    @property
    def word(self) -> tuple[float, float] | None:
        """The length and height of Terra's word rectangle (px), None without a box."""
        return None if self.box is None else word_size(self.box, self.angle)


@dataclass(frozen=True)
class Sizes:
    """The training words' measured size: their median letter height and their LENGTH_QUANTILE length, px."""
    height: float
    length: float

    @property
    def margin(self) -> int:
        """A crop's margin on every side, px."""
        return round(REACH * self.height)

    @property
    def point_half(self) -> int:
        """Half the side of a point window, px."""
        return math.ceil(self.length / 2) + self.margin


# --- the owner's answers --------------------------------------------------------------------------------------------

def cases_from(published: list[dict], answers: dict[str, dict]) -> list[dict]:
    """The code-check cases (`codecheck_cases_published.json`'s list) with the owner's answers
    (`codecheck_answers.json`): each case's id, sheet, page point (x, y), code (Sol's read), flag, `answer` (ok, fix,
    nolabel, unsure, or None) and `truth` (the owner's code: the case's for ok, the given one for fix, "" for
    nolabel; None otherwise), codes in `codes.canon`'s spelling."""
    out = []
    for c in published:
        a = answers.get(c["id"])
        kind = a["answer"] if a else None
        if kind not in (None, "ok", "fix", "nolabel", "unsure"):
            raise ValueError(f"code-check case {c['id']}: unknown answer {kind!r}")
        truth = {"ok": codes.canon(c["code"]), "fix": codes.canon((a or {}).get("code") or ""),
                 "nolabel": ""}.get(kind)
        out.append({"id": c["id"], "sheet": f"{paths.SERIES}-{c['sid']}", "x": float(c["x"]), "y": float(c["y"]),
                    "code": c["code"], "flag": bool(c["flag"]), "answer": kind, "truth": truth})
    return out


def load_cases() -> list[dict]:
    """The code-check cases and the owner's answers (`cases_from`), from config.INPUTS_DIR / CODE_CHECK."""
    folder = config.INPUTS_DIR / CODE_CHECK
    return cases_from(json.loads((folder / CASES_FILE).read_text()), json.loads((folder / ANSWERS_FILE).read_text()))


def join_cases(prints: pd.DataFrame, cases: list[dict]) -> np.ndarray:
    """Each case's print (its row in `prints`), or -1: the Sol-only print (kind 2) at the case's very point, else the
    print whose Terra box holds the point (edges included), the nearest by its point of several."""
    out = np.full(len(cases), -1, np.int64)
    rows = {s: np.flatnonzero(prints["sheet"].to_numpy() == s) for s in {c["sheet"] for c in cases}}
    px, py = prints["page_x"].to_numpy(float), prints["page_y"].to_numpy(float)
    kind = prints["kind"].to_numpy()
    b = prints[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
    for i, c in enumerate(cases):
        r = rows[c["sheet"]]
        x, y = c["x"], c["y"]
        exact = r[(kind[r] == 2) & (px[r] == x) & (py[r] == y)]
        if len(exact):
            out[i] = exact[0]
            continue
        inside = r[(b[r, 0] <= x) & (x <= b[r, 2]) & (b[r, 1] <= y) & (y <= b[r, 3])]
        if len(inside):
            out[i] = inside[np.argmin(np.hypot(px[inside] - x, py[inside] - y))]
    return out


# --- the labels -----------------------------------------------------------------------------------------------------

def printed(text) -> str:
    """A read as written, upper case with single spaces."""
    return " ".join(str(text or "").upper().split())


def _sol_text(row, use: list[dict], pts: np.ndarray) -> str:
    """Sol's read of a print as written: of a kind 0 print, the read of Sol's point inside its box nearest its centre
    (as `prints.match_prints` chose it); of a kind 2 print, the read at its point.  The print's own (`codes.canon`)
    spelling when no point is found."""
    if len(pts) and row.kind == 0:
        inside = np.flatnonzero((pts[:, 0] >= row.box_x0) & (pts[:, 0] <= row.box_x1) & (pts[:, 1] >= row.box_y0)
                                & (pts[:, 1] <= row.box_y1))
        if len(inside):
            return printed(use[inside[np.argmin(np.hypot(pts[inside, 0] - row.page_x,
                                                         pts[inside, 1] - row.page_y))]]["code"])
    if len(pts) and row.kind == 2:
        at = np.flatnonzero((pts[:, 0] == row.page_x) & (pts[:, 1] == row.page_y))
        if len(at):
            return printed(use[at[0]]["code"])
    return printed(row.sol_code)


def _block(lot_block, lot: int) -> str:
    return str(lot_block[lot]) if 0 <= lot < len(lot_block) else ""


def make_labels(prints: pd.DataFrame, terra: dict[str, list[dict]], sol: dict[str, list[dict]], cases: list[dict],
                lot_block, case_lots: dict[str, int]) -> list[Label]:
    """The labels (see the module), without their crops: one per row of `prints` (prints.parquet's table), in its
    order, then one per answered case joined to no print, in the cases' order.  `terra`: each sheet's Terra boxes
    (terra_boxes.json; a print's box is found there for Terra's text and reading angle); `sol`: each sheet's Sol labels
    (district_labels.json's "labels"); `cases`: `cases_from`; `lot_block`: each lot's block number (table.parquet's
    `block_num`, by lot index); `case_lots`: the lot under each answered case's own point (-1 where none)."""
    words = {(s, *map(float, b["box"])): b for s, bs in terra.items() for b in bs if b["kind"] == "zoning code"}
    joined = join_cases(prints, cases)
    by_print: dict[int, list[dict]] = {}
    for c, j in zip(cases, joined):
        by_print.setdefault(int(j), []).append(c)
    sol_points = {}
    for s, labs in sol.items():
        use = [lab for lab in labs if lab.get("kind", "use") == "use"]
        sol_points[s] = use, np.array([[lab["x"], lab["y"]] for lab in use], float).reshape(-1, 2)
    out = []
    for i, row in enumerate(prints.itertuples(index=False)):
        box = None if math.isnan(row.box_x0) else (row.box_x0, row.box_y0, row.box_x1, row.box_y1)
        word = words.get((row.sheet, *box)) if box is not None else None
        lab = Label(index=i, sheet=row.sheet, kind=int(row.kind), point=(float(row.page_x), float(row.page_y)),
                    box=box, direction=word["direction"] if word else "",
                    angle=float(word["angle_deg"] or 0) if word else 0.0,
                    terra_text=printed(word["text"]) if word else printed(row.terra_code),
                    sol_text=_sol_text(row, *sol_points.get(row.sheet, ([], np.zeros((0, 2))))),
                    reads={r: str(getattr(row, f"{r}_code")) for r in READERS}, lab_lot=int(row.lab_lot),
                    block=_block(lot_block, int(row.lab_lot)))
        _answer(lab, by_print.get(i, []), lot_block, case_lots)
        out.append(lab)
    for c in by_print.get(-1, []):
        if c["answer"] is None:
            continue
        lot = case_lots.get(c["id"], -1)
        lab = Label(index=-1, sheet=c["sheet"], kind=-1, point=(c["x"], c["y"]), sol_text=printed(c["code"]),
                    reads={r: codes.canon(c["code"]) if r == "sol" else "" for r in READERS}, lab_lot=lot,
                    block=_block(lot_block, lot))
        _answer(lab, [c], lot_block, case_lots)
        out.append(lab)
    assign_folds(out)
    mark_held_blocks(out)
    return out


def _answer(lab: Label, cases: list[dict], lot_block, case_lots: dict[str, int]) -> None:
    """Give a label its joined cases and the owner's answers among them (see the module)."""
    answered = [c for c in cases if c["answer"] is not None]
    truths = {c["truth"] for c in answered if c["truth"] is not None}
    lab.cases = tuple(c["id"] for c in cases)
    lab.answers = tuple((c["id"], c["truth"], c["flag"]) for c in answered)
    lab.answer = truths.pop() if len(truths) == 1 else None
    lab.held_out = bool(answered)
    lab.flagged = any(c["flag"] for c in answered)
    lab.case_blocks = tuple(sorted({b for c in answered if (b := _block(lot_block, case_lots.get(c["id"], -1)))}))


def assign_folds(labels: list[Label], seed: int = FOLD_SEED) -> None:
    """Give each label with an owner's code (or "no label") fold 0 or 1, and every other -1 (see the module)."""
    strata: dict[tuple[bool, bool], list[Label]] = {}
    for lab in labels:
        lab.fold = -1
        truths = [t for _, t, _ in lab.answers if t is not None]
        if truths:
            strata.setdefault((lab.flagged, "" in truths), []).append(lab)
    rng = np.random.default_rng(seed)
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda lab: (lab.sheet, lab.index, lab.point))
        for j, i in enumerate(rng.permutation(len(group))):
            group[i].fold = j % 2


def held_out_blocks(labels: list[Label]) -> set[str]:
    """The blocks of the held-out labels and of their answered cases' points."""
    return {b for lab in labels if lab.held_out for b in (lab.block, *lab.case_blocks) if b}


def mark_held_blocks(labels: list[Label]) -> None:
    """Mark each label in a held-out block (`held_out_blocks` of these labels) or with no block (`held_block`), so
    that a list of labels to learn from can be checked on its own."""
    held = held_out_blocks(labels)
    for lab in labels:
        lab.held_block = not lab.block or lab.block in held


def refuse_held(labels: list[Label]) -> None:
    """Raise ValueError when any of the labels may not be learned from: held out, or in a held-out block or none
    (`held_block`)."""
    held = [lab.index for lab in labels if lab.held_out or lab.held_block]
    if held:
        raise ValueError(f"{len(held)} held-out labels or labels in a held-out block given to learn from (first: "
                         f"print {held[0]}); pass labels.training(labels)")


def training(labels: list[Label]) -> list[Label]:
    """The labels to learn from: not held out, with a block, and in no held-out block (of these labels, nor marked
    so by `mark_held_blocks`)."""
    held = held_out_blocks(labels)
    return [lab for lab in labels if not lab.held_out and not lab.held_block and lab.block and lab.block not in held]


def _lot_blocks() -> np.ndarray:
    """Each lot's block number (table.parquet's `block_num`, in lot order)."""
    return pd.read_parquet(paths.LOTS_DIR / "table.parquet", columns=["block_num"])["block_num"].astype(str).to_numpy()


def case_lots(cases: list[dict]) -> dict[str, int]:
    """The lot under each case's own point: its map point through the sheet's placement, read at the grid px holding
    it (as `prints.under`) in `grown` (the nearest lot); -1 off the grid."""
    with np.load(paths.LOTS_DIR / "lots.npz") as z:
        grown = z["grown"]
    out = {}
    for sheet in sorted({c["sheet"] for c in cases}):
        mine = [c for c in cases if c["sheet"] == sheet]
        xy = data.load_transform(sheet).to_world(np.array([[c["x"], c["y"]] for c in mine], float))
        cr = np.floor(grid.to_grid(xy)).astype(np.int64)
        for c, (col, row) in zip(mine, cr):
            on = 0 <= row < grown.shape[0] and 0 <= col < grown.shape[1]
            out[c["id"]] = int(grown[row, col]) if on else -1
    return out


def load_labels(crops: bool = True) -> list[Label]:
    """Every label (`make_labels`) from the prints stage's output, the sheets' Sol labels and the owner's answers,
    with their crops (`add_crops`) unless `crops` is False."""
    pr = prints_stage.load_prints()
    terra = json.loads((paths.PRINTS_DIR / "terra_boxes.json").read_text())
    cases = load_cases()
    sol = {s: saved.sol_labels(s) for s in sorted(set(pr["sheet"]))}
    labels = make_labels(pr, terra, sol, cases, _lot_blocks(),
                         case_lots([c for c in cases if c["answer"] is not None]))
    if crops:
        add_crops(labels)
    return labels


# --- the code list --------------------------------------------------------------------------------------------------

def table_codes() -> list[str]:
    """The table's code names: every single-district code of the April 2000 table (lots.npz `table` where `known`),
    the list alone."""
    with np.load(paths.LOTS_DIR / "lots.npz") as z:
        return sorted(set(z["table"][z["known"]].tolist()))


def code_of(text, names) -> str:
    """The table code a printed form names (`codes.canon`, then ALIAS), or "" when it names none of `names`."""
    c = codes.canon(printed(text))
    c = ALIAS.get(c, c)
    return c if c in set(names) else ""


def code_list(labels: list[Label] | None = None, names=None) -> dict[str, str]:
    """Printed form -> code: every table code name (`table_codes`, or `names`) as itself, ALIAS's names, and every
    form the training labels' reads wrote (`training`; Terra's and Sol's as written, the others in their spelling)
    that names a table code (`code_of`).  `labels`: `load_labels` without crops when None."""
    names = set(table_codes() if names is None else names)
    labels = load_labels(crops=False) if labels is None else labels
    out = {n: n for n in names}
    out.update({k: v for k, v in ALIAS.items() if v in names})
    for lab in training(labels):
        for text in (lab.terra_text, lab.sol_text, *lab.reads.values()):
            form = printed(text)
            if form and form not in out and (code := code_of(form, names)):
                out[form] = code
    return dict(sorted(out.items()))


# --- the crops ------------------------------------------------------------------------------------------------------

def word_size(box, angle: float) -> tuple[float, float]:
    """The length and height (px) of the word rectangle of Terra's box at its reading angle: the box's sides for a
    quarter turn, `textnet.data.word_rect`'s otherwise."""
    x0, y0, x1, y1 = box
    turns = angle / 90
    if turns == round(turns):
        return (x1 - x0, y1 - y0) if round(turns) % 2 == 0 else (y1 - y0, x1 - x0)
    r = data.word_rect(box, angle)
    return float(np.hypot(*(r[1] - r[0]))), float(np.hypot(*(r[3] - r[0])))


def measure_sizes(labels: list[Label]) -> Sizes:
    """The size of the words of `labels` with a box (pass the training labels): the median height and the
    LENGTH_QUANTILE length of their word rectangles."""
    words = np.array([lab.word for lab in labels if lab.box is not None], float).reshape(-1, 2)
    if not len(words):
        raise ValueError("no label with a Terra box to measure the words by")
    return Sizes(height=float(np.median(words[:, 1])), length=float(np.quantile(words[:, 0], LENGTH_QUANTILE)))


def _axes(angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The reading direction and the direction down the letters on the page (x right, y down) for a reading angle
    (degrees counter-clockwise from left-to-right): unit vectors, exact for a quarter turn."""
    turns = angle / 90
    if turns == round(turns):
        eu = ((1, 0), (0, -1), (-1, 0), (0, 1))[round(turns) % 4]
    else:
        a = math.radians(angle)
        eu = (math.cos(a), -math.sin(a))
    return eu, (-eu[1], eu[0])


def frame(centre: tuple[float, float], width: int, height: int, angle: float) -> np.ndarray:
    """The (2, 3) map from a crop's px index (u, v, 1) to the page's (x, y) for a crop `width` x `height` px whose
    centre is the page px index `centre`, its u axis along the reading direction at `angle`, v down the letters."""
    (ux, uy), (vx, vy) = _axes(angle)
    uc, vc = (width - 1) / 2, (height - 1) / 2
    cx, cy = centre
    return np.array([[ux, vx, cx - ux * uc - vx * vc], [uy, vy, cy - uy * uc - vy * vc]], float)


def _footprint(to_page: np.ndarray, shape: tuple[int, int], pad: int, bounds: tuple[int, int]) -> tuple[int, ...]:
    """The page px (x0, y0, x1, y1; ends excluded) a crop of `shape` samples, `pad` px more each way, within a page of
    `bounds` (height, width); empty (x0 >= x1 or y0 >= y1) when it is off the page."""
    h, w = shape
    xy = to_page @ np.array([[0, w - 1, 0, w - 1], [0, 0, h - 1, h - 1], [1, 1, 1, 1]], float)
    x0, y0 = max(0, math.floor(xy[0].min()) - pad), max(0, math.floor(xy[1].min()) - pad)
    x1, y1 = min(bounds[1], math.ceil(xy[0].max()) + pad + 1), min(bounds[0], math.ceil(xy[1].max()) + pad + 1)
    return x0, y0, x1, y1


def sample(image: np.ndarray, to_page: np.ndarray, shape: tuple[int, int], fill, origin=(0, 0)) -> np.ndarray:
    """A crop of `shape` from `image` (uint8 or bool, its px (0, 0) the page's px `origin`) through `to_page` (see
    `frame`): an exact copy of px when the map is whole numbers (a quarter turn), bilinear otherwise (a bool image:
    True where any px the sample reads is); `fill` beyond the image."""
    a = np.asarray(to_page, float).copy()
    a[:, 2] -= origin
    h, w = shape
    if np.array_equal(a, np.rint(a)):
        a = np.rint(a).astype(np.int64)
        v, u = np.indices((h, w))
        x, y = a[0, 0] * u + a[0, 1] * v + a[0, 2], a[1, 0] * u + a[1, 1] * v + a[1, 2]
        ok = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        out = np.full((h, w), fill, image.dtype)
        out[ok] = image[y[ok], x[ok]]
        return out
    x0, y0, x1, y1 = _footprint(a, shape, 1, image.shape)
    if x0 >= x1 or y0 >= y1:
        return np.full((h, w), fill, image.dtype)
    flag = image.dtype == bool
    win = image[y0:y1, x0:x1]
    win = np.ascontiguousarray(win.view(np.uint8) * 255 if flag else win)
    a[:, 2] -= (x0, y0)
    out = cv2.warpAffine(win, a, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255 * int(fill) if flag else int(fill))
    return out > 0 if flag else out


def place_crop(label: Label, sizes: Sizes, page: np.ndarray) -> None:
    """Give a label its crop of `page` (grayscale uint8), `to_page`, `centre` and `code_rect` (see the module)."""
    if label.box is not None:
        x0, y0, x1, y1 = label.box
        length, height = word_size(label.box, label.angle)
        m = sizes.margin
        w, h = round(length + 2 * m), round(height + 2 * m)
        a = frame(((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2), w, h, label.angle)
        label.code_rect = ((w - length) / 2, (h - height) / 2, (w + length) / 2, (h + height) / 2)
        label.centre = (w / 2, h / 2)
    else:
        half = sizes.point_half
        px, py = label.point
        left, top = math.floor(px) - half, math.floor(py) - half
        w = h = 2 * half
        a = np.array([[1.0, 0.0, left], [0.0, 1.0, top]])
        label.code_rect = None
        label.centre = (px - left, py - top)
    label.to_page = a
    label.crop = sample(page, a, (h, w), 255)


def add_crops(labels: list[Label], sizes: Sizes | None = None, pages: dict[str, np.ndarray] | None = None) -> Sizes:
    """Give every label its crop (`place_crop`), sheet by sheet, sized by `sizes` (by default `measure_sizes` of the
    training labels); `pages`: each sheet's page (read from data/work when None).  Returns the sizes."""
    sizes = sizes or measure_sizes(training(labels))
    for sheet in sorted({lab.sheet for lab in labels}):
        page = data.load_page(sheet) if pages is None else pages[sheet]
        for lab in labels:
            if lab.sheet == sheet:
                place_crop(lab, sizes, page)
    return sizes


# --- the line masks -------------------------------------------------------------------------------------------------

def _stroke_ink(ink: np.ndarray, s: dict, keep: np.ndarray) -> np.ndarray:
    """The ink px (bool) of the strokes `keep` (bool per stroke) of `s` (`strokes.strokes` of `ink`, page px), each
    ink px going to its nearest centre-line px in its own piece of ink: `textnet.scores.long_lines`' rule."""
    off = s["off"]
    out = np.zeros(ink.shape, bool)
    if len(off) < 2:
        return out
    stroke = np.repeat(np.arange(len(off) - 1), np.diff(off))
    on_path = np.ones(len(stroke), bool)
    on_path[off[:-1]] = on_path[off[1:] - 1] = False          # the first and last points are its end nodes' centres
    col = np.floor(s["xy"][on_path, 0]).astype(np.intp)
    row = np.floor(s["xy"][on_path, 1]).astype(np.intp)
    kept = keep[stroke[on_path]]
    centre = np.zeros(ink.shape, bool)
    centre[row, col] = True
    on_kept = np.zeros(ink.shape, bool)
    on_kept[row[kept], col[kept]] = True
    _, nearest = cv2.distanceTransformWithLabels((~centre).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    _, piece = cv2.connectedComponents(ink.view(np.uint8), connectivity=8)
    is_kept = np.zeros(int(nearest.max()) + 1, bool)
    is_kept[nearest[centre]] = on_kept[centre]
    piece_of = np.zeros(len(is_kept), np.int32)
    piece_of[nearest[centre]] = piece[centre]
    return ink & is_kept[nearest] & (piece_of[nearest] == piece)


def width_lines(ink: np.ndarray, ft_per_px: float) -> np.ndarray:
    """The "width" rule's line ink (bool): `textnet.scores.long_lines`' strokes, and every stroke at least HEAVY_FT
    wide (median) whatever its length, so a heavy line cut short by junctions stays a line; none wider than
    THIN_LINE_FT (solid fills)."""
    ink = np.asarray(ink, bool)
    s = strokes.strokes(ink, ft_per_px, lambda colrow: np.asarray(colrow, float))
    keep = ((s["length"] >= scores.LONG_LINE_FT) | (s["wmed"] >= HEAVY_FT)) & (s["wmed"] <= scores.THIN_LINE_FT)
    return _stroke_ink(ink, s, keep)


def run_px(ft_per_px: float) -> int:
    """The "run" rule's length in px: the long lines' LONG_LINE_FT at the sheet's scale."""
    return round(scores.LONG_LINE_FT / ft_per_px)


def _line_kernel(length: int, angle: float) -> np.ndarray:
    """A one px line `length` px long through the centre of an odd square, at `angle` degrees."""
    k = length | 1
    m = np.zeros((k, k), np.uint8)
    c = k // 2
    dx, dy = math.cos(math.radians(angle)) * (length - 1) / 2, -math.sin(math.radians(angle)) * (length - 1) / 2
    cv2.line(m, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return m


def run_lines(ink: np.ndarray, length: int) -> np.ndarray:
    """The "run" rule's line ink (bool): the ink px on a straight run of ink at least `length` px long in one of
    RUN_ANGLES directions over a half turn (an opening by a one px line).  Junctions don't cut a run, and a letter's
    strokes are shorter than one."""
    u = np.ascontiguousarray(np.asarray(ink, bool)).view(np.uint8)
    out = np.zeros(u.shape, bool)
    for i in range(RUN_ANGLES):
        out |= cv2.morphologyEx(u, cv2.MORPH_OPEN, _line_kernel(length, 180 * i / RUN_ANGLES)) > 0
    return out


def find_lines(sd: data.SheetData, rule: str) -> np.ndarray:
    """A sheet's line ink by one of LINE_RULES (bool, the page's shape)."""
    if rule == "long":
        return scores.line_ink(sd)
    if rule == "width":
        return width_lines(data.page_ink(sd), sd.ft_per_px)
    if rule == "run":
        return run_lines(data.page_ink(sd), run_px(sd.ft_per_px))
    raise ValueError(f"a line rule is one of {', '.join(LINE_RULES)}, not {rule!r}")


_LINES: dict[tuple[str, str], tuple[np.ndarray, int, int]] = {}   # (sheet, rule) -> packed bits, height, width


def sheet_lines(sheet: str, rule: str = "long") -> tuple[np.ndarray, int, int]:
    """A sheet's line ink by `rule` (`find_lines`, at the sheet's scale), packed 8 px a byte along each row, and the
    page's height and width; found once per process."""
    if (sheet, rule) not in _LINES:
        sd = data.sheet_data(paths.SERIES, sheet, hold_out_validation=False, terra=[], others=[], placement=True)
        _LINES[sheet, rule] = np.packbits(find_lines(sd, rule), axis=1), sd.height, sd.width
    return _LINES[sheet, rule]


def line_mask(label: Label, lines: np.ndarray | None = None, rules: str | tuple[str, ...] = "long") -> np.ndarray:
    """The px of a label's crop on a line (bool, the crop's shape; see the module).  `lines`: the page's line ink
    (bool, before dilation); when None, the sheet's by each of `rules` (`sheet_lines`), united."""
    shape = label.crop.shape
    r = LINE_DILATE_PX
    if lines is not None:
        bounds = lines.shape
    else:
        found = [sheet_lines(label.sheet, rule) for rule in ((rules,) if isinstance(rules, str) else rules)]
        bounds = found[0][1:]
    x0, y0, x1, y1 = _footprint(label.to_page, shape, r + 2, bounds)
    if x0 >= x1 or y0 >= y1:
        return np.zeros(shape, bool)
    if lines is not None:
        win = lines[y0:y1, x0:x1].view(np.uint8)
    else:
        win = np.bitwise_or.reduce([np.unpackbits(bits[y0:y1], axis=1, count=width)[:, x0:x1]
                                    for bits, _, width in found])
    grown = cv2.dilate(np.ascontiguousarray(win), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    return sample(grown, label.to_page, shape, False, origin=(x0, y0))
