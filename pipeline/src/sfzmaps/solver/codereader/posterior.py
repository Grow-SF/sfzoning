"""The code reader's posterior: for one printed label, p(code | ink) over the codes, "none" (no text at the label)
and "other" (text that is no code), from its crop's ink, line px unobserved.

    p(h | ink)  oc  p(h) E_h,    E_h = p(ink | h) / p(ink | none),    E_none = 1

The likelihood (Kopec & Chou's document image decoding; Berg-Kirkpatrick et al.'s Ocular).  A hypothesis is text
laid out by the font (`glyphs`) at a placement: a scale (crop px per template px), an angle, a line (its offset `k`
across the reading direction, px from the print's point) and the pen of its first character along it (`t`, px).
  - Characters.  A character's log-likelihood ratio against the bare background with its centre at a px is one
    correlation of its rendered patch with the crop's ink (FFT, observed px only): `glyphs`' noise model, by the
    signed distance from the character's own strokes.  Every hypothesis is scored from these maps.  A string's ratio
    is the sum of its characters' at their pens (each pen the last plus the last character's advance, rounded to a
    px); px within 2 px of two characters' strokes count for both (on the training fits the characters' sums match
    the whole string's ratio to 0.3%).
  - Each character a little off its pen along the line (`_jitter`): the gaps between printed letters spread about
    the font's advances (`Font.spacing_sd`, measured in Task 2), so each character's shift is Gaussian with that
    spread / sqrt 2 and is summed over.  Without it a rigid layout misses the print's spacing and the open-vocabulary
    pass wins by spelling dots and spaces into the gaps.
  - Codes: each printed form of a code (`Prior.forms`), its characters laid out by the font's advances.
  - "other": an open-vocabulary pass over the font's characters (a space between words), the string's probability
    by a character bigram over the previous character's kind (`char_bigram`).  The best string from each cell (it
    ends where its bigram ends it) is found by a max-product pass along the line (`_viterbi`) and stands for the sum
    over strings there; its placement weight is judged at the end it carries.
  - Tempered: the px aren't independent (a stroke's edge is one event, not a run of independent px), so every ratio
    is divided by the temperature before anything is summed (a composite likelihood's magnitude adjustment; Ocular
    needed the same).  The pose prior carries a default; Task 4 fits it with the two masses.

The placement prior (`PosePrior`, measured on the training labels by `measure_pose_prior`).  Where the text is,
given the print:
  - With Terra's box (the crop reads along Terra's angle; `code_rect` is the word): the ink's left and right ends at
    the word's ends and the line's middle at the word's middle, each a Gaussian in word heights.  The two ends are two
    observations of where the text is: their product is a Gaussian on the text's centre times a likelihood of its
    length (scaled to 1 at the word's own length), so a code too long or too short for Terra's word pays for it.  The
    cap height over the word's height: log-normal.
  - Without a box (Sol's point, an answered case's point, the turned pass's box centre): the ink's centre about the
    point, along and across the line, per source; the cap height by the map's spread (`cap`).
  - Each a mixture of a core and a tail (the print a little off: Terra's box loose or swallowing a neighbour's
    letters, Sol's point off the word's middle), measured.  No uniform anywhere-in-the-crop component: in the
    measurement it holds the strings matched on other ink, which is what a summed px likelihood rewards (see below),
    not where labels are.  Cells where even the tail is below e^-REACH_LOG of its peak are not scored (a numerical
    tolerance, `_windows`).
  - Angles: Terra's reading angle and its half turn (Terra reads some bottom-to-top labels top-to-bottom: the half
    turn's rate is measured per Terra direction), or with no box every quarter turn (rates measured per source); a
    few degrees each way about each (`RESIDUALS`, rates measured).  Scales on `glyphs`' grid (`glyphs.search_for`).

Other text in the crop.  A summed px likelihood pays more for a match on more ink: under the bare background any
other text in the crop (a neighbour's letters, a line's unmasked stub) costs ~2.6 nats a px, so a hypothesis that
covers more of it gains.  The tight measured placement keeps codes at the print; the remaining exposure is "other"
strings and codes running a little past the label onto adjacent ink.  So the ink a hypothesis claims (the cells of
its characters: each its advance along the line by the lettering's height, leaned like the glyphs) may also be other
text, at the measured density of text in the crops (`PosePrior.foreign`, Terra's boxes of every kind), explaining
that ink about as well as the label's own print is explained.  Terra's word counts as one word of other text under
"none":

    E_h = L_h (1 + rho F(word - h)) / [(1 + rho F(word)) (1 + rho F(h - word))],    F(x) = exp(q G(x) / T)

with L the tempered ratio, G(x) the ink in x weighted by each character's typical ratio per ink px
(`PosePrior.char_gain`, measured on the training fits) and q the label's print quality (pass 1: the best code's
ratio over its G at the core placement).  Covering more ink of the label's quality then gains nothing; a code that
stops short of its word, or shrinks inside it, pays for the leftover as one more word; on paper and lines G is 0 and
E_h is L, which strokes on paper make tiny: "none".  The best "other" string carries its own G through the Viterbi
pass.  `PosePrior.foreign = 0` turns this off (the bare background alone).  Without a box there is no word: the term
is 1 / (1 + rho F(h)), and where the word ends comes from the ink along the line:

Where a word ends, without a box.  A reading that stops short of its word (RH-1 of RH-1(D), P on the R of RM-2)
leaves the rest of the word beside its end, on its line.  Past each end of a hypothesis's ink, a window of the
lettering band (`PosePrior.end_break["window"]`: a skip and a width, cap heights; leaned like the glyphs) holds some
density of observed ink; P(a word ends | that density) is measured on the training labels' fits (their strings' own ends against the
boundaries between their characters; isotonic, non-increasing in the density, a sample at the base rate added to
each step; the window kept by two-fold log loss: `end_breaks`).  Each end multiplies E_h by P(end | density) / P(end); the best "other" string
takes its last end's factor inside the Viterbi pass.  With a box Terra's word does this job.

The prior over hypotheses: p(code) = (1 - none - other) x `code_prior`, a code's printed forms weighted by
`form_prior` (how often the training strings print each), p(none), p(other): `Constants`, Task 4's to fit (two-fold
on the owner's answers).

Two passes (`evidence`): a coarse one (every other scale, each base angle's likeliest residual) finds the label's
print quality q (a statistic of the label, used as defined, at each temperature); the full pass scores every scale,
angle and cell within the placement prior's reach.  (Skipping scales and angles whose coarse score was far below the
label's best was tried and dropped: no cheap bound covers every hypothesis at every temperature, and it moved
posteriors.)

Deterministic: no randomness; torch's deterministic algorithms on THREADS threads (`settle_torch`, set by
`evidence`); on the CPU, or the GPU in one exclusive turn (`sfzmaps.gpu.gpu_turn`, taken by `evidence`; on the 2000
labels `mps` was far slower than the CPU).
"""
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch

from sfzmaps import gpu
from sfzmaps.solver import codes
from sfzmaps.solver.codereader import glyphs
from sfzmaps.solver.codereader import labels as labels_mod

log = logging.getLogger(__name__)

RULES = "run"              # the line rule whose px are unobserved (`labels.line_mask`), chosen on the training labels
RESIDUALS = glyphs.RESIDUALS
SPACE, START, END = " ", "^", "$"
BIGRAM_ADD = 0.5           # the bigram's counts are smoothed by this many in every allowed cell
CODE_ADD = 0.5             # the code prior's counts are smoothed by this many per code
FORM_ADD = 0.5             # p(form | code)'s counts are smoothed by this many per form
NONE_MASS = 0.02           # p(none) and p(other) until Task 4 fits them
OTHER_MASS = 0.03
REACH_LOG = 12.0           # cells whose placement prior is below e^-REACH_LOG of the tail's peak are not scored
# windows past a word's end tried (skip, width; cap heights): `end_breaks` keeps one by two-fold log loss
END_WINDOWS = tuple((a, b) for a in (0.0, 0.1, 0.2) for b in (0.4, 0.6, 0.9, 1.3))
SOURCES = ("sol", "turned")
THREADS = 1                # torch's threads while scoring (labels go in parallel processes): the same sums anywhere


# --- constants and priors --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Constants:
    """The three calibration constants: the temperature dividing every px log-likelihood ratio (None: the pose prior's
    measured one), and the prior masses of "none" and "other"."""
    temperature: float | None = None
    none_mass: float = NONE_MASS
    other_mass: float = OTHER_MASS


@dataclass
class PosePrior:
    """Where the text is, given the print, measured on the training labels (`measure_pose_prior`; see the module).
    Offsets in word heights (with a box) or in `height` (the training words' median height, px) without one; each
    (mean, core sd, tail sd)."""
    left: tuple = (0.0, 0.07, 0.4)       # ink's left end less the word's left end
    right: tuple = (0.0, 0.07, 0.4)      # ink's right end less the word's right end
    middle: tuple = (0.0, 0.04, 0.3)     # the line's middle less the word's middle (down the crop)
    box_tail: float = 0.1                # the tail's share with a box
    ratio: tuple = (0.0, 0.12)           # log(cap height / word height): mean, sd
    half_turn: dict = field(default_factory=dict)    # Terra's direction -> p(a half turn from Terra's angle)
    half_turn_other: float = 0.05        # a direction not in `half_turn`
    along: dict = field(default_factory=lambda: {s: (0.0, 0.15, 0.6) for s in SOURCES})    # point's centre
    across: dict = field(default_factory=lambda: {s: (0.0, 0.1, 0.4) for s in SOURCES})
    point_tail: dict = field(default_factory=lambda: {s: 0.2 for s in SOURCES})
    point_turns: dict = field(default_factory=lambda: {s: [0.7, 0.1, 0.1, 0.1] for s in SOURCES})
    residuals: list = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.2, 0.1])          # p over RESIDUALS
    cap: tuple = (math.log(27.0), 0.3)   # log cap height (px) over the map's labels: mean, sd
    height: float = 27.0                 # the training words' median height, px
    point_heights: tuple = ()            # the training words' heights (px) the point labels' scales span
    foreign: float = 5e-5                # other text per px^2 of crop (Terra's boxes of every kind)
    char_gain: dict = field(default_factory=dict)    # each character's mean ratio per ink px of its cell
    end_break: dict = field(default_factory=dict)    # P(a word ends | ink density past it), without a box (`end_breaks`)
    temperature: float = 30.0            # Constants' default: the design effect within one character
    ratios: tuple = tuple(glyphs.RATIOS)             # the scale grid's cap/word ratios with a box
    point_quantiles: tuple = glyphs.POINT_QUANTILES  # without one: cap heights over these quantiles of the heights
    measured: dict = field(default_factory=dict)     # how it was measured

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=float))

    @classmethod
    def from_dict(cls, d: dict) -> "PosePrior":
        tuples = ("left", "right", "middle", "ratio", "cap", "point_heights", "ratios", "point_quantiles")
        kw = {k: tuple(v) if k in tuples else v for k, v in d.items()}
        for k in ("along", "across"):
            kw[k] = {s: tuple(v) for s, v in kw.get(k, {}).items()}
        return cls(**kw)


def source_of(label: labels_mod.Label) -> str:
    """Where a label's point comes from: "box" (Terra's box), "turned" (the turned pass's box centre) or "sol" (Sol's
    point, or an answered case's)."""
    if label.code_rect is not None:
        return "box"
    return "turned" if label.kind == 3 else "sol"


@dataclass
class Prior:
    """What the posterior learns from the training labels besides the font: p(code) (`code_prior`), each code's
    printed forms (`printed_forms`) and p(form | code) (`form_prior`; empty: a code's forms alike), the "other" bigram
    over `chars` (`char_bigram`) and the pose prior."""
    codes: dict[str, float]
    forms: dict[str, str]
    bigram: np.ndarray
    chars: str
    pose: PosePrior
    form_weights: dict[str, float] = field(default_factory=dict)

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump({"codes": self.codes, "forms": self.forms, "form_weights": self.form_weights,
                       "bigram": self.bigram.tolist(), "chars": self.chars, "pose": self.pose.to_dict()}, fh, indent=1)

    @classmethod
    def load(cls, path) -> "Prior":
        with open(path) as fh:
            d = json.load(fh)
        return cls(codes=d["codes"], forms=d["forms"], bigram=np.array(d["bigram"], float), chars=d["chars"],
                   pose=PosePrior.from_dict(d["pose"]), form_weights=d.get("form_weights", {}))

    @classmethod
    def learn(cls, training_labels, font: glyphs.Font, noise: glyphs.NoiseModel, code_list: dict[str, str] | None = None,
              **measure) -> "Prior":
        """Everything from the training labels: the printed forms (`printed_forms` of `code_list`, by default
        `labels.code_list(training_labels)`), `code_prior`, `form_prior`, `char_bigram` over the font's characters
        and `measure_pose_prior` (keyword arguments go to it)."""
        _guard(training_labels)
        forms = printed_forms(labels_mod.code_list(training_labels) if code_list is None else code_list)
        return cls(codes=code_prior(training_labels, forms), forms=forms,
                   bigram=char_bigram(training_labels, font.chars, forms), chars=font.chars,
                   pose=measure_pose_prior(training_labels, font, noise, **measure),
                   form_weights=form_prior(training_labels, forms))


def _reads(label: labels_mod.Label) -> list[str]:
    """A label's reads as written (Terra's, Sol's) and the others in their spelling."""
    return [labels_mod.printed(t) for t in (label.terra_text, label.sol_text, *label.reads.values()) if t]


def _guard(labels: list[labels_mod.Label]) -> None:
    labels_mod.refuse_held(labels)


def named_district(name: str) -> bool:
    """Whether a table code is a named district (a word of four or more letters with a vowel: "POLK", "UNION",
    "24TH-NOE VALLEY"), whose label may read "<name> NCD"; abbreviations (CRNC, CCB, SLI) and patterned codes (RH-1,
    C-3-O(SD)) are not."""
    return any(re.search("[AEIOUY]", w) for w in re.findall("[A-Z]{4,}", name))


def printed_forms(code_list: dict[str, str]) -> dict[str, str]:
    """Printed form -> code: the code list (`labels.code_list`: the table's names, labels.ALIAS and the printed forms
    the training reads wrote), the other spellings `codes.canon` takes to one of its codes (codes.ALIAS), and for each
    named district (`named_district`) each of its forms with " NCD" after it.  From the code names alone, never from
    a held-out read."""
    out = dict(code_list)
    names = set(code_list.values())
    for alias, code in {**codes.ALIAS, **labels_mod.ALIAS}.items():
        if code in names:
            out.setdefault(alias, code)
    for form, code in list(out.items()):
        if named_district(code) and not form.endswith(" NCD"):
            out.setdefault(form + " NCD", code)
    return dict(sorted(out.items()))


def code_prior(training_labels: list[labels_mod.Label], forms: dict[str, str] | None = None) -> dict[str, float]:
    """p(code): how often each code is printed, one count per training label (its string, `glyphs.consensus`, by
    `forms` or `labels.code_of`; a string naming no code doesn't count), smoothed by CODE_ADD over every code of
    `forms` (printed form -> code; by default `printed_forms(labels.code_list(training_labels))`)."""
    _guard(training_labels)
    forms = printed_forms(labels_mod.code_list(training_labels)) if forms is None else forms
    names = sorted(set(forms.values()))
    counts: dict[str, float] = {}
    for lab in training_labels:
        text = labels_mod.printed(glyphs.consensus(lab))
        code = forms.get(text) or labels_mod.code_of(text, names)
        if code:
            counts[code] = counts.get(code, 0.0) + 1.0
    total = sum(counts.get(c, 0.0) + CODE_ADD for c in names)
    return {c: (counts.get(c, 0.0) + CODE_ADD) / total for c in names}


def form_prior(training_labels: list[labels_mod.Label], forms: dict[str, str]) -> dict[str, float]:
    """p(printed form | its code): one count per training label whose string (`glyphs.consensus`, printed) is the
    form, smoothed by FORM_ADD over each code's forms (`forms`: printed form -> code)."""
    _guard(training_labels)
    counts: dict[str, float] = {}
    for lab in training_labels:
        text = labels_mod.printed(glyphs.consensus(lab))
        if text in forms:
            counts[text] = counts.get(text, 0.0) + 1.0
    by_code: dict[str, list[str]] = {}
    for form, code in forms.items():
        by_code.setdefault(code, []).append(form)
    out = {}
    for fs in by_code.values():
        total = sum(counts.get(f, 0.0) + FORM_ADD for f in fs)
        out.update({f: (counts.get(f, 0.0) + FORM_ADD) / total for f in fs})
    return dict(sorted(out.items()))


def kind_of(char: str) -> str:
    """The bigram's kind of a previous character: START, "A" (a letter), "0" (a digit), or the character itself (the
    space and each punctuation mark)."""
    if char == START:
        return START
    if char.isalpha():
        return "A"
    if char.isdigit():
        return "0"
    return char


def kinds_of(chars: str) -> list[str]:
    """The bigram's rows: START, then every kind among `chars`."""
    return [START] + sorted({kind_of(c) for c in chars})


def char_bigram(training_labels: list[labels_mod.Label], chars: str, forms: dict[str, str]) -> np.ndarray:
    """"other"'s character bigram: log p(next | the previous character's kind), the next over `chars` (the font's,
    the space among them) and END (the last column), one row per kind (`kinds_of`).  Counted from the training
    labels whose string (`glyphs.consensus`) names no code of `forms`, their reads that name none either (as written;
    characters the font lacks break a string in two): other text, not a misspelt code.  Smoothed by BIGRAM_ADD in
    every allowed cell.  The previous characters are pooled by kind (letter, digit, space, each punctuation mark)
    because those reads are few.  Not allowed: a space first, last or twice, or no character."""
    _guard(training_labels)
    names = sorted(set(forms.values()))
    kinds = kinds_of(chars)
    row = {k: i for i, k in enumerate(kinds)}
    col = {c: i for i, c in enumerate(chars)}
    allowed = np.ones((len(kinds), len(chars) + 1), bool)
    allowed[row[START], -1] = False
    if SPACE in col:
        allowed[row[START], col[SPACE]] = allowed[row[SPACE], -1] = allowed[row[SPACE], col[SPACE]] = False
    counts = np.zeros(allowed.shape)
    for lab in training_labels:
        own = labels_mod.printed(glyphs.consensus(lab))
        if own in forms or labels_mod.code_of(own, names):
            continue
        for text in sorted(set(_reads(lab))):
            if text in forms or labels_mod.code_of(text, names):
                continue
            for piece in "".join(c if c in col else "\0" for c in text).split("\0"):
                piece = " ".join(piece.split())
                if piece:
                    for a, b in zip([START, *piece], [*piece, END]):
                        counts[row[kind_of(a)], col[b] if b != END else -1] += 1
    counts = np.where(allowed, counts + BIGRAM_ADD, 0.0)
    with np.errstate(divide="ignore"):
        return np.log(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-300))


# --- the engine ------------------------------------------------------------------------------------------------------

def _log_pdf(x, mean: float, sd: float) -> np.ndarray:
    """log of a Gaussian density at x, its variance widened by a px's rounding (1/12), so it is a fair probability
    of the px."""
    v = sd * sd + 1 / 12
    return -0.5 * (np.asarray(x, float) - mean) ** 2 / v - 0.5 * math.log(2 * math.pi * v)


def _log_mix(x, mean: float, core: float, tail: float, w_tail: float, scale: float) -> list[np.ndarray]:
    """The core and tail components' log densities (with their mixture weights) at x: offsets `scale` px a unit."""
    return [math.log(max(1 - w_tail, 1e-12)) + _log_pdf(x, mean * scale, core * scale),
            math.log(max(w_tail, 1e-12)) + _log_pdf(x, mean * scale, tail * scale)]


def _log_norm(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """log p of a Gaussian over a grid of `values`, normalised over the grid."""
    lp = -0.5 * ((np.asarray(values, float) - mean) / max(sd, 1e-6)) ** 2
    return lp - np.logaddexp.reduce(lp)


def _ink_extent(font: glyphs.Font, char: str) -> tuple[float, float]:
    """A character's ink, from its pen along the line: first and last px, template px (leaned)."""
    laid = font.laid(char)
    cols = np.flatnonzero((laid.sdf > 0).any(axis=0))
    if not len(cols):
        return 0.0, 0.0
    off = font.advance[char] / 2 - laid.ox
    return float(cols[0] + off), float(cols[-1] + off)


def _footprint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The px whose p(ink) a rendering changes from the background's (its noise weights not 0; `glyphs._ab` works
    in float32, so far from a stroke they are ~1e-9, not 0)."""
    return (np.abs(a) > 1e-6) | (np.abs(b) > 1e-6)


class _Label:
    """One label's crop for scoring: its observed ink and px, its frame, scales, angles and placement prior."""

    def __init__(self, label, observed, prior: Prior, device):
        self.label, self.device = label, device
        ink = label.ink
        obs = np.asarray(observed, bool)
        self.xo, self.of = (ink & obs).astype(np.float32), obs.astype(np.float32)
        self.h, self.w = ink.shape
        self.source = source_of(label)
        # the label's word: Terra's word rectangle (px edges -> the px whose centres it holds); none without a box
        self.word = np.zeros(ink.shape, np.float32)
        if label.code_rect is not None:
            u0, v0, u1, v1 = label.code_rect
            self.word[max(0, math.ceil(v0 - 0.5)):max(0, math.ceil(v1 - 0.5)),
                      max(0, math.ceil(u0 - 0.5)):max(0, math.ceil(u1 - 0.5))] = 1.0
        self.xw = self.xo * self.word
        self.word_ink = float(self.xw.sum())
        pose = prior.pose
        heights = np.asarray(pose.point_heights or (pose.height,), float)
        lo, hi = np.quantile(heights, pose.point_quantiles)
        r = pose.ratios
        step = (r[-1] / r[0]) ** (1 / max(1, len(r) - 1))
        n = max(1, math.ceil(math.log(hi * r[-1] / (lo * r[0])) / math.log(step))) + 1
        map_scales = tuple(max(glyphs.S_MIN, s / glyphs.SIZE) for s in np.geomspace(lo * r[0], hi * r[-1], n))
        # a slanted box's word rectangle (from Terra's axis box and angle) is no measure of the letters' height: on
        # the training labels its median is 14 px against 27 for words read a quarter turn
        self.slanted = label.code_rect is not None and (label.angle / 90) != round(label.angle / 90)
        if label.code_rect is not None:
            u0, v0, u1, v1 = label.code_rect
            self.box = (u0, v0, u1, v1)
            self.unit = pose.height if self.slanted else v1 - v0      # px a word height
            self.origin = ((u0 + u1 - 1) / 2, (v0 + v1 - 1) / 2)
            self.scales = map_scales if self.slanted else \
                tuple(max(glyphs.S_MIN, ratio * self.unit / glyphs.SIZE) for ratio in pose.ratios)
            p = pose.half_turn.get(label.direction, pose.half_turn_other)
            bases = ((0.0, 1 - p), (180.0, p))
            self.tail = pose.box_tail
        else:
            self.box = None
            self.unit = pose.height
            self.origin = (label.centre[0] - 0.5, label.centre[1] - 0.5)
            self.scales = map_scales
            turns = pose.point_turns.get(self.source, [0.25] * 4)
            bases = tuple((90.0 * i, turns[i]) for i in range(4))
            self.tail = pose.point_tail.get(self.source, 0.2)
        res = np.asarray(pose.residuals, float)
        res = res / res.sum()
        self.angles = [(b + d, math.log(max(pb, 1e-12)) + math.log(max(pr, 1e-12)))
                       for b, pb in bases for d, pr in zip(RESIDUALS, res)]
        caps = np.log(np.array(self.scales) * glyphs.SIZE)
        if self.box is not None and not self.slanted:
            self.log_scale = _log_norm(caps - math.log(self.unit), pose.ratio[0], pose.ratio[1])
        else:
            self.log_scale = _log_norm(caps, pose.cap[0], pose.cap[1])
        self.log_scale_map = _log_norm(caps, pose.cap[0], pose.cap[1])     # other text's sizes


def _next_fast(n: int) -> int:
    from scipy import fft as sfft
    return sfft.next_fast_len(int(n), real=True)


def _windows(lab: _Label, pose: PosePrior, s: float, angs: list, adv: list[int], lo: list[int], hi: list[int],
             longest: int):
    """The cells worth scoring at a scale for each angle (where the tail's prior is within e^-REACH_LOG of its
    peak): lines ks and pens ts, shared by the angles; and the starts the first pen may take."""
    u = lab.unit
    if lab.box is not None:
        reach_k = math.sqrt(2 * REACH_LOG) * pose.middle[2] * u + abs(pose.middle[0]) * u + 1
        reach_t = math.sqrt(2 * REACH_LOG) * max(pose.left[2], pose.right[2]) * u + \
            max(abs(pose.left[0]), abs(pose.right[0])) * u + 1
        u0, _, u1, _ = lab.box
        half = (u1 - u0) / 2
        ks = np.arange(-math.ceil(reach_k), math.ceil(reach_k) + 1)
        t_first = (-math.ceil(half + reach_t) - max(lo) - 1, math.ceil(-half + reach_t) - min(lo) + 1)
        ts = np.arange(t_first[0], math.ceil(half + reach_t) - min(hi) + max(adv) + 2)
    else:
        al = pose.along.get(lab.source, (0, 0.15, 0.6))
        ac = pose.across.get(lab.source, (0, 0.1, 0.4))
        reach_k = math.sqrt(2 * REACH_LOG) * ac[2] * u + abs(ac[0]) * u + 1
        reach_c = math.sqrt(2 * REACH_LOG) * al[2] * u + abs(al[0]) * u + 1
        ks = np.arange(-math.ceil(reach_k), math.ceil(reach_k) + 1)
        corners = np.array([[0, 0], [lab.w - 1, 0], [0, lab.h - 1], [lab.w - 1, lab.h - 1]], float) - lab.origin
        span = max(np.abs(corners @ np.array(glyphs._axes(a)[0])).max() for a, _ in angs)
        tt = math.ceil(min(span, reach_c + longest)) + max(adv) + 2
        ts = np.arange(-tt, tt + 1)
        t_first = (ts[0], math.ceil(reach_c) + 1)
    starts = np.flatnonzero((ts >= t_first[0]) & (ts <= t_first[1]))
    return ks, ts, starts


def _cell(font: glyphs.Font, char: str, scale: float, angle: float, band: tuple[float, float]) -> np.ndarray:
    """A character's cell, as crop px offsets from its centre (4 corners): its pen's advance along the line by the
    lettering's height (`band`: template rows from the line's middle, top and bottom, the noise's 2 px reach
    included), leaned like the glyphs.  A string's cells tile its line."""
    half = font.advance[char] / 2
    t = math.tan(math.radians(font.slant))
    lin = glyphs._linear(scale, angle, 0.0)
    corners = [(x - y * t, y) for x, y in ((-half, band[0]), (half, band[0]), (half, band[1]), (-half, band[1]))]
    return np.array([lin @ np.array(c) for c in corners])


def _window(font, noise, c: str, scale: float, a: float, band) -> tuple[np.ndarray, int, int]:
    """A character's glyph patch (the noise's weights a and b) and its cell (`_cell`) in one window [3, h, w], and
    the window px (cx, cy) of its centre."""
    corners = _cell(font, c, scale, a, band)
    if c in font.templates:
        d, cx, cy, _ = glyphs._patch(font.laid(c), scale, a)
        pa, pb = glyphs._ab(noise, d)
    else:
        pa = pb = np.zeros((1, 1), np.float32)
        cx = cy = 0
    top = min(-cy, math.floor(corners[:, 1].min()) - 1)
    left = min(-cx, math.floor(corners[:, 0].min()) - 1)
    bottom = max(pa.shape[0] - 1 - cy, math.ceil(corners[:, 1].max()) + 1)
    right = max(pa.shape[1] - 1 - cx, math.ceil(corners[:, 0].max()) + 1)
    h, w = bottom - top + 1, right - left + 1
    cy0, cx0 = -top, -left
    win = np.zeros((3, h, w), np.float32)
    if c in font.templates:
        win[0, cy0 - cy:cy0 - cy + pa.shape[0], cx0 - cx:cx0 - cx + pa.shape[1]] = pa
        win[1, cy0 - cy:cy0 - cy + pb.shape[0], cx0 - cx:cx0 - cx + pb.shape[1]] = pb
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint((corners + (cx0, cy0)) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    win[2] = mask
    return win, cx0, cy0


def _char_maps(lab: _Label, font, noise, sym: list[str], scale: float, angles: list[float], rect, band):
    """Every character's maps at a scale for each angle, for centres in the crop px `rect` (x0, y0, x1, y1): its
    log-likelihood ratio against the background (0 for the space), the observed ink in its cell (`_cell`) and the
    part of that inside the label's word.  Correlations by FFT against the crop's px those centres' patches can reach
    (outside the crop: unobserved).  Returns the maps [3, angle, char, S0, S1], each rolled so that a centre at crop px (x, y) is read at
    [(y - oy) mod S0, (x - ox) mod S1], valid where its patch meets that region; the region's origin (ox, oy) and
    size, and each character's patch: its centre px and size per angle."""
    pats = {(a, c): _window(font, noise, c, scale, a, band) for a in angles for c in sym}
    ph = max(p[0].shape[1] for p in pats.values())
    pw = max(p[0].shape[2] for p in pats.values())
    x0, y0, x1, y1 = rect
    ox, oy = max(0, x0 - pw), max(0, y0 - ph)
    ex, ey = min(lab.w, x1 + pw), min(lab.h, y1 + ph)
    dh, dw = max(1, ey - oy), max(1, ex - ox)
    s0, s1 = _next_fast(dh + ph), _next_fast(dw + pw)
    sub = np.zeros((3, s0, s1), np.float32)
    if ox < ex and oy < ey:
        sub[0, :ey - oy, :ex - ox] = lab.xo[oy:ey, ox:ex]
        sub[1, :ey - oy, :ex - ox] = lab.of[oy:ey, ox:ex]
        sub[2, :ey - oy, :ex - ox] = lab.xw[oy:ey, ox:ex]
    crop_ft = torch.fft.rfft2(torch.from_numpy(sub).to(lab.device))
    out = torch.empty((3, len(angles), len(sym), s0, s1), dtype=torch.float32, device=lab.device)
    geom = np.zeros((len(angles), len(sym), 4), np.int64)            # cx, cy, patch width, patch height
    for i, a in enumerate(angles):
        stack = np.zeros((3, len(sym), s0, s1), np.float32)
        for j, c in enumerate(sym):
            win, cx, cy = pats[a, c]
            stack[:, j, :win.shape[1], :win.shape[2]] = win
            geom[i, j] = (cx, cy, win.shape[2], win.shape[1])
        f = torch.conj(torch.fft.rfft2(torch.from_numpy(stack).to(lab.device)))
        both = torch.fft.irfft2(torch.stack([f[0] * crop_ft[0] + f[1] * crop_ft[1], f[2] * crop_ft[0],
                                             f[2] * crop_ft[2]]), s=(s0, s1))
        # both[m, j, y, x]: the patch's top-left at region px (y, x) (negative lags wrapped): its centre at
        # (y + cy, x + cx)
        for j in range(len(sym)):
            out[:, i, j] = torch.roll(both[:, j], shifts=(int(geom[i, j, 1]), int(geom[i, j, 0])), dims=(1, 2))
    return out, (ox, oy, dw, dh), geom


def _gather(maps: torch.Tensor, region, geom, lab: _Label, ks, ts, axes, centre_off: np.ndarray) -> torch.Tensor:
    """The characters' maps at every cell: [m, c, t, (angle, k)] with character c's pen at t on line k (its centre
    `centre_off[c]` px further along, rounded to a whole px along the line), 0 where its patch misses the region."""
    n_m, n_a, n_c, s0, s1 = maps.shape
    ox, oy, dw, dh = region
    n_k, n_t = len(ks), len(ts)
    shift = np.rint(centre_off).astype(np.int64)
    lo, hi = int(shift.min()), int(shift.max())
    tx = np.arange(ts[0] + lo, ts[-1] + hi + 1)
    out = torch.zeros((n_m, n_c, n_t, n_a * n_k), dtype=torch.float32, device=lab.device)
    for i, (eu, ev) in enumerate(axes):
        x = np.rint(lab.origin[0] + ks[None, :] * ev[0] + tx[:, None] * eu[0]).astype(np.int64) - ox
        y = np.rint(lab.origin[1] + ks[None, :] * ev[1] + tx[:, None] * eu[1]).astype(np.int64) - oy
        idx = torch.from_numpy((np.mod(y, s0) * s1 + np.mod(x, s1)).reshape(-1)).to(lab.device)
        vals = maps[:, i].reshape(n_m, n_c, -1)[:, :, idx].reshape(n_m, n_c, len(tx), n_k)
        cx, cy, pw, ph = (geom[i, :, q][:, None, None] for q in range(4))
        lag_x, lag_y = x[None] - cx, y[None] - cy                        # the patch's top-left in the region
        ok = (lag_x > -pw) & (lag_x < dw) & (lag_y > -ph) & (lag_y < dh)
        vals = torch.where(torch.from_numpy(ok).to(lab.device), vals, torch.zeros((), device=lab.device))
        for j in range(n_c):
            s = shift[j] - lo
            out[:, j, :, i * n_k:(i + 1) * n_k] = vals[:, j, s:s + n_t]
    return out


def _rect(lab: _Label, ks, ts, axes, centre_off) -> tuple[int, int, int, int]:
    """The crop px rectangle (x0, y0, x1, y1) holding every character centre the cells need."""
    t = np.array([ts[0] + centre_off.min(), ts[-1] + centre_off.max()])
    pts = []
    for eu, ev in axes:
        for k in (ks[0], ks[-1]):
            for tt in t:
                pts.append((lab.origin[0] + k * ev[0] + tt * eu[0], lab.origin[1] + k * ev[1] + tt * eu[1]))
    pts = np.array(pts)
    return (math.floor(pts[:, 0].min()) - 1, math.floor(pts[:, 1].min()) - 1,
            math.ceil(pts[:, 0].max()) + 2, math.ceil(pts[:, 1].max()) + 2)


def _viterbi(em: torch.Tensor, inks: torch.Tensor, adv: list[int], hi: list[int], kind: list[int],
             logq: torch.Tensor, end_w: torch.Tensor | None = None):
    """The best string (the kinds' bigram `logq`: [kind, next char + END]) whose first character c has its pen at t
    on line r: its score (the characters' emissions `em`, [c, t, r], its bigram and `end_w` [t, r, c] where it ends:
    its last character c's pen at t), its characters' ink maps (`inks`, [m, c, t, r], summed along it) and the t
    index of its last ink px: V [t, r, c], the inks [m, t, r, c] and Z [t, r, c].  `kind`: each character's row of
    `logq`.  Strings end inside the cells."""
    n_c, n_t, n_r = em.shape
    n_m = inks.shape[0]
    n_k = logq.shape[0]
    amax, amin = max(adv), min(adv)
    dev = em.device
    ninf = float("-inf")
    VIZ = torch.zeros((n_t + amax, n_r, n_c, n_m + 2), dtype=torch.float32, device=dev)   # score, inks, last px
    VIZ[..., 0] = ninf
    MS = torch.zeros((n_t + amax, n_r, n_k, n_m + 2), dtype=torch.float32, device=dev)    # the best by kind
    MS[..., 0] = ninf
    em_t, ink_t = em.permute(1, 2, 0), inks.permute(2, 3, 1, 0)                            # [t, r, c], [t, r, c, m]
    kinds = torch.tensor(kind, device=dev)
    lags = torch.tensor(adv, device=dev)
    endq = logq[kinds, n_c]
    nextq = logq[:, :n_c]
    hi_t = torch.tensor(hi, dtype=torch.float32, device=dev)
    rr = torch.arange(n_r, device=dev)[None, :, None]
    zero = torch.zeros((), device=dev)
    t1 = n_t
    while t1 > 0:
        t0 = max(0, t1 - amin)
        tt = torch.arange(t0, t1, device=dev)
        nxt = MS[(tt[:, None] + lags[None, :])[:, None, :], rr, kinds[None, None, :]]        # [b, r, c, m + 2]
        end = endq.expand_as(nxt[..., 0]) if end_w is None else endq + end_w[t0:t1]
        go_on = nxt[..., 0] > end
        here = tt.to(torch.float32)[:, None, None] + hi_t
        VIZ[t0:t1, ..., 0] = em_t[t0:t1] + torch.where(go_on, nxt[..., 0], end)
        VIZ[t0:t1, ..., 1:-1] = ink_t[t0:t1] + torch.where(go_on[..., None], nxt[..., 1:-1], zero)
        VIZ[t0:t1, ..., -1] = torch.where(go_on, nxt[..., -1], here)
        best, which = (VIZ[t0:t1, :, None, :, 0] + nextq).max(dim=3)                         # [b, r, kind]
        MS[t0:t1, ..., 0] = best
        MS[t0:t1, ..., 1:] = torch.gather(VIZ[t0:t1, ..., 1:], 2, which[..., None].expand(-1, -1, -1, n_m + 1))
        t1 = t0
    return VIZ[:n_t, ..., 0], VIZ[:n_t, ..., 1:-1].permute(3, 0, 1, 2), VIZ[:n_t, ..., -1]


@dataclass
class Evidence:
    """log E_h (the tempered likelihood ratio against "none", see the module) of every printed form and of "other",
    at each temperature; and the label's print quality (`gain`, per temperature)."""
    temperatures: tuple[float, ...]
    forms: dict[str, np.ndarray]
    other: np.ndarray
    gain: np.ndarray
    seconds: float = 0.0


def _lse(t: torch.Tensor) -> float:
    return float(torch.logsumexp(t.reshape(-1), 0)) if t.numel() else float("-inf")


class _Scorer:
    """The work for one label: its chunks (a scale, a set of angles) and every hypothesis's terms in them."""

    def __init__(self, label, font, noise, prior: Prior, observed, device):
        self.font, self.noise, self.prior, self.device = font, noise, prior, device
        self.pose = prior.pose
        self.lab = _Label(label, observed, prior, device)
        chars = prior.chars
        self.inked = [c for c in chars if c in font.templates]
        self.sym = self.inked + [c for c in chars if c not in font.templates]       # inkless (the space) last
        self.at = {c: i for i, c in enumerate(self.sym)}
        kinds = kinds_of(chars)
        cols = [list(chars).index(c) for c in self.sym] + [len(chars)]
        self.logq = torch.from_numpy(prior.bigram[:, cols].astype(np.float32)).to(device)
        self.kind = [kinds.index(kind_of(c)) for c in self.sym]
        self.start_q = self.logq[kinds.index(START), :len(self.sym)]
        gains = prior.pose.char_gain
        mean = float(np.mean(list(gains.values()))) if gains else 1.0
        self.char_gain = torch.tensor([gains.get(c, mean) for c in self.sym], dtype=torch.float32, device=device)
        self.mean_gain = mean
        self.band = lettering_band(font)
        self.lean = math.tan(math.radians(font.slant))
        self.forms = {f: code for f, code in prior.forms.items() if f and all(c in self.at for c in f)}
        missing = sorted(set(prior.forms) - set(self.forms))
        if missing:
            log.warning("forms with characters the font lacks, left out: %s", ", ".join(missing))

    def chunks(self, coarse: bool = False):
        """Every scale's cells for each base angle and its residuals, with the characters' maps (a generator).
        `coarse`: every other scale, and of each base angle only its likeliest residual."""
        lab, pose, font = self.lab, self.pose, self.font
        per_base = len(RESIDUALS)
        best_res = int(np.argmax(pose.residuals))
        for si, s in enumerate(lab.scales):
            if coarse and si % 2:
                continue
            adv = [max(1, round(font.advance[c] * s)) for c in self.sym]
            ext = {c: _ink_extent(font, c) for c in self.inked}
            lo = [round(ext[c][0] * s) if c in ext else 0 for c in self.sym]
            hi = [round(ext[c][1] * s) if c in ext else 0 for c in self.sym]
            centre_off = np.array([font.advance[c] * s / 2 for c in self.sym])
            layouts = {f: np.concatenate([[0], np.cumsum([adv[self.at[c]] for c in f])[:-1]]).astype(int)
                       for f in self.forms}
            longest = max((int(p[-1]) + adv[self.at[f[-1]]] for f, p in layouts.items()), default=0)
            for b0 in range(0, len(lab.angles), per_base):
                angs = lab.angles[b0:b0 + per_base]
                if coarse:
                    angs = angs[best_res:best_res + 1]
                axes = [glyphs._axes(a) for a, _ in angs]
                ks, ts, starts = _windows(lab, pose, s, angs, adv, lo, hi, longest)
                rect = _rect(lab, ks, ts, axes, centre_off)
                maps, region, geom = _char_maps(lab, font, self.noise, self.sym, s, [a for a, _ in angs], rect,
                                                (self.band[0] - 2 / s, self.band[1] + 2 / s))
                EI = _gather(maps, region, geom, lab, ks, ts, axes, centre_off)
                EI[1] *= self.char_gain[:, None, None]                  # the ink in its cell x its typical gain
                del maps
                yield si, s, angs, axes, ks, ts, starts, adv, lo, hi, layouts, longest, EI

    def placement(self, si, angs, axes, ks, ts, adv, lo, hi, longest):
        """The placement prior over a chunk's cells, per component (core, tail): each row's log prior, and the log
        weights of a string's first and last ink px along its line ([tx, r]); for point labels the centre's (mean,
        sd) along the line instead of the last px's weight.  And other text's log density per row."""
        lab, pose = self.lab, self.pose
        u = lab.unit
        log_angle = np.repeat([la for _, la in angs], len(ks))
        tx = np.arange(ts[0] + min(lo) - 1, ts[-1] + longest + max(hi) + 2)
        n_r = len(angs) * len(ks)
        rows, firsts, lasts, alongs = [], [], [], []
        for comp in (1, 2):
            w = math.log(max(1 - lab.tail, 1e-12)) if comp == 1 else math.log(max(lab.tail, 1e-12))
            if lab.box is not None:
                u0, _, u1, _ = lab.box
                mid, first, last = [], [], []
                for eu, ev in axes:
                    mid.append(_log_pdf(ks * ev[1], pose.middle[0] * u, pose.middle[comp] * u))
                    # the ink px's centre (glyphs' px-centre frame), its edges against the word's (px edges)
                    uu = lab.origin[0] + ks[None, :] * ev[0] + tx[:, None] * eu[0]
                    left = _log_pdf(uu - u0, pose.left[0] * u, pose.left[comp] * u)
                    right = _log_pdf(uu + 1 - u1, pose.right[0] * u, pose.right[comp] * u)
                    a, b = (left, right) if eu[0] >= 0 else (right, left)
                    first.append(a)
                    last.append(b)
                norm = 0.5 * math.log(2 * math.pi * ((pose.left[comp] * u) ** 2 + (pose.right[comp] * u) ** 2 + 1 / 6))
                rows.append(np.concatenate(mid) + log_angle + w + lab.log_scale[si])
                firsts.append(np.concatenate(first, axis=1) + norm)
                lasts.append(np.concatenate(last, axis=1))
                alongs.append(None)
            else:
                al = pose.along.get(lab.source, (0, 0.15, 0.6))
                ac = pose.across.get(lab.source, (0, 0.1, 0.4))
                rows.append(np.tile(_log_pdf(ks, ac[0] * u, ac[comp] * u), len(angs)) + log_angle + w +
                            lab.log_scale[si])
                firsts.append(np.zeros((len(tx), n_r)))
                lasts.append(None)
                alongs.append((al[0] * u, al[comp] * u))
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(self.device)
        log_rho = t(log_angle) + (math.log(pose.foreign) if pose.foreign > 0 else float("-inf")) + \
            float(lab.log_scale_map[si])                       # no other text (0): the bare background alone
        return ([t(r) for r in rows], [t(f) for f in firsts], [t(x) if x is not None else None for x in lasts],
                alongs, int(ts[0] - tx[0]), log_rho)

    def end_factors(self, s: float, axes, ks, x0: int, n_x: int):
        """Without a box: the end factor (`end_factor`) of a word starting just after each px along each line (from
        the ink just before it) and of one ending at it (from the ink just after it): [tx, r] each, n_x px from x0
        (the placement's px along), the rows as the cells' (angle, then line)."""
        eb = self.pose.end_break
        skip, width = _window_px(eb["window"], s)
        reach = skip + width
        band = band_px(self.band, s)
        tx = np.arange(x0 - reach, x0 + n_x + reach)
        col = column_ink(self.lab.xo, self.lab.origin, axes, ks, tx, band, self.lean).reshape(len(axes) * len(ks), -1)
        after, before = adjacent_ink(col, width, band[1] - band[0] + 1, skip)

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(end_factor(eb, a[:, reach:reach + n_x]).T, np.float32)) \
                .to(self.device)
        return t(before), t(after)

    def codes_at(self, EI, ts, starts, layouts, lo, hi, longest):
        """Every form's raw ratio and its ink maps (EI's channels summed over its characters) with its first pen at
        each start, [form, channel, start, r]; and the t of each one's first and last ink px, [form, start]."""
        names = list(self.forms)
        n_f, n_s = len(names), len(starts)
        width = max(len(f) for f in names)
        n_m, n_c, n_t = EI.shape[0], EI.shape[1], EI.shape[2]
        EIp = torch.zeros((n_m, n_c + 1, n_t + longest + 1, EI.shape[3]), dtype=torch.float32, device=self.device)
        EIp[:, :n_c, :n_t] = EI                     # past the cells, and the blank symbol n_c that pads short forms: 0
        cidx = np.full((n_f, width), n_c, np.int64)
        pens = np.zeros((n_f, width), np.int64)
        for i, f in enumerate(names):
            cidx[i, :len(f)] = [self.at[c] for c in f]
            pens[i, :len(f)] = layouts[f]
        t_at = torch.from_numpy(int(starts[0]) + pens[:, :, None] + np.arange(n_s)[None, None, :]).to(self.device)
        c_at = torch.from_numpy(cidx).to(self.device)
        tot = torch.zeros((n_m, n_f, n_s, EI.shape[3]), dtype=torch.float32, device=self.device)
        lengths = np.array([len(f) for f in names])
        for j in range(width):
            fs = torch.from_numpy(np.flatnonzero(lengths > j)).to(self.device)
            tot[:, fs] += EIp[:, c_at[fs, j, None], t_at[fs, j]]
        first = ts[starts][None, :] + np.array([lo[self.at[f[0]]] for f in names])[:, None]
        last = ts[starts][None, :] + np.array([layouts[f][-1] + hi[self.at[f[-1]]] for f in names])[:, None]
        return names, tot.permute(1, 0, 2, 3), first, last


def _jitter(EI: torch.Tensor, sd: float, temperature: float) -> torch.Tensor:
    """Each character anywhere near its pen along the line, its shift d (px, |d| <= 3.5 sd) Gaussian with `sd`
    (normalised over those shifts): the tempered ratio summed over the shifts, T log sum_d p(d) exp(E(t + d) / T)
    (so that over T it is the tempered ratio), and the ink maps (EI[1:]) averaged with the shifts' posterior
    weights.  An inkless character keeps 0.  sd 0: unchanged."""
    reach = math.ceil(3.5 * sd)
    if reach < 1:
        return EI
    n_t = EI.shape[2]
    pad = torch.zeros_like(EI[:, :, :reach])
    pad[0] = float("-inf")
    P = torch.cat([pad, EI, pad], dim=2)
    shifts = np.arange(-reach, reach + 1)
    log_p = -0.5 * shifts ** 2 / (sd * sd + 1 / 12)
    log_p = log_p - np.logaddexp.reduce(log_p)
    terms = torch.stack([P[0, :, reach + d:reach + d + n_t] / temperature + float(lp) for d, lp in zip(shifts, log_p)])
    total = torch.logsumexp(terms, dim=0)
    weights = torch.nan_to_num(torch.exp(terms - total), nan=0.0)
    inks = sum(weights[i] * P[1:, :, reach + d:reach + d + n_t] for i, d in enumerate(shifts))
    return torch.cat([(temperature * total)[None], inks])


def _place_weights(row, first_w, last_w, along, tx0, ts, first, last) -> torch.Tensor:
    """Forms' placement weights at each of their starts and rows: [form, start, r]; `first`, `last`: the t of each
    one's first and last ink px per start ([form, start]); `first_w`, `last_w`: [tx, r]."""
    dev = row.device
    fi = torch.from_numpy((first - ts[0] + tx0).astype(np.int64)).to(dev).clamp(0, len(first_w) - 1)
    w = row + first_w[fi]
    if last_w is not None:
        li = torch.from_numpy((last - ts[0] + tx0).astype(np.int64)).to(dev)
        off = (li >= len(last_w))[:, :, None]
        w = w + torch.where(off, torch.tensor(float("-inf"), device=dev), last_w[li.clamp(0, len(last_w) - 1)])
    else:
        w = w + torch.from_numpy(_log_pdf((first + last) / 2, *along).astype(np.float32)).to(dev)[:, :, None]
    return w


def _log_other_text(log_rho, g, G, W, mean_gain: float, word_ink: float) -> torch.Tensor:
    """log of what other text would make of the ink a hypothesis claims, against what it makes of it under "none"
    (see the module): the hypothesis's cells outside the label's word are one more word of other text, and whatever
    of the word it leaves is one too, where under "none" the word is a single word.  `G`: the claimed ink weighted by
    each character's gain; `W`: the claimed ink inside the word (raw); `word_ink`: the word's ink (0 without a box:
    then only the first term is left); g = the label's quality / T."""
    softplus = torch.nn.functional.softplus
    rest = torch.clamp(word_ink - W, min=0.0)
    return softplus(log_rho + g * (G - mean_gain * W)) + softplus(log_rho + g * mean_gain * word_ink) - \
        softplus(log_rho + g * mean_gain * rest)


def band_px(band: tuple[float, float], scale: float) -> tuple[int, int]:
    """The lettering band's first and last crop px across the line at a scale, from the line's middle."""
    return round(band[0] * scale), round(band[1] * scale)


def column_ink(xo: np.ndarray, origin, axes, ks, tx, band: tuple[int, int], lean: float) -> np.ndarray:
    """The observed ink across the lettering band at each px along each line, [angle, k, t]: the column at t on line
    k holds the px t - j * lean along and k + j across (j over `band`, crop px), leaned like the glyphs (`lean`: the
    slant's tangent); off the crop: no ink."""
    h, w = xo.shape
    out = np.zeros((len(axes), len(ks), len(tx)), np.float32)
    kk0 = np.asarray(ks, float)[:, None]
    tt0 = np.asarray(tx, float)[None, :]
    for i, (eu, ev) in enumerate(axes):
        for j in range(band[0], band[1] + 1):
            kk, tt = kk0 + j, tt0 - j * lean
            x = np.rint(origin[0] + kk * ev[0] + tt * eu[0]).astype(np.int64)
            y = np.rint(origin[1] + kk * ev[1] + tt * eu[1]).astype(np.int64)
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            out[i] += np.where(ok, xo[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)], 0.0)
    return out


def adjacent_ink(col: np.ndarray, width: int, rows: int, skip: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The ink density (ink px over px) in `width` px past each px along a line, `skip` px on (t+skip+1 ..
    t+skip+width), and as far before it (t-skip-width .. t-skip-1), over `rows` band px, from the columns `col`
    [.., t]; beyond `col`: paper."""
    reach = width + skip
    pad = np.zeros(col.shape[:-1] + (reach,), np.float64)
    c = np.concatenate([pad, col.astype(np.float64), pad], axis=-1)
    cs = np.concatenate([np.zeros(col.shape[:-1] + (1,)), np.cumsum(c, axis=-1)], axis=-1)
    t = np.arange(col.shape[-1]) + reach
    area = float(width * max(rows, 1))
    return ((cs[..., t + reach + 1] - cs[..., t + skip + 1]) / area,
            (cs[..., t - skip] - cs[..., t - reach]) / area)


def end_factor(end_break: dict, density) -> np.ndarray:
    """log P(a word ends | the ink density past the end) / P(a word ends) (`end_breaks`); 0 when not measured."""
    density = np.asarray(density, float)
    if not end_break:
        return np.zeros(density.shape)
    edges = np.asarray(end_break["edges"], float)
    i = np.clip(np.searchsorted(edges, density, side="left"), 0, len(edges) - 1)
    return np.asarray(end_break["log_factor"], float)[i]


def settle_torch() -> None:
    """torch's deterministic algorithms (it raises on an op that has none) on THREADS threads."""
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def evidence(label: labels_mod.Label, font: glyphs.Font, noise: glyphs.NoiseModel, prior: Prior,
             temperatures=(None,), *, observed: np.ndarray | None = None, rules=RULES, device="cpu") -> Evidence:
    """log E_h of every printed form of `prior` and of "other" (see the module), at each temperature (None: the pose
    prior's measured one).  `observed`: the crop's observed px (not on a line by `rules` when None).  torch runs
    deterministic on THREADS threads (`settle_torch`); on the GPU (`device` "mps") in one exclusive turn."""
    settle_torch()
    with gpu.gpu_turn(True, device):
        return _evidence(label, font, noise, prior, temperatures, observed, rules, device)


def _evidence(label, font, noise, prior: Prior, temperatures, observed, rules, device) -> Evidence:
    t_start = time.perf_counter()
    pose = prior.pose
    temps = tuple(float(pose.temperature if t is None else t) for t in temperatures)
    if observed is None:
        observed = ~labels_mod.line_mask(label, rules=rules)
    sc = _Scorer(label, font, noise, prior, observed, device)
    shift_sd = font.spacing_sd / math.sqrt(2)          # a pen's own shift: a gap is the difference of two
    # pass 1: the label's print quality, the best code's ratio per ink px in its footprint at the core placement
    # (every other scale, each base angle's likeliest residual)
    best = [(-math.inf, 0.0) for _ in temps]
    for si, s, angs, axes, ks, ts, starts, adv, lo, hi, layouts, longest, EI in sc.chunks(coarse=True):
        rows, firsts, lasts, alongs, tx0, _ = sc.placement(si, angs, axes, ks, ts, adv, lo, hi, longest)
        for ti, temp in enumerate(temps):
            names, tot, first, last = sc.codes_at(_jitter(EI, shift_sd * s, temp), ts, starts, layouts, lo, hi,
                                                  longest)
            score = tot[:, 0] / temp + _place_weights(rows[0], firsts[0], lasts[0], alongs[0], tx0, ts, first, last)
            i = int(torch.argmax(score))
            if float(score.reshape(-1)[i]) > best[ti][0]:
                ink = float(tot[:, 1].reshape(-1)[i])
                best[ti] = (float(score.reshape(-1)[i]), float(tot[:, 0].reshape(-1)[i]) / ink if ink > 0 else 0.0)
    gains = np.array([max(0.0, g) for _, g in best])
    # pass 2: every hypothesis at every scale, angle and cell
    acc_forms = [[] for _ in temps]
    acc_other = [[] for _ in temps]
    names = list(sc.forms)
    breaks = sc.lab.box is None and bool(pose.end_break)
    for si, s, angs, axes, ks, ts, starts, adv, lo, hi, layouts, longest, EI in sc.chunks():
        rows, firsts, lasts, alongs, tx0, log_rho = sc.placement(si, angs, axes, ks, ts, adv, lo, hi, longest)
        n_t, n_r = EI.shape[2], EI.shape[3]
        n_x = len(firsts[0])                                    # the placement's px along each line
        r_idx = torch.arange(n_r, device=device)[None, :, None]
        # each character's first and last ink px with its pen at t, as the placement's px: [t, c]
        at_lo = torch.from_numpy(np.arange(n_t)[:, None] + tx0 + np.array(lo)[None, :]).to(device).clamp(0, n_x - 1)
        at_hi = torch.from_numpy(np.arange(n_t)[:, None] + tx0 + np.array(hi)[None, :]).to(device).clamp(0, n_x - 1)
        start_w = [f[at_lo] for f in firsts]                    # each first character's start weight: [t, c, r]
        end_w = choose = left_w = right_w = None
        if breaks:                                              # where a word ends, from the ink along the line
            left_w, right_w = sc.end_factors(s, axes, ks, int(ts[0]) - tx0, n_x)          # [tx, r] each
            end_w = right_w[at_hi].permute(0, 2, 1)                                      # [t, r, c]
        elif lasts[0] is not None:
            # Terra's word: the weight that chooses where the best "other" string ends (the two components mixed by
            # their rows' weights); its exact weights are applied at the end it carries, below
            rw = torch.stack(rows)
            choose = torch.logsumexp(torch.stack(lasts) + rw[:, None, :], dim=0) - torch.logsumexp(rw, dim=0)
            end_w = choose[at_hi].permute(0, 2, 1)
        for ti, temp in enumerate(temps):
            g = gains[ti] / temp
            EJ = _jitter(EI, shift_sd * s, temp)
            _, tot, first, last = sc.codes_at(EJ, ts, starts, layouts, lo, hi, longest)
            val = tot[:, 0] / temp - _log_other_text(log_rho, g, tot[:, 1], tot[:, 2], sc.mean_gain, sc.lab.word_ink)
            if breaks:
                fi = torch.from_numpy((first - ts[0] + tx0).astype(np.int64)).to(device).clamp(0, n_x - 1)
                li = torch.from_numpy((last - ts[0] + tx0).astype(np.int64)).to(device).clamp(0, n_x - 1)
                val = val + left_w[fi] + right_w[li]                                     # [form, start, r]
            parts = torch.stack([torch.logsumexp(
                (val + _place_weights(rows[c], firsts[c], lasts[c], alongs[c], tx0, ts, first, last))
                .reshape(len(names), -1), dim=1) for c in range(2)])
            acc_forms[ti].append(torch.logsumexp(parts, dim=0).cpu().numpy())
            # other: the best string from each start (where its bigram, and its end's weight, end it), its own ink
            # and its last ink px carried along; the placement's weights judged at that end
            em = (EJ[0] - gains[ti] * EJ[1]) / temp
            V, inks, Zv = _viterbi(em, EJ[1:], adv, hi, sc.kind, sc.logq, end_w)
            V, Gv, Wv, Zv = V[starts], inks[0][starts], inks[1][starts], Zv[starts]          # [start, r, c]
            explained = g * Gv - _log_other_text(log_rho[None, :, None], g, Gv, Wv, sc.mean_gain, sc.lab.word_ink)
            at_end = (Zv.long() + tx0).clamp(0, n_x - 1)
            if choose is not None:
                V = V - choose[at_end, r_idx]                   # the choosing weight off: the exact ones below
            if breaks:
                V = V + left_w[at_lo[starts]].permute(0, 2, 1)  # the start's factor (the end's is in V)
            parts = []
            for c in range(2):
                val_o = V + sc.start_q + start_w[c][starts].permute(0, 2, 1) + explained
                if lasts[c] is not None:                      # Terra's word: the string's last ink px at its end
                    val_o = val_o + lasts[c][at_end, r_idx]
                if alongs[c] is not None:                     # point labels: the string's centre about the point
                    first_t = torch.from_numpy(ts[starts].astype(np.float32)).to(device)[:, None, None] + \
                        torch.tensor(lo, dtype=torch.float32, device=device)
                    centre = (first_t + float(ts[0]) + Zv) / 2
                    mean, sd = alongs[c]
                    var = sd * sd + 1 / 12
                    val_o = val_o - 0.5 * (centre - mean) ** 2 / var - 0.5 * math.log(2 * math.pi * var)
                parts.append(_lse(val_o + rows[c][None, :, None]))
            acc_other[ti].append(float(np.logaddexp.reduce(parts)))
    forms = {f: np.array([float(np.logaddexp.reduce([a[k] for a in acc_forms[ti]])) for ti in range(len(temps))])
             for k, f in enumerate(names)}
    other = np.array([float(np.logaddexp.reduce(acc_other[ti])) for ti in range(len(temps))])
    return Evidence(temperatures=temps, forms=forms, other=other, gain=gains, seconds=time.perf_counter() - t_start)


# --- measuring the pose prior on the training labels -----------------------------------------------------------------

def mixture_fit(z: np.ndarray, area: np.ndarray, floor, rounds: int = 300) -> dict:
    """A core and a tail Gaussian (one mean, a diagonal sd each, the tail's at least the core's) and a uniform over
    each row's `area` (the volume its z could take): fitted by EM.  The uniform takes what is neither (in the
    training fits: a string matched on other ink).  `floor`: no sd below it (per axis).  Returns the mean, the core's
    and the tail's sd, the three weights and each row's responsibilities (core, tail, uniform)."""
    z = np.asarray(z, float).reshape(len(z), -1)
    floor = np.maximum(np.asarray(floor, float), 1e-6)
    mu = np.median(z, axis=0)
    sc = np.maximum(floor, 1.4826 * np.median(np.abs(z - mu), axis=0))
    st = 4 * sc
    w = np.array([0.7, 0.2, 0.1])
    log_u = -np.log(np.asarray(area, float))
    r = np.zeros((len(z), 3))
    def logn(mean, s):
        return -0.5 * (((z - mean) / s) ** 2).sum(axis=1) - np.log(s).sum() - z.shape[1] / 2 * math.log(2 * math.pi)

    for _ in range(rounds):
        lp = np.stack([math.log(w[0]) + logn(mu, sc), math.log(w[1]) + logn(mu, st), math.log(w[2]) + log_u], axis=1)
        r = np.exp(lp - np.logaddexp.reduce(lp, axis=1, keepdims=True))
        w = np.clip(r.mean(axis=0), 1e-6, None)
        w /= w.sum()
        prec = r[:, :1] / sc ** 2 + r[:, 1:2] / st ** 2
        mu = (prec * z).sum(axis=0) / np.maximum(prec.sum(axis=0), 1e-12)
        sc = np.maximum(floor, np.sqrt((r[:, :1] * (z - mu) ** 2).sum(axis=0) / max(r[:, 0].sum(), 1e-12)))
        st = np.maximum(sc, np.sqrt((r[:, 1:2] * (z - mu) ** 2).sum(axis=0) / max(r[:, 1].sum(), 1e-12)))
    return {"mean": mu.tolist(), "core": sc.tolist(), "tail": st.tolist(), "weights": w.tolist(), "resp": r}


def _ink_edges(font, text, pose, shape) -> tuple[float, float, float, float] | None:
    """The drawn ink's box (px edges: x0, y0, x1, y1) of text at a pose, None when it draws nothing."""
    d = glyphs.render(font, text, pose, shape)
    ys, xs = np.nonzero(d > 0)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _to_crop(label, xy) -> np.ndarray:
    """Page px (edge frame) -> the label's crop (edge frame)."""
    a = label.to_page
    return np.linalg.solve(a[:, :2], (np.asarray(xy, float) - 0.5 - a[:, 2]).T).T + 0.5


def _word_offsets(labels, points_by_label) -> np.ndarray:
    """Terra's word centre less another reader's point (in the crop: along and across the reading direction, word
    heights), for each (label, page point)."""
    out = []
    for lab, xy in points_by_label:
        u0, v0, u1, v1 = lab.code_rect
        h = v1 - v0
        uv = _to_crop(lab, xy)
        out.append((((u0 + u1) / 2 - uv[0]) / h, ((v0 + v1) / 2 - uv[1]) / h))
    return np.array(out, float).reshape(-1, 2)


def sol_offsets(training_labels) -> np.ndarray:
    """Terra's word centre less Sol's point inside it (the one nearest its centre, as the prints were matched), on
    the training prints both read (kind 0): along and across, word heights."""
    from sfzmaps.solver import saved
    pairs = []
    for sheet in sorted({lab.sheet for lab in training_labels if lab.kind == 0}):
        use = [s for s in saved.sol_labels(sheet) if s.get("kind", "use") == "use"]
        pts = np.array([[s["x"], s["y"]] for s in use], float).reshape(-1, 2)
        for lab in training_labels:
            if lab.sheet != sheet or lab.kind != 0:
                continue
            x0, y0, x1, y1 = lab.box
            inside = np.flatnonzero((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
            if len(inside):
                j = inside[np.argmin(np.hypot(pts[inside, 0] - (x0 + x1) / 2, pts[inside, 1] - (y0 + y1) / 2))]
                pairs.append((lab, pts[j]))
    return _word_offsets([p[0] for p in pairs], pairs)


def turned_offsets(training_labels) -> np.ndarray:
    """Terra's word centre less the turned pass's zoning-code box centre on the training prints it read (its box
    holding the print's point or overlapping Terra's box by IoU > prints.TURNED_JOIN_IOU; of several, the farthest
    from its tile's edge, as the prints were joined): along and across, word heights."""
    from sfzmaps.solver import prints, saved
    from sfzmaps.solver.textnet import boxes as textboxes
    turned = [b for b in saved.answers(prints.TURNED_POSITIONS) if b["kind"] == "zoning code"]
    by_sheet: dict[str, list] = {}
    for b in turned:
        by_sheet.setdefault(b["sheet"], []).append(b)
    pairs = []
    for lab in training_labels:
        if lab.box is None:
            continue
        hit = [b for b in by_sheet.get(lab.sheet, [])
               if (b["box"][0] <= lab.point[0] <= b["box"][2] and b["box"][1] <= lab.point[1] <= b["box"][3])
               or textboxes.iou(b["box"], lab.box) > prints.TURNED_JOIN_IOU]
        if hit:
            b = max(hit, key=lambda b: b["edge"])
            pairs.append((lab, ((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)))
    return _word_offsets([p[0] for p in pairs], pairs)


def foreign_density(training_labels) -> tuple[float, int, float]:
    """Other text per px^2 of crop: Terra's boxes of every kind (the label's own left out) whose centre falls in a
    training label's crop, over the crops' area.  Returns the density, the boxes counted and the area."""
    from sfzmaps.solver import paths
    terra = json.loads((paths.PRINTS_DIR / "terra_boxes.json").read_text())
    count, area = 0, 0.0
    for lab in training_labels:
        bx = np.array([b["box"] for b in terra.get(lab.sheet, [])], float).reshape(-1, 4)
        if lab.box is not None:
            bx = bx[~np.all(np.isclose(bx, np.array(lab.box, float)), axis=1)]
        uv = _to_crop(lab, np.c_[(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2])
        h, w = lab.crop.shape
        count += int(((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)).sum())
        area += h * w
    return count / max(area, 1.0), count, area


def design_effect(font, noise, texts, fits, labels, observed, weights, reach: int | None) -> float:
    """The variance of a fit's summed px log-likelihood ratio over the sum of its px's variances: the temperature a
    composite likelihood needs.  From the fits' residuals (ink - p(ink)) times each px's log-odds weight, over the
    footprint's observed px, pooled with `weights`.  `reach`: px pairs up to so many px apart each way; None: every
    pair in the crop, the summed residual's variance across the fits (its mean, a fit's systematic misfit, left
    out) over the mean sum of squares."""
    import cv2
    num = den = 0.0
    sums, squares, ws = [], [], []
    for text, fit, lab, obs, w in zip(texts, fits, labels, observed, weights):
        if w <= 0:
            continue
        d = glyphs.render(font, text, fit.pose, lab.crop.shape)
        a, b = glyphs._ab(noise, d)
        on = np.asarray(obs, bool) & _footprint(a, b)
        u = np.where(on, (lab.ink.astype(np.float32) - noise.p_ink(d)) * a, 0.0).astype(np.float32)
        if reach is None:
            sums.append(float(u.sum()))
            squares.append(float((u * u).sum()))
            ws.append(w)
            continue
        near = cv2.boxFilter(u, -1, (2 * reach + 1, 2 * reach + 1), normalize=False, borderType=cv2.BORDER_CONSTANT)
        num += w * float((u * near).sum())
        den += w * float((u * u).sum())
    if reach is None:
        s, q, w = np.array(sums), np.array(squares), np.array(ws)
        mean = (w * s).sum() / w.sum()
        return float((w * (s - mean) ** 2).sum() / max((w * q).sum(), 1e-12))
    return num / max(den, 1e-12)


def lettering_band(font: glyphs.Font) -> tuple[float, float]:
    """The lettering's height: the first and last template rows any character inks (p >= 0.5), from the line's
    middle, template px."""
    rows = np.flatnonzero(np.any([(t >= 0.5).any(axis=1) for t in font.templates.values()], axis=0))
    return float(rows[0] - glyphs.MIDLINE), float(rows[-1] - glyphs.MIDLINE)


def char_gains(font, noise, texts, fits, labels, observed, weights) -> dict[str, float]:
    """Each character's mean log-likelihood ratio per observed ink px of its cell (`_cell`), over the fits (each
    character drawn alone at its laid-out spot, as the posterior scores it), pooled with `weights`, with the mean
    over every character counted as glyphs.PRIOR_WEIGHT instances (a rare character stays near it)."""
    ratio: dict[str, float] = {}
    ink: dict[str, float] = {}
    n: dict[str, float] = {}
    band = lettering_band(font)
    for text, fit, lab, obs, w in zip(texts, fits, labels, observed, weights):
        if w <= 0:
            continue
        chars, xs = font.layout(text)
        o = np.asarray(obs, bool)
        x = lab.ink & o
        s, lin = fit.pose.scale, glyphs._linear(fit.pose.scale, fit.pose.angle, 0.0)
        for c, pen in zip(chars, xs):
            if c not in font.templates:
                continue
            d = glyphs._warp(glyphs.lay(font, [c], np.array([pen]), np.zeros(1)), fit.pose, lab.crop.shape)
            a, b = glyphs._ab(noise, d)
            centre = np.array([fit.pose.x, fit.pose.y]) + lin @ np.array([pen + font.advance[c] / 2, 0.0])
            corners = centre + _cell(font, c, s, fit.pose.angle, (band[0] - 2 / s, band[1] + 2 / s))
            cell = np.zeros(lab.crop.shape, np.uint8)
            cv2.fillPoly(cell, [np.rint(corners * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
            ratio[c] = ratio.get(c, 0.0) + w * float(np.where(x, a, 0).sum() + b[o].sum())
            ink[c] = ink.get(c, 0.0) + w * float((x & (cell > 0)).sum())
            n[c] = n.get(c, 0.0) + w
    total_r, total_i, total_n = sum(ratio.values()), sum(ink.values()), sum(n.values())
    mean, per = total_r / max(total_i, 1e-9), total_i / max(total_n, 1e-9)
    k = glyphs.PRIOR_WEIGHT
    return {c: (ratio[c] + k * mean * per) / (ink[c] + k * per) for c in sorted(ratio)}


def _window_px(window, scale: float) -> tuple[int, int]:
    """A window's skip and width (cap heights) in crop px at a scale (the width at least 1 px)."""
    cap = scale * glyphs.SIZE
    return round(window[0] * cap), max(1, round(window[1] * cap))


def end_samples(font, texts, fits, labels, observed, weights, windows=END_WINDOWS) -> dict[tuple, np.ndarray]:
    """The ink density in a window of the lettering band past each fitted string's last ink px and before its first
    (the ends of a word), and likewise at each boundary between its characters (inside a word; a boundary at a space
    is left out), each character's ink extent and pen rounded as the posterior lays a string out at the fit's scale;
    observed ink only.  For each window (skip, width; cap heights): rows (density, 1 at a word's end, weight, the
    label's number), the labels pooled with `weights`."""
    band = lettering_band(font)
    lean = math.tan(math.radians(font.slant))
    out: dict[tuple, list] = {w: [] for w in windows}
    for n, (text, fit, lab, obs, wt) in enumerate(zip(texts, fits, labels, observed, weights)):
        if wt <= 0:
            continue
        s = fit.pose.scale
        chars, xs = font.layout(text)
        inked = [i for i, c in enumerate(chars) if c in font.templates]
        if not inked:
            continue
        adv = [max(1, round(font.advance[c] * s)) for c in chars]
        pens = round(xs[0] * s) + np.concatenate([[0], np.cumsum(adv)[:-1]]).astype(int)
        ext = {c: _ink_extent(font, c) for c in set(chars) if c in font.templates}
        first = {i: int(pens[i]) + round(ext[chars[i]][0] * s) for i in inked}
        last = {i: int(pens[i]) + round(ext[chars[i]][1] * s) for i in inked}
        px = {w: _window_px(w, s) for w in windows}
        pad = max(a + b for a, b in px.values())
        bp = band_px(band, s)
        t0 = min(first.values()) - pad - 1
        col = column_ink((lab.ink & np.asarray(obs, bool)).astype(np.float32), (fit.pose.x, fit.pose.y),
                         [glyphs._axes(fit.pose.angle)], [0], np.arange(t0, max(last.values()) + pad + 2), bp,
                         lean)[0, 0]
        for w, (skip, width) in px.items():
            after, before = adjacent_ink(col, width, bp[1] - bp[0] + 1, skip)
            for j, i in enumerate(inked):
                if not (i + 1 < len(chars) and chars[i + 1] == SPACE):
                    out[w].append((after[last[i] - t0], j == len(inked) - 1, wt, n))
                if not (i > 0 and chars[i - 1] == SPACE):
                    out[w].append((before[first[i] - t0], j == 0, wt, n))
    return {w: np.array(v, float).reshape(-1, 4) for w, v in out.items()}


def _isotonic_down(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted least-squares fit of y on x that never rises with x (pool adjacent violators), as blocks: each
    one's last x, its sum of w y and its sum of w, [block, 3]."""
    xs, inv = np.unique(x, return_inverse=True)
    sy, sw = np.bincount(inv, weights=w * y), np.bincount(inv, weights=w)
    blocks: list[list[float]] = []
    for xi, a, b in zip(xs, sy, sw):
        if b <= 0:
            continue
        blocks.append([float(xi), float(a), float(b)])
        while len(blocks) > 1 and blocks[-2][1] * blocks[-1][2] < blocks[-1][1] * blocks[-2][2]:
            xl, a2, b2 = blocks.pop()
            blocks[-1] = [xl, blocks[-1][1] + a2, blocks[-1][2] + b2]
    return np.array(blocks, float).reshape(-1, 3)


def fit_end_break(density: np.ndarray, ends: np.ndarray, weights: np.ndarray) -> dict:
    """P(a word ends | the density past it): isotonic (non-increasing in the density); and log P(end | density) /
    P(end), P(end) the samples' share of ends (`end_factor`'s table)."""
    base = float((weights * ends).sum() / weights.sum())
    blocks = _isotonic_down(density, ends, weights)
    # each block one sample at the base rate added (a block of a few light labels stays near it), then pooled again
    # where that broke the order
    p = (blocks[:, 1] + base) / (blocks[:, 2] + 1.0)
    blocks = _isotonic_down(blocks[:, 0], p, blocks[:, 2] + 1.0)
    p = blocks[:, 1] / blocks[:, 2]
    return {"edges": blocks[:, 0].tolist(), "p_end": p.tolist(), "log_factor": (np.log(p) - math.log(base)).tolist(),
            "base": base}


def _end_log_loss(model: dict, density, ends, weights) -> float:
    i = np.clip(np.searchsorted(np.asarray(model["edges"]), density, side="left"), 0, len(model["edges"]) - 1)
    p = np.asarray(model["p_end"])[i]
    return -float((weights * (ends * np.log(p) + (1 - ends) * np.log1p(-p))).sum())


def end_breaks(font, texts, fits, labels, observed, weights, windows=END_WINDOWS) -> dict:
    """Where a word ends, without a box (see the module): `end_samples` at each window, the window whose
    `fit_end_break` predicts the other half of the labels best (two folds by the labels' order, mean log loss each
    way), fitted on them all.  Returns `fit_end_break`'s table with the window (skip and width, cap heights) and
    every window's loss."""
    samples = end_samples(font, texts, fits, labels, observed, weights, windows)
    losses = {}
    for w, a in samples.items():
        if not len(a):
            continue
        m, y, wt, n = a.T
        half = (n % 2).astype(bool)
        losses[w] = sum(_end_log_loss(fit_end_break(m[f], y[f], wt[f]), m[~f], y[~f], wt[~f])
                        for f in (half, ~half)) / wt.sum()
    if not losses:
        return {}
    best = min(losses, key=losses.get)
    m, y, wt, _ = samples[best].T
    out = fit_end_break(m, y, wt)
    out.update(window=[float(best[0]), float(best[1])], log_loss={f"{k[0]}+{k[1]}": v for k, v in losses.items()},
               samples=len(m), ends=float((wt * y).sum()))
    return out


def measure_pose_prior(training_labels, font: glyphs.Font, noise: glyphs.NoiseModel, *, rules=RULES,
                       workers: int | None = None, reaches=(1, 2, 4, 8, 16, 32)) -> PosePrior:
    """The pose prior measured on the training labels (see the module):
      - each label's string (`glyphs.consensus`) aligned by `glyphs.align` under the broad first-round prior
        (`glyphs.search_for`), line px (`rules`) unobserved;
      - with a box: where its drawn ink's ends and line sit against Terra's word, in word heights, fitted by
        `mixture_fit` (core, tail, and a uniform over the crop for strings matched on other ink, which the scoring
        prior leaves out); the cap height over the word's height, the half turn's rate per Terra direction and the
        residual angles' rates from the core and tail's fits (Jeffreys' half count added);
      - without a box: Sol's points against Terra's word on the prints both read (`sol_offsets`) and the turned
        pass's boxes likewise (`turned_offsets`), each fitted by `mixture_fit`; the quarter turns' rates from the
        point labels' fits, weighted by how near their point they were found;
      - the map's cap heights; the density of other text in the crops (`foreign_density`); each character's ratio
        per ink px of its cell (`char_gains`); where a word ends without a box (`end_breaks`, on the fits the gains
        use: words read a quarter turn); the temperature as the design effect of the fits' residuals within one
        character (px pairs up to half the map's median cap height apart; `design_effect`, kept in `measured` with
        the effect at `reaches` px and over the whole crop to show how it grows)."""
    _guard(training_labels)
    t0 = time.perf_counter()
    labs = list(training_labels)
    heights = glyphs.word_heights(labs)
    observed = [~labels_mod.line_mask(lab, rules=rules) for lab in labs]
    texts = [glyphs.consensus(lab) for lab in labs]
    searches = [glyphs.search_for(lab, heights) for lab in labs]
    fits = glyphs._map(lambda i: glyphs.align(font, noise, texts[i], labs[i].ink, observed[i], searches[i]),
                       len(labs), workers)
    measured: dict = {"labels": len(labs), "align_seconds": round(time.perf_counter() - t0, 1)}
    pose = PosePrior(height=float(np.median(heights)), point_heights=tuple(float(h) for h in heights),
                     measured=measured)
    # with a box: the placement and the size measured on words read a quarter turn (a slanted box's word rectangle
    # is no measure of the letters' height); the turns and residual angles on every box
    box = [i for i, lab in enumerate(labs) if lab.code_rect is not None]
    z, area, extra, used, square = [], [], [], [], []
    for i in box:
        lab, fit = labs[i], fits[i]
        u0, v0, u1, v1 = lab.code_rect
        h = v1 - v0
        e = _ink_edges(font, texts[i], fit.pose, lab.crop.shape)
        if e is None:
            continue
        used.append(i)
        square.append((lab.angle / 90) == round(lab.angle / 90))
        z.append(((e[0] - u0) / h, (e[2] - u1) / h, (fit.pose.y - (v0 + v1 - 1) / 2) / h))
        hh, ww = lab.crop.shape
        area.append((ww / h) ** 2 * (hh / h))
        turn = abs(((fit.pose.angle + 90) % 360) - 90) >= 45
        base = 180.0 if turn else 0.0
        res = ((fit.pose.angle - base + 180) % 360) - 180
        extra.append((lab.direction, turn, res, math.log(fit.pose.scale * glyphs.SIZE / h),
                      math.log(fit.pose.scale * glyphs.SIZE)))
    square = np.array(square)
    step = 1 / pose.height
    mix = mixture_fit(np.array(z)[square], np.array(area)[square], [step / math.sqrt(12)] * 3)
    good = np.ones(len(z))                    # a slanted box's fit counts whole for the turns and residuals
    good[square] = mix["resp"][:, 0] + mix["resp"][:, 1]
    mean, core, tail = mix["mean"], mix["core"], mix["tail"]
    pose.left, pose.right, pose.middle = ((mean[i], core[i], tail[i]) for i in range(3))
    pose.box_tail = float(mix["weights"][1] / (mix["weights"][0] + mix["weights"][1]))
    gs = good[square]
    ratios = np.array([e[3] for e in extra])[square]
    m = float((gs * ratios).sum() / gs.sum())
    pose.ratio = (m, float(math.sqrt((gs * (ratios - m) ** 2).sum() / gs.sum())))
    caps = np.array([e[4] for e in extra])[square]
    mc = float((gs * caps).sum() / gs.sum())
    pose.cap = (mc, float(math.sqrt((gs * (caps - mc) ** 2).sum() / gs.sum())))
    turns: dict[str, list[float]] = {}
    res_count = np.full(len(RESIDUALS), 0.5)
    for (direction, turn, res, _, _), g in zip(extra, good):
        t = turns.setdefault(direction, [0.5, 0.5])
        t[int(turn)] += g
        res_count[int(np.argmin([abs(res - r) for r in RESIDUALS]))] += g
    pose.half_turn = {d: float(t[1] / sum(t)) for d, t in sorted(turns.items())}
    pose.residuals = (res_count / res_count.sum()).tolist()
    measured["box"] = {"fits": int(square.sum()), "slanted": int((~square).sum()), "weights": mix["weights"],
                       "turns": {d: t for d, t in turns.items()}}
    # without a box
    point_mix = {}
    for source, offs in (("sol", sol_offsets(labs)), ("turned", turned_offsets(labs))):
        if len(offs) < 10:
            continue
        pm = mixture_fit(offs, np.full(len(offs), (2 * labels_mod.REACH + 4) ** 2), [step / math.sqrt(12)] * 2)
        point_mix[source] = pm
        pose.along[source] = (pm["mean"][0], pm["core"][0], pm["tail"][0])
        pose.across[source] = (pm["mean"][1], pm["core"][1], pm["tail"][1])
        pose.point_tail[source] = float(pm["weights"][1] / (pm["weights"][0] + pm["weights"][1]))
        measured[source] = {"pairs": len(offs), "weights": pm["weights"]}
    for source in SOURCES:
        count = np.full(4, 0.5)
        for i, lab in enumerate(labs):
            if lab.code_rect is not None or source_of(lab) != source:
                continue
            e = _ink_edges(font, texts[i], fits[i].pose, lab.crop.shape)
            if e is None:
                continue
            al = pose.along.get(source, (0, 0.15, 0.6))
            ac = pose.across.get(source, (0, 0.1, 0.4))
            q = round(fits[i].pose.angle / 90) % 4
            eu, ev = glyphs._axes(90.0 * q)
            off = np.array([(e[0] + e[2]) / 2 - lab.centre[0], (e[1] + e[3]) / 2 - lab.centre[1]]) / pose.height
            near = math.exp(np.logaddexp.reduce(_log_mix(off @ np.array(eu), al[0], al[1], al[2],
                                                         pose.point_tail.get(source, 0.2), 1.0)) +
                            np.logaddexp.reduce(_log_mix(off @ np.array(ev), ac[0], ac[1], ac[2],
                                                         pose.point_tail.get(source, 0.2), 1.0)))
            count[q] += near / (near + 1.0 / ((2 * labels_mod.REACH + 4) ** 2))
        pose.point_turns[source] = (count / count.sum()).tolist()
    # other text, and the temperature
    pose.foreign, n_foreign, crop_area = foreign_density(labs)
    measured["foreign"] = {"boxes": n_foreign, "area": crop_area}
    w_all = np.zeros(len(labs))
    w_all[used] = np.where(square, good, 0.0)        # the temperature and the gains from words read a quarter turn
    # competing codes differ within a character or two, and a word's own misfit (its print bolder or thinner than
    # the templates) is shared by them all: the dependence that matters reaches over one character (half the cap
    # height each way), not the whole crop
    reach = max(1, round(math.exp(pose.cap[0]) / 2))
    effects = {str(r): design_effect(font, noise, texts, fits, labs, observed, w_all, r)
               for r in sorted({*reaches, reach})}
    effects["crop"] = design_effect(font, noise, texts, fits, labs, observed, w_all, None)
    measured["design_effect"] = effects
    measured["temperature_reach"] = reach
    pose.temperature = float(effects[str(reach)])
    pose.char_gain = char_gains(font, noise, texts, fits, labs, observed, w_all)
    pose.end_break = end_breaks(font, texts, fits, labs, observed, w_all)
    measured["seconds"] = round(time.perf_counter() - t0, 1)
    return pose


def posterior_of(ev: Evidence, prior: Prior, constants: Constants) -> dict[str, float]:
    """The posterior over the codes, "none" and "other" from a label's evidence at the constants' temperature (one
    of `ev.temperatures`): a code's evidence is its forms' weighted by `Prior.form_weights` (alike when empty)."""
    temp = prior.pose.temperature if constants.temperature is None else constants.temperature
    i = int(np.argmin([abs(t - temp) for t in ev.temperatures]))
    if not math.isclose(ev.temperatures[i], temp, rel_tol=1e-9):
        raise ValueError(f"no evidence at temperature {temp} (have {ev.temperatures})")
    by_code: dict[str, list[tuple[float, float]]] = {}
    for f, v in ev.forms.items():
        by_code.setdefault(prior.forms[f], []).append((float(v[i]), prior.form_weights.get(f, 1.0)))
    code_mass = 1 - constants.none_mass - constants.other_mass
    logs = {}
    for code, p in prior.codes.items():
        vals = by_code.get(code)
        if not vals or p <= 0:
            logs[code] = float("-inf")
            continue
        # p(form | code) over the forms scored (a form the font can't lay out is left out)
        w = np.array([x for _, x in vals], float)
        logs[code] = math.log(code_mass * p) + float(np.logaddexp.reduce(
            [v + math.log(x / w.sum()) for (v, _), x in zip(vals, w)]))
    logs["none"] = math.log(constants.none_mass)
    logs["other"] = math.log(constants.other_mass) + float(ev.other[i])
    m = max(logs.values())
    z = m + math.log(sum(math.exp(v - m) for v in logs.values()))
    return {h: math.exp(v - z) for h, v in logs.items()}


def score_label(label: labels_mod.Label, font: glyphs.Font, noise: glyphs.NoiseModel, prior: Prior,
                constants: Constants = Constants(), **kw) -> dict[str, float]:
    """The posterior over the codes plus "none" and "other" for one label (see the module); keyword arguments go to
    `evidence`."""
    return posterior_of(evidence(label, font, noise, prior, (constants.temperature,), **kw), prior, constants)
