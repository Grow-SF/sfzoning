"""The synthetic code crops the text reader adds to its real ones (spike S2's reader C), and the template reader's
inputs they are drawn with.

The template reader's inputs (`learn_template`), learned as the template reader learns them, from its training
labels (`codereader.labels.training`: every print that no owner's answer joined, in a block no answer reaches; one
font for the whole map, not one per quarter: the lettering is the same everywhere, so it carries glyph shape and no
label):
  font   `codereader.glyphs.learn_font` on the training labels' crops and their strings (`glyphs.consensus` of the
         saved reads), line px by glyphs.RULES unobserved, the characters of the code list besides the strings'
  prior  `codereader.posterior.Prior.learn` on the same labels and font: the code list's printed forms (the table's
         code NAMES, never a code at a label's spot; the aliases; the forms the training reads wrote), p(code) and
         p(form | code) from the training strings, and the pose prior (Terra's box about the drawn ink, the cap
         height over the word's height)
The code list the reader is scored on is this prior's `forms` (printed form -> code).

Each synthetic crop (`synth_crop`) is a printed code form drawn by the font and its noise, cut like a real crop
(`crops.rectify`'s pads, HEIGHT px tall).  Everything drawn is measured:
  the string     a code by p(code) half the time and uniformly the other half (rare codes seen), then one of its
                 printed forms by p(form | code) half the time and uniformly the other half
  the size       a training code word's height (the real crops, every scan, Terra's word) times the cap height over
                 word height, log-normal with the pose prior's measured mean and sd (`ratio`)
  the spacing    each character's pen moved by a Gaussian with the font's measured gap spread / sqrt 2
  the angle      a residual turn of N(0, RESIDUAL_SD) degrees
  the ink        each px ink with the noise's p(ink) at its signed distance from the drawn strokes, rescaled to run
                 from the pages' measured speck rate (`speck_rate`) far off the strokes to the noise's own value on
                 them (the noise's far value is the crops' clutter, not paper flips)
  Terra's box    its ends and middle off the drawn ink by the pose prior's measured core-and-tail Gaussians
  the lines      with the measured share of training code words whose Terra word holds line ink by the "run" rule
                 (`line_share`), a real non-text window of the 2000 scan laid over the crop (the darker px of the
                 two), cut at the synthetic word's own page size (`Overlays`)
Quarter q's reader gets its own set (`make_synth`): the heights, the line share and the line windows come from the
real crops it may train on (touching neither q nor the answered blocks).  The sampling is C's as the spike ran it,
known faults kept so that it is the recipe C was measured with: Terra's box offsets drawn with the opposite sign
(means 0.02-0.03 word heights), and line windows centred on ink (a line often runs through the word's middle).
Seeded per quarter; one thread.
"""
import math
import time

import cv2
import numpy as np
import pandas as pd

from sfzmaps.solver import paths
from sfzmaps.solver.codereader import glyphs, labels as L, posterior as P
from sfzmaps.solver.textnet import data
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import store

N = 50000                  # synthetic crops per quarter (reader C's count)
SEED = 20261117
RESIDUAL_SD = 1.5          # degrees
LINE_MIN_PX = 3            # a word holds line ink when at least this many of its px are on a "run" line
LINE_WORDS = 1500          # code words the line share is measured on, at most
SPECK_PX = 2               # an ink component this small (8-connected px) on a page is a speck


# --- the template reader's inputs -----------------------------------------------------------------------------------

def learn_template(labels: list, workers: int | None = None) -> tuple[glyphs.Font, P.Prior, dict]:
    """The font and the prior (see the module) from `labels` (`codereader.labels.load_labels()`, with crops), and
    what they were learned from."""
    t0 = time.perf_counter()
    train = L.training(labels)
    charset = "".join(sorted(set("".join(L.code_list(labels)))))
    observed = [~L.line_mask(lab, rules=glyphs.RULES) for lab in train]
    texts = [glyphs.consensus(lab) for lab in train]
    font = glyphs.learn_font(train, texts=texts, observed=observed, charset=charset, workers=workers)
    t1 = time.perf_counter()
    prior = P.Prior.learn(train, font, font.noise, workers=workers)
    info = {"training_labels": len(train), "labels": len(labels), "charset": charset,
            "strings_not_terra": int(sum(t != lab.terra_text for t, lab in zip(texts, train))),
            "font_trace": [{k: v for k, v in r.items() if k not in ("weights", "seconds")} for r in font.trace],
            "forms": len(prior.forms), "codes": len(prior.codes),
            "seconds": {"font": round(t1 - t0, 1), "prior": round(time.perf_counter() - t1, 1)}}
    return font, prior, info


# --- what the sampling measures -------------------------------------------------------------------------------------

def line_share(rows: pd.DataFrame, seed: int = 0) -> dict:
    """The share of `rows` (code words of the 2000 scan) whose Terra word rectangle holds line ink by the "run"
    rule, measured on the page around each (the window grown by the rule's run length), on at most LINE_WORDS."""
    m = rows.iloc[np.random.default_rng(seed).permutation(len(rows))[:LINE_WORDS]]
    hits = total = 0
    for sheet, part in m.groupby("sheet", sort=True):
        page = data.load_page(sheet)
        run = L.run_px(data.load_transform(sheet).median_scale_ft_per_px(page.shape))
        for b, angle in zip(part[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float), part["angle"]):
            rect = data.word_rect(b, float(angle))
            x0, y0 = np.floor(rect.min(axis=0)).astype(int) - run
            x1, y1 = np.ceil(rect.max(axis=0)).astype(int) + run + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, page.shape[1]), min(y1, page.shape[0])
            ink = page[y0:y1, x0:x1] < 128
            lines = L.run_lines(ink, run)
            inside = np.zeros(ink.shape, np.uint8)
            cv2.fillPoly(inside, [np.round(rect - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], 1)
            hits += int((lines & (inside > 0)).sum() >= LINE_MIN_PX)
            total += 1
    return {"words": total, "with_line_ink": hits, "share": hits / max(total, 1)}


def speck_rate(sheets) -> dict:
    """The share of paper px that are specks (ink px in 8-connected components of SPECK_PX px or fewer) over every px
    not in a larger component, on the given 2000 pages."""
    specks = paper = 0
    for sheet in sheets:
        ink = (data.load_page(sheet) < 128).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        size = stats[:, cv2.CC_STAT_AREA]
        small = size[1:] <= SPECK_PX
        specks += int(size[1:][small].sum())
        paper += int(ink.size - size[1:][~small].sum())
    return {"sheets": len(sheets), "speck_px": specks, "paper_px": paper, "rate": specks / max(paper, 1)}


class Overlays:
    """Line windows from non-text crops of the 2000 scan (`rows`), cut at a synthetic word's own page size."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.reset_index(drop=True)
        size = [L.word_size(b, float(a)) for b, a in
                zip(self.rows[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float), self.rows["angle"])]
        self.win_w = np.array([ln + 2 * K.PAD_X * h for ln, h in size]).reshape(-1)
        self.win_h = np.array([h * (1 + 2 * K.PAD_Y) for _, h in size]).reshape(-1)
        self.pages: dict[str, np.ndarray] = {}

    def cut(self, w: int, h: int, out_shape: tuple[int, int], rng) -> np.ndarray | None:
        """A page window w x h px from inside a checked-empty window at least that large, at that window's angle,
        resized like the crop to `out_shape`; None when no window is large enough."""
        ok = np.flatnonzero((self.win_w >= w) & (self.win_h >= h))
        if not len(ok):
            return None
        r = self.rows.iloc[int(ok[rng.integers(len(ok))])]
        if r["sheet"] not in self.pages:
            self.pages[r["sheet"]] = data.load_page(r["sheet"])
        b = (r["box_x0"], r["box_y0"], r["box_x1"], r["box_y1"])
        a = L.frame(((b[0] + b[2] - 1) / 2, (b[1] + b[3] - 1) / 2), w, h, float(r["angle"]))
        raw = L.sample(self.pages[r["sheet"]], a, (h, w), 255)
        return cv2.resize(raw, (out_shape[1], out_shape[0]),
                          interpolation=cv2.INTER_AREA if h > out_shape[0] else cv2.INTER_LINEAR)


# --- drawing --------------------------------------------------------------------------------------------------------

def draw_form(prior: P.Prior, rng) -> str:
    codes = sorted(prior.codes)
    p = np.array([prior.codes[c] for c in codes])
    code = codes[rng.choice(len(codes), p=p / p.sum())] if rng.random() < 0.5 else codes[rng.integers(len(codes))]
    forms = sorted(f for f, c in prior.forms.items() if c == code)
    w = np.array([prior.form_weights.get(f, 1.0) for f in forms])
    return forms[rng.choice(len(forms), p=w / w.sum())] if rng.random() < 0.5 else forms[rng.integers(len(forms))]


def _mix(rng, mean: float, core: float, tail: float, w_tail: float) -> float:
    return mean + rng.normal() * (tail if rng.random() < w_tail else core)


def synth_crop(font: glyphs.Font, p_ink, pose: P.PosePrior, text: str, word_h: float, rng,
               overlays: "Overlays | None") -> tuple[np.ndarray, bool]:
    """One synthetic crop of `text` (see the module), and whether a line window was laid over it."""
    cap = word_h * math.exp(rng.normal(pose.ratio[0], pose.ratio[1]))
    scale = cap / font.size
    chars, xs = font.layout(text)
    xs = xs + rng.normal(0.0, font.spacing_sd / math.sqrt(2), len(xs))
    laid = glyphs.lay(font, chars, xs, np.zeros(len(chars)))
    angle = rng.normal(0.0, RESIDUAL_SD)
    extent = (len(text) + 4) * font.size * scale + 4 * word_h
    w = int(math.ceil(extent)) + 8
    h = int(math.ceil(word_h * 4)) + 8
    d = glyphs._warp(laid, glyphs.Pose((w - 1) / 2, (h - 1) / 2, scale, angle), (h, w))
    ink = rng.random(d.shape) < p_ink(d)
    cols = np.flatnonzero((d > 0).any(axis=0))
    rows = np.flatnonzero((d > 0).any(axis=1))
    if not len(cols):
        cols, rows = np.array([w // 2]), np.array([h // 2])
    left = cols[0] + word_h * _mix(rng, *pose.left, pose.box_tail)
    right = cols[-1] + 1 + word_h * _mix(rng, *pose.right, pose.box_tail)
    if right - left < 0.5 * word_h:
        right = left + 0.5 * word_h
    mid = (rows[0] + rows[-1] + 1) / 2 + word_h * _mix(rng, *pose.middle, pose.box_tail)
    page = np.where(ink, 0, 255).astype(np.uint8)
    box = (left, mid - word_h / 2, right, mid + word_h / 2)
    crop, _, shape, _ = K.rectify(page, box, 0.0)
    if overlays is not None:
        ov = overlays.cut(shape[1], shape[0], crop.shape, rng)
        if ov is not None:
            return np.minimum(crop, ov), True
    return crop, False


def make_synth(q: int, font: glyphs.Font, prior: P.Prior, rows: pd.DataFrame, specks: dict, n: int = N) -> dict:
    """Quarter q's reader's synthetic crops (see the module): {"crops": list, "rows": DataFrame (text, code,
    overlay, overlay_wanted), "summary": dict}.  `rows`: the real crops' rows (`crops.make_crops`); `specks`:
    `speck_rate` of the 2000 pages."""
    t0 = time.perf_counter()
    cv2.setNumThreads(1)
    mine = rows.iloc[store.training_rows(rows, q)]
    codes = mine[mine["kind"] == "zoning code"]
    share = line_share(codes[codes["series"] == paths.SERIES])
    overlays = Overlays(mine[(mine["kind"] == "none") & (mine["series"] == paths.SERIES)])
    noise, sp = font.noise, specks["rate"]
    bg = noise.background

    def p_ink(dist):
        return sp + np.clip(noise.p_ink(dist) - bg, 0, None) / (1 - bg) * (1 - sp)

    code_h = codes["height"].to_numpy(float)
    rng = np.random.default_rng([SEED, q])
    out, meta = [], []
    for _ in range(n):
        text = draw_form(prior, rng)
        want = rng.random() < share["share"]
        crop, lined = synth_crop(font, p_ink, prior.pose, text, float(code_h[rng.integers(len(code_h))]), rng,
                                 overlays if want else None)
        out.append(crop)
        meta.append({"text": text, "code": prior.forms[text], "overlay": lined, "overlay_wanted": bool(want)})
    df = pd.DataFrame(meta)
    summary = {"quarter": paths.QUARTERS[q], "crops": n, "seed": [SEED, q], "line_share": share,
               "code_heights": int(len(code_h)), "line_windows": int(len(overlays.rows)),
               "residual_sd": RESIDUAL_SD, "overlaid": int(df["overlay"].sum()),
               "codes": int(df["code"].nunique()), "seconds": round(time.perf_counter() - t0, 1)}
    return {"crops": out, "rows": df, "summary": summary}
