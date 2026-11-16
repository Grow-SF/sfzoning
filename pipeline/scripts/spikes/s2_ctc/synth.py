"""Spike S2: the synthetic crops reader C adds (survey section 4.6, step 2 C; section 4.2 "a synthetic stream").

    uv run python scripts/spikes/s2_ctc/synth.py [N]     # writes data/solver/experiments/spike-s2/synth.npz + meta

Each crop is a printed code form drawn by the generative reader's own learned font and noise (`codereader.glyphs`:
`inputs/font_real.npz`, the font its records were scored with; templates learned from the training labels' crops) and
cut like a real crop (crops.py `rectify`'s pads, HEIGHT px tall).  Everything drawn is measured:
  the string     a code by p(code) half the time and uniformly the other half (rare codes seen), then one of its
                 printed forms by p(form | code) half the time and uniformly the other half (prior3.json, learned from
                 the training labels' reads)
  the size       a training code word's height (crops.py's training split, every scan; Terra's word) times the cap
                 height over word height, log-normal with the pose prior's measured mean and sd (prior3 `ratio`)
  the spacing    each character's pen moved by a Gaussian with the font's measured gap spread / sqrt 2
                 (`Font.spacing_sd`, as the posterior's `_jitter`)
  the angle      a residual turn of N(0, RESIDUAL_SD) degrees: Terra's angle is a few degrees off on slanted words
  the ink        each px ink with the noise's p(ink) at its signed distance from the drawn strokes (Kanungo-style,
                 `NoiseModel.p_ink`), so ragged edges come from the measured noise.  The noise's far value (7%) is
                 the crops' clutter (lines the masks miss, other text), not paper flips: drawn as independent px it
                 is salt the scan doesn't have.  So p(ink) is rescaled to run from the pages' measured speck rate
                 (`speck_rate`: ink px in components of SPECK_PX px or fewer, of all paper) far off the strokes to the
                 noise's own value on them
  Terra's box    its ends and middle off the drawn ink by the pose prior's measured core-and-tail Gaussians
                 (prior3 `left`, `right`, `middle`, `box_tail`; word heights)
  the lines      with the measured share of training code words whose Terra word holds line ink by the "run" rule
                 (`codereader.labels.run_lines`, the generative reader's own rule; `line_share`), the page under a
                 real non-text training window of the 2000 scan (crops.py: ink outside every word and every unknown
                 px, mostly lot and district lines) laid over the crop (the darker px of the two): the same page px
                 window as the synthetic word's, cut at the same scale from inside a window that was checked empty
                 of text (so line widths are the page's own)
Seeded; one thread.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import crops as K  # noqa: E402
from sfzmaps.solver.codereader import glyphs, labels as L, posterior as P  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402

cv2.setNumThreads(1)
N = 50000
SEED = 20261117
RESIDUAL_SD = 1.5          # degrees
LINE_MIN_PX = 3            # a word holds line ink when at least this many of its px are on a "run" line
SPECK_PX = 2               # an ink component this small (8-connected px) on a page is a speck


def line_share(meta: pd.DataFrame, limit: int = 1500, seed: int = 0) -> dict:
    """The share of training code words (2000 scan) whose Terra word rectangle holds line ink by the "run" rule,
    measured on the page around each (the window grown by the rule's run length)."""
    m = meta[(meta["split"] == "train") & (meta["kind"] == "zoning code") & (meta["series"] == K.paths.SERIES)]
    m = m.iloc[np.random.default_rng(seed).permutation(len(m))[:limit]]
    hits = total = 0
    for sheet, part in m.groupby("sheet"):
        page = data.load_page(sheet)
        ft = data.load_transform(sheet).median_scale_ft_per_px(page.shape)
        run = L.run_px(ft)
        for b, angle in zip(part["box"], part["angle"]):
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
    """The share of paper px that are specks: ink px in 8-connected components of SPECK_PX px or fewer, over every
    px that is not in a larger component, on the given 2000 pages."""
    specks = paper = 0
    for sheet in sheets:
        ink = (data.load_page(sheet) < 128).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        size = stats[:, cv2.CC_STAT_AREA]
        small = size[1:] <= SPECK_PX
        specks += int(size[1:][small].sum())
        paper += int(ink.size - size[1:][~small].sum())
    return {"sheets": len(sheets), "speck_px": specks, "paper_px": paper, "rate": specks / max(paper, 1)}


class Overlays:
    """Line windows from the 2000 scan's training non-text crops, cut at a synthetic word's own page size."""

    def __init__(self, meta: pd.DataFrame):
        m = meta[(meta["split"] == "train") & (meta["kind"] == "none") & (meta["series"] == K.paths.SERIES)]
        self.rows = m.reset_index(drop=True)
        size = [L.word_size(b, float(a)) for b, a in zip(self.rows["box"], self.rows["angle"])]
        self.win_w = np.array([ln + 2 * K.PAD_X * h for ln, h in size])
        self.win_h = np.array([h * (1 + 2 * K.PAD_Y) for _, h in size])
        self.pages: dict[str, np.ndarray] = {}

    def cut(self, w: int, h: int, out_shape: tuple[int, int], rng) -> np.ndarray | None:
        """A page window w x h px (a synthetic word's, pads included) from inside a checked-empty window at least
        that large, at that window's angle, resized like the crop to `out_shape`; None when no window is large
        enough."""
        ok = np.flatnonzero((self.win_w >= w) & (self.win_h >= h))
        if not len(ok):
            return None
        r = self.rows.iloc[int(ok[rng.integers(len(ok))])]
        if r["sheet"] not in self.pages:
            self.pages[r["sheet"]] = data.load_page(r["sheet"])
        b = r["box"]
        a = L.frame(((b[0] + b[2] - 1) / 2, (b[1] + b[3] - 1) / 2), w, h, float(r["angle"]))
        raw = L.sample(self.pages[r["sheet"]], a, (h, w), 255)
        return cv2.resize(raw, (out_shape[1], out_shape[0]),
                          interpolation=cv2.INTER_AREA if h > out_shape[0] else cv2.INTER_LINEAR)


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
    # the drawn ink's extent at this scale on a canvas big enough for it and the crop's pads
    extent = (len(text) + 4) * font.size * scale + 4 * word_h
    w = int(math.ceil(extent)) + 8
    h = int(math.ceil(word_h * 4)) + 8
    d = glyphs._warp(laid, glyphs.Pose((w - 1) / 2, (h - 1) / 2, scale, angle), (h, w))
    ink = rng.random(d.shape) < p_ink(d)
    cols = np.flatnonzero((d > 0).any(axis=0))
    rows = np.flatnonzero((d > 0).any(axis=1))
    if not len(cols):
        cols, rows = np.array([w // 2]), np.array([h // 2])
    # Terra's box about the drawn ink: ends and middle off by the measured spreads (word heights)
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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N
    t0 = time.time()
    font = glyphs.Font.load(C.INPUTS / "font_real.npz")
    noise = font.noise
    prior = P.Prior.load(C.INPUTS / "prior3.json")
    crops = C.load_crops()
    m = crops.meta
    share = line_share(m)
    specks = speck_rate(K.paths.SHEETS)
    print(f"line share {share}; specks {specks} ({time.time() - t0:.0f} s)", flush=True)
    bg, sp = noise.background, specks["rate"]

    def p_ink(d):
        return sp + np.clip(noise.p_ink(d) - bg, 0, None) / (1 - bg) * (1 - sp)

    code_h = m.loc[(m["split"] == "train") & (m["kind"] == "zoning code"), "height"].to_numpy(float)
    overlays = Overlays(m)
    rng = np.random.default_rng(SEED)
    out, meta = [], []
    for i in range(n):
        text = draw_form(prior, rng)
        want = rng.random() < share["share"]
        crop, lined = synth_crop(font, p_ink, prior.pose, text, float(code_h[rng.integers(len(code_h))]), rng,
                                 overlays if want else None)
        out.append(crop)
        meta.append({"text": text, "code": prior.forms[text], "overlay": lined, "overlay_wanted": bool(want)})
        if (i + 1) % 10000 == 0:
            print(f"{i + 1} crops ({time.time() - t0:.0f} s)", flush=True)
    widths = np.array([c.shape[1] for c in out], np.int32)
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    np.savez(C.OUT / "synth.npz", flat=np.concatenate(out, axis=1), offsets=offsets, widths=widths)
    pd.DataFrame(meta).to_pickle(C.OUT / "synth_meta.pkl")
    summary = {"crops": n, "line_share": share, "specks": specks, "seed": SEED, "residual_sd": RESIDUAL_SD,
               "overlaid": int(sum(r["overlay"] for r in meta)),
               "overlay_wanted": int(sum(r["overlay_wanted"] for r in meta)),
               "codes": int(pd.DataFrame(meta)["code"].nunique()), "seconds": round(time.time() - t0, 1)}
    (C.OUT / "synth_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(summary)
    pick = rng.permutation(n)[:32]
    C.contact_sheet([out[i] for i in pick], [f"{meta[i]['text']} {'+line' if meta[i]['overlay'] else ''}"
                                             for i in pick], C.OUT / "synth_examples.png", cols=4,
                    title="spike S2: synthetic crops for reader C (glyph font + measured noise, sizes, box, lines)")


if __name__ == "__main__":
    main()
