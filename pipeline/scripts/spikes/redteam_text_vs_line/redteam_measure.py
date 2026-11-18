"""Red-team measurements for the text-vs-line design. CPU, one thread, read-only (writes only to the scratchpad).

  real     per 2000 sheet: Terra-word ink split by the stroke rule (M1's base: scores.long_lines) and the run rule
  synth    synthetic RM-2-style cases with exact truth: what each line rule calls line, glyph ink included
  picture  the RM-2 window of picture 0 with each rule's line ink coloured
  texture  i.i.d.-rendered glyph ink (textread.synth's way) against real code ink: ragged-edge statistics
"""
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
          "OPENCV_FOR_THREADS_NUM", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import json
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.codereader import glyphs, labels as L  # noqa: E402
from sfzmaps.solver.textnet import data, scores  # noqa: E402

data.LOAD_THREADS = 1
OUT = Path(__file__).parent / "redteam-out"
OUT.mkdir(exist_ok=True)
LINES = runs.EXPERIMENTS_DIR / "detector-overcall" / "lines" / paths.SERIES


def stroke_long(sd):
    f = LINES / f"{sd.sheet}.npz"
    with np.load(f) as z:
        return np.unpackbits(z["long"], axis=1, count=sd.width).view(bool)


def real(names):
    rows = []
    for name in names:
        sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
        ink = data.page_ink(sd)
        long = stroke_long(sd)
        run = L.run_lines(ink, L.run_px(sd.ft_per_px))
        np.savez_compressed(OUT / f"run-{name}.npz", run=np.packbits(run, axis=1))
        word = data._words(sd, 0, 0, sd.height, sd.width)
        inbox = ink & (word > 0)
        r = {"sheet": name, "ft_per_px": sd.ft_per_px, "run_px": L.run_px(sd.ft_per_px),
             "inbox": int(inbox.sum()), "inbox_long": int((inbox & long).sum()), "inbox_run": int((inbox & run).sum()),
             "inbox_run_not_long": int((inbox & run & ~long).sum()), "inbox_long_not_run": int((inbox & long & ~run).sum())}
        # per word (code words only, kind 0): of the word's run-rule ink, the share the stroke rule calls long
        per = []
        lo, hi = sd.words.min(axis=1), sd.words.max(axis=1) + 1
        for j in range(len(sd.words)):
            (x0, y0), (x1, y1) = np.maximum(lo[j], 0), np.minimum(hi[j], [sd.width, sd.height])
            if x1 <= x0 or y1 <= y0:
                continue
            rect = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(rect, [(sd.words[j] - [x0, y0]).reshape(-1, 1, 2)], 1)
            m = rect.view(bool) & ink[y0:y1, x0:x1]
            nr = int((m & run[y0:y1, x0:x1]).sum())
            if nr >= 3:
                per.append((int(sd.kinds[j]), nr, int((m & run[y0:y1, x0:x1] & long[y0:y1, x0:x1]).sum()),
                            int((m & long[y0:y1, x0:x1]).sum()), int(m.sum())))
        r["words"] = len(sd.words)
        r["words_with_run_ink"] = len(per)
        per = np.array(per, float).reshape(-1, 5)
        for k, kn in data.KINDS.items():
            p = per[per[:, 0] == kn]
            if len(p):
                cover = p[:, 2] / p[:, 1]
                r[f"{k}: words with run ink"] = len(p)
                r[f"{k}: share of their run ink that is stroke-long (pooled)"] = round(p[:, 2].sum() / p[:, 1].sum(), 3)
                r[f"{k}: words where stroke rule catches <half of run ink"] = round(float((cover < 0.5).mean()), 3)
        rows.append(r)
        print(json.dumps(r), flush=True)
    (OUT / "real.json").write_text(json.dumps(rows, indent=1))


def synth():
    """Exact-truth cases at the 2000 scale: a code with a line through, along, or touching it."""
    font = glyphs.Font.load(paths.SOLVER_DIR / "text-crops" / "font.npz") if hasattr(paths, "SOLVER_DIR") else \
        glyphs.Font.load(Path("data/solver/text-crops/font.npz"))
    ftpx = 3.98
    runlen = L.run_px(ftpx)
    out = []
    for text in ("RM-2", "RH-1(D)", "NC-3", "C-3-O", "RM-4", "P"):
        for cap in (20.0, 27.0):
            H, W = 400, 400
            scale = cap / font.size
            d = glyphs.render(font, text, glyphs.Pose(W / 2, H / 2, scale, 0.0), (H, W))
            glyph = d > 0
            ys, xs = np.nonzero(glyph)
            top, bot, left, right = ys.min(), ys.max(), xs.min(), xs.max()
            cases = {
                "horizontal line through the middle, 6 px": [((0, H // 2), (W - 1, H // 2), 6)],
                "horizontal line along the baseline, 4 px": [((0, bot + 1), (W - 1, bot + 1), 4)],
                "vertical line through the 2nd character, 6 px": [((left + int(cap * 0.9), 0),
                                                                  (left + int(cap * 0.9), H - 1), 6)],
                "vertical line touching the word's end, 4 px": [((right + 2, 0), (right + 2, H - 1), 4)],
                "block corner: L of two 4-px lines touching": [((0, bot + 2), (right + 3, bot + 2), 4),
                                                               ((right + 3, 0), (right + 3, bot + 2), 4)],
                "two parallel street lines, word between, 4 px": [((0, top - 3), (W - 1, top - 3), 4),
                                                                  ((0, bot + 3), (W - 1, bot + 3), 4)],
            }
            for cname, segs in cases.items():
                line = np.zeros((H, W), np.uint8)
                for p, q, w in segs:
                    cv2.line(line, p, q, 1, w)
                line = line.view(bool)
                ink = glyph | line
                box = np.zeros((H, W), bool)
                box[max(top - 3, 0):bot + 4, max(left - 3, 0):right + 4] = True
                lo = scores.long_lines(ink, ftpx)
                ru = L.run_lines(ink, runlen)
                lonly, gonly = line & ~glyph, glyph & ~line
                r = {"text": text, "cap": cap, "case": cname}
                for rn, m in (("stroke", lo), ("run", ru)):
                    r[f"{rn}: line ink in box caught"] = round((m & lonly & box).sum() / max((lonly & box).sum(), 1), 3)
                    r[f"{rn}: glyph ink called line"] = round((m & gonly).sum() / max(gonly.sum(), 1), 3)
                out.append(r)
                if text == "RM-2" and cap == 27.0:
                    vis = np.full((H, W, 3), 255, np.uint8)
                    vis[ink] = (160, 160, 160)
                    vis[lo & ru] = (200, 0, 200)
                    vis[lo & ~ru] = (0, 0, 220)
                    vis[ru & ~lo] = (220, 120, 0)
                    cv2.imwrite(str(OUT / f"synth-{cname[:30].replace(' ', '_').replace(',', '')}.png"),
                                cv2.resize(vis[100:300, 80:320], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    (OUT / "synth.json").write_text(json.dumps(out, indent=1))
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        agg[r["case"]].append(r)
    for c, rs in agg.items():
        print(c)
        for k in rs[0]:
            if ":" in k:
                print(f"   {k}: mean {np.mean([r[k] for r in rs]):.3f}  min {min(r[k] for r in rs):.3f}  "
                      f"max {max(r[k] for r in rs):.3f}")


def picture():
    name = "2000-digital-zoning-p02"
    sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
    ink = data.page_ink(sd)
    long = stroke_long(sd)
    f = OUT / f"run-{name}.npz"
    run = np.unpackbits(np.load(f)["run"], axis=1, count=sd.width).view(bool) if f.exists() else \
        L.run_lines(ink, L.run_px(sd.ft_per_px))
    y, x, n = 1600, 1920, 640
    s = np.s_[y:y + n, x:x + n]
    vis = np.full((n, n, 3), 255, np.uint8)
    i, lo, ru = ink[s], long[s], run[s]
    vis[i] = (150, 150, 150)
    vis[i & lo & ru] = (200, 0, 200)        # both: magenta
    vis[i & lo & ~ru] = (0, 0, 230)         # stroke rule only: red
    vis[i & ru & ~lo] = (230, 120, 0)       # run rule only: blue
    shift = np.array([x, y], np.int32)
    for r in sd.words:
        cv2.polylines(vis, [(r - shift).reshape(-1, 1, 2)], True, (0, 170, 0), 1)
    cv2.imwrite(str(OUT / "rm2-window-rules.png"), vis)
    for nm, (cy, cx) in {"rm2-vertical": (1850, 2505), "rm2-top": (1745, 2175), "rm2-mid": (1868, 2218)}.items():
        sub = vis[cy - y - 70:cy - y + 70, cx - x - 70:cx - x + 70]
        cv2.imwrite(str(OUT / f"{nm}.png"), cv2.resize(sub, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))


def texture():
    """Ragged-edge statistics: real code-word ink (off run lines) against glyphs rendered i.i.d. from the noise."""
    font = glyphs.Font.load(Path("data/solver/text-crops/font.npz"))
    noise = font.noise
    rng = np.random.default_rng(0)

    def stats(ink, mask):
        u = ink.astype(np.uint8)
        nb = cv2.filter2D(u, -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32),
                          borderType=cv2.BORDER_CONSTANT)
        m = ink & mask
        iso = (m & (nb <= 1)).sum()
        # paper px inside the word with >= 6 ink neighbours (pinholes in strokes)
        pin = (~ink & mask & (nb >= 6)).sum()
        n, _ = cv2.connectedComponents(u, connectivity=8)
        return int(m.sum()), int(iso), int(pin), n - 1

    real_t = np.zeros(4)
    names = ["2000-digital-zoning-p02", "2000-digital-zoning-p05", "2000-digital-zoning-p12"]
    nwords = 0
    for name in names:
        sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
        ink = data.page_ink(sd)
        f = OUT / f"run-{name}.npz"
        run = np.unpackbits(np.load(f)["run"], axis=1, count=sd.width).view(bool) if f.exists() else \
            L.run_lines(ink, L.run_px(sd.ft_per_px))
        clean = ink & ~cv2.dilate(run.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
        lo, hi = sd.words.min(axis=1), sd.words.max(axis=1) + 1
        for j in np.flatnonzero(sd.kinds == 0):
            (x0, y0), (x1, y1) = np.maximum(lo[j], 0), np.minimum(hi[j], [sd.width, sd.height])
            if x1 - x0 < 5 or y1 - y0 < 5:
                continue
            win = ink[y0:y1, x0:x1]
            if (run[y0:y1, x0:x1] & win).any():
                continue                     # words with no line ink at all: glyph ink alone
            rect = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(rect, [(sd.words[j] - [x0, y0]).reshape(-1, 1, 2)], 1)
            real_t += stats(clean[y0:y1, x0:x1], rect.view(bool))
            nwords += 1
    sp = 1e-4
    bg = noise.background

    def p_ink(dd):
        return sp + np.clip(noise.p_ink(dd) - bg, 0, None) / (1 - bg) * (1 - sp)

    syn_t = np.zeros(4)
    texts = ["RM-2", "RH-1(D)", "NC-3", "C-3-O", "RM-4", "RH-2", "NC-1", "P", "RH-3", "M-1"]
    for k in range(nwords):
        text = texts[k % len(texts)]
        cap = float(np.exp(rng.normal(np.log(26.0), 0.15)))
        d = glyphs.render(font, text, glyphs.Pose(150, 60, cap / font.size, 0.0), (120, 300))
        ink = rng.random(d.shape) < p_ink(d)
        ys, xs = np.nonzero(d > -1)
        mask = np.zeros(d.shape, bool)
        mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        syn_t += stats(ink, mask)
    res = {"real code words without line ink": nwords,
           "real": {"ink": real_t[0], "isolated ink px share": real_t[1] / real_t[0],
                    "pinholes per 1000 ink px": 1000 * real_t[2] / real_t[0], "components per word": real_t[3] / nwords},
           "iid rendered": {"ink": syn_t[0], "isolated ink px share": syn_t[1] / syn_t[0],
                            "pinholes per 1000 ink px": 1000 * syn_t[2] / syn_t[0],
                            "components per word": syn_t[3] / nwords},
           "noise p at knots": noise.p.round(3).tolist()}
    print(json.dumps(res, indent=1, default=float))
    (OUT / "texture.json").write_text(json.dumps(res, indent=1, default=float))
    # one strip of each for the eye
    font_img = []
    for text in texts[:4]:
        d = glyphs.render(font, text, glyphs.Pose(150, 40, 26 / font.size, 0.0), (80, 300))
        font_img.append(np.where(rng.random(d.shape) < p_ink(d), 0, 255).astype(np.uint8))
    cv2.imwrite(str(OUT / "iid-rendered.png"), cv2.resize(np.vstack(font_img), None, fx=3, fy=3,
                                                          interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "real":
        real(sys.argv[2:] or list(paths.SHEETS))
    elif what == "synth":
        synth()
    elif what == "picture":
        picture()
    elif what == "texture":
        texture()
