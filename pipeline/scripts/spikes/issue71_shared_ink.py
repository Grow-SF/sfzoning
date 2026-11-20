"""Issue #71 spike: the printed frame as the ink a sheet shares with the other sheets of its edition.

No placement, no frame shape, no hand list of pages. For one series:

1. Every page (data/work/<series>/<sheet>/page.png, as scanned) becomes soft ink, 1 - luminance.
2. Every ordered pair (A, B) is lined up page-to-page: B turned by k quarter turns and scaled over a small grid, the
   shift from an FFT cross-correlation of blurred ink, then an affine ECC refinement. Nothing about the city is used;
   the shared template is what the match locks onto.
3. On A's ink px, sibling B "agrees" where B (moved onto A) has ink nearby.
4. A two-class mixture per sheet A (frame / city) over its ink px, each sibling a witness (Dawid-Skene):
     P(agree | frame) = f_b, fitted;
     P(agree | city)  = 1 - (1 - n_b(x)) (1 - d_b): n_b(x) is the MEASURED chance that B has ink near a random spot
       around x (the local mean of B's agreement map), so dense hatching and street grids, which match by chance,
       carry no evidence; d_b is B's fitted rate of sharing city ink (a duplicate scan: d_b ~ 1, no weight).
   Fitted by EM.  A text page gets f_b ~ chance and so carries little weight; no list of pages is written by hand.
5. Pictures: A's ink coloured by P(frame) (red frame, blue city).

Spike only: the agreement radius and working scale are fixed here, not measured.

    uv run python scripts/spikes/issue71_shared_ink.py SERIES [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "work"
OUT = ROOT / "data" / "research" / "issue71" / os.environ.get("ISSUE71_OUT", "shared_ink")

COARSE = 8  # coarse search scale (page px per working px)
FINE = 2  # agreement scale
SCALES = (0.95, 0.975, 1.0, 1.025, 1.05)
AGREE_R = int(os.environ.get("ISSUE71_R", "1"))  # agreement radius in FINE px (spike choice)
NULL_HALF = 12  # half-width (FINE px) of the window the chance rate n_b(x) is measured over (spike choice)
F0 = float(os.environ.get("ISSUE71_F0", "0.6"))  # starting frame agree rate (spike)


def load_ink(path: Path, down: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    im = im.resize((max(1, w // down), max(1, h // down)), Image.BOX)
    return 1.0 - np.asarray(im, dtype=np.float32) / 255.0


def blur(x: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), s)


def coarse_match(a: np.ndarray, b: np.ndarray):
    """Best (k, scale, dx, dy, score) mapping b (turned k quarter turns, scaled) onto a, by FFT cross-correlation."""
    ab = blur(a, 1.5)
    ab = ab - ab.mean()
    H, W = ab.shape
    best = None
    for k in range(4):
        bk = np.ascontiguousarray(np.rot90(b, k))
        for s in SCALES:
            bs = cv2.resize(bk, (max(1, round(bk.shape[1] * s)), max(1, round(bk.shape[0] * s))), interpolation=cv2.INTER_AREA)
            bb = blur(bs, 1.5)
            bb = bb - bb.mean()
            ph, pw = H + bb.shape[0], W + bb.shape[1]
            fa = np.fft.rfft2(ab, (ph, pw))
            fb = np.fft.rfft2(bb, (ph, pw))
            cc = np.fft.irfft2(fa * np.conj(fb), (ph, pw))
            idx = np.unravel_index(np.argmax(cc), cc.shape)
            score = cc[idx] / (np.linalg.norm(ab) * np.linalg.norm(bb) + 1e-9)
            dy, dx = idx
            if dy > ph // 2:
                dy -= ph
            if dx > pw // 2:
                dx -= pw
            if best is None or score > best[4]:
                best = (k, s, dx, dy, float(score))
    return best


def rot_matrix(k: int, shape_hw) -> np.ndarray:
    """2x3 affine taking px of the original b to px of np.rot90(b, k)."""
    h, w = shape_hw
    # np.rot90 turns counter-clockwise: (x, y) -> (y, w-1-x) for k=1
    m = np.eye(3)
    for _ in range(k % 4):
        r = np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=float)
        m = r @ m
        h, w = w, h
    return m


def pair_transform(sa: str, sb: str, inks_c: dict, inks_f: dict):
    """Affine (2x3, FINE px) taking B's FINE px to A's FINE px, and the ECC score."""
    a, b = inks_c[sa], inks_c[sb]
    k, s, dx, dy, score = coarse_match(a, b)
    # coarse: b px -> rot -> scale -> shift (dx, dy) in COARSE px
    R = rot_matrix(k, b.shape)
    S = np.diag([s, s, 1.0])
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)
    Mc = T @ S @ R  # COARSE b -> COARSE a
    f = COARSE / FINE
    D = np.diag([f, f, 1.0])
    Mf = D @ Mc @ np.linalg.inv(D)  # FINE b -> FINE a
    A, B = inks_f[sa], inks_f[sb]
    # ECC at a mid scale (FINE*2) for speed, blurred ink
    mid = 2
    Dm = np.diag([1 / mid, 1 / mid, 1.0])
    Am = cv2.resize(A, (A.shape[1] // mid, A.shape[0] // mid), interpolation=cv2.INTER_AREA)
    Bm = cv2.resize(B, (B.shape[1] // mid, B.shape[0] // mid), interpolation=cv2.INTER_AREA)
    Mm = Dm @ Mf @ np.linalg.inv(Dm)
    # ECC wants the map template(a) -> input(b)
    warp = np.linalg.inv(Mm)[:2].astype(np.float32)
    ecc = float("nan")
    try:
        for sig in (4.0, 1.5):
            ecc, warp = cv2.findTransformECC(
                blur(Am, sig), blur(Bm, sig), warp, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 1,
            )
        Wm = np.vstack([warp, [0, 0, 1]])
        Mf = np.linalg.inv(Dm) @ np.linalg.inv(Wm) @ Dm
    except cv2.error:
        pass
    return Mf[:2], dict(k=k, scale=s, coarse_score=score, ecc=ecc)


def agreement(sa: str, sb: str, M: np.ndarray, inks_f: dict) -> np.ndarray:
    A, B = inks_f[sa], inks_f[sb]
    moved = cv2.warpAffine(B, M.astype(np.float32), (A.shape[1], A.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * AGREE_R + 1, 2 * AGREE_R + 1))
    return cv2.dilate(moved, kern)


def fit_mixture(agree: np.ndarray, null: np.ndarray, weight: np.ndarray, iters: int = 300):
    """agree, null: (n_px, n_sib) in [0,1]; weight: A's ink per px.  Returns P(frame) per px, f_b, d_b, pi."""
    n, m = agree.shape
    eps = 1e-4
    null = np.clip(null, eps, 1 - eps)
    # start from parameters, not responsibilities: siblings print the frame (f = F0), share no city ink (d = 0)
    f = np.full(m, F0)
    d = np.zeros(m)
    pi = 0.2
    r = None
    for _ in range(iters):
        if r is not None:
            wr = weight * r
            wq = weight * (1 - r)
            f = np.clip((wr[:, None] * agree).sum(0) / (wr.sum() + 1e-9), eps, 1 - eps)
            num = (wq[:, None] * (agree - null)).sum(0)
            den = (wq[:, None] * (1 - null)).sum(0) + 1e-9
            d = np.clip(num / den, 0.0, 1 - eps)
            pi = np.clip(wr.sum() / weight.sum(), eps, 1 - eps)
        pc = np.clip(1 - (1 - null) * (1 - d), eps, 1 - eps)
        lf = np.log(pi) + (agree * np.log(f) + (1 - agree) * np.log(1 - f)).sum(1)
        lc = np.log(1 - pi) + (agree * np.log(pc) + (1 - agree) * np.log(1 - pc)).sum(1)
        r_new = 1 / (1 + np.exp(np.clip(lc - lf, -50, 50)))
        done = r is not None and np.max(np.abs(r_new - r)) < 1e-5
        r = r_new
        if done:
            break
    return r, f, d, pi


def picture(A: np.ndarray, p: np.ndarray) -> Image.Image:
    ink = np.clip(A, 0, 1)
    base = np.full(A.shape + (3,), 255.0)
    red = np.array([215, 25, 28.0])
    blue = np.array([44, 123, 182.0])
    col = p[..., None] * red + (1 - p[..., None]) * blue
    out = base * (1 - ink[..., None]) + col * ink[..., None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def run_sheet(args):
    sa, sheets, series = args
    od = OUT / series
    od.mkdir(parents=True, exist_ok=True)
    cache = od / f"{sa}.align.json"
    aligned = json.load(open(cache)) if cache.exists() else {}
    inks_f = {s: load_ink(WORK / series / s / "page.png", FINE) for s in sheets}
    sibs = [s for s in sheets if s != sa]
    if any(sb not in aligned for sb in sibs):
        inks_c = {s: load_ink(WORK / series / s / "page.png", COARSE) for s in sheets}
        for sb in sibs:
            if sb not in aligned:
                M, info = pair_transform(sa, sb, inks_c, inks_f)
                aligned[sb] = dict(M=M.tolist(), **info)
        json.dump(aligned, open(cache, "w"), indent=1)
    A = inks_f[sa]
    mask = A > 0.25  # px the mixture is fitted on (ink)
    ag, nl = [], []
    k = 2 * NULL_HALF + 1
    for sb in sibs:
        D = agreement(sa, sb, np.array(aligned[sb]["M"]), inks_f)
        ag.append(D[mask])
        nl.append(cv2.blur(D, (k, k))[mask])
    ag = np.stack(ag, -1)
    nl = np.stack(nl, -1)
    r, f, d, pi = fit_mixture(ag, nl, A[mask])
    P = np.zeros_like(A)
    P[mask] = r
    picture(A, P).save(od / f"{sa}.png")
    np.save(od / f"{sa}_pframe.npy", P.astype(np.float16))
    meta = {sb: dict(k=aligned[sb]["k"], scale=aligned[sb]["scale"], ecc=aligned[sb]["ecc"], f=float(f[i]),
                     d=float(d[i]), chance=float(nl[:, i].mean())) for i, sb in enumerate(sibs)}
    json.dump(dict(sheet=sa, pi=float(pi), siblings=meta), open(od / f"{sa}.json", "w"), indent=1)
    return sa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    allow = OUT.parent / "ONLY_SERIES.txt"  # spike control: series outside this list exit at once
    if allow.exists() and a.series not in allow.read_text().split():
        print("skipped", a.series, "(not in ONLY_SERIES.txt)")
        return
    sheets = sorted(p.parent.name for p in (WORK / a.series).glob("*/page.png"))
    todo = a.only or sheets
    from multiprocessing import Pool

    with Pool(a.workers) as pool:
        for s in pool.imap_unordered(run_sheet, [(s, sheets, a.series) for s in todo]):
            print("done", a.series, s, flush=True)


if __name__ == "__main__":
    main()
