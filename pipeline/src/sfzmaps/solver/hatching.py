"""The 1921 hatching per drawn region, step 1: texture measures of square patches of every 1921 scan, at the scan's own
resolution, and the share of each patch that each solver lot holds.

The 1921 map draws its districts as fill patterns (the legend on section 1): 1R single diagonal hatching, 2R blank, C
solid, LI diagonal cross-hatch, HI square grid, U checkerboard.  A lot-by-lot read stalls because a small lot falls in
the white between 1R's hatch lines, text covers lots and lot outlines sit a few px off.  So the scans are measured in
square PATCHES sized from each sheet's own hatch spacing; a lot then takes the evidence of the patches it overlaps
(`patch_lots`: its share of each patch's in-block px), and the joint solve pools it over whole regions.

Each sheet is measured as scanned (data/work/1921/<sheet>/page.png, ink where darker than `grid.INK_BELOW`), never
resampled.  The solver's page (4 ft/px) is too coarse for hatch lines, so the solver's grid is only used to say which
lot (`lots.npz` `lot`) and which block each page px lies in, through the sheet's placement (data/warp, as the merged
page reads it: `to_world` every `page.STEP` px, bilinear between).

Every number is measured on the scans, derived from such a measure, or is a binning:

  period    the sheet's 1R hatch period (`sheet_period`): the strongest wave of the ink inside the lots the pattern
            reader read 1R on that sheet (its reads are used for this and nothing else).  It counts only when it
            stands out of the background of its frequency ring by more than the largest of that many bins of noise
            would (exponential power: ln of the number of bins searched), and when its lines lie nearer 1R's own
            angle (the orientation of the legend's 1R swatch, turned to the sheet: `paper_turn`) than the
            perpendicular (a cross-hatch: LI, HI or U).  A sheet whose 1R lots show no such wave takes the median of
            the other sheets' periods carried to its scan resolution (by ft per px); the legend's sheet takes its own
            1R swatch (`fallback_period`).
  patches   squares of side 2 x period (two periods: the least that shows a period) at a stride of one period
            (`patch_grid`), so every px away from the page's edge lies in four.
  stroke    the printed line's width (`stroke_width`, across the edges of the white regions the line work traced:
            blocks_px.geojson).  The in-block mask (`inblock_mask`) leaves out the band one stroke wide outside the
            blocks' paper, the printed block outlines found on the scan (`outline_mask`; `block_paper`: the regions of
            paper lying mostly on the solver's blocks, split at gaps narrower than two strokes and walled by the line
            work's traced edges, joined across one stroke; the lines inside a block, its hatch, stay), so the
            outline's ink stays out of the measure however far the solver's lots sit from the
            print (the lots' own mask, which is also eroded by it, does not: they sit up to tens of px off it).  It is
            also the width of the taper at a region's edge (the spike's TAPER_PX), so the cut edge adds no lines.
  band      the hatch band runs from the legend's finest period / 2 (`legend_scale`: the finest hatch the map uses,
            measured on the hatched swatches of the legend, `LEGENDS_FILE`, on sheet 20180102094142210; in px of
            that sheet, the same on every sheet) to 2 x period.
  sectors   the periodicity is read at the strongest period the patch can show (at most half its side: two periods
            across), at the lags its own spectrum can't tell from that peak: the peak's frequency +- 1 / (2 x side),
            and its angle within atan(that / frequency).  It is normalised by the whole window's overlap with itself
            (the spike: the region's, read only where it was over a quarter), so a region narrower than a period
            across the lines shows little periodicity instead of a noisy ratio; it is not clipped (`periodicity`).
  BANDS, N_ORI, WHITE_R, BLACK_R   the spike's binning of the spectrum (period bands, 15 degree orientations) and the
            scales of the morphology: features, not cutoffs.

Orientation is kept in two frames, each angle term weighted by its strength (|m2| for one family of lines, |m4| for a
right-angle cross: m2 = |m2| e^{2 i theta}): the page (turned upright where the sheet was scanned on its side:
`paper_turn`) and the block's major axis (the minimum rotated rectangle of the block's lots in the solver's lot table,
carried to page px through the placement at the patch's centre).

A patch at a seam is measured once: on the sheet whose neatline (data/solver/1921/page/crops/<sheet>.png, the
neatline interior less the notes, as the merged page crops it) holds its centre, and where several do, on the one the
published map draws on top (`stitch.draw_order`).

Writes, under data/solver/experiments/hatch-regions/:
  patches/<sheet>.parquet     one row per patch: sheet, patch (its index in the sheet's `patch_grid`), x0, y0, side
                              (page px), cx, cy (its centre, EPSG:2227 ft), block (its block piece), block_phi,
                              block_elongation, period, ft_per_px, and the measures (`measure_patch`)
  patch_lots/<sheet>.parquet  patch, lot (the solver's lot index), share (of the patch's in-block px)
  sheets.json                 per sheet: the period and how it was found, the stroke, the turn, the patch counts;
                              and the legend's measures
  swatches.parquet            the legend swatches measured as patches

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatching --series 1921 --workers 6
    uv run python -m sfzmaps.solver.hatching --series 1921 --look SHEET SHEET     (the look picture)

$0: no API calls.  Ported from the spike (branch worktree-agent-ab10cd06a6d7b4538, scripts/spikes/hatch1921/measure.py).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from scipy import fft as sfft
from scipy import ndimage
from scipy.ndimage import map_coordinates

from sfzmaps import config
from sfzmaps.solver import grid, page
from sfzmaps.transform import SheetTransform

SERIES = "1921"
WORK = config.WORK_DIR / SERIES
WARP = config.DATA_DIR / "warp" / SERIES
SOLVER_1921 = config.DATA_DIR / "solver" / SERIES     # this series' folders, whatever series the solver is set to
EXPERIMENT = config.DATA_DIR / "solver" / "experiments" / "hatch-regions"
LEGENDS_FILE = config.PIPELINE_DIR / "src" / "sfzmaps" / "districts" / "legends.json"
SKIP_VIA = ("off_sheet", "no_tile")      # reads of a lot not on the sheet (the reader's own marks)
LEGEND_HATCHED = ("1R", "LI", "HI", "U")  # the legend's patterns drawn in lines (2R is blank paper, C solid ink)
BANDS = (3, 5, 7, 10, 14, 20, 28, 40)    # period band edges, px: the spike's binning of the spectrum
N_ORI = 12                                # orientation bins of 15 degrees
WHITE_R = (1, 2, 3, 5)                    # the morphology's scales (disk radii, px)
BLACK_R = (1, 2, 3)
LATTICE = page.STEP                       # page px between the points where a placement is evaluated
# A patch's spectrum is padded to PAD x its side (at least 2 x, so its autocorrelation does not wrap round).  PAD is
# where the measure has converged: at 2 x a 1R hatch a little finer than the half-side limit (`spectrum`) often landed
# on the bin beyond it and its periodicity was read at a harmonic; the review measured the periodicity AUC (1R against
# 2R, a sample of one sheet's patches) at 0.806, 0.800 and 0.803 for PAD 4, 8 and 16, and the 90th percentile of
# |periodicity(PAD 4) - periodicity(PAD 8)| at 0.016.
PAD = 4
INK_BELOW = grid.INK_BELOW


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


@dataclass(frozen=True)
class Frame:
    """What a patch is measured against: its sheet's 1R period (px), the hatch band's shortest period (`low`, px), the
    taper at a region's edge (px), 1R's wave direction on the page (radians, page px: x right, y down; NaN: unknown),
    the quarter turns from the page to the sheet upright, and the block's major axis on the page (radians; NaN: no
    block)."""
    period: float
    low: float = 2.0             # Nyquist: no finer period exists on a px grid
    taper: float = 1.0           # 1 px: no taper (a region's edge px keep their whole weight)
    wave_1r: float = np.nan
    turn: int = 0
    phi: float = np.nan

    @property
    def band(self) -> tuple[float, float]:
        return self.low, 2.0 * self.period


# --- the spectrum ---------------------------------------------------------------------------------------------------

_BINS: dict[int, tuple] = {}


def bins(n: int):
    """For an n x n rfft2: band index (-1 outside BANDS), orientation bin, e^{2i beta}, e^{4i beta}, period, weight,
    and the flat (band, orientation) index of the bins inside BANDS with their positions."""
    if n not in _BINS:
        ky = np.fft.fftfreq(n)[:, None]
        kx = np.fft.rfftfreq(n)[None, :]
        f = np.hypot(kx, ky)
        with np.errstate(divide="ignore"):
            period = np.where(f > 0, 1.0 / np.maximum(f, 1e-12), np.inf)
        beta = np.arctan2(ky, kx) + np.pi / 2            # the lines run perpendicular to the wave vector
        band = np.digitize(period, BANDS) - 1
        band[(period < BANDS[0]) | (period >= BANDS[-1])] = -1
        ori = np.round(np.mod(beta, np.pi) / (np.pi / N_ORI)).astype(int) % N_ORI
        w = np.ones_like(f)
        w[:, 0] = 0.5                                      # rfft: the kx = 0 and Nyquist columns hold both halves
        if n % 2 == 0:
            w[:, -1] = 0.5
        wave = np.arctan2(ky, kx) * np.ones_like(f)
        ok = np.flatnonzero(band >= 0)
        cell = (band.ravel()[ok] * N_ORI + ori.ravel()[ok])
        _BINS[n] = (band, ori, np.exp(2j * beta), np.exp(4j * beta), period, w, f * np.ones_like(f), wave, ok, cell)
    return _BINS[n]


_LAGS: dict[int, tuple] = {}


def lags(n: int):
    """For an n x n autocorrelation (irfft2 layout): each cell's lag (dx, dy) as its length, angle, |dx| and |dy|."""
    if n not in _LAGS:
        d = (np.arange(n) + n // 2) % n - n // 2
        dx, dy = np.meshgrid(d, d)
        _LAGS[n] = (np.hypot(dx, dy), np.arctan2(dy, dx), np.abs(dx), np.abs(dy))
    return _LAGS[n]


def taper_weights(mask: np.ndarray, taper: float) -> np.ndarray:
    """1 inside the region, falling to 0 over `taper` px at its edge (the window's edge counts as one)."""
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    return np.clip(dist / max(float(taper), 1e-9), 0, 1).astype(np.float64)


_SQUARE: dict[tuple, np.ndarray] = {}


def _square_acf(shape: tuple[int, int], n: int, taper: float) -> np.ndarray:
    """The autocorrelation of the whole window's own weights: what a region filling the window overlaps itself by."""
    key = (tuple(shape), n, float(taper))
    if key not in _SQUARE:
        w = taper_weights(np.ones(shape, bool), taper)
        _SQUARE[key] = np.fft.irfft2(np.abs(np.fft.rfft2(w, s=(n, n))) ** 2, s=(n, n))
    return _SQUARE[key]


def _sector(n: int, shape, period: float, wave: float, side: float) -> np.ndarray:
    """The lags the window's own spectrum can't tell from one `period` across waves at `wave`: lengths between
    1 / (f + d) and 1 / (f - d) and angles within atan(d / f) of the wave, f = 1 / period, d = 1 / (2 side) (the
    window's frequency resolution), inside the window."""
    r, a, adx, ady = lags(n)
    f, d = 1.0 / period, 0.5 / side
    rmin = 1.0 / (f + d)
    rmax = 1.0 / (f - d) if f > d else np.inf
    da = np.abs(np.mod(a - wave + np.pi / 2, np.pi) - np.pi / 2)
    return (r >= rmin) & (r <= rmax) & (da <= np.arctan2(d, f)) & (adx < shape[1]) & (ady < shape[0])


def periodicity(acf: np.ndarray, wsq: np.ndarray, n: int, shape, period: float, wave_angle: float) -> float:
    """How periodic the region's ink is: its autocorrelation one period across the lines of a wave (normalised by its
    value at lag 0 and by the whole window's overlap with itself at that lag), the best over the lags the window's
    spectrum can't tell from that period and direction (`_sector`).  About 1 for a hatch filling the window, about 0
    for blank paper, a single line, text or solid ink; a region narrower than a period across the lines can't show
    one (its overlap there is empty), so it scores low.  Not clipped: acf[lag] / acf[0] lies in [-1, 1]
    (Cauchy-Schwarz), so the value lies within +-1 / (the window's overlap fraction at that lag); it passes 1 only
    when the ink correlates more strongly inside the overlap than a stationary pattern would.  0 with no ink or no
    lag to read."""
    if acf[0, 0] <= 1e-12 or not np.isfinite(period):
        return 0.0
    sector = _sector(n, shape, period, wave_angle, max(shape))
    if not sector.any():
        return 0.0
    ratio = (acf[sector] / acf[0, 0]) / (wsq[sector] / wsq[0, 0])
    return float(ratio.max())


def spectrum(ink: np.ndarray, mask: np.ndarray, frame: Frame) -> dict | None:
    """The region's ink minus its mean, tapered `frame.taper` px inside the region (so its edge adds no lines),
    padded to PAD times the window and FFT'd: the power (as a share of the region's weighted variance, so it doesn't
    depend on the padding) binned by BANDS and
    the orientation of the LINES (N_ORI bins, page px: x right, y down, angle atan2(dy, dx)).  Over the hatch band
    (`frame.band`) the complex moments m2 = sum P e^{2 i beta} / sum P (one family of lines: its strength and
    orientation) and m4 = sum P e^{4 i beta} / sum P (a cross-hatch at right angles), and the power-weighted log
    period; the strongest period the window can show (at most half its side: two periods across, so two block
    outlines further apart than that are no period) and the periodicity there; 1R's own hatch, sought at the sheet's
    1R period and direction (`dir1r_acf`, the periodicity there, and `dir1r_share`, the band's power within what the
    window can't tell from it).  None when the region has no weight."""
    if not mask.any():
        return None
    w = taper_weights(mask, frame.taper)
    sw = w.sum()
    if sw <= 0:
        return None
    x = (ink - (ink * w).sum() / sw) * w
    side = max(ink.shape)
    n = 1 << int(np.ceil(np.log2(PAD * side)))
    P = np.abs(np.fft.rfft2(x, s=(n, n))) ** 2
    acf = np.fft.irfft2(P, s=(n, n))
    wsq = _square_acf(ink.shape, n, frame.taper)
    band, ori, e2, e4, period, wt, freq, wave, ok, cell = bins(n)
    p = P / (w ** 2).sum() * wt / (n * n / 2)          # Parseval: sums to the weighted variance, whatever the padding
    out = {"grid": np.bincount(cell, p.ravel()[ok], minlength=(len(BANDS) - 1) * N_ORI).reshape(-1, N_ORI),
           "periodicity": 0.0}
    lo, hi = frame.band
    hb = (period >= lo) & (period < hi)
    ph = p[hb]
    tot = ph.sum()
    out["hatch_power"] = float(tot)
    out["var"] = float(p.sum())
    if tot > 0:
        out["m2"] = complex((ph * e2[hb]).sum() / tot)
        out["m4"] = complex((ph * e4[hb]).sum() / tot)
        out["logperiod"] = float((ph * np.log(period[hb])).sum() / tot)
    else:
        out["m2"] = out["m4"] = 0j
        out["logperiod"] = np.nan
    shows = hb & (period <= side / 2)
    out["peak_period"] = np.nan
    if (p[shows] > 0).any():
        k = np.argmax(p[shows])
        out["peak_period"] = float(period[shows][k])
        out["periodicity"] = periodicity(acf, wsq, n, ink.shape, out["peak_period"], float(wave[shows][k]))
    if np.isfinite(frame.wave_1r):
        out["dir1r_acf"] = periodicity(acf, wsq, n, ink.shape, frame.period, frame.wave_1r)
        d = 0.5 / side
        da = np.abs(np.mod(wave - frame.wave_1r + np.pi / 2, np.pi) - np.pi / 2)
        cone = hb & (np.abs(freq - 1.0 / frame.period) <= d) & (da <= np.arctan2(d, 1.0 / frame.period))
        out["dir1r_share"] = float(p[cone].sum() / tot) if tot > 0 else 0.0
    else:
        out["dir1r_acf"] = out["dir1r_share"] = np.nan
    return out


def spec_feats(sp: dict | None, frame: Frame) -> dict:
    """The spectrum's measures as columns: the orientation measures in the page's frame (the sheet upright:
    `frame.turn` quarter turns; m4 is the same either way) and in the block's (m2 e^{-2 i phi}, m4 e^{-4 i phi}),
    and the strongest line orientation of the page frame (degrees, from m2)."""
    if sp is None:
        sp = {"grid": np.full((len(BANDS) - 1, N_ORI), np.nan), "hatch_power": np.nan, "var": np.nan,
              "m2": complex(np.nan, np.nan), "m4": complex(np.nan, np.nan), "logperiod": np.nan,
              "peak_period": np.nan, "periodicity": np.nan, "dir1r_acf": np.nan, "dir1r_share": np.nan}
    turn = int(frame.turn) % 4
    g = np.roll(sp["grid"], turn * N_ORI // 2, axis=1)      # a quarter turn moves a line 90 degrees: N_ORI / 2 bins
    m2_page, m4 = sp["m2"], sp["m4"]
    m2 = m2_page * (-1) ** turn
    out = {"periodicity": sp["periodicity"], "dir1r_acf": sp["dir1r_acf"], "dir1r_share": sp["dir1r_share"]}
    for b in range(len(BANDS) - 1):
        for o in range(N_ORI):
            out[f"g{b}_{o}"] = float(g[b, o])
    out["hatch_power"], out["var"] = sp["hatch_power"], sp["var"]
    out["m2_re"], out["m2_im"] = m2.real, m2.imag
    out["m4_re"], out["m4_im"] = m4.real, m4.imag
    if np.isfinite(frame.phi):
        b2, b4 = m2_page * np.exp(-2j * frame.phi), m4 * np.exp(-4j * frame.phi)
    else:
        b2 = b4 = complex(np.nan, np.nan)
    out["blk_m2_re"], out["blk_m2_im"] = b2.real, b2.imag
    out["blk_m4_re"], out["blk_m4_im"] = b4.real, b4.imag
    out["logperiod"], out["peak_period"] = sp["logperiod"], sp["peak_period"]
    out["line_angle"] = float(np.degrees(np.angle(m2) / 2)) if np.isfinite(m2) and abs(m2) > 0 else np.nan
    return out


# --- the morphology -------------------------------------------------------------------------------------------------

def morph_stack(ink: np.ndarray) -> np.ndarray:
    """uint8 H x W x (1 + len(WHITE_R) + len(BLACK_R)): ink, white opened by each disk, ink opened by each disk."""
    ink = ink.astype(np.uint8)
    white = (1 - ink).astype(np.uint8)
    ch = [ink]
    ch += [cv2.morphologyEx(white, cv2.MORPH_OPEN, disk(r)) for r in WHITE_R]
    ch += [cv2.morphologyEx(ink, cv2.MORPH_OPEN, disk(r)) for r in BLACK_R]
    return np.dstack(ch)


MORPH_KEYS = ("density", *(f"white_open{r}" for r in WHITE_R), *(f"ink_open{r}" for r in BLACK_R))


def morph_feats(stack: np.ndarray, mask: np.ndarray) -> dict:
    """The region's ink density and the share of its white (holes of the lattice: LI diamonds, HI squares, U dots, 2R
    paper) and of its ink (lines versus solid) that survives an opening by each disk."""
    s = stack[mask].astype(np.float64)
    if not len(s):
        return {k: np.nan for k in MORPH_KEYS}
    n = len(s)
    ink = s[:, 0].sum()
    white = n - ink
    out = {"density": float(ink / n)}
    for k, r in enumerate(WHITE_R):
        out[f"white_open{r}"] = float(s[:, 1 + k].sum() / white) if white > 0 else 0.0
    for k, r in enumerate(BLACK_R):
        out[f"ink_open{r}"] = float(s[:, 1 + len(WHITE_R) + k].sum() / ink) if ink > 0 else 0.0
    return out


def _window_stack(ink: np.ndarray, box) -> np.ndarray:
    """morph_stack's values inside `box`, from the box grown by the farthest an opening reads (twice the largest
    radius), so they equal the whole page's."""
    x0, y0, x1, y1 = box
    reach = 2 * max(WHITE_R + BLACK_R)
    h, w = ink.shape
    gx0, gy0, gx1, gy1 = max(x0 - reach, 0), max(y0 - reach, 0), min(x1 + reach, w), min(y1 + reach, h)
    st = morph_stack(ink[gy0:gy1, gx0:gx1])
    return st[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]


def measure_patch(ink: np.ndarray, inblock: np.ndarray, box, frame: Frame) -> dict:
    """The measures of one patch, `box` = (x0, y0, x1, y1) page px (ends exclusive), over its in-block px: `area` (the
    in-block px), the morphology (`morph_feats`) and the spectrum (`spectrum`, `spec_feats`).  `ink`: the page's ink
    (H x W, 1 = ink), or its `morph_stack` (H x W x C, the ink first) so the openings are made once per page."""
    x0, y0, x1, y1 = box
    m = np.asarray(inblock[y0:y1, x0:x1], bool)
    if ink.ndim == 3:
        st, ic = ink[y0:y1, x0:x1], ink[y0:y1, x0:x1, 0]
    else:
        st, ic = _window_stack(ink, box), ink[y0:y1, x0:x1]
    out = {"area": int(m.sum())}
    out.update(morph_feats(st, m))
    out.update(spec_feats(spectrum(ic.astype(np.float64), m, frame), frame))
    return out


# --- the sheet's scale ----------------------------------------------------------------------------------------------

def _pow2(v: int) -> int:
    return 1 << int(np.ceil(np.log2(max(int(v), 1))))


def hatch_peak(ink: np.ndarray, mask: np.ndarray, low: float = 2.0, taper: float = 1.0) -> dict | None:
    """The strongest wave of the region's ink (minus its mean, tapered `taper` px, padded to a square twice the
    region's longer side, so the peak is located between the region's own frequency steps), at periods of `low` px or
    more: its period (px), its lines' angle (degrees, page px, (-90, 90]), its strength (its power over the
    background of its frequency ring: the ring's median power / ln 2, the mean of exponential noise with that
    median) and the number of frequency bins searched.  None for an empty region.  Single precision: a whole
    sheet's region is large."""
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    ic = ink[y0:y1, x0:x1].astype(np.float64)
    w = taper_weights(m, taper)
    sw = w.sum()
    if sw <= 0:
        return None
    x = ((ic - (ic * w).sum() / sw) * w).astype(np.float32)
    n0 = n1 = _pow2(2 * max(x.shape))       # square: the same frequency step every way, as `spectrum`
    X = sfft.rfft2(x, s=(n0, n1), workers=1)
    del x
    P = (X.real ** 2 + X.imag ** 2).astype(np.float32)
    del X
    ky = np.fft.fftfreq(n0).astype(np.float32)[:, None]
    kx = np.fft.rfftfreq(n1).astype(np.float32)[None, :]
    f = np.hypot(kx, ky)
    searched = (f > 0) & (f <= 1.0 / low)
    if not searched.any():
        return None
    k = np.unravel_index(np.argmax(np.where(searched, P, -1.0)), P.shape)
    ring = searched & (np.abs(f - f[k]) <= 0.5 * max(1.0 / n0, 1.0 / n1))
    background = float(np.median(P[ring])) / np.log(2)
    fy, fx = float(ky[k[0], 0]), float(kx[0, k[1]])
    angle = np.degrees(np.arctan2(fy, fx) + np.pi / 2)
    return {"period": float(1.0 / np.hypot(fx, fy)), "line_angle": float((angle + 90) % 180 - 90),
            "strength": float(float(P[k]) / background) if background > 0 else np.inf, "bins": int(searched.sum())}


def _angle_apart(a: float, b: float) -> float:
    """How far two line angles (degrees) are apart, mod 180: 0-90."""
    return float(abs((a - b + 90) % 180 - 90))


def accept_period(peak: dict | None, line_angle: float | None = None) -> float:
    """The peak's period when it is a 1R hatch: stronger than the largest of its bins of noise would be (ln of their
    number) and, with 1R's line angle on the page given, its lines nearer that angle than the perpendicular (whose
    lines belong to a cross-hatch).  NaN otherwise."""
    if peak is None or not peak["strength"] > np.log(peak["bins"]):
        return np.nan
    if line_angle is not None and np.isfinite(line_angle) and _angle_apart(peak["line_angle"], line_angle) >= 45:
        return np.nan
    return peak["period"]


def fallback_period(sheet: str, ft_per_px: float, measured: dict[str, tuple[float, float]], legend_sheet: str,
                    legend_1r: float) -> tuple[float, str]:
    """The period of a sheet whose 1R lots show no 1R hatch, and where it came from.  The legend's sheet takes its own
    measured 1R swatch.  Any other takes the median of the measured sheets' periods (`measured`: sheet -> (period px,
    ft per px)), each carried to this sheet's scan resolution (period x its ft per px / this sheet's), so scans at
    another resolution count at this one's."""
    if sheet == legend_sheet:
        return float(legend_1r), "legend 1R swatch"
    vals = [t * f / ft_per_px for s, (t, f) in sorted(measured.items()) if s != sheet]
    return float(np.median(vals)), "median"


def sheet_period(ink: np.ndarray, mask_1r: np.ndarray, low: float = 2.0, line_angle: float | None = None,
                 taper: float = 1.0) -> float:
    """The dominant hatch period, in px, of the sheet's ink inside the reader's 1R lots (`hatch_peak`, `accept_period`);
    NaN when they show no 1R hatch (the caller takes the other sheets' median)."""
    return accept_period(hatch_peak(ink, mask_1r, low, taper), line_angle)


def stroke_width(ink: np.ndarray, rings: list[np.ndarray]) -> int:
    """The printed block outline's width, px: at every px along the outlines (`rings`, page px, traced on the scan's
    line work) where the page is ink, the run of ink through it along the outline's normal; the most common run (a
    hatch line meeting the outline, or a solid block, makes a longer one now and then)."""
    pts, nrm = [], []
    for r in rings:
        r = np.asarray(r, float)[:, :2]
        a, b = r[:-1], r[1:]
        for p, q in zip(a, b):
            d = q - p
            length = float(np.hypot(*d))
            if length <= 0:
                continue
            k = max(1, int(length))
            t = (np.arange(k) + 0.5)[:, None] / k
            pts.append(p + t * d)
            nrm.append(np.repeat([[-d[1] / length, d[0] / length]], k, axis=0))
    if not pts:
        return 1
    P, N = np.concatenate(pts), np.concatenate(nrm)
    h, w = ink.shape

    def is_ink(q):
        xi, yi = np.round(q[:, 0]).astype(np.int64), np.round(q[:, 1]).astype(np.int64)
        on = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros(len(q), bool)
        out[on] = ink[yi[on], xi[on]] > 0
        return out

    keep = is_ink(P)
    P, N = P[keep], N[keep]
    if not len(P):
        return 1
    run = np.ones(len(P), np.int64)
    for sign in (1.0, -1.0):
        active = np.arange(len(P))
        k = 1
        while len(active):
            more = is_ink(P[active] + sign * k * N[active])
            active = active[more]
            run[active] += 1
            k += 1
    vals, counts = np.unique(run, return_counts=True)
    return int(vals[np.argmax(counts)])


def paper_turn(quad) -> int:
    """Quarter turns from the page to the printed sheet upright: every 1921 section is drawn on a portrait sheet (15
    of the 16 neatlines are taller than wide), so a neatline wider than tall was scanned on its side (20180102100525362,
    whose 1R lines lie at +45 degrees on the page where the others' lie at -45).  Which way it turned doesn't matter
    to a line's angle mod 180, nor to a cross's mod 90."""
    q = np.asarray(quad, float)
    width = np.hypot(*(q[1] - q[0])) + np.hypot(*(q[2] - q[3]))
    height = np.hypot(*(q[3] - q[0])) + np.hypot(*(q[2] - q[1]))
    return int(width > height)


# --- page px and the solver's grid ----------------------------------------------------------------------------------

class PageWorld:
    """A sheet's page px -> EPSG:2227 ft: its placement evaluated every LATTICE px (as the merged page reads a
    sheet), bilinear between.  `box` (x0, y0, x1, y1): only the nodes that cover it; the values are the whole page's."""

    def __init__(self, transform, shape: tuple[int, int], box=None):
        h, w = shape
        x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
        self.nx0, self.ny0 = x0 // LATTICE * LATTICE, y0 // LATTICE * LATTICE
        xs = np.arange(self.nx0, max(x1 - 1, x0) + LATTICE, LATTICE, dtype=float)
        ys = np.arange(self.ny0, max(y1 - 1, y0) + LATTICE, LATTICE, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        self.nodes = np.asarray(transform.to_world(np.c_[gx.ravel(), gy.ravel()]), float).reshape(gy.shape + (2,))

    def at(self, xy: np.ndarray) -> np.ndarray:
        """World ft (n, 2) of page px (n, 2) (x, y; px centres at integers)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        rc = [(xy[:, 1] - self.ny0) / LATTICE, (xy[:, 0] - self.nx0) / LATTICE]
        return np.stack([map_coordinates(self.nodes[..., k], rc, order=1, mode="nearest") for k in (0, 1)], -1)

    def block(self, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
        """World x and y (each (y1 - y0, x1 - x0)) of the px of a box."""
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = self.at(np.c_[xx.ravel(), yy.ravel()])
        return w[:, 0].reshape(yy.shape), w[:, 1].reshape(yy.shape)


def grid_values(X: np.ndarray, Y: np.ndarray, raster: np.ndarray, offset=(0, 0), fill=-1) -> np.ndarray:
    """The value of a grid raster (rows and columns from `offset` (row, col) of the solver's grid) at world points;
    `fill` off it."""
    c = np.floor((X - grid.X0) / grid.FT_PER_PX)
    r = np.floor((grid.Y1 - Y) / grid.FT_PER_PX)
    ok = np.isfinite(c) & np.isfinite(r)
    ci = np.where(ok, c, -1).astype(np.int64) - offset[1]
    ri = np.where(ok, r, -1).astype(np.int64) - offset[0]
    ok &= (ci >= 0) & (ci < raster.shape[1]) & (ri >= 0) & (ri < raster.shape[0])
    out = np.full(X.shape, fill, raster.dtype)
    out[ok] = raster[ri[ok], ci[ok]]
    return out


def page_values(pw: PageWorld, shape: tuple[int, int], rasters: list[tuple[np.ndarray, int]], offset=(0, 0),
                rows: int = 512) -> list[np.ndarray]:
    """Grid rasters (each with its fill off it) read at every px of a page (bands of `rows` rows at once: memory
    only)."""
    h, w = shape
    outs = [np.full(shape, fill, r.dtype) for r, fill in rasters]
    for y0 in range(0, h, rows):
        X, Y = pw.block(0, y0, w, min(h, y0 + rows))
        for o, (r, fill) in zip(outs, rasters):
            o[y0:y0 + rows] = grid_values(X, Y, r, offset, fill)
    return outs


def _shares(lots: np.ndarray, mask: np.ndarray) -> list[tuple[int, float]]:
    n = int(mask.sum())
    if not n:
        return []
    v = lots[mask]
    ids, counts = np.unique(v[v >= 0], return_counts=True)
    return [(int(i), float(c) / n) for i, c in zip(ids, counts)]


def patch_lots(box, sheet_transform, lot_raster: np.ndarray, inblock: np.ndarray | None = None,
               offset=(0, 0)) -> list[tuple[int, float]]:
    """(solver lot, share of the patch's in-block px) for each lot under a patch: every px of `box` (x0, y0, x1, y1,
    page px) carried to the solver's grid through the placement (`PageWorld`), `lot_raster` read there (from `offset`
    (row, col) of the grid; -1 is no lot).  `inblock`: the page's in-block mask (all of the box when None).  A px in a
    block but on no lot counts in the patch and in no lot's share."""
    x0, y0, x1, y1 = box
    pw = PageWorld(sheet_transform, (y1, x1), box)
    X, Y = pw.block(x0, y0, x1, y1)
    lots = grid_values(X, Y, lot_raster, offset)
    m = np.ones(lots.shape, bool) if inblock is None else np.asarray(inblock[y0:y1, x0:x1], bool)
    return _shares(lots, m)


def patch_grid(shape: tuple[int, int], period: float) -> list[tuple[int, int, int, int]]:
    """Squares (x0, y0, x1, y1, page px, ends exclusive) of side 2 x period at a stride of one period (rounded to whole
    px), row by row: two periods across is the least that shows a period, and every px but the last stride at the
    page's right and bottom edge lies in four."""
    s = max(1, int(round(period)))
    h, w = shape
    return [(x0, y0, x0 + 2 * s, y0 + 2 * s) for y0 in range(0, h - 2 * s + 1, s) for x0 in range(0, w - 2 * s + 1, s)]


# --- seams ----------------------------------------------------------------------------------------------------------

@dataclass
class Above:
    """A sheet drawn above another: its placement, its neatline crop and that crop's bounds in world ft."""
    transform: object
    crop: np.ndarray
    bbox: tuple[float, float, float, float]

    @classmethod
    def of(cls, transform, crop: np.ndarray) -> "Above":
        return cls(transform, crop, crop_bbox(transform, crop))


def crop_bbox(transform, crop: np.ndarray) -> tuple[float, float, float, float]:
    """The world bounds (ft) of a crop: its outline's px carried through the placement."""
    cs, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return (np.inf, np.inf, -np.inf, -np.inf)
    w = np.asarray(transform.to_world(np.concatenate([c.reshape(-1, 2) for c in cs]).astype(float)))
    return (float(w[:, 0].min()), float(w[:, 1].min()), float(w[:, 0].max()), float(w[:, 1].max()))


def _in_crop(crop: np.ndarray, xy: np.ndarray) -> np.ndarray:
    ok = np.isfinite(xy).all(axis=1)
    xi = np.round(np.where(ok, xy[:, 0], -1)).astype(np.int64)
    yi = np.round(np.where(ok, xy[:, 1], -1)).astype(np.int64)
    ok &= (xi >= 0) & (xi < crop.shape[1]) & (yi >= 0) & (yi < crop.shape[0])
    out = np.zeros(len(xy), bool)
    out[ok] = crop[yi[ok], xi[ok]]
    return out


def owned(centres_px: np.ndarray, centres_world: np.ndarray, crop: np.ndarray, above: list[Above]) -> np.ndarray:
    """Which patches a sheet writes: those whose centre lies inside its neatline (`crop`, at the centre's nearest px)
    and inside no neatline of a sheet drawn above it (`above`), so each place is measured once."""
    ok = _in_crop(crop, np.asarray(centres_px, float))
    cw = np.asarray(centres_world, float)
    for t in above:
        x0, y0, x1, y1 = t.bbox
        cand = np.flatnonzero(ok & (cw[:, 0] >= x0) & (cw[:, 0] <= x1) & (cw[:, 1] >= y0) & (cw[:, 1] <= y1))
        if not len(cand):
            continue
        ok[cand[_in_crop(t.crop, np.asarray(t.transform.grid_to_pixel(cw[cand], t.crop.shape)))]] = False
    return ok


# --- blocks ---------------------------------------------------------------------------------------------------------

def block_pieces(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Physical blocks from the solver's lot table (`block_num`, `geometry` WKB; row = lot index): each block number's
    lots joined across gaps narrower than one px of the solver's grid (it can't show them), split into its pieces,
    largest first; a lot belongs to the piece holding its representative point.  Returns the pieces (id
    "<block>.<j>", axis_world: the major axis of the minimum rotated rectangle, radians from world east;
    elongation: long side / short side) and each lot's piece (-1: no geometry)."""
    geoms = shapely.from_wkb(table["geometry"].to_numpy()) if len(table) and isinstance(
        table["geometry"].iloc[0], (bytes, bytearray)) else np.asarray(table["geometry"].to_numpy())
    good = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    half = grid.FT_PER_PX / 2
    piece_of = np.full(len(table), -1, np.int64)
    ids, axes, elongs = [], [], []
    blocks = table["block_num"].astype(str).to_numpy()
    order = np.argsort(blocks, kind="stable")
    starts = np.flatnonzero(np.r_[True, blocks[order][1:] != blocks[order][:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        idx = order[a:b]
        idx = idx[good[idx]]
        if not len(idx):
            continue
        g = geoms[idx]
        joined = shapely.buffer(shapely.union_all(shapely.buffer(g, half)), -half)
        parts = [p for p in shapely.get_parts(joined) if p.geom_type == "Polygon" and not p.is_empty]
        parts.sort(key=lambda p: (-p.area, p.bounds))
        pts = shapely.point_on_surface(g)
        px, py = shapely.get_x(pts), shapely.get_y(pts)
        mine = np.full(len(idx), -1, np.int64)
        for j, p in enumerate(parts):
            hit = (mine < 0) & shapely.contains_xy(p, px, py)
            mine[hit] = len(ids) + j
        if (mine < 0).any() and parts:                  # a point on a joined piece's edge: its nearest piece
            far = np.flatnonzero(mine < 0)
            dist = np.array([[p.distance(q) for p in parts] for q in pts[far]])
            mine[far] = len(ids) + np.argmin(dist, axis=1)
        piece_of[idx] = mine
        for j, p in enumerate(parts):
            with np.errstate(divide="ignore", invalid="ignore"):     # GEOS divides by an axis-aligned edge's 0 dx
                r = np.asarray(shapely.minimum_rotated_rectangle(p).exterior.coords)[:4]
            e1, e2 = r[1] - r[0], r[2] - r[1]
            la, lb = np.hypot(*e1), np.hypot(*e2)
            major = e1 if la >= lb else e2
            ids.append(f"{blocks[idx[0]]}.{j}")
            axes.append(float(np.arctan2(major[1], major[0])))
            elongs.append(float(max(la, lb) / max(min(la, lb), 1e-6)))
    return pd.DataFrame({"block": ids, "axis_world": axes, "elongation": elongs}), piece_of


def page_axes(pw: PageWorld, centres_px: np.ndarray, axis_world: np.ndarray) -> np.ndarray:
    """World directions (radians from east) carried to page px (radians, x right, y down) at page points, through the
    placement's local linear map there (one px steps)."""
    c = np.asarray(centres_px, float)
    w0, wx, wy = pw.at(c), pw.at(c + [1.0, 0.0]), pw.at(c + [0.0, 1.0])
    a, b = wx - w0, wy - w0                              # d world / d x, d world / d y
    u = np.c_[np.cos(axis_world), np.sin(axis_world)]
    det = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    vx = (b[:, 1] * u[:, 0] - b[:, 0] * u[:, 1]) / det
    vy = (-a[:, 1] * u[:, 0] + a[:, 0] * u[:, 1]) / det
    return np.arctan2(vy, vx)


# --- the run --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sources:
    """Where the run reads: the pages, reads and printed blocks (work), the placements (warp), the solver's 1921 page
    and lots (solver), and the legend's swatch boxes."""
    work: Path
    warp: Path
    solver: Path
    legends: Path


def sources() -> Sources:
    return Sources(Path(WORK), Path(WARP), Path(SOLVER_1921), Path(LEGENDS_FILE))


def sheet_names(src: Sources) -> list[str]:
    """The sheets with a page, the pattern reader's reads, printed blocks, a placement and a neatline crop."""
    frames = json.loads((src.solver / "page" / "frames.json").read_text())
    out = []
    for d in sorted(p for p in src.work.iterdir() if p.is_dir()):
        if all((d / f).is_file() for f in ("page.png", "district_pattern.json", "blocks_px.geojson")) and \
                (src.warp / d.name / "transform.json").is_file() and d.name in frames:
            out.append(d.name)
    return out


def draw_order(sheets: list[str]) -> list[str]:
    """The sheets bottom first, as the published map draws them (`stitch.draw_order`); any it doesn't rank at the
    bottom, by name."""
    from sfzmaps import stitch
    ranked = [s for s in stitch.draw_order(SERIES, stitch.load_scores(SERIES)) if s in sheets]
    return sorted(set(sheets) - set(ranked)) + ranked


def load_page(src: Sources, sheet: str) -> np.ndarray:
    f = src.work / sheet / "page.png"
    pg = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if pg is None:
        raise FileNotFoundError(f"can't read {f}")
    return (pg < INK_BELOW).astype(np.uint8)


def load_transform(src: Sources, sheet: str) -> SheetTransform:
    folder = src.warp / sheet
    return SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)


def load_crop(src: Sources, sheet: str, shape, quad) -> np.ndarray:
    """The sheet's neatline interior less its notes, as the merged page crops it (page/crops/<sheet>.png); the quad of
    frames.json when there is no crop file."""
    f = src.solver / "page" / "crops" / f"{sheet}.png"
    if f.is_file():
        crop = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0
        if crop.shape != tuple(shape):
            raise ValueError(f"{f} is {crop.shape}, its page {tuple(shape)}")
        return crop
    return page.crop_mask(tuple(shape), np.asarray(quad, float), [])


def printed_rings(src: Sources, sheet: str) -> list[np.ndarray]:
    """The printed block outlines the line work traced (blocks_px.geojson), page px."""
    fc = json.loads((src.work / sheet / "blocks_px.geojson").read_text())
    rings = []
    for f in fc["features"]:
        geom = shapely.geometry.shape(f["geometry"])
        for p in shapely.get_parts(geom):
            if p.geom_type == "Polygon" and not p.is_empty:
                rings.append(np.asarray(p.exterior.coords)[:, :2])
    return rings


def reads_1r(src: Sources, sheet: str) -> set[str]:
    """The blklots the pattern reader read 1R on this sheet (training labels: used only to measure the period)."""
    out = set()
    for p in json.loads((src.work / sheet / "district_pattern.json").read_text())["parcels"]:
        code = p.get("code")
        if isinstance(code, str) and code.strip() == "1R" and p.get("via") not in SKIP_VIA:
            out.add(str(p["blklot"]))
    return out


def legend_scale(src: Sources, frames: dict) -> dict:
    """The legend's measures (`LEGENDS_FILE`'s swatch boxes on its sheet): its sheet's outline stroke, each hatched
    swatch's period and the angle of its strongest wave (`hatch_peak` from Nyquist, inside the box shrunk by the
    stroke), the finest of those periods and the hatch band's low end (half of it), and 1R's line angle on the sheet
    upright: the orientation of the 1R swatch's m2 over the band from that low end to twice its own period (the
    swatch is too small, about 36 x 16 px inside its taper, for its single strongest bin to place the angle: that
    bin says -41 degrees where m2 and every sheet's 1R lots say -44 to -47)."""
    leg = json.loads(src.legends.read_text())[SERIES]
    sheet = leg["sheet"]
    ink = load_page(src, sheet)
    stroke = stroke_width(ink, printed_rings(src, sheet))
    swatches, regions = {}, {}
    for code in LEGEND_HATCHED:
        if code not in leg["swatches"]:
            continue
        x0, y0, x1, y1 = leg["swatches"][code]
        regions[code] = ink[y0 + stroke:y1 - stroke, x0 + stroke:x1 - stroke].astype(np.float64)
        swatches[code] = hatch_peak(regions[code], np.ones(regions[code].shape, bool), taper=stroke)
    finest = min(v["period"] for v in swatches.values())
    turn = paper_turn(frames[sheet])
    one = regions["1R"]
    sp = spectrum(one, np.ones(one.shape, bool), Frame(period=swatches["1R"]["period"], low=finest / 2, taper=stroke))
    angle = float(np.degrees(np.angle(sp["m2"]) / 2)) + 90.0 * turn
    return {"sheet": sheet, "stroke": stroke, "turn": turn, "swatches": swatches, "finest": finest,
            "low": finest / 2, "line_angle_1r": (angle + 90) % 180 - 90, "m2_1r": abs(sp["m2"]),
            "boxes": leg["swatches"]}


def _ring_points(r) -> np.ndarray | None:
    r = np.asarray(r, float)[:, :2]
    if len(r) < 3 or not np.isfinite(r).all():
        return None
    return np.round(r * 16).astype(np.int32)


def _paper(ink: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """The scan's paper with the edges of the white regions the line work traced (`rings`, page px) drawn in as ink:
    the line work closes the gaps in a block's outline.  8-connected edges, which 4-connected paper can't cross."""
    barrier = (np.asarray(ink) > 0).astype(np.uint8)
    pts = [p for p in (_ring_points(r) for r in rings) if p is not None]
    if pts:
        cv2.polylines(barrier, pts, True, 1, 1, lineType=cv2.LINE_8, shift=4)
    return barrier == 0


def paper_regions(paper: np.ndarray, stroke: int) -> np.ndarray:
    """Paper split at its necks (int32 labels, 0 off the paper).  The cores are the paper more than one stroke from
    any ink; a core's region is the paper within one stroke of it (the morphological opening: what a disk of one
    stroke's radius sweeps from the core, which can't pass a neck narrower than two strokes nor cross a line), so a
    block and the street a gap in its outline opens onto are two regions.  The paper no core reaches (a strip or a
    cell thinner than two strokes, the gap itself, a tight corner) makes regions of its own, by its connections."""
    dist = cv2.distanceTransform(paper.astype(np.uint8), cv2.DIST_L2, 5)
    core = dist > stroke
    regions = np.zeros(paper.shape, np.int32)
    if core.any():
        near, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_CCOMP)
        swept = paper & (near <= stroke)
        regions[swept] = lab[swept]
    rest = paper & (regions == 0)
    _, other = cv2.connectedComponents(rest.astype(np.uint8), connectivity=4)
    regions[rest] = other[rest] + int(regions.max())
    return regions


def _mostly_on(labels: np.ndarray, on: np.ndarray) -> np.ndarray:
    """For each label, whether most of its px lie on `on` (label 0 never)."""
    n = int(labels.max()) + 1
    total = np.bincount(labels.ravel(), minlength=n)
    inside = np.bincount(labels[on], minlength=n)
    owned = inside > total - inside
    owned[0] = False
    return owned


def block_paper(ink: np.ndarray, rings: list[np.ndarray], in_blocks: np.ndarray, stroke: int) -> np.ndarray:
    """The paper inside the printed blocks: each connected region of paper (4-neighbours, `_paper`: a blank block,
    one strip between two hatch lines, one cell of a cross-hatch, a street) lying mostly on the solver's blocks
    (`in_blocks`), less the parts of it, split at its necks (`paper_regions`), that lie mostly off them.  A gap in an
    outline the line work didn't close joins a block to the street; where that makes one region lying mostly on the
    blocks, the split hands the street back.  A street network, lying mostly off the blocks, stays one region and
    unowned whatever parts of it today's lots cover (splitting it too would own every stretch of a 1921 street the
    lots now overlay).  The scan's own regions hold the strips the line work left untraced."""
    paper = _paper(ink, rings)
    on = np.asarray(in_blocks, bool)
    _, comps = cv2.connectedComponents(paper.astype(np.uint8), connectivity=4)
    regions = paper_regions(paper, stroke)
    return _mostly_on(comps, on)[comps] & _mostly_on(regions, on)[regions]


def outline_mask(ink: np.ndarray, rings: list[np.ndarray], stroke: int, in_blocks: np.ndarray) -> np.ndarray:
    """The printed block outlines' ink, and only that: the band one stroke wide outside the edge of the blocks' paper
    (`block_paper`) joined across one line of ink (closed by one stroke, so the strips on either side of a hatch
    line, or the cells of a cross-hatch, make one block) with its holes filled.  The lines inside a block stay.  Two
    blocks don't join across a street: it is at least two outlines and some paper wide, more than one closing
    bridges.  Only outside: the paper's edge is the scan's own, so the outline starts there; a band one stroke inside
    too took the hatch lines' ends, and scored lower on every one of the five 1R-against-2R AUCs over all sheets."""
    s = int(stroke)
    paper = block_paper(ink, rings, in_blocks, s).astype(np.uint8)
    joined = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, disk(s))
    filled = ndimage.binary_fill_holes(joined > 0).astype(np.uint8)
    # one stroke out in every direction (a square: a pen-drawn outline's outer corner lies up to s * sqrt 2 from the
    # paper's corner; on a diagonal outline the square reaches about a px past it, onto the street's paper)
    return (cv2.dilate(filled, np.ones((2 * s + 1, 2 * s + 1), np.uint8)) > 0) & ~(filled > 0)


def inblock_mask(in_blocks: np.ndarray, ink: np.ndarray, rings: list[np.ndarray], stroke: int,
                 crop: np.ndarray) -> np.ndarray:
    """The page px a patch measures: in a block by the solver's lots (`in_blocks`, eroded by the stroke), inside the
    neatline (`crop`) and off the printed block outlines found on the scan itself (`outline_mask`), so the outlines
    stay out however far the lots sit from the print, and a block's hatch stays in."""
    eroded = cv2.erode(np.asarray(in_blocks, np.uint8), disk(int(stroke))).astype(bool)
    return eroded & np.asarray(crop, bool) & ~outline_mask(ink, rings, stroke, in_blocks)


def _setup(src: Sources, sheet: str, quad, lots: np.ndarray, blocks: np.ndarray, offset) -> dict:
    """A sheet's page, placement, neatline, lot of every px, outline stroke and in-block mask (`inblock_mask`: the page
    px whose grid px is in a block, `blocks` being the lots with their enclosed holes)."""
    ink = load_page(src, sheet)
    tr = load_transform(src, sheet)
    crop = load_crop(src, sheet, ink.shape, quad)
    pw = PageWorld(tr, ink.shape)
    lot_px, in_blocks = page_values(pw, ink.shape, [(lots, -1), (blocks.astype(np.uint8), 0)], offset)
    rings = printed_rings(src, sheet)
    stroke = stroke_width(ink, rings)
    inblock = inblock_mask(in_blocks, ink, rings, stroke, crop)
    return {"ink": ink, "tr": tr, "crop": crop, "pw": pw, "lot_px": lot_px, "stroke": stroke, "inblock": inblock,
            "turn": paper_turn(quad)}


def _init_worker():
    cv2.setNumThreads(1)


def scale_job(job) -> dict:
    """Pass 1, one sheet: its stroke, its 1R period (`sheet_period`), its turn, ft per px and the world bounds of its
    neatline."""
    sheet, src, quad, lots, blocks, offset, is1r, low, line_angle_1r = job
    s = _setup(src, sheet, quad, lots, blocks, offset)
    lot_px = s["lot_px"]
    mask = s["inblock"] & (lot_px >= 0)
    mask[mask] = is1r[lot_px[mask]]
    expected = line_angle_1r - 90.0 * s["turn"]
    expected = (expected + 90) % 180 - 90
    pk = hatch_peak(s["ink"], mask, low, s["stroke"])
    return {"sheet": sheet, "stroke": s["stroke"], "turn": s["turn"], "px_1r": int(mask.sum()),
            "peak": pk, "expected_line_angle_1r": expected, "period_measured": accept_period(pk, expected),
            "ft_per_px": float(s["tr"].median_scale_ft_per_px(s["ink"].shape)),
            "bbox": crop_bbox(s["tr"], s["crop"]), "shape": list(s["ink"].shape)}


def patch_job(job) -> dict:
    """Pass 2, one sheet: every patch it owns (`owned`), measured (`measure_patch`) with its lots' shares
    (`patch_lots`); writes patches/<sheet>.parquet and patch_lots/<sheet>.parquet."""
    (sheet, src, quad, lots, blocks, offset, scale, period, low, line_angle_1r, pieces, piece_of, above,
     out) = job
    s = _setup(src, sheet, quad, lots, blocks, offset)
    ink, pw, lot_px, inblock, turn = s["ink"], s["pw"], s["lot_px"], s["inblock"], s["turn"]
    boxes = np.array(patch_grid(ink.shape, period), np.int64).reshape(-1, 4)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2.0 - 0.5          # px centres sit at integers
    world = pw.at(centres)
    tops = [Above.of(load_transform(src, a), load_crop(src, a, shp, q)) for a, q, shp in above]
    mine = owned(centres, world, s["crop"], tops)
    wave_1r = np.radians(line_angle_1r - 90.0 * turn) + np.pi / 2
    stack = morph_stack(ink)
    rows, lrows = [], []
    keep = [k for k in np.flatnonzero(mine) if inblock[boxes[k, 1]:boxes[k, 3], boxes[k, 0]:boxes[k, 2]].any()]
    shares = {}
    piece = np.full(len(boxes), -1, np.int64)
    for k in keep:
        x0, y0, x1, y1 = boxes[k]
        sh = _shares(lot_px[y0:y1, x0:x1], inblock[y0:y1, x0:x1])
        shares[k] = sh
        if sh:
            ps = piece_of[[lot for lot, _ in sh]]
            w = np.array([v for _, v in sh])
            have = ps >= 0
            if have.any():
                u, inv = np.unique(ps[have], return_inverse=True)
                piece[k] = u[np.argmax(np.bincount(inv, w[have]))]
    keep = np.array(keep, np.int64)
    phi = np.full(len(boxes), np.nan)
    has = keep[piece[keep] >= 0] if len(keep) else keep
    if len(has):
        phi[has] = page_axes(pw, centres[has], pieces["axis_world"].to_numpy()[piece[has]])
    for k in keep:
        x0, y0, x1, y1 = (int(v) for v in boxes[k])
        frame = Frame(period=period, low=low, taper=s["stroke"], wave_1r=wave_1r, turn=turn, phi=float(phi[k]))
        p = int(piece[k])
        row = {"sheet": sheet, "patch": int(k), "x0": x0, "y0": y0, "side": x1 - x0,
               "cx": float(world[k, 0]), "cy": float(world[k, 1]),
               "block": pieces["block"].iat[p] if p >= 0 else "",
               "block_phi": float(phi[k]), "block_elongation": float(pieces["elongation"].iat[p]) if p >= 0 else np.nan,
               "period": float(period), "ft_per_px": float(scale["ft_per_px"])}
        row.update(measure_patch(stack, inblock, (x0, y0, x1, y1), frame))
        rows.append(row)
        lrows += [(int(k), lot, share) for lot, share in shares[k]]
    df = pd.DataFrame(rows) if rows else _no_patches()
    pl = pd.DataFrame(lrows, columns=["patch", "lot", "share"]).astype({"patch": np.int64, "lot": np.int64,
                                                                        "share": np.float64})
    for name, d in (("patches", df.sort_values("patch", kind="stable")),
                    ("patch_lots", pl.sort_values(["patch", "lot"], kind="stable"))):
        (out / name).mkdir(parents=True, exist_ok=True)
        d.reset_index(drop=True).to_parquet(out / name / f"{sheet}.parquet", index=False)
    return {"sheet": sheet, "lattice": int(len(boxes)), "owned": int(mine.sum()), "patches": int(len(rows)),
            "patch_lots": int(len(pl))}


def _no_patches() -> pd.DataFrame:
    """The patches table of a sheet that owns none: no rows, the columns and types of one that does."""
    row = {"sheet": "", "patch": 0, "x0": 0, "y0": 0, "side": 0, "cx": 0.0, "cy": 0.0, "block": "",
           "block_phi": 0.0, "block_elongation": 0.0, "period": 0.0, "ft_per_px": 0.0}
    row.update(measure_patch(np.zeros((8, 8), np.uint8), np.ones((8, 8), bool), (0, 0, 8, 8), Frame(period=4.0)))
    return pd.DataFrame([row]).astype({k: np.float64 for k, v in row.items() if isinstance(v, float)}).iloc[:0]


def measure_swatches(src: Sources, legend: dict, period: float) -> pd.DataFrame:
    """The legend's swatches measured as patches (the box shrunk by the stroke), each in its own frame: its box's
    major axis horizontal (phi 0), the legend sheet's period, band and 1R direction."""
    sheet = legend["sheet"]
    stack = morph_stack(load_page(src, sheet))
    wave = np.radians(legend["line_angle_1r"] - 90.0 * legend["turn"]) + np.pi / 2
    frame = Frame(period=period, low=legend["low"], taper=legend["stroke"], wave_1r=wave, turn=legend["turn"],
                  phi=0.0)
    rows = []
    st = legend["stroke"]
    for code, (x0, y0, x1, y1) in sorted(legend["boxes"].items()):
        box = (x0 + st, y0 + st, x1 - st, y1 - st)
        rows.append({"code": code, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                     "elongation": (box[2] - box[0]) / (box[3] - box[1]),
                     **measure_patch(stack, np.ones(stack.shape[:2], bool), box, frame)})
    return pd.DataFrame(rows)


def _map(fn, jobs, workers: int):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
        return list(pool.imap_unordered(fn, jobs))


def _slice(tr, quad, shape) -> tuple[int, int, int, int]:
    return page.grid_box(tr, np.asarray(quad, float), shape)


def run(out: Path, workers: int, only: list[str] | None = None) -> dict:
    """Measure every 1921 sheet (or `only` those, still taking every sheet's period and seams into account); see the
    module."""
    src = sources()
    names = sheet_names(src)
    if not names:
        raise FileNotFoundError(f"no 1921 sheets with pages, reads, printed blocks and placements under {src.work}")
    frames = json.loads((src.solver / "page" / "frames.json").read_text())
    order = draw_order(names)
    with np.load(src.solver / "lots" / "lots.npz") as z:
        lots = z["lot"]
    table = pd.read_parquet(src.solver / "lots" / "table.parquet", columns=["blklot", "block_num", "geometry"])
    blocks = ndimage.binary_fill_holes(lots >= 0)
    pieces, piece_of = block_pieces(table)
    legend = legend_scale(src, frames)
    blklots = table["blklot"].astype(str).to_numpy()
    boxes = {}
    for s in names:
        boxes[s] = _slice(load_transform(src, s), frames[s], lots.shape)

    def cut(s):
        r0, r1, c0, c1 = boxes[s]
        return lots[r0:r1, c0:c1], blocks[r0:r1, c0:c1], (r0, c0)

    jobs = []
    for s in names:
        lot_s, blk_s, off = cut(s)
        jobs.append((s, src, frames[s], lot_s, blk_s, off, np.isin(blklots, list(reads_1r(src, s))), legend["low"],
                     legend["line_angle_1r"]))
    scales = {r["sheet"]: r for r in _map(scale_job, jobs, workers)}
    measured = {s: (v["period_measured"], v["ft_per_px"]) for s, v in scales.items()
                if np.isfinite(v["period_measured"])}
    if not measured:
        raise ValueError("no 1921 sheet's 1R lots show a 1R hatch: no period to measure the patches by")
    periods, sources_ = {}, {}
    for s, v in scales.items():
        if s in measured:
            periods[s], sources_[s] = measured[s][0], "measured"
        else:
            periods[s], sources_[s] = fallback_period(s, v["ft_per_px"], measured, legend["sheet"],
                                                      legend["swatches"]["1R"]["period"])
    todo = [s for s in names if not only or s in only]
    jobs = []
    for s in todo:
        lot_s, blk_s, off = cut(s)
        above = []
        sb = scales[s]["bbox"]
        for a in order[order.index(s) + 1:]:
            ab = scales[a]["bbox"]
            if ab[0] <= sb[2] and ab[2] >= sb[0] and ab[1] <= sb[3] and ab[3] >= sb[1]:
                above.append((a, frames[a], tuple(scales[a]["shape"])))
        jobs.append((s, src, frames[s], lot_s, blk_s, off, scales[s], periods[s], legend["low"],
                     legend["line_angle_1r"], pieces, piece_of, above, out))
    jobs.sort(key=lambda j: -int(np.prod(scales[j[0]]["shape"])))
    out.mkdir(parents=True, exist_ok=True)
    counts = {r["sheet"]: r for r in _map(patch_job, jobs, workers)}
    measure_swatches(src, legend, periods[legend["sheet"]]).to_parquet(out / "swatches.parquet", index=False)
    summary = {"legend": {k: v for k, v in legend.items() if k != "boxes"}, "draw_order": order, "sheets": {}}
    for s in names:
        v = scales[s]
        summary["sheets"][s] = {
            "period": periods[s], "period_measured": s in measured, "period_source": sources_[s],
            "peak": v["peak"], "expected_line_angle_1r": v["expected_line_angle_1r"], "px_1r": v["px_1r"],
            "stroke": v["stroke"], "turn": v["turn"], "ft_per_px": v["ft_per_px"],
            "side": 2 * max(1, round(periods[s])),
            **{k: counts[s][k] for k in ("lattice", "owned", "patches", "patch_lots") if s in counts}}
    (out / "sheets.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=_json) + "\n")
    return summary


def _json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    raise TypeError(type(v))


# --- the look picture -----------------------------------------------------------------------------------------------

def look(out: Path, sheets: list[str], dest: Path) -> Path:
    """For checking the periods and angles by eye: each sheet's scan with its patches (centre squares of one stride)
    coloured by their strongest line orientation in the page frame (hue: the angle mod 180; brightness: |m2|, how
    much one family of lines there is) and a tick along that orientation, with the sheet's period."""
    src = sources()
    meta = json.loads((out / "sheets.json").read_text())["sheets"]
    panels = []
    for sheet in sheets:
        ink = load_page(src, sheet)
        df = pd.read_parquet(out / "patches" / f"{sheet}.parquet")
        turn = meta[sheet]["turn"]
        img = np.where(ink > 0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        over = img.copy()
        s = max(1, round(meta[sheet]["period"]))
        strength = np.hypot(df["m2_re"], df["m2_im"]).to_numpy()
        top = np.nanquantile(strength, 0.95) if np.isfinite(strength).any() else 1.0
        for r, m in zip(df.itertuples(), strength):
            if not np.isfinite(r.line_angle):
                continue
            a = np.radians(r.line_angle - 90.0 * turn)                       # back to the page
            hue = int(((np.degrees(a) + 90) % 180))                          # OpenCV hue runs 0-179
            val = int(np.clip(m / top, 0, 1) * 255) if top > 0 else 0
            bgr = cv2.cvtColor(np.uint8([[[hue, 255, max(val, 60)]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cx, cy = r.x0 + r.side // 2, r.y0 + r.side // 2
            cv2.rectangle(over, (cx - s // 2, cy - s // 2), (cx + s - s // 2 - 1, cy + s - s // 2 - 1), bgr, -1)
        img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
        for r, m in zip(df.itertuples(), strength):
            if not np.isfinite(r.line_angle) or not (m >= top / 4):
                continue
            a = np.radians(r.line_angle - 90.0 * turn)
            cx, cy = r.x0 + r.side // 2, r.y0 + r.side // 2
            d = np.array([np.cos(a), np.sin(a)]) * s * 0.45
            cv2.line(img, (int(cx - d[0]), int(cy - d[1])), (int(cx + d[0]), int(cy + d[1])), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{sheet}  period {meta[sheet]['period']:.2f} px  "
                         f"({meta[sheet]['period_source']})  "
                         f"patch {2 * s} px  stroke {meta[sheet]['stroke']} px", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3, cv2.LINE_AA)
        panels.append(img)
    h = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, h - p.shape[0]), (0, 20), (0, 0)), constant_values=255) for p in panels]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.hstack(panels))
    return dest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", choices=[SERIES], default=SERIES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sheets", nargs="*", help="measure only these sheets (periods and seams still use them all)")
    ap.add_argument("--out", type=Path, default=EXPERIMENT)
    ap.add_argument("--look", nargs="+", metavar="SHEET", help="draw the look picture of these measured sheets")
    a = ap.parse_args(argv)
    if a.look:
        print(look(a.out, a.look, a.out / "look_patches.png"))
        return
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")                 # the workers start with one maths thread each
    summary = run(a.out, a.workers, a.sheets)
    for s, v in summary["sheets"].items():
        print(f"{s}: period {v['period']:.2f} px ({v['period_source']}), stroke "
              f"{v['stroke']}, turn {v['turn']}, {v.get('patches', 0)} patches", flush=True)


if __name__ == "__main__":
    main()
