"""What the sheets need before a detector trains by the label rule "lines-unknown" (`data.crop_targets(...,
labels="lines-unknown")`, `train.Recipe.labels`), made once, before the GPU's turn (`prepare`):

  zones      only the label points of Sol's that no Terra word of the same sheet holds, or lies within half that
             word's height of (`matched_points`), keep their unknown disc: around a matched point the word is taught
             already.  Matching is sheet by sheet: a word on the next sheet across a seam matches nothing here.  The
             discs' radius (`zone_radius`) is half the 99th percentile of the code words' lengths (Terra words of
             kind "zoning code", each its rectangle's first edge, as `reader_stage.measure_radius` measures them: the
             words whose centre is in an area held out left out), one for every map: a point on its word's middle
             reaches the ends of nearly every code word.  `data._unknown` draws them as before.
  line px    the run rule's line ink (`linerules.line_px` at the sheet's scale) grown by GROW_PX px (the 8 around each
             px), packed as the ink (`data.SheetData.line_bits`): inside a word they are unknown, a lesson neither of
             text nor of not-text.  Kept on disk per sheet (`<cache>/<series>/<sheet>.npz`, LINES_DIR for the
             experiments), keyed by the hash of linerules.py, GROW_PX, the run rule's length and the hash of the
             sheet's ink: found again when any of them differs.

`label_shares` gives, per map and kind, the share of the ink inside word boxes the rule makes unknown; `write_rule`
writes it with the radius and the points' counts (`python -m sfzmaps.solver.textnet.zones` for the training maps, to
RULE_FILE).

No stage imports this module (`data` and `train` don't), so no stage's import closure changes with it.
"""
import dataclasses
import hashlib
import json
import logging
import math
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import data, linerules

log = logging.getLogger(__name__)

TEXTLINES_DIR = paths.SOLVER_DIR / "experiments" / "textlines"
LINES_DIR = TEXTLINES_DIR / "lines"           # the sheets' line px, kept for the experiments (never shipped)
RULE_FILE = TEXTLINES_DIR / "labels_rule.json"
GROW_PX = 1                # line ink is grown this many px (8-neighbours) before it is unknown inside a word
RADIUS_QUANTILE = 0.99     # of the code words' lengths: the zone radius is half of it
POINT_CHUNK = 256          # Sol's points matched against a sheet's words at a time
THREADS = 4                # sheets whose line px `write_rule`'s command finds at once


# --- the zones -------------------------------------------------------------------------------------------------------

def matched_points(sd: data.SheetData) -> np.ndarray:
    """Which of Sol's points on the sheet (bool, one per `sd.points`) a Terra word of the same sheet holds or lies
    within half that word's height of: the distance from the point to the word's rectangle (its corners' mean, its
    first edge the length, its second the height; 0 inside) is at most half the height."""
    pts = sd.points.astype(float)
    out = np.zeros(len(pts), bool)
    if not len(pts) or not len(sd.words):
        return out
    r = sd.words.astype(float)
    centre = r.mean(axis=1)
    along, across = r[:, 1] - r[:, 0], r[:, 2] - r[:, 1]
    length = np.maximum(np.hypot(*along.T), 1e-9)
    height = np.maximum(np.hypot(*across.T), 1e-9)
    v, n = along / length[:, None], across / height[:, None]
    for i in range(0, len(pts), POINT_CHUNK):
        d = pts[i:i + POINT_CHUNK, None, :] - centre[None]                    # (points, words, 2)
        u = np.maximum(np.abs((d * v).sum(-1)) - length / 2, 0)
        w = np.maximum(np.abs((d * n).sum(-1)) - height / 2, 0)
        out[i:i + POINT_CHUNK] = (np.hypot(u, w) <= height / 2).any(axis=1)
    return out


def zone_radius(sheets: list[data.SheetData], areas=()) -> dict:
    """The zones' radius (see the module) from `sheets`, the code words whose centre px (their rectangle's corners'
    mean, rounded, kept on the page) is in one of `areas` left out: `px` half the RADIUS_QUANTILE quantile of the
    lengths, rounded; `q50`, `q90`, `q99` the lengths' quantiles (px, linear); `words` the code words measured.
    ValueError when there is none."""
    code = data.KINDS["zoning code"]
    lengths = []
    for sd in sheets:
        for j in np.flatnonzero(sd.kinds == code):
            rect = sd.words[j].astype(float)
            x, y = np.clip(np.rint(rect.mean(axis=0)).astype(int), 0, [sd.width - 1, sd.height - 1])
            if any(data.in_area(sd, a, int(y), int(x), 1, 1)[0, 0] for a in areas):
                continue
            lengths.append(float(np.hypot(*(rect[1] - rect[0]))))
    if not lengths:
        raise ValueError("no Terra word of kind zoning code to measure the zone radius by")
    q50, q90, q99 = (float(np.quantile(lengths, q)) for q in (0.5, 0.9, RADIUS_QUANTILE))
    return {"rule": "q99/2", "px": int(round(q99 / 2)), "q50": round(q50, 3), "q90": round(q90, 3),
            "q99": round(q99, 3), "words": len(lengths)}


# --- the line px -----------------------------------------------------------------------------------------------------

def rule_key() -> str:
    """The hash of linerules.py (sha256): the line px kept on disk are found again when it changes."""
    return hashlib.sha256(Path(linerules.__file__).read_bytes()).hexdigest()


def _scale(sd: data.SheetData) -> float:
    if not (math.isfinite(sd.ft_per_px) and sd.ft_per_px > 0):
        raise ValueError(f"{sd.series} {sd.sheet} was read without its placement: its scale, which the run rule's "
                         "length is in, is not known (read it with placement=True)")
    return sd.ft_per_px


def find_lines(sd: data.SheetData) -> np.ndarray:
    """A sheet's line px (see the module), packed as its ink."""
    lines = linerules.line_px(data.page_ink(sd), _scale(sd))
    grown = cv2.dilate(lines.view(np.uint8), np.ones((2 * GROW_PX + 1, 2 * GROW_PX + 1), np.uint8))
    return np.packbits(grown > 0, axis=1)


def _key(sd: data.SheetData) -> dict:
    return {"rule": rule_key(), "grow": GROW_PX, "run_px": linerules.run_px(_scale(sd)),
            "ink": hashlib.sha256(np.ascontiguousarray(sd.ink_bits)).hexdigest()}


def _read(f: Path, key: dict, shape) -> np.ndarray | None:
    """The line px kept in `f` when they were found under `key` for a page of `shape` (packed), else None."""
    try:
        with np.load(f) as z:
            bits = z["line_bits"]
            if bits.shape == shape and bits.dtype == np.uint8 and all(str(z[k]) == str(v) for k, v in key.items()):
                return bits
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        pass
    return None


def _write(f: Path, bits: np.ndarray, key: dict) -> None:
    """Keep the line px in `f`, whole: written beside it first, then renamed into place."""
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as out:
        np.savez_compressed(out, line_bits=bits, **{k: np.array(v) for k, v in key.items()})
    os.replace(tmp, f)


def sheet_lines(sd: data.SheetData, cache: Path | None) -> np.ndarray:
    """A sheet's line px (`find_lines`), read from `cache` when kept there under the same key, else found (and kept
    there, unless `cache` is None)."""
    key = _key(sd)
    f = None if cache is None else Path(cache) / sd.series / f"{sd.sheet}.npz"
    if f is not None and f.is_file():
        bits = _read(f, key, sd.ink_bits.shape)
        if bits is not None:
            return bits
    t0 = time.perf_counter()
    bits = find_lines(sd)
    log.info("%s %s: line px found in %.1f s", sd.series, sd.sheet, time.perf_counter() - t0)
    if f is not None:
        _write(f, bits, key)
    return bits


def prepare(sheets: list[data.SheetData], areas, cache: Path | None, threads: int = 1) -> list[data.SheetData]:
    """The sheets as the label rule "lines-unknown" trains on them (see the module): each with only its unmatched
    points (`matched_points`), the zone radius of `sheets` with `areas` left out (`zone_radius`) and its line px
    (`sheet_lines`, kept in `cache`; `threads` sheets at a time).  Everything else is shared with `sheets`.  Run it
    before the GPU's turn."""
    radius = zone_radius(sheets, areas)["px"]
    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            lines = list(ex.map(lambda sd: sheet_lines(sd, cache), sheets))
    else:
        lines = [sheet_lines(sd, cache) for sd in sheets]
    return [dataclasses.replace(sd, points=sd.points[~matched_points(sd)], point_radius=radius, line_bits=b)
            for sd, b in zip(sheets, lines, strict=True)]


# --- what the rule leaves unknown ------------------------------------------------------------------------------------

def _share(ink: int, unknown: int) -> dict:
    return {"ink": int(ink), "unknown": int(unknown), "share": float(unknown / ink) if ink else None}


def label_shares(sheets: list[data.SheetData]) -> dict:
    """Per map (series) and kind (`data.KINDS`, and "all"): the ink px inside word boxes (each px the kind of the
    word drawn last on it, as the detector learns it), those of them the rule makes unknown (line px) and their
    share (None without ink).  ValueError for a sheet without its line px."""
    kinds = len(data.KINDS)
    counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sd in sheets:
        data.line_window(sd, 0, 0, 1, 1)                 # stops on a sheet without its line px
        ink_n, unknown_n = counts.setdefault(sd.series, (np.zeros(kinds, np.int64), np.zeros(kinds, np.int64)))
        for y0 in range(0, sd.height, data.BAND):
            h = min(data.BAND, sd.height - y0)
            word = data._words(sd, y0, 0, h, sd.width)
            in_box = data._ink_window(sd, y0, 0, h, sd.width) & (word > 0)
            unknown = in_box & data.line_window(sd, y0, 0, h, sd.width)
            ink_n += np.bincount(sd.kinds[word[in_box].astype(np.int64) - 1], minlength=kinds)
            unknown_n += np.bincount(sd.kinds[word[unknown].astype(np.int64) - 1], minlength=kinds)
    return {series: {**{name: _share(i[k], u[k]) for name, k in data.KINDS.items()}, "all": _share(i.sum(), u.sum())}
            for series, (i, u) in counts.items()}


def point_counts(sheets: list[data.SheetData]) -> dict:
    """Per map: Sol's points, those a Terra word matches (`matched_points`) and those keeping a zone."""
    out: dict[str, dict] = {}
    for sd in sheets:
        m = matched_points(sd)
        c = out.setdefault(sd.series, {"points": 0, "matched": 0, "zones": 0})
        c["points"] += len(m)
        c["matched"] += int(m.sum())
        c["zones"] += int((~m).sum())
    return out


def write_rule(sheets: list[data.SheetData], areas, out: Path = RULE_FILE, cache: Path | None = LINES_DIR,
               threads: int = 1, extra: dict | None = None) -> dict:
    """Prepare the sheets (`prepare`, timed) and write to `out`, whole (beside it first, then renamed into place):
    the zone radius, the share of in-box ink made unknown per map and kind (`label_shares`), the points' counts
    (`point_counts`), the line rule's key and the line px's bytes, with `extra` beside them.  Returns what it wrote."""
    t0 = time.perf_counter()
    prepared = prepare(sheets, areas, cache, threads)
    t1 = time.perf_counter()
    shares = label_shares(prepared)
    t2 = time.perf_counter()
    rule = {"zone_radius": zone_radius(sheets, areas), "label_shares": shares, "points": point_counts(sheets),
            "sheets": {s: sum(sd.series == s for sd in sheets) for s in dict.fromkeys(sd.series for sd in sheets)},
            "rule_key": rule_key(), "grow_px": GROW_PX, "line_bytes": int(sum(sd.line_bits.nbytes for sd in prepared)),
            "seconds": {"prepare": round(t1 - t0, 1), "label_shares": round(t2 - t1, 1)}, "threads": threads,
            **(extra or {})}
    _write_json(Path(out), rule)
    return rule


def _write_json(out: Path, obj) -> None:
    """Write `obj` to `out` as JSON, whole: beside it first, then renamed into place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, out)


def main() -> None:
    """The rule on the training maps' sheets (`paths.TEXT_TRAIN_SERIES`), the owner's answered blocks held out of the
    radius (as the held-out detectors' radius is measured), to RULE_FILE; the line px kept in LINES_DIR.  Beside it:
    the radius with each quarter held out too, and today's unknown radius (`reader_stage.measure_radius`)."""
    from sfzmaps.solver.textnet import reader_stage       # reads the labels and the lots: only here

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    blocks = reader_stage.load_block_area(reader_stage.held_blocks())
    loaded = time.perf_counter() - t0
    by_quarter = {name: zone_radius(sheets, (data.Quarter(q), blocks)) for q, name in enumerate(paths.QUARTERS)}
    today = reader_stage.measure_radius(sheets, (blocks,))
    extra = {"areas": "the answered blocks (reader_stage.held_blocks)", "zone_radius_by_quarter_held": by_quarter,
             "unknown_radius_today": today, "seconds_load": round(loaded, 1),
             "lines_kept_before": sum(1 for _ in LINES_DIR.glob("*/*.npz"))}
    rule = write_rule(sheets, (blocks,), RULE_FILE, LINES_DIR, THREADS, extra)
    t1 = time.perf_counter()
    prepare(sheets, (blocks,), LINES_DIR, 1)          # again, every sheet's line px read from LINES_DIR
    rule["seconds"]["prepare_kept"] = round(time.perf_counter() - t1, 1)
    _write_json(RULE_FILE, rule)
    log.info("wrote %s: zone radius %s px", RULE_FILE, rule["zone_radius"]["px"])


if __name__ == "__main__":
    main()
