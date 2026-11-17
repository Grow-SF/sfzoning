"""The composites' measures and checks on the training maps (`textnet.composite`), made before a composites run, on the
CPU, sheets 4 at a time, one thread each:

    uv run python scripts/textlines/composites.py

Writes to `data/solver/experiments/textlines/`:
  distributions.json  `composite.measure` of the 2000 sheets with each quarter and the owner's answered blocks held out
                      (`by_held_quarter`); at the top NE's, the quarter the composites run first (`cross_by_kind`,
                      `cross_by_kind_ink`, `clean_words`, `clean_word_texture`, `line_texture`); `p` per quarter, the
                      word-weighted share of words crossed over all kinds (the composites' share, `--composites`);
                      `banks` per quarter, each map's bank (`Bank.sizes`: clean words, pieces, words crossed by kind)
  composites.json     100 batches of `train.draw` (stage.DEFAULT_RECIPE with the label rule lines-unknown, seed 0)
                      timed at composites p and at 0, alone and with `batch_tensors` (the prefetch thread's work); what
                      the composites were (`Bank.tally`); the texture check: `texture_stats` of the text ink of
                      composites and of the real crops at the same centres (page scale, not turned), and of clean words
  composites.png      24 composites as the training sees them (12 word onto lines, 12 line onto a word): text red, other
                      ink grey, px with no lesson (unknown) tinted blue, the pasted word's frame or the clean word's box
                      green
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import dataclasses  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import composite, data, reader_stage, stage, train, zones  # noqa: E402

OUT = zones.TEXTLINES_DIR
THREADS = 4
FIRST = "ne"              # the quarter the composites run first: its numbers are the file's top level
BATCHES = 100             # batches timed
SEED = 0
PICTURES = 12             # composites of each kind in the picture
CHECKS = 150              # composites of each kind (and real crops) in the texture check
log = logging.getLogger("composites")


def _p(m: dict) -> float:
    return m["cross_by_kind"]["all"]["share"]


def _centre(pools, rng):
    """A crop's centre, scale and turns as `train.draw` draws them."""
    pool = pools[0] if rng.random() < train.TEXT_SHARE else pools[1]
    i, y, x = pool[rng.integers(len(pool))]
    scale = float(np.exp(rng.uniform(np.log(train.SCALE[0]), np.log(train.SCALE[1]))))
    return i, y, x, scale, int(rng.integers(4))


def time_draw(sheets, pools, recipe, bank) -> dict:
    """BATCHES batches of `train.draw` from `default_rng(SEED)`, timed (ms a batch), alone and with `batch_tensors`."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    batches = [train.draw(sheets, pools, recipe, rng, bank) for _ in range(BATCHES)]
    drawn = time.perf_counter() - t0
    t0 = time.perf_counter()
    for b in batches:
        train.batch_tensors(b, recipe.ink_only)
    tensors = time.perf_counter() - t0
    return {"composites": recipe.composites, "batches": BATCHES, "batch": recipe.batch, "crop": recipe.crop,
            "draw_ms_per_batch": round(1000 * drawn / BATCHES, 1),
            "draw_and_tensors_ms_per_batch": round(1000 * (drawn + tensors) / BATCHES, 1)}


def _to_crop(r: float, c: float, page: tuple, size: int, turns: int) -> tuple[float, float]:
    """A window px (row, col) of a crop's page window (y0, x0, src) where it lies in the finished crop: resized to
    size and turned `turns` quarter turns counter-clockwise."""
    _, _, src = page
    r, c = r * size / src, c * size / src
    for _ in range(turns % 4):
        r, c = size - 1 - c, r
    return r, c


def _tile(c: dict, info: dict, bank, size: int, turns: int, caption: str) -> np.ndarray:
    ink, text, known = c["ink"], c["text"].astype(bool), c["known"]
    img = np.full((*ink.shape, 3), 255, np.uint8)
    img[~known & ~ink] = (255, 228, 205)                       # no lesson, paper: pale blue (BGR)
    img[ink & ~text] = (125, 125, 125)                         # ink, not text: grey
    img[ink & ~text & ~known] = (200, 170, 150)                # ink with no lesson: blue-grey
    img[text & known] = (40, 40, 225)                          # text: red
    img[text & ~known] = (150, 150, 235)
    if info.get("mode") == "a":
        oy, ox, fh, fw = info["placed"]
        corners = [(oy, ox), (oy, ox + fw - 1), (oy + fh - 1, ox + fw - 1), (oy + fh - 1, ox)]
    elif info.get("mode") == "b":
        y0, x0, _ = info["window"]
        rect = bank.words[bank.sheets[info["sheet"]].series].rect[info["word"]]
        corners = [(float(py - y0), float(px - x0)) for px, py in rect]
    else:
        corners = []
    if corners:
        pts = np.array([_to_crop(r, cc, info["window"], size, turns)[::-1] for r, cc in corners], np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (0, 150, 0), 1)
    band = np.full((22, ink.shape[1], 3), 255, np.uint8)
    cv2.putText(band, caption, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, band])


def picture(sheets, pools, bank, recipe, out) -> list[dict]:
    """PICTURES composites of each kind (`picture`'s rng seeded 5), drawn as `train.draw` would, to `out`."""
    rng = np.random.default_rng(5)
    tiles, made = [], []
    kinds = {v: k for k, v in data.KINDS.items()}
    for mode, share in (("a", 1.0), ("b", 0.0)):
        forced = dataclasses.replace(bank, a_share=share)
        n = 0
        while n < PICTURES:
            i, y, x, scale, turns = _centre(pools, rng)
            c = forced.crop(rng, sheets[i], y, x, recipe.crop, scale, turns)
            info = forced.last
            if info["mode"] != mode:
                continue
            words = forced.words[sheets[i].series]
            kind = kinds[int(words.kind[info["word"]])]
            where = ("on line" if info["on_line"] else "off line") if mode == "a" else "line across"
            caption = f"({mode}) {where}, {kind}, {sheets[i].series[:9]}, x{scale:.2f}, {turns} turn"
            tiles.append(_tile(c, info, forced, recipe.crop, turns, caption))
            made.append({"mode": mode, "series": sheets[i].series, "kind": kind, "scale": round(scale, 3),
                         "turns": turns, **{k: v for k, v in info.items() if k in ("on_line", "window")}})
            n += 1
    gap = 6
    rows = []
    for r in range(0, len(tiles), 6):
        row = tiles[r:r + 6]
        rows.append(np.hstack([np.pad(t, ((0, gap), (0, gap), (0, 0)), constant_values=255) for t in row]))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(out), sheet):
        raise OSError(f"can't write {out}")
    return made


def texture_check(sheets, pools, bank) -> dict:
    """`texture_stats` of the text ink of CHECKS composites of each kind and of the real crops (lines-unknown, text
    ink with a lesson) at the same centres, at page scale and not turned (320 px)."""
    out = {}
    for mode, share in (("a", 1.0), ("b", 0.0)):
        forced = dataclasses.replace(bank, a_share=share)
        rng = np.random.default_rng(11)
        comp, real = [], []
        while len(comp) < CHECKS:
            i, y, x, _, _ = _centre(pools, rng)
            c = forced.crop(rng, sheets[i], y, x, 320, 1.0, 0)
            if forced.last["mode"] != mode:
                continue
            comp.append(c["ink"] & c["text"].astype(bool))
            if mode == "b":
                y, x = forced.last["centre"]
                i = forced.last["sheet"]
            r = data.crop_targets(bank.sheets[i], y, x, 320, 1.0, 0, labels="lines-unknown")
            real.append(r["ink"] & r["text"].astype(bool) & r["known"])
        out[mode] = {"composites": composite.texture_stats(comp), "real crops": composite.texture_stats(real)}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    blocks = reader_stage.load_block_area(reader_stage.held_blocks())
    log.info("loaded %d sheets in %.0f s", len(sheets), time.perf_counter() - t0)
    by_quarter, banks, seconds = {}, {}, {}
    keep = None
    for q, name in enumerate(paths.QUARTERS):
        areas = (data.Quarter(q), blocks)
        prepared = zones.prepare(sheets, areas, zones.LINES_DIR, THREADS)
        t0 = time.perf_counter()
        by_quarter[name] = composite.measure([sd for sd in prepared if sd.series == paths.SERIES], areas, THREADS)
        t1 = time.perf_counter()
        bank = composite.Bank.build(prepared, areas, SEED, THREADS)
        seconds[name] = {"measure": round(t1 - t0, 1), "bank": round(time.perf_counter() - t1, 1)}
        banks[name] = bank.sizes()
        log.info("%s held: p %.4f, bank %s (%.0f s)", name, _p(by_quarter[name]),
                 {s: (sum(v["clean_words"].values()), v["pieces"]) for s, v in banks[name].items()},
                 time.perf_counter() - t0)
        if name == FIRST:
            keep = (prepared, areas, bank)
        del prepared, bank
    first = by_quarter[FIRST]
    p = {name: _p(m) for name, m in by_quarter.items()}
    zones._write_json(OUT / "distributions.json", {
        "held_out": f"the quarter {FIRST} and the answered blocks (reader_stage.held_blocks); by_held_quarter for each",
        "rules": {"cross_px": composite.CROSS_PX, "clean_px": composite.CLEAN_PX, "piece_px": composite.PIECE_PX,
                  "tile": composite.TILE, "line_px": "SheetData.line_bits (linerules.line_px grown by 1 px)"},
        **{k: first[k] for k in ("cross_by_kind", "cross_by_kind_ink", "clean_words", "clean_word_texture",
                                 "line_texture", "sheets")},
        "p": p, "by_held_quarter": by_quarter, "banks": banks, "seconds": seconds})
    log.info("wrote %s: p %s", OUT / "distributions.json", p)

    prepared, areas, bank = keep
    held = data.hold_out(prepared, areas)
    t0 = time.perf_counter()
    pools = train.centre_pools(held)
    pooled = time.perf_counter() - t0
    base = dataclasses.replace(stage.DEFAULT_RECIPE, labels="lines-unknown")
    with_p = dataclasses.replace(base, composites=round(p[FIRST], 4))
    timing = [time_draw(held, pools, with_p, bank), time_draw(held, pools, base, None)]
    made = dict(sorted(bank.tally.items()))
    log.info("draw: %s", timing)
    bank.tally.clear()
    shown = picture(held, pools, bank, with_p, OUT / "composites.png")
    check = texture_check(held, pools, bank)
    zones._write_json(OUT / "composites.json", {
        "recipe": dataclasses.asdict(with_p), "timing": timing, "gpu_step_ms": 95,
        "composites_made_in_timing": made, "pools_seconds": round(pooled, 1), "picture": shown,
        "texture_check": check, "clean_word_texture": first["clean_word_texture"]})
    log.info("wrote %s and %s", OUT / "composites.json", OUT / "composites.png")


if __name__ == "__main__":
    main()
