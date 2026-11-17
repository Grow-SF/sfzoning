"""The reader's training crops from our own boxes, sloppy crops of Terra's words and masking outside the word
(`wordread.readercrops`; the off-by-default options of `textread.train` and `textread.read`), on synthetic pages."""
import hashlib
import math
import platform
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sfzmaps.solver import paths, wordboxes
from sfzmaps.solver.textnet import composite, data
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import model, read, store
from sfzmaps.solver.textread import train as T
from sfzmaps.solver.wordread import readercrops as RC

# --- the reader's training: today's batches, pinned -----------------------------------------------------------------

# sha256 of the first 20 batches `train.fit` feeds the reader and the 0/180 classifier on `_pool()` at seed 0, recorded
# with train.py as at d627640 (unchanged since 5b0b9d6), before the training options existed.  TEXTS: the batches'
# strings in order (the draws of the seeded generator alone: the same on every machine).  BYTES: every crop's shape and
# bytes too (the resampling of `augment` is not promised alike elsewhere), on the owner's Mac only.
PINNED_TEXTS = {"reader": "080864c3a5f358f58c246b60a291af3941f984f22e58a4a707a11249507c8b3e",
                "direction": "6def254067c14f7f42b54e130ca1123c6eb3a1512aaa626e0d30e8a5c93713b3"}
PINNED_BYTES = {("Darwin", "arm64"): {"reader": "699abc2e8c28c0af03f5fe43c8ee1b6c1996432dce1ca80d5c9450b83d6af9ab",
                                      "direction": "403dbefbae2549a31ab0b44886a87215c89f03070d00b935bc87542cd437a0db"}}
STEPS = 20


class _TinyReader(nn.Module):
    """A stand-in for the reader with its output's shape (n, frames, CLASSES): the batches don't depend on the
    network, and a tiny one keeps the test fast."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, model.CLASSES, (model.HEIGHT, model.STRIDE), stride=(model.HEIGHT, model.STRIDE))

    def forward(self, x):
        return self.conv(x)[:, :, 0].transpose(1, 2)


class _TinyDirection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x[:, :1].mean(dim=(1, 2, 3))[:, None])[:, 0]


def _crops(n: int, seed: int, ink: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = np.full((model.HEIGHT, int(rng.integers(20, 140))), 255, np.uint8)
        c[rng.random(c.shape) < 0.2] = ink
        out.append(c)
    return out


def _pool() -> T.Pool:
    """A pool of 60 real crops (two of them weighted 0.5) and 30 synthetic ones."""
    real = store.Crops.of(_crops(60, 1), model.HEIGHT)
    rows = pd.DataFrame({"text": [("RH-1", "C-2", "")[k % 3] for k in range(60)],
                         "weight": [0.5 if k in (3, 7) else 1.0 for k in range(60)]})
    synth = store.Crops.of(_crops(30, 2), model.HEIGHT)
    return T.Pool(real, rows, np.arange(60), synth, ["P"] * 30)


def _extra(n_own: int = 20, n_sloppy: int = 20) -> dict:
    """Own-box crops (target "OWN", each with a mask keeping its left half) and sloppy crops ("SLOP", no mask)."""
    own = []
    for c in _crops(n_own, 3):
        m = np.zeros(c.shape, bool)
        m[:, :c.shape[1] // 2] = True
        own.append((c, "OWN", {"mask": m}))
    return {"own": own, "sloppy": [(c, "SLOP", {}) for c in _crops(n_sloppy, 4)]}


def _fit_batches(monkeypatch, kind: str, pool, recipe: dict, **kw) -> list[tuple[str, np.ndarray]]:
    """Every (string, crop) `fit` feeds the network in `STEPS` steps (the crops as they enter `model.to_input`)."""
    monkeypatch.setattr(model, "Reader", _TinyReader)
    monkeypatch.setattr(model, "Direction", _TinyDirection)
    seen, texts = [], []
    real_input, real_targets = model.to_input, T.targets

    def to_input(batch, quantize=False):
        seen.append([np.array(c) for c in batch])
        return real_input(batch, quantize)

    def targets(t):
        texts.append(list(t))
        return real_targets(t)

    monkeypatch.setattr(model, "to_input", to_input)
    monkeypatch.setattr(T, "targets", targets)
    T.fit(kind, pool, {**recipe, "steps": STEPS}, "cpu", seed=0, **kw)
    if kind == "direction":
        texts = [[""] * len(b) for b in seen]
    return [(t, c) for tb, cb in zip(texts, seen, strict=True) for t, c in zip(tb, cb, strict=True)]


def _digests(got) -> tuple[str, str]:
    a, b = hashlib.sha256(), hashlib.sha256()
    for text, crop in got:
        a.update(text.encode() + b"|")
        b.update(f"{text}|{crop.shape}|".encode() + np.ascontiguousarray(crop).tobytes())
    return a.hexdigest(), b.hexdigest()


def _assert_pinned(kind, got):
    texts, whole = _digests(got)
    assert texts == PINNED_TEXTS[kind]
    want = PINNED_BYTES.get((platform.system(), platform.machine()))
    if want is not None:
        assert whole == want[kind]


@pytest.mark.parametrize("kind", ["reader", "direction"])
def test_fit_with_the_default_options_feeds_todays_batches(monkeypatch, kind):
    torch.set_num_threads(1)
    recipe = T.READER if kind == "reader" else T.DIRECTION
    _assert_pinned(kind, _fit_batches(monkeypatch, kind, _pool(), recipe))


def test_fit_with_extra_crops_at_share_zero_feeds_todays_batches(monkeypatch):
    torch.set_num_threads(1)
    got = _fit_batches(monkeypatch, "reader", _pool(), {**T.READER, **T.READER_OPTIONS}, extra=_extra())
    _assert_pinned("reader", got)


def test_fit_draws_own_box_and_sloppy_crops_at_their_shares(monkeypatch):
    torch.set_num_threads(1)
    got = _fit_batches(monkeypatch, "reader", _pool(), {**T.READER, "own": 0.3, "sloppy": 0.2}, extra=_extra())
    texts = pd.Series([t for t, _ in got])
    n = len(texts)
    assert n == STEPS * T.BATCH
    share = texts.value_counts() / n
    sd = lambda p: math.sqrt(p * (1 - p) / n)     # noqa: E731
    assert abs(share["OWN"] - 0.3) < 4 * sd(0.3)
    assert abs(share["SLOP"] - 0.2) < 4 * sd(0.2)
    assert abs(share["P"] - T.SYNTH_SHARE) < 4 * sd(T.SYNTH_SHARE)
    real = 1 - 0.3 - 0.2 - T.SYNTH_SHARE
    assert abs(share[["RH-1", "C-2", ""]].sum() - real) < 4 * sd(real)


def test_fit_stops_on_a_share_with_no_crops(monkeypatch):
    monkeypatch.setattr(model, "Reader", _TinyReader)
    with pytest.raises(ValueError, match="own"):
        T.fit("reader", _pool(), {**T.READER, "steps": 2, "own": 0.3}, "cpu")
    with pytest.raises(ValueError, match="sloppy"):
        T.fit("reader", _pool(), {**T.READER, "steps": 2, "sloppy": 0.3}, "cpu", extra={"own": _extra()["own"]})


def test_a_masked_training_blanks_the_own_box_crops_outside_their_mask():
    extra = _extra(3, 2)
    plain, masked = _pool().joined(extra, mask=False), _pool().joined(extra, mask=True)
    base = len(_pool().items)
    assert plain.n_own == masked.n_own == 3 and plain.n_sloppy == masked.n_sloppy == 2
    for k, (c, _, meta) in enumerate(extra["own"]):
        np.testing.assert_array_equal(plain.items[base + k], c)
        np.testing.assert_array_equal(masked.items[base + k], K.mask_outside(c, meta["mask"]))
    for k, (c, _, _) in enumerate(extra["sloppy"]):          # no mask: as cut
        np.testing.assert_array_equal(masked.items[base + 3 + k], c)
    assert masked.texts[base:] == ["OWN"] * 3 + ["SLOP"] * 2 and masked.weights[base:] == [1.0] * 5


# --- masking ----------------------------------------------------------------------------------------------------------

def test_mask_outside_blanks_exactly_the_px_outside_the_mask():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (32, 50)).astype(np.uint8)
    mask = rng.random(crop.shape) < 0.4
    got = RC.mask_outside(crop, mask)
    assert got.dtype == np.uint8
    assert (got[~mask] == 255).all()
    np.testing.assert_array_equal(got[mask], crop[mask])
    with pytest.raises(ValueError):
        RC.mask_outside(crop, mask[:, :-1])


def _ink_page():
    page = np.full((120, 200), 255, np.uint8)
    rng = np.random.default_rng(1)
    page[40:80, 30:170][rng.random((40, 140)) < 0.3] = 0
    return page


def test_a_word_mask_is_cut_as_the_box_is():
    """At a quarter turn and a crop of the page's own height the cut is an exact copy: the mask of the ink cut as
    the box is cut is the crop's ink, from the whole page or from a window of it."""
    page = _ink_page()
    height = model.HEIGHT / (1 + 2 * K.PAD_Y)           # the crop is the page's own px tall: no resizing
    corners = np.array([[50, 60 - height / 2], [150, 60 - height / 2], [150, 60 + height / 2],
                        [50, 60 + height / 2]])
    for axis in (0.0, 180.0, 90.0):
        crop = K.box_crop(page, corners, 100, height, axis)
        whole = K.box_mask(page < 128, corners, 100, height, axis)
        assert whole.dtype == bool and whole.shape == crop.shape
        if axis != 90.0:
            np.testing.assert_array_equal(whole, crop < 128)
        win = (page < 128)[20:110, 10:190]
        np.testing.assert_array_equal(K.box_mask(win, corners, 100, height, axis, origin=(10, 20)), whole)


RECTS = ((0, 0, 12, 12), (15, 25, 86, 55), (115, 55, 205, 80))    # x0, y0, x1, y1 of the high chance


def _comp_page():
    """Ink: a speck (fewer than MIN_PX px) first in row order, then two words; a line runs through the first and on
    past it, where the chance is low (the detector's line).  The chance is high only in RECTS."""
    page = np.full((100, 220), 255, np.uint8)
    page[5:8, 5:8] = 0                                   # 9 px: a speck
    page[30:50, 20:80:3] = 0                             # word one: thin bars, joined by the INK_R gate
    page[40, :] = 0                                      # the line through it
    page[60:70, 120:200:4] = 0                           # word two
    chance = np.zeros(page.shape, np.uint8)
    for x0, y0, x1, y1 in RECTS:
        chance[y0:y1, x0:x1] = 255
    return page, chance


def _grouped(page, chance):
    ink = page < 128
    g = wordboxes.group(chance, ink, wordboxes.ink_gate(ink), 0.99)
    rows = []
    for k in range(len(g["axis"])):
        rows.append({"row": 10 + k, "i": k, "level": 0.99, "ink_px": int(g["ink_px"][k]), "axis": float(g["axis"][k]),
                     "t_length": float(g["t_length"][k]), "t_height": float(g["t_height"][k]),
                     **{f"t{j}{a}": float(g["t"][k, j, n]) for j in range(4) for n, a in enumerate("xy")}})
    return rows


def test_each_box_is_masked_by_its_own_component_grown_by_ink_r():
    page, chance = _comp_page()
    rows = _grouped(page, chance)
    assert len(rows) == 2
    got = read.word_masks("s", rows, page=page, chance=chance)
    ink = page < 128
    gate = wordboxes.ink_gate(ink)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wordboxes.INK_R + 1, 2 * wordboxes.INK_R + 1))
    rects = []
    for x0, y0, x1, y1 in RECTS[1:]:
        r = np.zeros(ink.shape, bool)
        r[y0:y1, x0:x1] = True
        rects.append(r)
    for r, rect, other in zip(rows, rects, rects[::-1]):
        m, (x0, y0) = got[r["row"]]
        full = np.zeros(ink.shape, bool)
        full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        np.testing.assert_array_equal(full, cv2.dilate((rect & gate).view(np.uint8), disc) > 0)
        assert full[ink & rect].all() and not full[ink & other].any()
    with pytest.raises(ValueError, match="ink"):
        read.word_masks("s", [{**rows[0], "ink_px": rows[0]["ink_px"] + 1}], page=page, chance=chance)


def test_reading_unmasked_cuts_todays_crops_and_masked_blanks_outside_the_word():
    page, chance = _comp_page()
    rows = _grouped(page, chance)
    plain = read.box_crops(page, rows)
    for r, c in zip(rows, plain):
        corners = np.array([[r[f"t{i}x"], r[f"t{i}y"]] for i in range(4)])
        np.testing.assert_array_equal(c, K.box_crop(page, corners, r["t_length"], r["t_height"], r["axis"]))
    masks = read.word_masks("s", rows, page=page, chance=chance)
    masked = read.box_crops(page, rows, masks)
    for r, c, m in zip(rows, plain, masked):
        corners = np.array([[r[f"t{i}x"], r[f"t{i}y"]] for i in range(4)])
        win, origin = masks[r["row"]]
        keep = K.box_mask(win, corners, r["t_length"], r["t_height"], r["axis"], origin=origin)
        np.testing.assert_array_equal(m, K.mask_outside(c, keep))
    assert (masked[0] != plain[0]).any()                 # the line's ends past word one are blanked


def test_word_masks_hold_every_box_of_a_real_sheet():
    """A real-data check: every box of one sheet finds its component (its ink px count) from today's chance."""
    f = wordboxes.BOXES_DIR / "boxes.parquet"
    if not f.is_file() or not (paths.READER_TEXT_DIR / paths.SHEETS[0] / "chance.png").is_file():
        pytest.skip("solver-boxes has not run")
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    boxes["row"] = np.arange(len(boxes))
    rows = boxes[boxes["sheet"] == paths.SHEETS[0]].to_dict("records")
    got = read.word_masks(paths.SHEETS[0], rows)
    assert len(got) == len(rows) and all(m.any() for m, _ in got.values())


# --- own boxes and sloppy crops on a synthetic page -------------------------------------------------------------------

SHEET = "s"
BLOCKS = (120, 170, 260, 260)        # x0, y0, x1, y1: the answered blocks' px
SPLIT = 300                          # x: nw (quarter 0) left of it, ne (quarter 1) right of it


def _glyphs(page, x0, y0, n=4, h=24, flip=False):
    """n glyphs (an L: a stem and a foot, asymmetric under a half turn), 6 px wide with 4 px gaps, from (x0, y0);
    `flip`: turned a half turn in place (the word reads at 180 degrees)."""
    w = n * 6 + (n - 1) * 4
    word = np.full((h, w), 255, np.uint8)
    for k in range(n):
        word[:, k * 10:k * 10 + 2] = 0
        word[h - 2:, k * 10:k * 10 + 6] = 0
    page[y0:y0 + h, x0:x0 + w] = np.rot90(word, 2) if flip else word
    return x0, y0, x0 + w, y0 + h


def _page():
    page = np.full((300, 600), 255, np.uint8)
    boxes = {"A": _glyphs(page, 82, 55), "B": _glyphs(page, 402, 55, flip=True), "C": _glyphs(page, 162, 205),
             "D": _glyphs(page, 402, 205), "E": _glyphs(page, 120, 55, n=2)}
    return page, boxes


def _touch(sheet, window, shape):
    x0, y0, x1, y1 = window
    bx0, by0, bx1, by1 = BLOCKS
    return {"touch_nw": x0 < SPLIT, "touch_ne": x1 > SPLIT, "touch_sw": False, "touch_se": False,
            "touch_blocks": x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0}


KINDS = {"A": "zoning code", "B": "zoning code", "C": "zoning code", "D": "zoning code", "E": "block number"}


def _terra(boxes, line=None):
    """crops.parquet-like rows of the words: A (nw), B (ne, read upside down), C (nw, in the answered blocks),
    D (ne), E (nw, a block number right of A).  `line`: per word (line ink px, word ink px, line thickness)."""
    rows = []
    for k, (name, text, angle) in enumerate((("A", "RH-1", 0.0), ("B", "C-2", 180.0), ("C", "P", 0.0),
                                             ("D", "M-1", 0.0), ("E", "NC", 0.0))):
        x0, y0, x1, y1 = boxes[name]
        rows.append({"series": paths.SERIES, "sheet": SHEET, "word": k, "kind": KINDS[name], "terra": text,
                     "text": text, "angle": angle, "box_x0": x0, "box_y0": y0, "box_x1": x1, "box_y1": y1,
                     "length": float(x1 - x0), "height": float(y1 - y0), "weight": 1.0,
                     **_touch(SHEET, (x0, y0, x1, y1), None), "quarter": int(x0 >= SPLIT), "block": "",
                     **(dict(zip(("line_ink", "word_ink", "line_thick"), line[k])) if line else {})})
    rows.append({**rows[0], "series": "1980-1990-zoning"})          # another map's word: not ours to match
    rows.append({**rows[0], "word": -1, "kind": "none", "text": ""})  # a non-text window
    return RC.terra_frame(pd.DataFrame(rows))


def _box(cx, cy, length, height, axis=0.0, quarter=0, i=0):
    """One of our boxes centred on the page px index (cx, cy), its grouped box its trimmed box, as `wordboxes.group`
    writes corners (page px edge coordinates)."""
    a = math.radians(axis)
    v, nrm = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])
    c = np.array([cx + 0.5, cy + 0.5])
    pts = [c - v * length / 2 - nrm * height / 2, c + v * length / 2 - nrm * height / 2,
           c + v * length / 2 + nrm * height / 2, c - v * length / 2 + nrm * height / 2]
    corners = {f"{k}{j}{n}": float(p[m]) for k in "gt" for j, p in enumerate(pts) for m, n in enumerate("xy")}
    return {"sheet": SHEET, "i": i, "level": 0.99, "quarter": quarter, "axis": axis % 180.0, "length": length,
            "height": height, "t_length": length, "t_height": height, "ink_px": 50, **corners}


def _boxes(boxes):
    """Our boxes: A's exactly; B's at axis 0 (B reads at 180); C's; one loose on D (IoU about 0.35); one on nothing."""
    def centre(name):
        x0, y0, x1, y1 = boxes[name]
        return (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    rows = [_box(*centre("A"), 36, 24), _box(*centre("B"), 36, 24, quarter=1), _box(*centre("C"), 36, 24),
            _box(centre("D")[0] + 18, centre("D")[1], 36, 24, quarter=1), _box(300, 280, 30, 10, quarter=1)]
    return pd.DataFrame(rows, index=[100, 101, 102, 103, 104])




def test_own_box_crops_carry_the_matched_terra_string_and_only_training_quarter_boxes():
    page, where = _page()
    terra, boxes = _terra(where), _boxes(where)
    kw = {"pages": {SHEET: page}, "touch": _touch}
    for quarter, want in ((1, ["A"]), (0, ["B"]), (2, ["A", "B"])):
        got = RC.own_box_crops(boxes, terra, quarter=quarter, **kw)
        assert [m["word"] for _, _, m in got] == [{"A": 0, "B": 1}[w] for w in want]
        for crop, text, meta in got:
            assert text == {0: "RH-1", 1: "C-2"}[meta["word"]]
            assert meta["iou"] >= 0.5 and not meta["touch_blocks"]
            assert not meta[f"touch_{paths.QUARTERS[quarter]}"]
            b = boxes.loc[meta["row"]]
            corners = b[[f"t{j}{a}" for j in range(4) for a in "xy"]].to_numpy(float).reshape(4, 2)
            cut = K.box_crop(page, corners, b["t_length"], b["t_height"], b["axis"])
            # B reads at 180 degrees: its crop is turned a half turn, to read as Terra's word does
            np.testing.assert_array_equal(crop, np.rot90(cut, 2) if meta["word"] == 1 else cut)
            assert meta["turned"] == (meta["word"] == 1)
            # our box is Terra's exactly here: the crop is Terra's crop
            t = terra[terra["word"] == meta["word"]].iloc[0]
            box = t[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
            np.testing.assert_array_equal(crop, K.rectify(page, box, t["angle"])[0])
    # a word whose own row touches the answered blocks is left out, whatever its crop's footprint touches
    terra2 = terra.copy()
    terra2.loc[terra2["word"] == 0, "touch_blocks"] = True
    assert [m["word"] for _, _, m in RC.own_box_crops(boxes, terra2, quarter=2, **kw)] == [1]
    with pytest.raises(ValueError, match="quarter"):
        RC.own_box_crops(boxes, terra, quarter=None, **kw)


def test_own_box_crops_carry_their_mask_turned_with_them():
    page, where = _page()
    terra, boxes = _terra(where), _boxes(where)
    ink = page < 128

    def masks(sheet, rows):
        return {r["row"]: (ink[40:90, 60:460], (60, 40)) for r in rows}

    got = RC.own_box_crops(boxes, terra, quarter=2, pages={SHEET: page}, touch=_touch, word_masks=masks)
    assert len(got) == 2
    for crop, _, meta in got:
        b = boxes.loc[meta["row"]]
        corners = b[[f"t{j}{a}" for j in range(4) for a in "xy"]].to_numpy(float).reshape(4, 2)
        m = K.box_mask(ink[40:90, 60:460], corners, b["t_length"], b["t_height"], b["axis"], origin=(60, 40))
        np.testing.assert_array_equal(meta["mask"], np.rot90(m, 2) if meta["turned"] else m)
        assert meta["mask"].shape == crop.shape


LINE = [(5, 50, 4.0), (0, 56, 0.0), (0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0)]   # A crossed: 5 of its 50 ink px, 4 px


def test_offset_distribution_measures_ends_cross_angle_height_lines_neighbours_and_kinds():
    """A's box starts 3 px early, ends 6 px late, sits 2.4 px low and is 30 px tall (A: 36 x 24); B's (reading at 180
    degrees) runs 2 px past B's end (its left).  A holds line ink and a quarter of E (beside A, not on it); B
    neither."""
    page, where = _page()
    terra = _terra(where, line=LINE)
    boxes = pd.DataFrame([_box(101.0, 68.9, 45, 30), _box(418.5, 66.5, 38, 24, quarter=1)])
    d = RC.offset_distribution(boxes, terra, quarter=2)
    assert d["n"] == 2
    off = d["offsets"]
    np.testing.assert_allclose(off["start"], [-3 / 24, 0.0], atol=1e-9)
    np.testing.assert_allclose(off["end"], [6 / 24, 2 / 24], atol=1e-9)
    np.testing.assert_allclose(off["cross"], [0.1, 0.0], atol=1e-9)
    np.testing.assert_allclose(off["angle"], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(off["height_ratio"], [1.25, 1.0], atol=1e-9)
    np.testing.assert_allclose(off["neighbour"], [0.25, 0.0], atol=1e-9)        # drawn with its pair's offsets
    np.testing.assert_allclose(d["mean"]["start"], -3 / 48, atol=1e-9)
    assert d["line_share"] == 0.5
    assert d["line_thickness"]["q99"] == 4.0 and d["line_cover"]["q99"] == pytest.approx(0.1)
    assert d["neighbour_share"] == 0.5
    np.testing.assert_allclose(d["neighbour_part"], [0.25], atol=1e-9)
    assert d["neighbour_on_word"]["q99"] == 0.0
    # the kinds of the training quarters' Terra words (C is in the answered blocks; quarter 1 holds B and D out)
    assert d["kind_shares"] == {"zoning code": 0.75, "block number": 0.25, "street name": 0.0, "other": 0.0}
    assert RC.offset_distribution(boxes, terra, quarter=1)["kind_shares"]["block number"] == 0.5
    with pytest.raises(ValueError, match="quarter"):
        RC.offset_distribution(boxes, terra, quarter=None)


def test_a_sloppy_crop_drawn_at_a_pairs_own_offsets_is_that_boxs_own_crop():
    page, where = _page()
    terra = _terra(where)
    boxes = pd.DataFrame([_box(101.0, 68.9, 45, 30), _box(418.5, 66.5, 38, 24, quarter=1)])
    own = RC.own_box_crops(boxes, terra, quarter=2, pages={SHEET: page}, touch=_touch)
    assert len(own) == 2
    for k, (crop, _, meta) in enumerate(own):
        d = RC.offset_distribution(boxes.iloc[[k]], terra, quarter=2)
        d.update(line_share=0.0)
        word = terra[terra["word"] == meta["word"]].iloc[0]          # (no neighbour column: none is pasted)
        got = RC.sloppy_crop(page, word, d, [], np.random.default_rng(0))
        np.testing.assert_array_equal(got, crop)


def test_a_tilted_box_measures_its_tilt_counter_clockwise_and_is_cut_back_the_same():
    """Boxes on A (reading at 0) at axes 3 and 177 degrees, and on B (reading at 180) at axis 3: the axis 177 and B's
    are read a half turn round (183 and 357 degrees), so the angle offsets are +3, -3 and +3; Terra's word cut at a
    box's offsets is that box's crop."""
    page, where = _page()
    terra = _terra(where)
    boxes = pd.DataFrame([_box(99.5, 66.5, 40, 26, axis=3.0), _box(99.5, 66.5, 40, 26, axis=177.0),
                          _box(419.5, 66.5, 40, 26, axis=3.0, quarter=1)])
    d = RC.offset_distribution(boxes, terra, quarter=2)
    np.testing.assert_allclose(d["offsets"]["angle"], [3.0, -3.0, 3.0], atol=1e-9)
    own = RC.own_box_crops(boxes, terra, quarter=2, pages={SHEET: page}, touch=_touch)
    assert [m["turned"] for _, _, m in own] == [False, True, True]
    for k, (crop, _, meta) in enumerate(own):
        one = RC.offset_distribution(boxes.iloc[[k]], terra, quarter=2)
        one.update(line_share=0.0)
        word = terra[terra["word"] == meta["word"]].iloc[0]
        got = RC.sloppy_crop(page, word, one, [], np.random.default_rng(0))
        assert got.shape == crop.shape
        assert np.abs(got.astype(int) - crop.astype(int)).max() <= 1


def _dist(rows=200, line=0.0, neighbour=0.0, seed=0, cover=None):
    """Offsets drawn well away from 0; a `neighbour` share of the rows hold a neighbour (a tenth or 0.3 of it)."""
    rng = np.random.default_rng(seed)
    off = {"start": rng.normal(-0.4, 0.1, rows), "end": rng.normal(0.5, 0.1, rows),
           "cross": rng.normal(0.2, 0.05, rows), "angle": rng.normal(5.0, 2.0, rows),
           "height_ratio": rng.normal(1.3, 0.1, rows)}
    mean = {k: float(np.mean(v)) for k, v in off.items()}
    off["neighbour"] = np.where(rng.random(rows) < neighbour, rng.choice([0.1, 0.3], rows), 0.0)
    return {"n": rows, "offsets": {k: v.tolist() for k, v in off.items()}, "mean": mean, "line_share": line,
            "line_cover": {"q99": cover}, "neighbour_on_word": {"q99": 0.0}}


def _line_piece():
    p = np.full((60, 3), 255, np.uint8)
    p[:, 1] = 0
    return p


def _blob():
    return np.zeros((40, 40), np.uint8)                 # a solid square of ink


def test_sloppy_crops_offsets_follow_the_distribution():
    page, where = _page()
    terra = RC.with_neighbours(_terra(where))
    word = terra[terra["word"] == 0].iloc[0]
    dist = _dist(line=0.3, neighbour=0.4, cover=0.3)
    held = np.mean(np.array(dist["offsets"]["neighbour"]) > 0)
    rng = np.random.default_rng(5)
    drawn = []
    for _ in range(500):
        info = {}
        crop = RC.sloppy_crop(page, word, dist, [_line_piece(), _blob()], rng, info)
        assert crop.dtype == np.uint8 and crop.shape[0] == model.HEIGHT
        drawn.append(info)
    got = pd.DataFrame(drawn)
    for k, m in dist["mean"].items():
        assert abs(got[k].mean() - m) <= 0.1 * abs(m), k
    # a rejected line piece is drawn again: the line rate stays line_share
    for k, p in (("line", 0.3), ("neighbour", held)):
        assert abs(got[k].mean() - p) < 4 * math.sqrt(p * (1 - p) / len(got)), k
    assert got["line_rejected"].sum() > 0 and not got["line_gave_up"].any()
    assert (got.loc[got["line"], "line_cover"] <= 0.3).all()
    assert (got.loc[got["neighbour"], "on_word"] <= 1e-9).all()        # never on the word


def _zero(line=0.0, neighbour=0.0, end=0.0, cover=0.3):
    return {"n": 1, "offsets": {k: [v] for k, v in (("start", 0.0), ("end", end), ("cross", 0.0), ("angle", 0.0),
                                                     ("height_ratio", 1.0), ("neighbour", neighbour))},
            "line_share": line, "line_cover": {"q99": cover}, "neighbour_on_word": {"q99": 0.0}}


def test_a_line_piece_goes_through_the_letters():
    page, where = _page()
    terra = RC.with_neighbours(_terra(where))
    word = terra[terra["word"] == 0].iloc[0]
    before = page.copy()
    plain = RC.sloppy_crop(page, word, _zero(), [_line_piece()], np.random.default_rng(1))
    np.testing.assert_array_equal(plain, K.rectify(page, where["A"], 0.0)[0])
    for seed in range(5):
        info = {}
        got = RC.sloppy_crop(page, word, _zero(line=1.0), [_line_piece()], np.random.default_rng(seed), info)
        assert info["line"] and not info["neighbour"] and info["line_rejected"] == 0
        x, y = info["line_at"]
        x0, y0, x1, y1 = where["A"]
        assert page[y, x] < 128 and x0 <= x < x1 and y0 <= y < y1      # on a letter of the word
        assert (got < 128).sum() > (plain < 128).sum()
    np.testing.assert_array_equal(page, before)                     # the page is untouched


def test_a_line_piece_covering_more_of_the_word_than_measured_is_drawn_again():
    page, where = _page()
    word = RC.with_neighbours(_terra(where)).iloc[0]
    rejected = 0
    for seed in range(20):
        info = {}
        RC.sloppy_crop(page, word, _zero(line=1.0), [_line_piece(), _blob()], np.random.default_rng(seed), info)
        assert info["line"] and info["line_cover"] <= 0.3
        rejected += info["line_rejected"]
    assert rejected > 0
    # a piece always covers its own letter px: at a cap of 0 every draw is rejected, and the word is left clean
    info = {}
    got = RC.sloppy_crop(page, word, _zero(line=1.0, cover=0.0), [_line_piece()], np.random.default_rng(0), info)
    assert not info["line"] and info["line_gave_up"] and info["line_rejected"] == RC.LINE_TRIES
    np.testing.assert_array_equal(got, K.rectify(page, where["A"], 0.0)[0])


def test_a_neighbour_is_moved_beside_the_word_never_onto_it():
    """A's box runs half a height past A's end and holds a quarter of a word: E (right of A) is moved along the
    reading axis until a quarter of it is inside the box, all of it off A; its old place is paper."""
    page, where = _page()
    before = page.copy()
    word = RC.with_neighbours(_terra(where)).iloc[0]
    assert word["n_word"] == 4                                          # A's nearest word is E
    info = {}
    got = RC.sloppy_crop(page, word, _zero(neighbour=0.25, end=0.5), [], np.random.default_rng(0), info)
    assert info["neighbour"] and info["side"] == "along" and info["part"] == 0.25
    assert abs(info["part_got"] - 0.25) < 0.05 and info["on_word"] <= 1e-9
    assert info["shift"] == (6, 0)
    moved = page.copy()
    x0, y0, x1, y1 = where["E"]
    moved[y0:y1, x0:x1] = 255
    moved[y0:y1, x0 + 6:x1 + 6] = np.minimum(moved[y0:y1, x0 + 6:x1 + 6], page[y0:y1, x0:x1])
    np.testing.assert_array_equal(got, K.cut(moved, *info["frame"])[0])
    # no room beside A inside its box: the drawn share is not reached, and A is still not covered
    info = {}
    RC.sloppy_crop(page, word, _zero(neighbour=0.5), [], np.random.default_rng(0), info)
    assert info["neighbour"] and info["on_word"] <= 1e-9 and info["part_got"] < 0.5
    np.testing.assert_array_equal(page, before)


def test_neighbours_are_the_nearest_other_word_of_the_frame_given():
    page, where = _page()
    terra = _terra(where)
    t = RC.with_neighbours(terra)
    by = dict(zip(t["word"], t["n_word"]))
    assert by[0] == 4 and by[4] == 0
    alone = RC.with_neighbours(terra[terra["word"].isin([0, 1])])
    assert dict(zip(alone["word"], alone["n_word"])) == {0: 1, 1: 0}
    assert RC.with_neighbours(terra[terra["word"] == 0])["n_word"].tolist() == [-1]


def test_sloppy_crops_hold_out_the_quarter_and_the_answered_blocks():
    page, where = _page()
    terra = _terra(where)
    dist = _dist(neighbour=1.0)
    for quarter, want in ((0, {1, 3}), (1, {0, 4})):
        got = RC.sloppy_crops(terra, dist, [_line_piece()], seed=3, quarter=quarter, copies=2,
                              pages={SHEET: page}, touch=_touch)
        assert {m["word"] for _, _, m in got} == want
        assert len(got) == 4
        for crop, text, meta in got:
            assert not meta["touch_blocks"] and not meta[f"touch_{paths.QUARTERS[quarter]}"]
            assert meta["n_word"] in want                     # the neighbour pasted in is a training word too
            assert text == terra.loc[terra["word"] == meta["word"], "text"].iloc[0]
            assert "mask" not in meta
    again = RC.sloppy_crops(terra, dist, [_line_piece()], seed=3, quarter=1, copies=2, pages={SHEET: page},
                            touch=_touch)
    for (a, _, _), (b, _, _) in zip(got, again, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="quarter"):
        RC.sloppy_crops(terra, dist, [], seed=3, quarter=None, pages={SHEET: page}, touch=_touch)


def test_sloppy_crops_carry_the_mask_of_their_words_components_cut_as_the_crop():
    page, where = _page()
    ink = page < 128

    def masks(sheet, words):
        return {int(w): (ink[30:110, 50:200], (50, 30)) for w in words["word"] if w == 0}

    got = RC.sloppy_crops(_terra(where), _dist(), [], seed=3, quarter=1, pages={SHEET: page}, touch=_touch,
                          masks=masks)
    assert {m["word"] for _, _, m in got} == {0, 4}
    for crop, _, meta in got:
        if meta["word"] == 0:
            np.testing.assert_array_equal(meta["mask"], K.cut(ink[30:110, 50:200], *meta["frame"], fill=False,
                                                              origin=(50, 30))[0])
            assert meta["mask"].shape == crop.shape
        else:
            assert "mask" not in meta                         # no component meets E: left as cut


def _bank(held):
    sd = SimpleNamespace(series=paths.SERIES, sheet=SHEET, held=held)
    pieces = composite.Pieces(np.zeros(2, np.int32), np.array([[10, 10, 190, 20], [10, 40, 190, 60]], np.int32),
                              np.array([[10, 10], [10, 40]], np.int32))
    return SimpleNamespace(sheets=[sd], pieces={paths.SERIES: pieces})


def test_line_pieces_come_from_a_bank_holding_the_areas_out_and_no_thicker_than_measured():
    page = np.full((80, 200), 255, np.uint8)
    page[10:20, 10:190] = 0                              # 10 px thick: kept
    page[40:60, 10:190] = 0                              # 20 px thick: thicker than the measured crossings
    area = data.GridArea(np.ones((2, 2), bool), 5, 7)
    kw = {"pages": {SHEET: page}, "block_area": area, "max_thickness": 11.0,
          "piece_mask": lambda sd, box, seed: page[box[1]:box[3], box[0]:box[2]] < 128}
    info = {}
    got = RC.piece_patches(_bank((data.Quarter(1), area)), quarter=1, info=info, **kw)
    assert len(got) == 1 and info == {"pieces": 2, "kept": 1, "too_thick": 1}
    np.testing.assert_array_equal(got[0], page[10:20, 10:190])
    assert RC.thickness(page[10:20, 10:190] < 128) == 10.0
    for bank, quarter in ((_bank((data.Quarter(1), area)), 0), (_bank((data.Quarter(1),)), 1),
                          (_bank((data.Quarter(1), data.GridArea(np.ones((2, 2), bool), 0, 0))), 1)):
        with pytest.raises(ValueError, match="held"):
            RC.piece_patches(bank, quarter=quarter, **kw)


def _own_kinds():
    return [(c, "Z", {"kind": "zoning code"}) for c in _crops(2, 5)] + \
        [(c, "B", {"kind": "block number"}) for c in _crops(8, 6)]


SHARES = {"zoning code": 0.5, "block number": 0.3, "street name": 0.2, "other": 0.0}


def test_kind_weights_draw_own_crops_in_terras_kind_shares(monkeypatch):
    """Own crops of two kinds only: the kinds present share Terra's shares between them (0.5 : 0.3)."""
    own = _own_kinds()
    p = RC.own_draw_weights(own, SHARES)
    assert p.sum() == pytest.approx(1.0)
    assert p[:2].sum() == pytest.approx(0.625) and p[2:].sum() == pytest.approx(0.375)
    assert "own_p" not in RC.extra_pool(own, [])                          # off by default
    with pytest.raises(ValueError):
        RC.extra_pool(own, [], kind_weights="terra")
    extra = RC.extra_pool(own, [], kind_weights="terra", kind_shares=SHARES)
    torch.set_num_threads(1)
    got = pd.Series([t for t, _ in _fit_batches(monkeypatch, "reader", _pool(), {**T.READER, "own": 0.5},
                                               extra=extra)])
    z, b = (got == "Z").sum(), (got == "B").sum()
    assert abs(z / (z + b) - 0.625) < 4 * math.sqrt(0.625 * 0.375 / (z + b))


def test_a_masked_training_blanks_terras_and_the_sloppy_crops_by_their_components_too():
    pool = _pool()
    rng = np.random.default_rng(9)
    real_masks = {k: rng.random(pool.items[k].shape) < 0.5 for k in (3, 10)}
    sloppy = [(c, "SLOP", {"mask": rng.random(c.shape) < 0.5}) for c in _crops(2, 4)] + [(_crops(1, 8)[0], "SLOP", {})]
    extra = {**_extra(2, 0), "sloppy": sloppy, "real_masks": real_masks}
    plain, masked = _pool().joined(extra, mask=False), _pool().joined(extra, mask=True)
    for k in range(len(pool.items)):
        want = K.mask_outside(pool.items[k], real_masks[k]) if k in real_masks else pool.items[k]
        np.testing.assert_array_equal(masked.items[k], want)
        np.testing.assert_array_equal(plain.items[k], pool.items[k])
    base = len(pool.items) + 2
    for j, (c, _, meta) in enumerate(sloppy):
        want = K.mask_outside(c, meta["mask"]) if "mask" in meta else c
        np.testing.assert_array_equal(masked.items[base + j], want)
        np.testing.assert_array_equal(plain.items[base + j], c)


def _rows_for(page_boxes, quarter=0):
    rows = []
    for k, (x0, y0, x1, y1) in enumerate(page_boxes):
        rows.append({"series": paths.SERIES, "sheet": "s", "word": k, "kind": "zoning code", "text": "X",
                     "angle": 0.0, "box_x0": x0, "box_y0": y0, "box_x1": x1, "box_y1": y1,
                     "length": float(x1 - x0), "height": float(y1 - y0), "quarter": quarter})
    rows.append({**rows[0], "series": "1980-1990-zoning"})        # another map's: no components, never masked
    return pd.DataFrame(rows)


def test_terras_crops_are_masked_by_the_components_meeting_their_rectangle():
    page, chance = _comp_page()
    rows = _rows_for([(20, 30, 78, 50), (150, 5, 200, 20)])        # word one; blank paper
    got = RC.real_masks(rows, pages={"s": page}, chances={"s": chance}, levels={paths.QUARTERS[0]: 0.99})
    assert set(got) == {0}
    ink = page < 128
    x0, y0, x1, y1 = RECTS[1]
    comp = np.zeros(ink.shape, bool)
    comp[y0:y1, x0:x1] = True
    comp &= wordboxes.ink_gate(ink)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wordboxes.INK_R + 1, 2 * wordboxes.INK_R + 1))
    grown = cv2.dilate(comp.view(np.uint8), disc) > 0
    crop = K.rectify(page, (20, 30, 78, 50), 0.0)[0]
    np.testing.assert_array_equal(got[0], K.cut(grown, (48.5, 39.5), 58, 20, 0.0, fill=False)[0])
    assert got[0].shape == crop.shape
