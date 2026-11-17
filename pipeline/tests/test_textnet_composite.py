"""Composites inside real crops (`textnet.composite`): real clean words ORed onto real line work, and real line pieces
ORed across clean words, with exact masks; the bank of references they come from, which holds nothing held out; the
measures the bank rests on; and their way into `train.draw` (none when the recipe's share is 0)."""
import dataclasses
import hashlib
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_solver_textnet import OLD, TINY, maps  # noqa: F401

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import composite, data, train, zones

# sha256 (`_digest`) of draw_n(maps, train.Recipe(), seed=3) with train.py at 102f422, before composites existed
PINNED_BATCHES_SEED3 = "dd2f5a8193a451c7b9774f45bb76b9d8beeccea87465456dd8af6c0935517820"
FT_PER_PX = 4.0           # the run rule's length is round(200 / 4) = 50 px here
ORIGIN = (6000000.0, 2100000.0)                 # the map point (EPSG:2227 ft) of page px (0, 0)
PAGE = 640
SHEET = "2000-digital-zoning-p01"
LINES = (200, 440)        # rows and columns of the printed line grid, 2 px
CLEAN = (("RM-2", (40, 100), "zoning code"), ("RH-1", (250, 100), "zoning code"),
         ("12", (70, 330), "block number"), ("ELM ST", (250, 330), "street name"))
CROSSED = ("RC-3", (470, 100), "zoning code")   # a line through its middle: not clean
ZONE_POINT = (490, 490)   # (x, y): one of Sol's points on no word, so it keeps an unknown zone
CENTRE = (440, 440)       # (row, col): a crop of 128 px here holds the lines' crossing, part of the zone, no word
SERIES = paths.SERIES


# --- helpers ---------------------------------------------------------------------------------------------------------

def _digest(batches) -> str:
    """sha256 of batches of crops: every key, dtype, shape and byte, in order."""
    h = hashlib.sha256()
    for batch in batches:
        for crop in batch:
            for k in sorted(crop):
                v = np.ascontiguousarray(crop[k])
                h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
                h.update(v.tobytes())
    return h.hexdigest()


def batches_equal(a, b) -> bool:
    """Whether two lists of batches (or their `_digest`s) are the same, byte for byte."""
    return (a if isinstance(a, str) else _digest(a)) == (b if isinstance(b, str) else _digest(b))


def draw_n(maps_or_sheets, recipe, seed, bank=None, n=3) -> list:
    """`n` batches drawn by `train.draw` from one generator seeded `seed`, on the given sheets or (a fixture's folder)
    on the `maps` fixture's two sheets with the validation area held out."""
    sheets = maps_or_sheets if isinstance(maps_or_sheets, list) else \
        data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    pools = train.centre_pools(sheets)
    rng = np.random.default_rng(seed)
    extra = {} if bank is None else {"bank": bank}
    return [train.draw(sheets, pools, recipe, rng, **extra) for _ in range(n)]


def _sheet(gray, words=(), points=(), sheet=SHEET) -> data.SheetData:
    """A 2000 SheetData made in memory: the page `gray`, placed north up at FT_PER_PX ft per page px with page px
    (0, 0) at ORIGIN; Terra's `words` (axis box, kind, reading angle) and Sol's `points` (x, y)."""
    h, w = gray.shape
    step = data.PLACEMENT_STEP
    gx, gy = np.meshgrid(np.arange(0, w - 1 + step, step, dtype=float), np.arange(0, h - 1 + step, step, dtype=float))
    world = np.stack([ORIGIN[0] + FT_PER_PX * gx, ORIGIN[1] - FT_PER_PX * gy], -1)
    rects = [data.word_rect(box, angle) for box, _, angle in words]
    return data.SheetData(
        series=SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.array([data.KINDS[k] for _, k, _ in words], np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.array(points, np.int32).reshape(-1, 2), world=world, ft_per_px=FT_PER_PX)


def _glyphs(text, org, scale=0.8, thick=2) -> np.ndarray:
    """A word's glyphs alone (cv2.putText, Hershey simplex), bool PAGE x PAGE."""
    pg = np.full((PAGE, PAGE), 255, np.uint8)
    cv2.putText(pg, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick)
    return pg < grid.INK_BELOW


def _box_of(mask, margin=2) -> list[int]:
    """The axis box (x0, y0, x1, y1) of a mask's px, grown by `margin`: a word box as Terra gives one."""
    rows, cols = np.nonzero(mask)
    return [int(cols.min()) - margin, int(rows.min()) - margin, int(cols.max()) + margin, int(rows.max()) + margin]


def _centre(box) -> tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def _grid_lines() -> np.ndarray:
    pg = np.full((PAGE, PAGE), 255, np.uint8)
    for v in LINES:
        cv2.line(pg, (0, v), (PAGE - 1, v), 0, 2)
        cv2.line(pg, (v, 0), (v, PAGE - 1), 0, 2)
    return pg < grid.INK_BELOW


def _prepared(gray, words, points) -> data.SheetData:
    return zones.prepare([_sheet(gray, words, points)], (), cache=None)[0]


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Each px's distance (px, Euclidean) to the nearest px of `mask` (inf with none)."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return cv2.distanceTransform((~mask).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def bank_fx():
    """One 2000 sheet: a grid of printed lines, four clean words (two codes, a block number, a street name) inside the
    grid's cells, a code crossed by a line, Sol's points on every word and one on none (a zone); its bank."""
    ink = _grid_lines()
    words, points, clean_ink = [], [], []
    for text, org, kind in (*CLEAN, CROSSED):
        g = _glyphs(text, org)
        ink |= g
        box = _box_of(g)
        words.append((box, kind, 0))
        points.append(_centre(box))
        if (text, org, kind) != CROSSED:
            clean_ink.append(g[box[1]:box[3] + 1, box[0]:box[2] + 1])
    row = _centre(words[-1][0])[1]
    line = np.full((PAGE, PAGE), 255, np.uint8)
    cv2.line(line, (450, row), (630, row), 0, 2)
    ink |= line < grid.INK_BELOW
    sd = _prepared(np.where(ink, 0, 255).astype(np.uint8), words, [*points, ZONE_POINT])
    bank = composite.Bank.build([sd], (), seed=0)
    return SimpleNamespace(sd=sd, sheets=[sd], bank=bank, centre=CENTRE, words=words,
                           real_texture=composite.texture_stats(clean_ink))


@pytest.fixture
def bank_with_one_huge_word():
    """A sheet with two clean words: one too big for a 64 px crop (its strokes long enough for the run rule) and a
    small one; a printed line far from both; the page's top-left 64 x 64 px blank."""
    huge = _glyphs("RM-2", (200, 420), scale=3.0, thick=6)
    small = _glyphs("12", (100, 200))
    ink = huge | small
    ink[600:602, 20:620] = True
    words = [(_box_of(huge), "zoning code", 0), (_box_of(small), "block number", 0)]
    sd = _prepared(np.where(ink, 0, 255).astype(np.uint8), words, [_centre(b) for b, _, _ in words])
    bank = composite.Bank.build([sd], (), seed=0)
    return SimpleNamespace(sd=sd, bank=bank, huge=_box_of(huge), small=_box_of(small))


def _forced(bank, a_share=None, cross=None):
    """The bank making only composites of kind (a) (a_share 1) or (b) (0), and placing words on line ink with the
    share `cross` for every kind."""
    changes = {}
    if a_share is not None:
        changes["a_share"] = a_share
    if cross is not None:
        changes["cross"] = {s: {k: cross for k in data.KINDS.values()} for s in bank.cross}
    return dataclasses.replace(bank, **changes)


# --- the brief's tests -----------------------------------------------------------------------------------------------

def test_a_word_onto_lines_keeps_exact_masks(bank_fx, rng):
    c = bank_fx.bank.crop(rng, bank_fx.sd, *bank_fx.centre, 128, 1.0, 0)
    assert (c["text"].astype(bool) <= c["ink"]).all()
    assert (c["ink"] & ~c["text"].astype(bool) & c["known"]).any()


def test_composites_zero_leaves_batches_unchanged(maps):  # noqa: F811
    assert batches_equal(draw_n(maps, train.Recipe(composites=0.0), seed=3), PINNED_BATCHES_SEED3)


def test_composites_repeat_from_the_seed(bank_fx):
    r = train.Recipe(composites=0.5)
    assert batches_equal(draw_n(bank_fx.sheets, r, 7, bank_fx.bank), draw_n(bank_fx.sheets, r, 7, bank_fx.bank))


def test_composite_word_ink_is_real_word_ink(bank_fx, rng):
    c = bank_fx.bank.crop(rng, bank_fx.sd, *bank_fx.centre, 128, 1.0, 0)
    t = composite.texture_stats([c["ink"] & c["text"].astype(bool)])
    assert t["pinholes_per_1000"] <= 3 * max(bank_fx.real_texture["pinholes_per_1000"], 1)


def test_a_word_that_fits_nowhere_takes_the_next_candidate(bank_with_one_huge_word, rng):
    fx = bank_with_one_huge_word
    bank = _forced(fx.bank, a_share=1.0)
    before = bank.tally["words_failed"]
    for _ in range(12):
        c = bank.crop(rng, fx.sd, 32, 32, 64, 1.0, 0)
        # every crop got the small word, the huge one never fitting
        assert c["text"].sum() == (_glyphs("12", (100, 200))).sum()
    assert bank.tally["words_failed"] > before


# --- the bank --------------------------------------------------------------------------------------------------------

def test_the_bank_holds_the_clean_words_and_long_line_pieces_by_reference(bank_fx):
    words = bank_fx.bank.words[SERIES]
    assert sorted(words.kind.tolist()) == sorted(data.KINDS[k] for _, _, k in CLEAN)
    assert words.rect.shape == (4, 4, 2) and set(words.sheet.tolist()) == {0}
    pieces = bank_fx.bank.pieces[SERIES]
    assert len(pieces) > 0
    for (x0, y0, x1, y1), (sx, sy) in zip(pieces.box, pieces.seed, strict=True):
        assert np.hypot(x1 - x0, y1 - y0) >= composite.PIECE_PX and x0 <= sx < x1 and y0 <= sy < y1
        m = composite.piece_mask(bank_fx.sd, (x0, y0, x1, y1), (sx, sy))
        assert m.shape == (y1 - y0, x1 - x0) and m[sy - y0, sx - x0]
        # a piece is line ink and nothing else
        assert (data._ink_window(bank_fx.sd, y0, x0, y1 - y0, x1 - x0) >= m).all()
        assert (data.line_window(bank_fx.sd, y0, x0, y1 - y0, x1 - x0) >= m).all()
    # the crossed code is the share crossed: one code of three, no block number, no street name
    assert bank_fx.bank.cross[SERIES][data.KINDS["zoning code"]] == pytest.approx(1 / 3)
    assert bank_fx.bank.cross[SERIES][data.KINDS["block number"]] == 0.0
    sizes = bank_fx.bank.sizes()
    assert sizes[SERIES]["clean_words"]["zoning code"] == 2 and sizes[SERIES]["pieces"] == len(pieces)


def test_the_bank_takes_nothing_from_an_area_held_out(bank_fx):
    right = data.MapRect(ORIGIN[0] + FT_PER_PX * 320, ORIGIN[1] - FT_PER_PX * 1000, ORIGIN[0] + FT_PER_PX * 1000,
                         ORIGIN[1] + 10)              # page columns 320 on
    bank = composite.Bank.build(bank_fx.sheets, (right,), seed=0)
    words = bank.words[SERIES]
    assert len(words) and (words.rect[..., 0].max(axis=1) + composite.CLEAN_PX < 320).all()
    assert len(words) < len(bank_fx.bank.words[SERIES])
    pieces = bank.pieces[SERIES]
    assert len(pieces) and (pieces.box[:, 2] <= 320 - composite.CLEAN_PX).all()
    # the crossing share counts only the words whose centre is outside it: the crossed code is in it
    assert bank.cross[SERIES][data.KINDS["zoning code"]] == 0.0
    # a composite made of it holds the area out too: no lesson there
    held = data.in_area(bank.sheets[0], right, 0, 0, PAGE, PAGE)
    only_b = _forced(bank, a_share=0.0)
    for seed in range(6):
        c = only_b.crop(np.random.default_rng(seed), bank_fx.sd, *CENTRE, 512, 1.0, 0)
        assert only_b.last["mode"] == "b"
        y0, x0, src = only_b.last["window"]
        assert held[y0:y0 + src, x0:x0 + src].sum() > 1000
        assert not c["known"][held[y0:y0 + src, x0:x0 + src]].any()


def test_a_bank_needs_the_sheets_line_px():
    sd = _sheet(np.full((PAGE, PAGE), 255, np.uint8))
    with pytest.raises(ValueError, match="line px"):
        composite.Bank.build([sd], (), seed=0)


def test_a_bank_with_no_word_that_fits_leaves_the_crop_as_it_is(bank_with_one_huge_word, rng):
    fx = bank_with_one_huge_word
    words = fx.bank.words[SERIES]
    only_huge = dataclasses.replace(fx.bank, a_share=1.0,
                                    words={SERIES: words.take(np.flatnonzero(words.kind == data.KINDS["zoning code"]))})
    c = only_huge.crop(rng, fx.sd, 32, 32, 64, 1.0, 0)
    base = data.crop_targets(fx.sd, 32, 32, 64, 1.0, 0, labels="lines-unknown")
    assert all(np.array_equal(c[k], base[k]) for k in base)
    assert only_huge.last["mode"] == "none"


# --- (a) a word onto lines -------------------------------------------------------------------------------------------

def test_a_word_goes_on_line_ink_clear_of_words_boxes_and_zones(bank_fx):
    bank = _forced(bank_fx.bank, a_share=1.0, cross=1.0)
    base = data.crop_targets(bank_fx.sd, *CENTRE, 128, 1.0, 0, labels="lines-unknown")
    y0, x0 = CENTRE[0] - 64, CENTRE[1] - 64
    line_ink = data.line_window(bank_fx.sd, y0, x0, 128, 128) & base["ink"]
    to_forbidden = _distance_to((base["text"] > 0) | ~base["known"])
    assert np.isfinite(to_forbidden).any()      # the zone reaches into the crop
    overlaps = 0
    for seed in range(10):
        c = bank.crop(np.random.default_rng(seed), bank_fx.sd, *CENTRE, 128, 1.0, 0)
        assert bank.last["mode"] == "a" and bank.last["on_line"]
        new = c["text"].astype(bool) & ~base["text"].astype(bool)
        assert new.any() and (new <= c["ink"]).all() and c["known"][new].all()
        assert to_forbidden[new].min() > composite.CLEAN_PX
        word = bank.words[SERIES]
        assert set(c["kind"][new].tolist()) == {int(word.kind[bank.last["word"]])}
        # on the line: line ink under the word's box, taught not text where the letters are not
        py, px, ph, pw = bank.last["placed"]
        assert line_ink[py:py + ph, px:px + pw].any()
        assert (c["known"] & line_ink & ~c["text"].astype(bool)).any()
        overlaps += int((new & line_ink).any())
    assert overlaps                              # some letters lie on the line: those px are text


def test_a_word_off_the_lines_keeps_clear_of_line_px(bank_fx):
    bank = _forced(bank_fx.bank, a_share=1.0, cross=0.0)
    base = data.crop_targets(bank_fx.sd, *CENTRE, 128, 1.0, 0, labels="lines-unknown")
    to_line = _distance_to(data.line_window(bank_fx.sd, CENTRE[0] - 64, CENTRE[1] - 64, 128, 128))
    placed = 0
    for seed in range(10):
        c = bank.crop(np.random.default_rng(seed), bank_fx.sd, *CENTRE, 128, 1.0, 0)
        new = c["text"].astype(bool) & ~base["text"].astype(bool)
        if bank.last["mode"] == "a":
            placed += 1
            assert not bank.last["on_line"] and to_line[new].min() > composite.CLEAN_PX
    assert placed


# --- (b) a line onto a word ------------------------------------------------------------------------------------------

def test_a_line_across_a_word_is_ink_not_text_and_the_word_stays_text(bank_fx):
    bank = _forced(bank_fx.bank, a_share=0.0)
    for seed in range(10):
        c = bank.crop(np.random.default_rng(seed), bank_fx.sd, *CENTRE, 128, 1.0, 0)
        info = bank.last
        assert info["mode"] == "b"
        y0, x0, src = info["window"]
        wsd = bank.sheets[info["sheet"]]
        base = data.crop_targets(wsd, info["centre"][0], info["centre"][1], 128, 1.0, 0, labels="lines-unknown")
        rect = bank.words[SERIES].rect[info["word"]]
        in_rect = data._paint(rect[None], np.ones(1, np.int32), y0, x0, src, src) > 0
        word_ink = base["ink"] & in_rect
        new_ink = c["ink"] & ~base["ink"]
        assert new_ink.any()                                   # the piece, ORed in
        assert not c["text"][new_ink].any() and c["known"][new_ink].all()
        assert c["text"][word_ink].all() and c["known"][word_ink].all()
        assert (c["text"].astype(bool) <= c["ink"]).all()
        assert (new_ink & in_rect).any()                       # across the word's box


def test_a_line_onto_a_word_teaches_its_long_strokes_as_text(bank_with_one_huge_word):
    fx = bank_with_one_huge_word
    words = fx.bank.words[SERIES]
    huge = np.flatnonzero(words.kind == data.KINDS["zoning code"])
    bank = dataclasses.replace(fx.bank, a_share=0.0, words={SERIES: words.take(huge)})
    c = bank.crop(np.random.default_rng(1), fx.sd, 32, 32, 320, 1.0, 0)
    y0, x0, src = bank.last["window"]
    base = data.crop_targets(fx.sd, *bank.last["centre"], 320, 1.0, 0, labels="lines-unknown")
    glyph = _glyphs("RM-2", (200, 420), scale=3.0, thick=6)[y0:y0 + src, x0:x0 + src]
    assert (~base["known"] & glyph).sum() > 100             # the label rule leaves its long strokes unknown
    assert c["text"][glyph].all() and c["known"][glyph].all()


# --- the page-scale path ---------------------------------------------------------------------------------------------

def test_the_page_scale_path_is_crop_targets(bank_fx):
    for size, scale, turns in ((128, 1.0, 0), (128, 0.6, 1), (96, 1.6, 2), (101, 1.3, 3)):
        parts = composite._page_parts(bank_fx.sd, *CENTRE, size, scale)
        got = composite._finish(parts, size, turns)
        want = data.crop_targets(bank_fx.sd, *CENTRE, size, scale, turns, labels="lines-unknown")
        assert sorted(got) == sorted(want)
        assert all(got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]) for k in want)


# --- the measures ----------------------------------------------------------------------------------------------------

def test_measure_counts_the_words_crossed_and_the_clean_words_by_kind(bank_fx):
    m = composite.measure(bank_fx.sheets)
    cross = m["cross_by_kind"]
    assert cross["zoning code"] == {"words": 3, "crossed": 1, "share": pytest.approx(1 / 3)}
    assert cross["block number"] == {"words": 1, "crossed": 0, "share": 0.0}
    assert cross["other"] == {"words": 0, "crossed": 0, "share": None}
    assert cross["all"] == {"words": 5, "crossed": 1, "share": pytest.approx(0.2)}
    assert m["clean_words"] == {"zoning code": 2, "block number": 1, "street name": 1, "other": 0}
    assert m["clean_word_texture"]["masks"] == 4
    assert m["line_texture"]["3"]["pieces"] >= 4         # the grid's lines (cv2.line at thickness 2 draws 3 px)
    right = data.MapRect(ORIGIN[0] + FT_PER_PX * 320, ORIGIN[1] - FT_PER_PX * 1000, ORIGIN[0] + FT_PER_PX * 1000,
                         ORIGIN[1] + 10)
    m = composite.measure(bank_fx.sheets, (right,))
    assert m["cross_by_kind"]["zoning code"] == {"words": 2, "crossed": 0, "share": 0.0}


def test_texture_stats_counts_isolated_px_pinholes_and_pieces():
    a = np.zeros((20, 20), bool)
    a[2:12, 2:12] = True
    a[6, 6] = False               # a pinhole
    a[16, 16] = True              # an isolated px
    b = np.zeros((5, 5), bool)
    b[1:4, 1:4] = True
    t = composite.texture_stats([a, b])
    ink = 99 + 1 + 9
    assert t["masks"] == 2 and t["ink_px"] == ink
    assert t["isolated_share"] == pytest.approx(1 / ink)
    assert t["pinholes_per_1000"] == pytest.approx(1000 / ink)
    assert t["pieces_per_word"] == pytest.approx(1.5)


def _bar(top_bumps: np.ndarray, width=4, length=200) -> np.ndarray:
    """A horizontal bar `width` px thick and `length` px long, its top edge raised 1 px at the columns where
    `top_bumps` is 1."""
    m = np.zeros((width + 6, length + 4), bool)
    m[3:3 + width, 2:2 + length] = True
    m[2, 2:2 + length] = top_bumps.astype(bool)
    return m


def test_line_texture_is_the_edges_offsets_from_a_straight_fit():
    straight = composite.line_texture([_bar(np.zeros(200))])
    assert straight["4"]["pieces"] == 1 and straight["4"]["sd"] == pytest.approx(0.0, abs=1e-6)
    bumps = (np.random.default_rng(0).random(200) < 0.2).astype(float)
    t = composite.line_texture([_bar(bumps)])["4"]
    # pooled over the two edges: the top's offsets from its fit (the bumps less their mean), the bottom's none
    assert t["sd"] == pytest.approx(np.sqrt(np.var(bumps) / 2), rel=0.05)
    assert abs(t["ac1"]) < 0.25
    blocky = np.repeat((np.random.default_rng(1).random(25) < 0.3).astype(float), 8)
    assert composite.line_texture([_bar(blocky)])["4"]["ac1"] > 0.6
    # a piece that is not one straight stroke (an L) is left out, and counted
    ell = np.zeros((120, 120), bool)
    ell[100:104, 0:110] = True
    ell[0:104, 100:104] = True
    t = composite.line_texture([ell])
    assert t["not_straight"] == 1 and not any(v["pieces"] for k, v in t.items() if k != "not_straight")


# --- into the training -----------------------------------------------------------------------------------------------

def test_the_composite_share_is_a_recipe_setting_named_only_when_set():
    assert train.Recipe().composites == 0.0 and train.Recipe().name() == "default"
    assert train.Recipe(composites=0.25).name() == "composites=0.25"
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="composites"):
            train.Recipe(composites=bad)


class _FakeBank:
    """Any object with `.crop(rng, sd, y, x, size, scale, turns)`: records its calls and draws once from the rng."""

    def __init__(self):
        self.calls = []

    def crop(self, rng, sd, y, x, size, scale, turns):
        self.calls.append((threading.current_thread().name, sd.sheet, y, x, size, scale, turns))
        rng.random()
        return data.crop_targets(sd, y, x, size, scale, turns)


def test_draw_asks_the_bank_for_a_share_of_its_crops(maps):  # noqa: F811
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    pools = train.centre_pools(sheets)
    bank = _FakeBank()
    train.draw(sheets, pools, train.Recipe(batch=6, crop=64, composites=0.0), np.random.default_rng(0), bank=bank)
    assert bank.calls == []
    train.draw(sheets, pools, train.Recipe(batch=6, crop=64, composites=1.0), np.random.default_rng(0), bank=bank)
    assert len(bank.calls) == 6 and all(size == 64 and 0.6 <= scale <= 1.6 for *_, size, scale, _ in bank.calls)
    with pytest.raises(ValueError, match="bank"):
        train.draw(sheets, pools, train.Recipe(batch=6, crop=64, composites=1.0), np.random.default_rng(0))


def test_fit_with_composites_needs_a_bank_and_trains_with_one(bank_fx):
    recipe = dataclasses.replace(TINY, composites=0.5, labels="lines-unknown")
    with pytest.raises(ValueError, match="bank"):
        train.fit(bank_fx.sheets, recipe, device="cpu", seed=0)
    before = bank_fx.bank.tally["a"] + bank_fx.bank.tally["b"]
    net = train.fit(bank_fx.sheets, recipe, device="cpu", seed=0, bank=bank_fx.bank)
    assert all(np.isfinite(v.numpy()).all() for v in net.state_dict().values() if v.is_floating_point())
    assert bank_fx.bank.tally["a"] + bank_fx.bank.tally["b"] > before
