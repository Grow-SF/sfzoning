"""Word boxes from the detector's word heads (`wordread.words`): the components of the core chance, split where a
two-word model of their px tags is likelier than a one-word model (PAN's tags, Gaussian about each word's mean), each
grown back by DBNet's unclip or rebuilt from its px's geometry votes (EAST's locality-aware NMS), and the level chosen
per quarter by the headline over Terra's code words.

The maps are synthetic: Terra-like words on a blank page, and the outputs a perfect net would give, drawn by the
detector's own target code (`data.crop_targets(..., words=True)`: the cores, and the distances from each core px to
its word's sides), so a box that comes back right has inverted exactly what the heads learn."""
import dataclasses

import cv2
import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver import grid, paths, wordboxes
from sfzmaps.solver.textnet import data
from sfzmaps.solver.wordread import shrink, words

R = 0.4           # the ruled shrink ratio
SIZE = 240        # the synthetic pages are SIZE x SIZE px
LEVEL = 0.5
REFINES = ("unclip", "votes")
# the measured parameters, here test values: the within-word tag spread, the log prior odds of a second word, the IoU
# at which two votes are merged (Task 8 measures all three on the training quarters' predictions)
KW = {"tag_sigma": 0.3, "split_prior": -20.0, "merge_iou": 0.5}


def _rect(cx, cy, length, height, angle=0.0) -> np.ndarray:
    """A word's rectangle (4 corners, the first edge along the reading direction, as `boxes.word_rect` lays them)."""
    a = np.deg2rad(angle)
    v = np.array([np.cos(a), -np.sin(a)])
    n = np.array([-v[1], v[0]])
    c = np.array([cx, cy], float)
    return np.array([c - v * length / 2 - n * height / 2, c + v * length / 2 - n * height / 2,
                     c + v * length / 2 + n * height / 2, c - v * length / 2 + n * height / 2])


def _sheet(rects, lines=(), cores=None) -> data.SheetData:
    """An in-memory sheet: each word's ink a block inside it (its glyphs), plus `lines` ((x0, y0), (x1, y1),
    thickness) of ink; the words' cores shrunk at R unless `cores` are given."""
    gray = np.full((SIZE, SIZE), 255, np.uint8)
    for r in rects:                                 # the glyphs: the word's middle 60% of height, 3 px in from each end
        c = r.mean(0)
        v, n = r[1] - r[0], r[3] - r[0]
        half_l, half_h = np.linalg.norm(v) / 2 - 3, np.linalg.norm(n) * 0.3
        v, n = v / np.linalg.norm(v), n / np.linalg.norm(n)
        inner = [c - v * half_l - n * half_h, c + v * half_l - n * half_h, c + v * half_l + n * half_h,
                 c - v * half_l + n * half_h]
        cv2.fillPoly(gray, [np.round(inner).astype(np.int32)], 0)
    for p, q, thick in lines:
        cv2.line(gray, p, q, 0, thick)
    rects = np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32)
    sd = data.SheetData(
        series=paths.SERIES, sheet="s", height=SIZE, width=SIZE, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=rects, kinds=np.zeros(len(rects), np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)), ft_per_px=float("nan"))
    if cores is None:
        return shrink.prepare([sd], R)[0]
    return dataclasses.replace(sd, cores=np.asarray(cores, np.int32).reshape(-1, 4, 2))


def _heads(sd: data.SheetData, tags=None, noise=0.0, seed=0):
    """What a perfect net gives on the sheet: core chance, geo (the four distances and the reading angle), tags (one
    vector per word: `tags[k]`, or the word's number x 10 on the first channel; plus Gaussian noise of sd `noise` on
    every px, seeded), ink."""
    t = data.crop_targets(sd, SIZE // 2, SIZE // 2, SIZE, 1.0, 0, words=True)
    n = len(sd.words)
    vecs = np.zeros((n + 1, 4), np.float32)
    vecs[1:] = np.asarray(tags, np.float32) if tags is not None else np.c_[10.0 * np.arange(n), np.zeros((n, 3))]
    tag = np.moveaxis(vecs[t["tag_id"]], -1, 0)
    tag += noise * np.random.default_rng([seed, 1]).standard_normal(tag.shape).astype(np.float32)
    return t["core"].astype(np.float32), t["geo"].astype(np.float32), tag, t["ink"]


def _g(df: pd.DataFrame, k: int, key="g") -> np.ndarray:
    return df[[f"{key}{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float)[k].reshape(4, 2)


def _iou(a, b) -> float:
    inter = cv2.intersectConvexConvex(np.float32(a), np.float32(b))[0]
    area = wordboxes.poly_area(np.array([a, b], float))
    return float(inter / (area.sum() - inter))


def _turn(a, b) -> float:
    """The angle between two reading directions (degrees), in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


def _matches(df, rects):
    """Each rectangle's box (the best IoU) and that IoU."""
    out = []
    for r in rects:
        ious = [_iou(_g(df, k), r) for k in range(len(df))]
        out.append((int(np.argmax(ious)), max(ious)))
    return out


def _right_words(df, sd, refine):
    """Each word has its own box: by its votes, the word's rectangle; by unclip, its core grown back at R (these
    tests' cores touch, so they are drawn by hand, not shrunk at R: unclip grows what the net would give)."""
    want = sd.words.astype(float) if refine == "votes" else shrink.unclip(sd.cores.astype(float), R)
    found = _matches(df, want)
    assert sorted(k for k, _ in found) == list(range(len(want)))
    for (k, iou), r in zip(found, want):
        if refine == "votes":
            assert np.abs(_g(df, k) - r).max() < 1e-3
        else:
            assert iou > 0.9            # the painted px of a core (one column of each pair shared) grown back


# --- words that touch --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("refine", REFINES)
def test_two_touching_words_whose_ink_joins_become_two_boxes(refine):
    rects = [_rect(70, 70, 60, 20), _rect(130, 70, 60, 20)]      # side by side, their glyphs joined by a stroke
    sd = _sheet(rects, lines=[((90, 70), (110, 70), 3)])
    core, geo, tag, ink = _heads(sd)
    _, lab = cv2.connectedComponents(ink.view(np.uint8), connectivity=8)
    assert len(np.unique(lab[ink])) == 1                           # one connected piece of ink: today's one box

    df = words.core_words(core, geo, tag, ink, LEVEL, R, refine, **KW)

    assert len(df) == 2
    for (k, iou), r in zip(_matches(df, sd.words.astype(float)), sd.words.astype(float)):
        # votes: the targets' own rectangle; unclip: the painted core grown back (px-rounded), within 1 px
        assert np.abs(_g(df, k) - r).max() < (1e-3 if refine == "votes" else 1.0)
    assert sorted(_matches(df, sd.words.astype(float))[i][0] for i in range(2)) == [0, 1]


@pytest.mark.parametrize("refine", REFINES)
def test_touching_cores_split_when_two_tag_clouds_are_likelier_than_one(refine):
    rects = [_rect(70, 70, 60, 20), _rect(130, 70, 60, 20)]
    cores = [_rect(71, 70, 58, 12), _rect(129, 70, 58, 12)]         # meeting at x = 100: one component of core
    sd = _sheet(rects, cores=np.round(cores))
    # two clouds of tags 2 apart, each spread 0.3 about its word's mean
    core, geo, tag, ink = _heads(sd, tags=[[0, 0, 0, 0], [2, 0, 0, 0]], noise=0.3)
    _, lab = cv2.connectedComponents((core >= LEVEL).view(np.uint8), connectivity=8)
    assert lab.max() == 1

    # the separation is large against the measured spread: two words
    df = words.core_words(core, geo, tag, ink, LEVEL, R, refine, **KW)
    assert len(df) == 2
    assert set(df["form"]) == {f"{refine}+split"}
    assert (df["split_llr"] > -KW["split_prior"]).all()
    assert df["tag_spread"].to_numpy() == pytest.approx([0.3, 0.3], rel=0.1)       # each word's own px: sd 0.3
    _right_words(df, sd, refine)

    # the same clouds, the spread measured large (3): one word
    one = words.core_words(core, geo, tag, ink, LEVEL, R, refine, **{**KW, "tag_sigma": 3.0})
    assert len(one) == 1
    assert one["form"].tolist() == [refine]
    assert np.isnan(one["split_llr"].iloc[0])
    # ... and with a second word a priori far less likely than the clouds make it: one word
    assert len(words.core_words(core, geo, tag, ink, LEVEL, R, refine, **{**KW, "split_prior": -1e9})) == 1


@pytest.mark.parametrize("refine", REFINES)
def test_a_core_whose_tags_split_in_three_gives_three_words(refine):
    rects = [_rect(50, 70, 50, 20), _rect(100, 70, 50, 20), _rect(150, 70, 50, 20)]
    cores = [_rect(51, 70, 48, 12), _rect(100, 70, 50, 12), _rect(149, 70, 48, 12)]     # one strip of core
    sd = _sheet(rects, cores=np.round(cores))
    core, geo, tag, ink = _heads(sd, tags=[[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]], noise=0.3)
    df = words.core_words(core, geo, tag, ink, LEVEL, R, refine, **KW)
    assert len(df) == 3
    _right_words(df, sd, refine)
    assert (df["split_llr"] > -KW["split_prior"]).all()
    assert df["tag_spread"].to_numpy() == pytest.approx([0.3] * 3, rel=0.1)


# --- lines ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("refine", REFINES)
def test_a_line_touching_or_crossing_a_word_neither_extends_its_box_nor_splits_it(refine):
    rect = _rect(120, 100, 80, 24)
    lines = [((0, 112), (SIZE - 1, 112), 2),        # along the word's foot, across the whole page
             ((120, 0), (120, SIZE - 1), 2)]        # crossing the word's middle, top to bottom
    sd = _sheet([rect], lines=lines)
    core, geo, tag, ink = _heads(sd)
    df = words.core_words(core, geo, tag, ink, LEVEL, R, refine, **KW)
    assert len(df) == 1
    assert np.abs(_g(df, 0) - sd.words[0]).max() < (1e-3 if refine == "votes" else 1.0)
    t = _g(df, 0, "t")
    lo, hi = sd.words[0].min(0), sd.words[0].max(0)
    assert (t.min(0) >= lo - 1e-6).all() and (t.max(0) <= hi + 1e-6).all()     # the trimmed box stays in the word
    assert df["ink_px"].iloc[0] > 0


def test_votes_keep_the_word_when_the_core_bleeds_onto_a_line():
    """The core runs on along a line touching the word (above the level), its px voting for thin boxes along the line:
    the votes of the word's own px still decide its box."""
    rect = _rect(100, 100, 80, 24)
    sd = _sheet([rect], lines=[((140, 104), (220, 104), 2)])
    core, geo, tag, ink = _heads(sd)
    bleed = np.zeros_like(core, bool)
    bleed[103:106, 133:200] = True                  # touches the core (x 133...) and runs 60 px on along the line
    bleed &= core < LEVEL
    core[bleed] = 0.9
    geo[:4, bleed] = np.array([1.0, 1.0, 30.0, 30.0], np.float32)[:, None]     # thin boxes 60 px long, on the line
    geo[4, bleed] = 0
    tag[:, bleed] = 0
    _, lab = cv2.connectedComponents((core >= LEVEL).view(np.uint8), connectivity=8)
    assert lab.max() == 1
    df = words.core_words(core, geo, tag, ink, LEVEL, R, "votes", **KW)
    assert len(df) == 1
    assert np.abs(_g(df, 0) - sd.words[0]).max() < 1e-3


# --- the page edge ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("refine", REFINES)
@pytest.mark.parametrize("where", ["left", "bottom"])
def test_a_core_at_the_page_edge_gives_a_clipped_box(refine, where):
    rect = _rect(10, 70, 60, 20) if where == "left" else _rect(120, SIZE - 2, 60, 20)    # part off the page
    sd = _sheet([rect])
    core, geo, tag, ink = _heads(sd)
    df = words.core_words(core, geo, tag, ink, LEVEL, R, refine, **KW)
    assert len(df) == 1                             # not dropped
    g, t = _g(df, 0), _g(df, 0, "t")
    for box in (g, t):                              # clipped: never past the page, never wrapped to its far side
        assert (box >= -1e-6).all() and (box <= SIZE + 1e-6).all()
    word = sd.words[0].astype(float)
    # the other sides: votes give the word's own; unclip grows a core the page cut by less than the word's D
    tol = 1e-3 if refine == "votes" else 2.0
    if where == "left":
        assert g[:, 0].min() == pytest.approx(0.0, abs=1e-6)
        assert g[:, 0].max() == pytest.approx(word[:, 0].max(), abs=tol)
        assert g[:, 1].min() == pytest.approx(word[:, 1].min(), abs=tol)
    else:
        assert g[:, 1].max() == pytest.approx(SIZE, abs=1e-6)
        assert g[:, 1].min() == pytest.approx(word[:, 1].min(), abs=tol)
        assert g[:, 0].min() == pytest.approx(word[:, 0].min(), abs=tol)
    assert df["ink_px"].iloc[0] > 0
    assert data.reading_angle(g) == 0               # still read left to right


# --- rotated words ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [30, 90, 150, -45, 180])
def test_a_rotated_words_box_comes_back_from_its_votes(angle):
    """Within 1 px per corner of the rectangle the targets were drawn from (its corners rounded to px, so not quite a
    rectangle: each vote gives the rectangle at the word's whole-degree angle around those corners), the first edge
    along the reading direction."""
    rect = _rect(120, 120, 100, 24, angle)
    sd = _sheet([rect])
    core, geo, tag, ink = _heads(sd)
    df = words.core_words(core, geo, tag, ink, LEVEL, R, "votes", **KW)
    assert len(df) == 1
    g = _g(df, 0)
    assert np.abs(g - sd.words[0]).max() < 1.0
    assert _turn(data.reading_angle(g), angle) <= 1
    assert wordboxes.axis_error(df["axis"].iloc[0], angle) < 1.0
    # unclip: the painted core grown back, oriented by the predicted reading direction
    un = words.core_words(core, geo, tag, ink, LEVEL, R, "unclip", **KW)
    assert _iou(_g(un, 0), sd.words[0]) > 0.85
    assert _turn(data.reading_angle(_g(un, 0)), angle) <= 1


def test_the_direction_given_as_cos_and_sin_reads_the_same_as_degrees():
    rects = [_rect(80, 80, 90, 22, 30), _rect(170, 170, 70, 20, -60)]
    sd = _sheet(rects)
    core, geo, tag, ink = _heads(sd)
    a = np.deg2rad(geo[4])
    six = np.concatenate([geo[:4], 3 * np.cos(a)[None], 3 * np.sin(a)[None]])     # any length: the net's channels 5-6
    for refine in REFINES:
        pd.testing.assert_frame_equal(words.core_words(core, geo, tag, ink, LEVEL, R, refine, **KW),
                                      words.core_words(core, six, tag, ink, LEVEL, R, refine, **KW), atol=1e-6)


# --- the table ---------------------------------------------------------------------------------------------------

def test_the_boxes_table_has_the_word_boxes_columns_with_their_meaning():
    rects = [_rect(70, 70, 60, 20), _rect(160, 160, 100, 26, 90)]
    sd = _sheet(rects)
    core, geo, tag, ink = _heads(sd)
    kind_p = np.zeros((4, SIZE, SIZE), np.float32)
    kind_p[0] = 0.75
    kind_p[2] = 0.25
    df = words.core_words(core, geo, tag, ink, LEVEL, R, "votes", kind_p=kind_p, sheet="p07", **KW)
    assert list(df.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]
    assert words.EXTRA_COLUMNS == ("form", "core_px", "tag_spread", "p_word", "split_llr")
    assert df["sheet"].tolist() == ["p07", "p07"]
    assert df["i"].tolist() == [0, 1]
    assert (df["level"] == LEVEL).all()
    assert df["form"].tolist() == ["votes", "votes"]
    tag_id = data.crop_targets(sd, SIZE // 2, SIZE // 2, SIZE, 1.0, 0, words=True)["tag_id"]
    assert df["core_px"].tolist() == [int((tag_id == k + 1).sum()) for k in range(2)]
    # not placed on the map yet: the caller places the boxes (`prints._place`, as `wordboxes.sheet_boxes` does)
    assert (df["quarter"] == -1).all() and df["x"].isna().all() and (df["lab_lot"] == -1).all()
    assert not df["street"].any()
    for k in range(2):
        g, t = _g(df, k), _g(df, k, "t")
        assert (df["page_x"].iloc[k], df["page_y"].iloc[k]) == pytest.approx(tuple(g.mean(0)))
        assert df["length"].iloc[k] == pytest.approx(np.linalg.norm(g[1] - g[0]))
        assert df["height"].iloc[k] == pytest.approx(np.linalg.norm(g[3] - g[0]))
        assert df["t_length"].iloc[k] == pytest.approx(np.linalg.norm(t[1] - t[0]))
        assert df["t_height"].iloc[k] == pytest.approx(np.linalg.norm(t[3] - t[0]))
        assert df["t_length"].iloc[k] < df["length"].iloc[k]        # the glyphs, trimmed inside the box
        assert wordboxes.axis_error(df["axis"].iloc[k], data.reading_angle(g)) < 0.5
        assert df["mean_chance"].iloc[k] == df["p_word"].iloc[k] == pytest.approx(1.0)
        assert np.isnan(df["split_llr"].iloc[k]) and df["tag_spread"].iloc[k] == 0
        assert (df["p_zoning"].iloc[k], df["p_street"].iloc[k]) == pytest.approx((0.75, 0.25))
    assert wordboxes.axis_error(df["axis"].to_numpy(), [0.0, 90.0]).max() < 0.5


def test_refine_must_be_unclip_or_votes_and_votes_need_their_merge_iou():
    sd = _sheet([_rect(70, 70, 60, 20)])
    core, geo, tag, ink = _heads(sd)
    with pytest.raises(ValueError, match="refine"):
        words.core_words(core, geo, tag, ink, LEVEL, R, "both", **KW)
    with pytest.raises(ValueError, match="merge_iou"):
        words.core_words(core, geo, tag, ink, LEVEL, R, "votes", tag_sigma=0.3, split_prior=-20.0)


def test_no_core_gives_an_empty_table():
    z = np.zeros((SIZE, SIZE), np.float32)
    df = words.core_words(z, np.zeros((5, SIZE, SIZE), np.float32), np.zeros((4, SIZE, SIZE), np.float32),
                          z > 0, LEVEL, R, "votes", **KW)
    assert len(df) == 0
    assert list(df.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]


# --- the level -------------------------------------------------------------------------------------------------------

def _terra_and_boxes():
    """Terra's words on one sheet, two code words a quarter (and a street name, not a code), and each level's boxes
    and reads.  Level 0.5 reads right: nw 2/2, ne 0/2 (one word has no box at all), sw 1/2, se 1/2; level 0.7: nw 0/2,
    ne 2/2, sw 1/2, se 1/2."""
    rows, centres = [], []
    for q in range(4):
        for j in range(2):
            cx, cy = 100 + 200 * q, 100 + 100 * j
            r = _rect(cx, cy, 40, 16)
            rows.append({"sheet": "s", "text": "RH-2", "kind": "zoning code", "quarter": q,
                         **dict(zip(words.measures.C_COLS, r.reshape(8).tolist()))})
            centres.append((cx, cy))
    street = _rect(1000, 100, 120, 20)
    rows.append({"sheet": "s", "text": "MISSION ST", "kind": "street name", "quarter": 0,
                 **dict(zip(words.measures.C_COLS, street.reshape(8).tolist()))})
    terra = pd.DataFrame(rows)
    right = {0.5: [1, 1, 0, None, 1, 0, 1, 0], 0.7: [0, 0, 1, 1, 1, 0, 1, 0]}     # None: no box holds the word
    boxes, reads = {}, {}
    for level, marks in right.items():
        kept = [(k, m) for k, m in enumerate(marks) if m is not None] + [(8, 1)]    # the street name has a box too
        g = [(_rect(*centres[k], 44, 20) if k < 8 else _rect(1000, 100, 124, 24)).reshape(8) for k, _ in kept]
        df = pd.DataFrame(np.array(g), columns=words.measures.G_COLS, index=[10 * k + 3 for k, _ in kept])
        df.insert(0, "sheet", "s")
        boxes[level] = df
        reads[level] = {10 * k + 3: {"closed": {"RH-2": 0.9 if m else 0.1, "NC-3": 0.1 if m else 0.9}, "top": "RH-2"}
                        for k, m in kept}
    return terra, boxes, reads


def test_choose_level_scores_each_quarter_on_the_other_three_with_a_missed_box_wrong():
    terra, boxes, reads = _terra_and_boxes()
    got = words.choose_level(boxes, reads, terra, forms={"RH-2": "RH-2", "NC-3": "NC-3"})
    # nw on ne, sw, se: 0.5 gets 2/6, 0.7 gets 4/6; ne on nw, sw, se: 4/6 against 2/6; sw and se: 3/6 each (the
    # word with no box counted wrong; dropped, 0.5 would get 3/5 on sw and se): a tie, to the higher level
    assert got["level"] == {"nw": 0.7, "ne": 0.5, "sw": 0.7, "se": 0.7}
    assert got["right_on_others"]["nw"] == {"0.500": pytest.approx(2 / 6), "0.700": pytest.approx(4 / 6)}
    assert got["right_on_others"]["sw"] == {"0.500": pytest.approx(3 / 6), "0.700": pytest.approx(3 / 6)}
    assert got["words_on_others"] == {"nw": 6, "ne": 6, "sw": 6, "se": 6}       # the street name is not a code


def test_choose_level_reads_terras_words_not_the_owners_answers():
    terra, boxes, reads = _terra_and_boxes()
    flipped = terra.assign(text=terra["text"].replace({"RH-2": "NC-3"}))       # Terra's strings decide what is right
    got = words.choose_level(boxes, reads, flipped, forms={"RH-2": "RH-2", "NC-3": "NC-3"})
    # every read right before is now wrong and the other way round; the word with no box stays wrong
    assert got["right_on_others"]["nw"] == {"0.500": pytest.approx(3 / 6), "0.700": pytest.approx(2 / 6)}
    assert got["level"]["nw"] == 0.5
