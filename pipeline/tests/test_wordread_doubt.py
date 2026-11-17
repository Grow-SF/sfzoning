"""Doubt weights for Terra's words (`wordread.doubt`): the features of each word (its ink outside its box, the forced
height of a near-45-degree word, its kind, and the other tile's read of a word Terra was shown twice), the fit of
P(the two reads' strings agree) and of the pair's box IoU on those double reads, the weights (expected IoU, P(string
right)) with no cut, and the word heads' loss weighing each word's px by its weight (`data.crop_targets`,
`train.word_tensors`, `train.word_loss`)."""
import contextlib
import dataclasses
import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import boxes, data, model, train
from sfzmaps.solver.wordread import doubt

ORIGIN = (6000000.0, 2110000.0)       # the map point of page px (0, 0)
FT = 2.0                              # ft per page px


@contextlib.contextmanager
def _torch_kept():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng)


@pytest.fixture(autouse=True)
def _torch_as_it_was():
    with _torch_kept():
        yield


def _world(h, w) -> np.ndarray:
    """Placement samples of a page placed north up at FT ft per px, page px (0, 0) at ORIGIN."""
    step = data.PLACEMENT_STEP
    gx, gy = np.meshgrid(np.arange(0, w - 1 + step, step, dtype=float), np.arange(0, h - 1 + step, step, dtype=float))
    return np.stack([ORIGIN[0] + FT * gx, ORIGIN[1] - FT * gy], -1)


def _map_rect(x0, y0, x1, y1) -> data.MapRect:
    """The map rectangle holding page px x0..x1 - 1, y0..y1 - 1 (by their map points)."""
    return data.MapRect(ORIGIN[0] + FT * x0 - FT / 2, ORIGIN[1] - FT * y1 + FT / 2,
                        ORIGIN[0] + FT * x1 - FT / 2, ORIGIN[1] - FT * y0 + FT / 2)


def _sheet(words, gray=None, size=(240, 320), held=(), kinds=None, sheet="s", lines=None) -> data.SheetData:
    """An in-memory sheet placed north up: Terra's `words` (axis box, reading angle), ink `gray` (default: a stroke
    along each word's middle), the areas `held` held out, every word's core its own rectangle, its line px `lines`
    (bool page; default none)."""
    h, w = size
    if gray is None:
        gray = np.full((h, w), 255, np.uint8)
        for box, angle in words:
            r = data.word_rect(box, angle)
            a, b = (r[0] + r[3]) / 2, (r[1] + r[2]) / 2
            cv2.line(gray, tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b)), 0, 3)
    rects = np.round(np.array([data.word_rect(b, a) for b, a in words], float).reshape(-1, 4, 2)).astype(np.int32)
    lines = np.zeros((h, w), bool) if lines is None else lines
    return data.SheetData(
        series=paths.SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=rects, kinds=np.zeros(len(words), np.uint8) if kinds is None else np.array(kinds, np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.zeros((0, 2), np.int32), world=_world(h, w), ft_per_px=FT, held=tuple(held), cores=rects.copy(),
        line_bits=np.packbits(lines, axis=1))


def _item(text, box, kind="zoning code", angle=0, tile=(0, 0)):
    x0, y0, x1, y1 = box
    return {"text": text, "kind": kind, "x0": x0 - tile[0], "y0": y0 - tile[1], "x1": x1 - tile[0],
            "y1": y1 - tile[1], "direction": "left-to-right", "angle_deg": angle}


def _records(tiles, sheet="s", window="0,0,500,300"):
    """Terra's saved answers for a sheet: `tiles` [(page_box, [(text, page box, kind, angle), ...]), ...]."""
    return [{"series": paths.SERIES, "sheet": sheet, "page_box": list(pb),
             "script": {"args": ["terra", paths.SERIES, sheet, window, "out"]},
             "answer": {"items": [_item(t, b, k, a, tile=pb[:2]) for t, b, k, a in items]}} for pb, items in tiles]


def _sheet_of(records, gray, sheet="s", held=(), lines=None) -> data.SheetData:
    found = boxes.terra_boxes(paths.SERIES, sheet, records)
    return _sheet([(f["box"], float(f["angle_deg"])) for f in found], gray=gray, size=gray.shape, held=held,
                  kinds=[data.KINDS[f["kind"]] for f in found], sheet=sheet, lines=lines)


# --- the features ----------------------------------------------------------------------------------------------------

def test_forced_height_is_word_rects_near_diagonal_case():
    # word_rect forces the height (TYPICAL_HEIGHT) whatever the box: two different boxes give it exactly
    for angle in range(-180, 181):
        heights = [float(np.linalg.norm(r[3] - r[0])) for r in (boxes.word_rect((0, 0, 200, 150), angle),
                                                                  boxes.word_rect((0, 0, 170, 190), angle))]
        assert bool(doubt.forced_height(angle)) == all(abs(h - boxes.TYPICAL_HEIGHT) < 1e-9 for h in heights), angle
    assert doubt.forced_height(45) and not doubt.forced_height(0) and not doubt.forced_height(90)
    # the fit reads the continuous |cos^2 - sin^2| (1 along or across the page, 0 at 45 degrees), not the case
    angles = np.arange(-180, 181)
    a = np.deg2rad(angles)
    assert np.allclose(doubt.axis_of(angles), np.abs(np.cos(2 * a)))
    assert np.array_equal(doubt.forced_height(angles), doubt.axis_of(angles) <= doubt.NEAR_DIAGONAL)


def test_ink_outside_counts_the_letters_ink_and_line_share_the_line_ink_touching_the_word():
    gray = np.full((200, 300), 255, np.uint8)
    lines = np.zeros(gray.shape, bool)
    gray[60:63, 100:140] = 0                    # a bar of 120 px, half under word 0's box (columns 100-119)
    gray[100:110, 30:34] = 0                    # word 1's letter, wholly inside its box
    gray[100:110, 40:44] = 0
    gray[150:153, 200:260] = 0                  # word 2's letters...
    gray[140:180, 230:232] = 0                  # ... crossed by a line of 2 x 40 px, rows 140-179
    lines[140:180, 229:233] = True              # the line's px, grown by 1 px across (as `zones.find_lines` grows them)
    words = {"HALF": (100, 55, 119, 70), "WHOLE": (25, 95, 50, 115), "LINED": (195, 145, 265, 159),
             "EMPTY": (10, 10, 40, 30)}
    recs = _records([((0, 0, 300, 200), [(t, b, "zoning code", 0) for t, b in words.items()])], window="0,0,300,200")
    sd = _sheet_of(recs, gray, lines=lines)
    f = doubt.doubt_features(sd, recs)
    assert list(f["word"]) == [0, 1, 2, 3]                     # one row per word of the sheet, in its order
    f = f.set_index("text")
    assert f.loc["HALF", "ink_outside"] == pytest.approx(0.5) and f.loc["HALF", "ink_px"] == 60
    assert f.loc["HALF", "line_share"] == 0 and f.loc["WHOLE", "ink_outside"] == 0
    # the line doesn't count as the word's ink outside its box (its letters, cut in two by the line, are inside); it
    # is the line ink inside the rectangle (rows 145-159, columns 195-265): the line's 2 x 15 px and the bar's 2 x 3
    # px under the grown line px, of the rectangle's 71 x 15 px and of its 180 + 30 - 6 ink px
    assert f.loc["LINED", "ink_outside"] == 0
    assert f.loc["LINED", "line_share"] == pytest.approx(36 / (71 * 15))
    assert f.loc["LINED", "line_of_ink"] == pytest.approx(36 / (180 + 30 - 6))
    # no ink: all of it is outside, and no line ink
    assert f.loc["EMPTY", "ink_px"] == 0 and f.loc["EMPTY", "ink_outside"] == 1 and f.loc["EMPTY", "line_share"] == 0
    assert not f["forced_height"].any() and (f["axis"] == 1).all() and set(f["kind"]) == {"zoning code"}
    # Terra's distance from its tile's edge (the tile is the page, 300 x 200)
    assert list(f.loc[["HALF", "WHOLE", "LINED", "EMPTY"], "edge"]) == [55, 25, 35, 10]
    # the same page with no line px: the line is a piece of word 2's ink, 50 of its 254 px outside its box
    plain = doubt.doubt_features(_sheet_of(recs, gray), recs).set_index("text")
    assert plain.loc["LINED", "ink_outside"] == pytest.approx(2 * (5 + 20) / (180 + 80 - 6))
    assert plain.loc["LINED", "line_share"] == 0
    with pytest.raises(ValueError, match="line px"):
        doubt.doubt_features(dataclasses.replace(sd, line_bits=np.zeros((0, 0), np.uint8)), recs)


def test_a_word_shown_in_two_tiles_takes_the_other_tiles_best_overlapping_read():
    gray = np.full((300, 500), 255, np.uint8)
    a, b = (0, 0, 300, 300), (200, 0, 500, 300)           # two tiles overlapping in columns 200-299
    tiles = [(a, [("RH-2", (250, 100, 290, 120), "zoning code", 0),            # in both tiles, 10 px from a's edge
                  ("RH", (249, 101, 271, 119), "zoning code", 0),              # a worse read of it in tile a
                  ("P", (50, 50, 70, 70), "zoning code", 0),                   # tile a's alone
                  ("OF THE CIT", (210, 200, 296, 215), "other", 0)]),          # cut at tile a's edge
             (b, [("rh-2 ", (250, 100, 291, 121), "zoning code", 0),           # tile b's read of it, 50 px in: kept
                  ("CITY AND", (280, 200, 380, 215), "other", 0),              # read whole in tile b only
                  ("RM-1", (330, 150, 360, 165), "zoning code", 0)])]          # tile b's alone
    recs = _records(tiles)
    sd = _sheet_of(recs, gray)
    f = doubt.doubt_features(sd, recs).set_index("text")
    assert sorted(f.index) == ["CITY AND", "OF THE CIT", "P", "RM-1", "rh-2 "]
    # the read farthest from its tile's edge is kept (tile b's); its other read is tile a's best overlapping one,
    # chosen by IoU with no cut, its strings agreeing once folded
    rh = f.loc["rh-2 "]
    want = doubt.rect_iou(boxes.word_rect((250, 100, 291, 121), 0), boxes.word_rect((250, 100, 290, 120), 0))
    assert rh["read_twice"] and rh["paired"] and rh["other_text"] == "RH-2" and rh["other_tiles"] == 1
    assert rh["other_iou"] == pytest.approx(want) and 0.9 < want < 1 and rh["agree"] == 1
    # a word in one tile only was read once
    for t in ("P", "RM-1", "CITY AND"):
        assert not f.loc[t, "read_twice"] and not f.loc[t, "paired"] and np.isnan(f.loc[t, "other_iou"])
    # the read cut at tile a's edge lies whole in tile b too: its other read is tile b's whole one, overlapping it
    # little (no IoU cut), the strings differing
    cit = f.loc["OF THE CIT"]
    assert cit["read_twice"] and cit["other_text"] == "CITY AND" and cit["agree"] == 0 and 0 < cit["other_iou"] < 0.3
    # identical reads are the same rectangle: IoU exactly 1
    r = boxes.word_rect((230, 100, 271, 121), 30)
    assert doubt.rect_iou(r, r.copy()) == 1.0


def test_a_word_shown_twice_but_read_once_has_no_pair():
    gray = np.full((300, 500), 255, np.uint8)
    tiles = [((0, 0, 300, 300), [("NC-1", (230, 100, 270, 120), "zoning code", 0)]),
             ((200, 0, 500, 300), [("RM-2", (400, 100, 430, 120), "zoning code", 0)])]
    recs = _records(tiles)
    f = doubt.doubt_features(_sheet_of(recs, gray), recs).set_index("text")
    assert f.loc["NC-1", "read_twice"] and not f.loc["NC-1", "paired"] and np.isnan(f.loc["NC-1", "agree"])


def test_features_name_the_held_out_areas_each_word_touches_and_must_be_the_sheets_words():
    gray = np.full((300, 500), 255, np.uint8)
    recs = _records([((0, 0, 500, 300), [("R-4", (100, 100, 140, 120), "zoning code", 0),
                                          ("ANZA ST", (300, 100, 380, 120), "street name", 0)])])
    sd = _sheet_of(recs, gray)
    areas = {"touch_x": _map_rect(139, 0, 200, 300), "touch_y": _map_rect(0, 0, 50, 50)}
    f = doubt.doubt_features(sd, recs, areas)
    assert list(f["touch_x"]) == [True, False] and list(f["touch_y"]) == [False, False]
    assert list(f["kind"]) == ["zoning code", "street name"]
    with pytest.raises(ValueError, match="Terra's words"):
        doubt.doubt_features(dataclasses.replace(sd, words=sd.words[:1]), recs)


# --- the fit ---------------------------------------------------------------------------------------------------------

# A known model over FEATURES (zoning, block, street, other, ink_outside, line_share, axis, axis2, log_edge)
TRUE = {"paired": [1.0, 0.5, 0.8, 1.2, -1.5, -4.0, -1.0, 1.0, 0.4],
        "agree": [0.5, 0.0, -0.2, -0.3, -1.0, -3.0, -1.0, 1.2, 0.5],
        "one": [-3.0, -3.0, -3.5, -2.5, -1.5, -3.0, -3.0, 3.0, 0.1],
        "same": [-1.0, -0.5, -1.5, -0.5, -1.5, -3.0, -2.0, 2.5, 0.8],
        "mean": [3.0, 2.8, 2.6, 2.7, -0.6, -1.5, -2.0, 2.0, -0.2],
        "precision": [3.5, 3.4, 3.0, 2.9, -0.5, -1.0, -1.0, 1.0, -0.1]}
NEIGHBOUR = {"mean": -0.7, "precision": 1.1}


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _double_reads(n, seed=0) -> pd.DataFrame:
    """Synthetic words shown twice, drawn from the TRUE model: paired by its logistic; for a pair, agreement by its
    logistic, the IoU 1 by `one`, otherwise the same word's beta (by `same`; mean and precision by x) or a
    neighbour's (NEIGHBOUR)."""
    rng = np.random.default_rng([seed, 7])
    kinds = np.array(list(data.KINDS))[rng.integers(4, size=n)]
    axis = np.where(rng.random(n) < 0.7, 1.0, rng.uniform(0, 1, n))
    f = pd.DataFrame({"kind": kinds, "ink_outside": rng.uniform(0, 1, n), "line_share": rng.uniform(0, 0.4, n),
                      "axis": axis, "edge": rng.uniform(0, 300, n)})
    x = doubt.design(f)
    paired = rng.random(n) < _sig(x @ TRUE["paired"])
    mu, phi = _sig(x @ TRUE["mean"]), np.exp(x @ TRUE["precision"])
    m, p = _sig(NEIGHBOUR["mean"]), np.exp(NEIGHBOUR["precision"])
    inner = np.where(rng.random(n) < _sig(x @ TRUE["same"]), rng.beta(mu * phi, (1 - mu) * phi),
                     rng.beta(m * p, (1 - m) * p, n))
    iou = np.where(rng.random(n) < _sig(x @ TRUE["one"]), 1.0, np.clip(inner, 1e-12, 1 - 1e-12))
    agree = (rng.random(n) < _sig(x @ TRUE["agree"])).astype(float)
    return f.assign(read_twice=True, paired=paired, agree=np.where(paired, agree, np.nan),
                    other_iou=np.where(paired, iou, np.nan), held_out=False, answered=False)


def _params(**changed):
    return {"features": list(doubt.FEATURES), "neighbour": dict(NEIGHBOUR),
            **{part: dict(zip(doubt.FEATURES, map(float, changed.get(part, v)))) for part, v in TRUE.items()}}


def test_the_fit_recovers_a_known_model_on_synthetic_words_shown_twice():
    f = _double_reads(60000)
    # a word read once was not shown twice: it says nothing about a second read, and the fit doesn't read it
    once = f.iloc[:500].assign(read_twice=False, paired=False, agree=np.nan, other_iou=np.nan)
    p = doubt.fit_doubt(pd.concat([f, once], ignore_index=True))
    assert p["rows"] == 60000 and p["paired_rows"] == int(f["paired"].sum()) < 60000
    assert p["features"] == list(doubt.FEATURES)
    # every part's chance (or the interior's mean) at the words' own x, as the TRUE model gives it: the mean error
    # within about 3 x the largest of three seeds' (measured: paired 0.0055, agree 0.0043, one 0.0020, same 0.0137,
    # mean 0.0024, precision 2.4%, box 0.0028, string 0.0033; the neighbour's logit mean -0.56 to -0.81)
    x = doubt.design(f)
    for part, tol in (("paired", 0.015), ("agree", 0.015), ("one", 0.006), ("same", 0.04), ("mean", 0.008)):
        got, want = _sig(x @ doubt._coef(p, part)), _sig(x @ np.array(TRUE[part]))
        assert np.abs(got - want).mean() < tol, part
    assert np.abs(np.exp(x @ doubt._coef(p, "precision")) / np.exp(x @ np.array(TRUE["precision"])) - 1).mean() < 0.08
    assert p["neighbour"]["mean"] == pytest.approx(NEIGHBOUR["mean"], abs=0.3)
    assert p["neighbour"]["precision"] == pytest.approx(NEIGHBOUR["precision"], abs=0.3)
    # the weights it gives are the TRUE model's
    box, string = doubt.weights_of(f, p)
    true_box, true_string = doubt.weights_of(f, _params())
    assert np.abs(box - true_box).mean() < 0.01 and np.abs(string - true_string).mean() < 0.01


def test_fit_doubt_refuses_answered_block_or_held_quarter_words():
    f = _double_reads(400)
    for flag in ("held_out", "answered"):
        with pytest.raises(ValueError, match=flag):
            doubt.fit_doubt(f.drop(columns=flag))
        bad = f.copy()
        bad.loc[3, flag] = True
        with pytest.raises(ValueError, match=flag):
            doubt.fit_doubt(bad)


# --- the weights -----------------------------------------------------------------------------------------------------

def test_a_second_read_missed_is_doubt_the_weights_carry_p_paired():
    f = _double_reads(50)
    box, string = doubt.weights_of(f, _params())
    x = doubt.design(f)
    p_paired = _sig(x @ np.array(TRUE["paired"]))
    p_one, p_same = _sig(x @ np.array(TRUE["one"])), _sig(x @ np.array(TRUE["same"]))
    inner = p_same * _sig(x @ np.array(TRUE["mean"])) + (1 - p_same) * _sig(NEIGHBOUR["mean"])
    assert np.allclose(box, p_paired * (p_one + (1 - p_one) * inner))
    assert np.allclose(string, p_paired * _sig(x @ np.array(TRUE["agree"])))
    # a word that is never read twice weighs nothing
    never = _params(paired=[-60.0] * 4 + [0.0] * 5)
    assert np.allclose(doubt.weights_of(f, never), 0)


def test_more_ink_outside_or_line_ink_gives_a_lower_weight_smoothly():
    grid_ = np.linspace(0, 1, 1001)
    params = _params()

    def bound(feature, parts):          # the steepest the product of logistics can be: |b| / 4 per unit, each part
        return sum(abs(params[p][feature]) for p in parts) / 4 * (grid_[1] - grid_[0]) + 1e-12

    for kind in data.KINDS:
        for axis in (1.0, 0.3):
            for feature, other in (("ink_outside", "line_share"), ("line_share", "ink_outside")):
                f = pd.DataFrame({"kind": kind, feature: grid_, other: 0.1, "axis": axis, "edge": 120.0})
                box, string = doubt.doubt_weights(None, params, features=f)
                for w, parts in ((box, ("paired", "one", "same", "mean")), (string, ("paired", "agree"))):
                    assert ((w > 0) & (w < 1)).all()
                    step = np.diff(w)
                    assert (step < 0).all()                             # lower with every bit more
                    assert np.abs(step).max() <= bound(feature, parts)  # and no step
    # on a page: the box drawn shorter over the same bar of ink, its weights fall with it
    gray = np.full((100, 200), 255, np.uint8)
    gray[40:43, 50:150] = 0
    got = []
    for x1 in (149, 139, 129, 119, 109):
        recs = _records([((0, 0, 200, 100), [("RH-1", (50, 35, x1, 48), "zoning code", 0)])], window="0,0,200,100")
        sd = _sheet_of(recs, gray)
        got.append(doubt.doubt_weights(sd, params, records=recs))
    box = np.array([g[0][0] for g in got])
    string = np.array([g[1][0] for g in got])
    assert (np.diff(box) < 0).all() and (np.diff(string) < 0).all()


def test_the_tile_edge_is_read_as_log_1_plus_edge_with_no_clip():
    f = pd.DataFrame({"kind": "zoning code", "ink_outside": 0.1, "line_share": 0.0, "axis": 1.0,
                      "edge": [0.0, 10.0, 100.0, 251.0, 369.0, 1000.0]})
    assert np.array_equal(doubt.design(f)[:, -1], np.log1p(f["edge"].to_numpy()))
    _, string = doubt.weights_of(f, _params())
    assert (np.diff(string) > 0).all()     # farther from its tile's edge, surer; beyond the fit's edges too (no clip)
    with pytest.raises(ValueError, match="edge"):
        doubt.design(f.assign(edge=-1.0))


def test_prepare_gives_each_sheet_its_own_words_weights():
    sd = _sheet([((40, 60, 100, 80), 0), ((120, 60, 160, 80), 0)])
    other = _sheet([((40, 60, 100, 80), 0)], sheet="t")

    def entry(s, box):
        return {"words": doubt.words_hash(s), "box": box, "string": [b / 2 for b in box]}

    weights = {f"{sd.series}/s": entry(sd, [0.25, 0.5]), f"{sd.series}/t": entry(other, [0.75])}
    got = doubt.prepare([sd, other], weights)
    assert got[0].weights.dtype == np.float32 and list(got[0].weights) == [0.25, 0.5] and list(got[1].weights) == [0.75]
    assert got[0].ink_bits is sd.ink_bits and sd.weights.shape == (0,)
    assert list(doubt.prepare([sd], weights, which="string")[0].weights) == [0.125, 0.25]
    # the weights fit only the words they were made for: the same number of other words is refused
    moved = _sheet([((40, 60, 100, 80), 0), ((120, 61, 160, 80), 0)])
    assert doubt.words_hash(moved) != doubt.words_hash(sd)
    with pytest.raises(ValueError, match="other words"):
        doubt.prepare([moved], weights)
    with pytest.raises(ValueError, match="2 words"):
        doubt.prepare([sd], {f"{sd.series}/s": entry(sd, [0.25])})
    with pytest.raises(ValueError, match="no weights"):
        doubt.prepare([sd], {})


def test_load_weights_names_a_fold_of_the_split(tmp_path):
    f = tmp_path / "doubt.json"
    f.write_text(json.dumps({"folds": {"ne": {"weights": {"m/s": {"words": "h", "box": [0.5], "string": [0.25]}}}}}))
    got = doubt.load_weights("ne", f)
    assert got["m/s"]["words"] == "h" and got["m/s"]["box"].dtype == np.float32 and list(got["m/s"]["string"]) == [0.25]
    with pytest.raises(ValueError, match="not one of"):
        doubt.load_weights("NE", f)
    with pytest.raises(ValueError, match="holds no fold"):
        doubt.load_weights("sw", f)


def test_each_folds_fit_leaves_out_the_words_touching_its_quarter_or_an_answered_block(monkeypatch):
    rng = np.random.default_rng(5)
    f = _double_reads(400).drop(columns=["held_out", "answered"])
    f["read_twice"] = rng.random(len(f)) < 0.8
    f.loc[~f["read_twice"], ["paired", "agree", "other_iou"]] = [False, np.nan, np.nan]
    f = f.assign(series="m", sheet=np.where(np.arange(len(f)) < 200, "a", "b"), ink_px=10, forced_height=False,
                 line_of_ink=0.0, **{f"touch_{q}": rng.random(len(f)) < 0.3 for q in paths.QUARTERS},
                 touch_blocks=rng.random(len(f)) < 0.1)
    seen = {}

    def fake(rows):
        seen[len(seen)] = rows
        return {**_params(), "rows": len(rows), "paired_rows": int(rows["paired"].sum())}

    monkeypatch.setattr(doubt, "fit_doubt", fake)
    folds = doubt.fit_folds(f, {("m", "a"): "ha", ("m", "b"): "hb"})
    assert list(folds) == list(paths.QUARTERS)
    for i, q in enumerate(paths.QUARTERS):
        rows = seen[i]
        free = ~f[f"touch_{q}"] & ~f["touch_blocks"]
        # exactly the words touching neither the fold's quarter nor an answered block, said so in the flags
        assert sorted(rows.index) == sorted(f.index[free])
        assert not rows[f"touch_{q}"].any() and not rows["touch_blocks"].any()
        assert not rows["held_out"].any() and not rows["answered"].any()
        # and every word, held out or not, gets its weights, with its sheet's words' hash
        w = folds[q]["weights"]
        assert w["m/a"]["words"] == "ha" and len(w["m/a"]["box"]) == 200 and len(w["m/b"]["string"]) == 200
        assert folds[q]["edge_support"]["fit_edge_max"] == f.loc[free & f["read_twice"], "edge"].max()


# --- the word targets and losses -------------------------------------------------------------------------------------

def test_each_core_px_carries_its_words_weight_and_without_words_nothing_changes():
    sd = _sheet([((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0), ((160, 40, 180, 160), 90)])
    weighted = dataclasses.replace(sd, weights=np.array([0.25, 0.5, 0.75], np.float32))
    for y, x, size, scale, turns in ((90, 110, 64, 0.5, 0), (100, 150, 64, 1.3, 1), (70, 120, 48, 1.0, 3)):
        plain = data.crop_targets(sd, y, x, size, scale, turns)
        # the targets without the word heads are the same with weights or without, byte for byte
        got = data.crop_targets(weighted, y, x, size, scale, turns)
        assert sorted(got) == sorted(plain)
        for k in plain:
            assert got[k].dtype == plain[k].dtype and np.array_equal(got[k], plain[k])
        # with them: no weight key without weights; with weights, each core px its word's weight, 1 elsewhere
        bare = data.crop_targets(sd, y, x, size, scale, turns, words=True)
        heavy = data.crop_targets(weighted, y, x, size, scale, turns, words=True)
        assert "weight" not in bare and sorted(heavy) == sorted([*bare, "weight"])
        for k in bare:
            assert np.array_equal(heavy[k], bare[k])
        want = np.array([1, 0.25, 0.5, 0.75], np.float32)[heavy["tag_id"]]
        assert heavy["weight"].dtype == np.float32 and np.array_equal(heavy["weight"], want)
    with pytest.raises(ValueError, match="weights"):
        data.crop_targets(dataclasses.replace(sd, weights=np.ones(2, np.float32)), 90, 110, 64, 1.0, 0, words=True)
    with pytest.raises(ValueError, match="weights"):
        data.crop_targets(dataclasses.replace(sd, weights=np.array([1, -1, 1], np.float32)), 90, 110, 64, 1.0, 0,
                          words=True)


def test_a_word_reaching_into_a_held_out_area_has_no_geometry():
    words = [((40, 60, 120, 80), 0), ((40, 100, 120, 120), 0)]
    free = _sheet(words)
    held = _sheet(words, held=(_map_rect(110, 0, 320, 70),))     # holds the end of word 0's top rows only
    a = data.crop_targets(free, 90, 70, 64, 1.0, 0, words=True)  # page columns 38-101, rows 58-121: no held px
    b = data.crop_targets(held, 90, 70, 64, 1.0, 0, words=True)
    assert b["known"].all() and np.array_equal(a["core"], b["core"]) and np.array_equal(a["tag_id"], b["tag_id"])
    zero = b["tag_id"] == 1
    assert zero.any() and a["geo"][3][zero].min() > 0           # without the hold-out: a word's size at each px
    assert not b["geo"][:4][:, zero].any()                       # word 0: no distances, the angle kept
    assert np.array_equal(b["geo"][4], a["geo"][4])
    two = b["tag_id"] == 2
    assert np.array_equal(b["geo"][:, two], a["geo"][:, two])    # word 1 touches nothing held
    # which px's distances the loss may read: word 1's, not word 0's (without the hold-out, both)
    assert not b["geo_known"][zero].any() and b["geo_known"][two].all() and np.array_equal(a["geo_known"], a["core"])


def _batch(weights=None, turns=(0, 1)):
    sd = _sheet([((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0), ((160, 40, 180, 160), 90)])
    if weights is not None:
        sd = dataclasses.replace(sd, weights=np.array(weights, np.float32))
    return [data.crop_targets(sd, 90, 110, 64, 0.5, turns[0], words=True),
            data.crop_targets(sd, 100, 150, 64, 1.3, turns[1], words=True)]


def test_word_tensors_weigh_each_words_px_and_leave_a_weightless_word_untagged():
    plain = train.word_tensors(_batch())
    ones = train.word_tensors(_batch([1, 1, 1]))
    for p, o in zip(plain, ones):
        assert p.dtype == o.dtype and torch.equal(p, o)
    core, geo, tag, known, geo_known, groups = train.word_tensors(_batch([0.5, 0, 1]))
    crops = _batch([0.5, 0, 1])
    ids = torch.from_numpy(np.stack([c["tag_id"] for c in crops]).astype(np.int64))
    assert torch.equal(known, torch.tensor([1.0, 0.5, 0.0, 1.0])[ids] * plain[3])
    assert not tag[ids == 2].any() and (tag[ids == 1] > 0).all() and torch.equal(groups, torch.arange(1, 3))


def _out(seed=1):
    return torch.randn(2, 16, 64, 64, generator=torch.Generator().manual_seed(seed))


def test_word_losses_scale_with_the_weights_and_a_weightless_word_gives_no_gradient():
    crops = _batch()
    ids = torch.from_numpy(np.stack([c["tag_id"] for c in crops]).astype(np.int64))
    # weight 0: nothing of the word's px reaches the loss; the others' do
    out = _out().requires_grad_()
    loss, _ = train.word_loss(out, *train.word_tensors(_batch([0, 1, 1])))
    loss.backward()
    g = out.grad.abs().sum(1)
    assert not g[ids == 1].any() and (g[ids == 2] > 0).all()
    # all ones: the loss without weights, bit for bit
    same = train.word_loss(_out(), *train.word_tensors(_batch([1, 1, 1])))
    plain = train.word_loss(_out(), *train.word_tensors(crops))
    assert torch.equal(same[0], plain[0]) and torch.equal(same[1], plain[1])
    # the geometry and pull are weighted means over the words: the parts with weights (a, b, c) are the single
    # words' parts weighed by their weight and px (geometry) or by their weight (pull, one crop each word)

    def parts(w):
        return dict(zip(train.WORD_PARTS, train.word_loss(_out(), *train.word_tensors(_batch(w)))[1].tolist()))

    def px(w):
        return float(train.word_tensors(_batch(w))[3].mul(torch.from_numpy(
            np.stack([c["core"] for c in _batch(w)]).astype(np.float32))).sum())

    single = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    weights = [0.2, 0.5, 0.9]
    mass = [px(s) for s in single]
    for part in ("centre", "size", "axis"):
        want = sum(w * m * parts(s)[part] for w, m, s in zip(weights, mass, single)) / sum(
            w * m for w, m in zip(weights, mass))
        assert parts(weights)[part] == pytest.approx(want, rel=1e-5)


def test_the_tag_term_weighs_each_core_px_by_its_words_weight():
    # one crop holding two words: the tag term is the mean of the core px's cross-entropies, each px weighing its
    # word's weight.  So with weights (a, b) it is (a X + b Y) / (a n0 + b n1), X and Y the words' summed
    # cross-entropies and n0, n1 their px: two weightings find X and Y, a third must agree; and weights scaled alike
    # change nothing
    sd = _sheet([((40, 60, 100, 80), 0), ((110, 60, 170, 80), 0)])
    out = torch.randn(1, 16, 96, 96, generator=torch.Generator().manual_seed(3))

    def tag(w):
        crop = data.crop_targets(dataclasses.replace(sd, weights=np.array(w, np.float32)), 70, 105, 96, 1.0, 0,
                                 words=True)
        return train.word_loss(out, *train.word_tensors([crop]))[1][-1].item()

    crop = data.crop_targets(sd, 70, 105, 96, 1.0, 0, words=True)
    n0, n1 = float((crop["tag_id"] == 1).sum()), float((crop["tag_id"] == 2).sum())
    xy = np.linalg.solve([[1, 1], [0.3, 0.6]], [tag([1, 1]) * (n0 + n1), tag([0.3, 0.6]) * (0.3 * n0 + 0.6 * n1)])
    assert tag([0.9, 0.2]) == pytest.approx((0.9 * xy[0] + 0.2 * xy[1]) / (0.9 * n0 + 0.2 * n1), rel=1e-5)
    assert tag([0.6, 1.2]) == pytest.approx(tag([0.3, 0.6]), rel=1e-6)


def test_a_word_with_no_geometry_teaches_its_core_and_direction_but_not_its_box():
    crops = _batch()
    core, geo, tag, known, geo_known, groups = train.word_tensors(crops)
    ids = torch.from_numpy(np.stack([c["tag_id"] for c in crops]).astype(np.int64))
    geo_known = torch.where(ids == 1, torch.zeros_like(geo_known), geo_known)      # word 0: its distances not known
    out = _out().requires_grad_()
    loss, _ = train.word_loss(out, core, geo, tag, known, geo_known, groups)
    loss.backward()
    on = ids == 1
    assert not out.grad[:, model.GEO].abs().sum(1)[on].any()
    assert (out.grad[:, 5:7].abs().sum(1)[on] > 0).all()                    # its angle is kept
    assert (out.grad[:, model.CORE][on] != 0).any() and (out.grad[:, model.GEO].abs().sum(1)[ids == 2] > 0).all()


def test_the_core_cross_entropy_counts_a_weighted_core_px_by_its_weight():
    target = torch.zeros(1, 2, 4)
    target[0, 0, :2] = 1
    known = torch.ones(1, 2, 4)
    known[0, 0, 0] = 0.5                                  # two core px, weights 0.5 and 1; six background px
    logit = torch.tensor([[[0.0, 3.0, 2.0, 1.0], [-1.0, -2.0, -3.0, -4.0]]])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    want = ((0.5 * bce[0, 0, 0] + bce[0, 0, 1]) / 1.5 + bce[target == 0].mean()) / 2
    assert torch.allclose(train._balanced_bce(logit, target, known), want)
