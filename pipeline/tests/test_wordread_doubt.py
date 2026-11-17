"""Doubt weights for Terra's words (`wordread.doubt`): the features of each word (its ink outside its box, the forced
height of a near-45-degree word, its kind, and the other tile's read of a word Terra was shown twice), the fit of
P(the two reads' strings agree) and of the pair's box IoU on those double reads, the weights (expected IoU, P(string
right)) with no cut, and the word heads' loss weighing each word's px by its weight (`data.crop_targets`,
`train.word_tensors`, `train.word_loss`)."""
import contextlib
import dataclasses

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


def _sheet(words, gray=None, size=(240, 320), held=(), kinds=None, sheet="s") -> data.SheetData:
    """An in-memory sheet placed north up: Terra's `words` (axis box, reading angle), ink `gray` (default: a stroke
    along each word's middle), the areas `held` held out, every word's core its own rectangle."""
    h, w = size
    if gray is None:
        gray = np.full((h, w), 255, np.uint8)
        for box, angle in words:
            r = data.word_rect(box, angle)
            a, b = (r[0] + r[3]) / 2, (r[1] + r[2]) / 2
            cv2.line(gray, tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b)), 0, 3)
    rects = np.round(np.array([data.word_rect(b, a) for b, a in words], float).reshape(-1, 4, 2)).astype(np.int32)
    return data.SheetData(
        series=paths.SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=rects, kinds=np.zeros(len(words), np.uint8) if kinds is None else np.array(kinds, np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.zeros((0, 2), np.int32), world=_world(h, w), ft_per_px=FT, held=tuple(held), cores=rects.copy())


def _item(text, box, kind="zoning code", angle=0, tile=(0, 0)):
    x0, y0, x1, y1 = box
    return {"text": text, "kind": kind, "x0": x0 - tile[0], "y0": y0 - tile[1], "x1": x1 - tile[0],
            "y1": y1 - tile[1], "direction": "left-to-right", "angle_deg": angle}


def _records(tiles, sheet="s", window="0,0,500,300"):
    """Terra's saved answers for a sheet: `tiles` [(page_box, [(text, page box, kind, angle), ...]), ...]."""
    return [{"series": paths.SERIES, "sheet": sheet, "page_box": list(pb),
             "script": {"args": ["terra", paths.SERIES, sheet, window, "out"]},
             "answer": {"items": [_item(t, b, k, a, tile=pb[:2]) for t, b, k, a in items]}} for pb, items in tiles]


def _sheet_of(records, gray, sheet="s", held=()) -> data.SheetData:
    found = boxes.terra_boxes(paths.SERIES, sheet, records)
    return _sheet([(f["box"], float(f["angle_deg"])) for f in found], gray=gray, size=gray.shape, held=held,
                  kinds=[data.KINDS[f["kind"]] for f in found], sheet=sheet)


# --- the features ----------------------------------------------------------------------------------------------------

def test_forced_height_is_word_rects_near_diagonal_case():
    # word_rect forces the height (TYPICAL_HEIGHT) whatever the box: two different boxes give it exactly
    for angle in range(-180, 181):
        heights = [float(np.linalg.norm(r[3] - r[0])) for r in (boxes.word_rect((0, 0, 200, 150), angle),
                                                                  boxes.word_rect((0, 0, 170, 190), angle))]
        assert bool(doubt.forced_height(angle)) == all(abs(h - boxes.TYPICAL_HEIGHT) < 1e-9 for h in heights), angle
    assert doubt.forced_height(45) and not doubt.forced_height(0) and not doubt.forced_height(90)


def test_ink_outside_is_the_share_of_the_words_ink_pieces_px_outside_its_box():
    gray = np.full((200, 300), 255, np.uint8)
    gray[60:63, 100:140] = 0                    # a bar of 120 px, half under word 0's box (columns 100-119)
    gray[100:110, 30:34] = 0                    # word 1's letter, wholly inside its box
    gray[100:110, 40:44] = 0
    gray[150:153, 200:260] = 0                  # word 2's letters...
    gray[140:180, 230:232] = 0                  # ... joined to a line of which 80 of 200 px fall outside its box
    words = {"HALF": (100, 55, 119, 70), "WHOLE": (25, 95, 50, 115), "LINED": (195, 145, 265, 159),
             "EMPTY": (10, 10, 40, 30)}
    recs = _records([((0, 0, 300, 200), [(t, b, "zoning code", 0) for t, b in words.items()])], window="0,0,300,200")
    sd = _sheet_of(recs, gray)
    f = doubt.doubt_features(sd, recs)
    assert list(f["word"]) == [0, 1, 2, 3]                     # one row per word of the sheet, in its order
    f = f.set_index("text")
    assert f.loc["HALF", "ink_outside"] == pytest.approx(0.5) and f.loc["HALF", "ink_px"] == 60
    assert f.loc["WHOLE", "ink_outside"] == 0
    piece = 180 + 80 - 6                                  # the letters' bar and the line (6 px shared): one piece
    outside = 2 * ((145 - 140) + (180 - 160))             # the line's rows above and below the box (rows 145-159)
    assert f.loc["LINED", "ink_outside"] == pytest.approx(outside / piece)
    assert f.loc["EMPTY", "ink_px"] == 0 and f.loc["EMPTY", "ink_outside"] == 1     # no ink: all its ink is outside
    assert not f["forced_height"].any() and set(f["kind"]) == {"zoning code"}


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

TRUE = {"agree": [2.5, 2.0, 1.5, 1.0, -3.0, -1.0], "one": [-2.0, -2.5, -3.0, -2.0, -2.0, -0.5],
        "mean": [2.0, 1.8, 1.5, 1.2, -1.5, -0.8], "precision": [3.0, 3.4, 2.4, 2.0, -1.0, -0.4]}


def _double_reads(n, seed=0) -> pd.DataFrame:
    """Synthetic double reads drawn from the TRUE model: agreement by the logistic, the IoU 1 by the logistic `one`
    and otherwise from Beta(mu phi, (1 - mu) phi) with logit mu by `mean` and log phi by `precision`."""
    rng = np.random.default_rng([seed, 7])
    kinds = np.array(list(data.KINDS))[rng.integers(4, size=n)]
    f = pd.DataFrame({"kind": kinds, "ink_outside": rng.uniform(0, 1, n), "forced_height": rng.random(n) < 0.2})
    x = doubt.design(f)
    sig = lambda z: 1 / (1 + np.exp(-z))                                     # noqa: E731
    phi = np.exp(x @ TRUE["precision"])
    mu = sig(x @ TRUE["mean"])
    iou = np.where(rng.random(n) < sig(x @ TRUE["one"]), 1.0, rng.beta(mu * phi, (1 - mu) * phi))
    return f.assign(read_twice=True, paired=True, agree=(rng.random(n) < sig(x @ TRUE["agree"])).astype(float),
                    other_iou=iou, held_out=False, answered=False)


def test_the_fit_recovers_known_coefficients_on_synthetic_double_reads():
    f = _double_reads(40000)
    # a word read once, or shown twice and read once, says nothing about agreement: left out
    extra = f.iloc[:500].assign(paired=False, agree=np.nan, other_iou=np.nan)
    p = doubt.fit_doubt(pd.concat([f, extra], ignore_index=True))
    assert p["rows"] == 40000 and p["features"] == list(doubt.FEATURES)
    for part, tol in (("agree", 0.15), ("one", 0.3), ("mean", 0.06), ("precision", 0.15)):   # about 4 standard errors
        got = np.array([p[part][k] for k in doubt.FEATURES])
        assert np.abs(got - TRUE[part]).max() < tol, (part, got)


def test_fit_doubt_refuses_answered_block_or_held_quarter_words():
    f = _double_reads(200)
    for flag in ("held_out", "answered"):
        with pytest.raises(ValueError, match=flag):
            doubt.fit_doubt(f.drop(columns=flag))
        bad = f.copy()
        bad.loc[3, flag] = True
        with pytest.raises(ValueError, match=flag):
            doubt.fit_doubt(bad)


# --- the weights -----------------------------------------------------------------------------------------------------

def _params():
    return {"features": list(doubt.FEATURES),
            **{part: dict(zip(doubt.FEATURES, map(float, TRUE[part]))) for part in TRUE}}


def test_more_ink_outside_gives_a_lower_weight_smoothly():
    grid_ = np.linspace(0, 1, 1001)
    params = _params()
    for kind in data.KINDS:
        for forced in (False, True):
            f = pd.DataFrame({"kind": kind, "ink_outside": grid_, "forced_height": forced})
            box, string = doubt.doubt_weights(None, params, features=f)
            for w, part in ((box, "mean"), (string, "agree")):
                assert ((w > 0) & (w < 1)).all()
                step = np.diff(w)
                assert (step < 0).all()                                 # lower with every bit more ink outside
                # no step: never steeper than the logistics' steepest slope allows (|b| / 4 per unit, each part)
                bound = sum(abs(params[p]["ink_outside"]) for p in ("one", "mean")) / 4 if part == "mean" \
                    else abs(params["agree"]["ink_outside"]) / 4
                assert np.abs(step).max() <= bound * (grid_[1] - grid_[0]) + 1e-12
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


def test_prepare_gives_each_sheet_its_words_box_weights():
    sd = _sheet([((40, 60, 100, 80), 0), ((120, 60, 160, 80), 0)])
    other = _sheet([((40, 60, 100, 80), 0)], sheet="t")
    got = doubt.prepare([sd, other], {f"{sd.series}/s": [0.25, 0.5], f"{sd.series}/t": [0.75]})
    assert got[0].weights.dtype == np.float32 and list(got[0].weights) == [0.25, 0.5] and list(got[1].weights) == [0.75]
    assert got[0].ink_bits is sd.ink_bits and sd.weights.shape == (0,)
    with pytest.raises(ValueError, match="2 words"):
        doubt.prepare([sd], {f"{sd.series}/s": [0.25]})
    with pytest.raises(ValueError, match="no weights"):
        doubt.prepare([sd], {})


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
    assert zero.any() and (a["geo"][0] + a["geo"][1])[zero].min() > 0       # without the hold-out: a box at each px
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
    for part in ("iou", "direction"):
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
