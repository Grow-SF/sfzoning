"""The detector's word heads (`train.Recipe.heads="words"`) and their targets (`data.crop_targets(..., words=True)`,
`wordread.shrink.prepare`): word cores (each word's own rectangle, no shrink), the box each core px votes for (EAST)
and word tags (PAN); and the record-only overlap measurement of shrunk cores (`shrink.core`, `unclip`, `measure_r`)."""
import contextlib
import dataclasses

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, model, train
from sfzmaps.solver.wordread import shrink


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


def _rect(x0, y0, x1, y1, angle=0.0) -> np.ndarray:
    return data.word_rect([x0, y0, x1, y1], angle)


def _sheet(words, size=(240, 320), sheet="s") -> data.SheetData:
    """An in-memory sheet, no placement and nothing held out: a blank page with a stroke of ink along each word's
    middle, and Terra's `words` (axis box (x0, y0, x1, y1), reading angle)."""
    h, w = size
    gray = np.full((h, w), 255, np.uint8)
    rects = [_rect(*box, angle) for box, angle in words]
    for r in rects:
        a, b = (r[0] + r[3]) / 2, (r[1] + r[2]) / 2
        cv2.line(gray, tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b)), 0, 3)
    return data.SheetData(
        series=paths.SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.zeros(len(words), np.uint8), angles=np.array([data.reading_angle(r) for r in rects], np.int16),
        other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)),
        ft_per_px=float("nan"))


# --- shrinking and unclipping ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", [0.3, 0.4, 0.55, 0.7])
def test_unclip_undoes_core_within_half_a_px(r):
    rects = np.array([_rect(10, 20, 70, 40), _rect(100, 50, 140, 210, 90), _rect(0, 0, 300, 300, 30),
                      _rect(5, 5, 9, 60, -45), _rect(50, 60, 250, 90, 170), _rect(0, 0, 2.5, 2.5)])
    for rect in rects:
        assert np.abs(shrink.unclip(shrink.core(rect, r), r) - rect).max() < 0.5
    # many at once, the same numbers
    assert np.allclose(shrink.unclip(shrink.core(rects, r), r), rects, atol=1e-6)


def test_core_is_the_word_shrunk_by_dbnets_offset_and_keeps_its_reading_direction():
    rect = _rect(100, 100, 160, 120)                    # 60 x 20 px, read left to right
    got = shrink.core(rect, 0.4)
    d = 60 * 20 * (1 - 0.4 ** 2) / (2 * (60 + 20))     # D = A (1 - r^2) / L = 6.3 px
    assert np.allclose(got, [[100 + d, 100 + d], [160 - d, 100 + d], [160 - d, 120 - d], [100 + d, 120 - d]])
    turned = _rect(300, 100, 330, 300, 90)              # read bottom to top
    assert data.reading_angle(shrink.core(turned, 0.5)) == data.reading_angle(turned) == 90


def _pairs(overlapping: int, touching: int, w=1000, h=200, o=120) -> np.ndarray:
    """`touching` pairs of w x h words side by side, and `overlapping` pairs whose second word starts `o` px inside
    the first; each pair a row of its own, rows 1.5 h apart (beyond one word height: no neighbours across rows)."""
    words = []
    for i in range(overlapping + touching):
        y = i * (h + 300)
        shift = o if i < overlapping else 0
        words += [_rect(0, y, w, y + h), _rect(w - shift, y, 2 * w - shift, y + h)]
    return np.round(np.array(words)).astype(np.int32)


def test_measure_r_is_the_largest_r_whose_cores_overlap_on_at_most_one_percent_of_neighbouring_pairs():
    # two 1000 x 200 words overlapping by 120 px: shrunk by D = 83.3 (1 - r^2) each, their cores are 5 px apart at
    # r = 0.5 and overlap by 3.75 px at 0.55
    two = shrink.measure_r({"a": _pairs(2, 58), "b": _pairs(0, 40)})
    assert two["pairs"] == 100 and two["words"] == 200
    assert two["r"] == 0.5
    curve = {c["r"]: c for c in two["curve"]}
    assert sorted(curve) == [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    assert [curve[r]["overlapping"] for r in sorted(curve)] == [0] * 5 + [2] * 4
    assert curve[0.55]["share"] == pytest.approx(0.02)
    # one in a hundred is allowed
    assert shrink.measure_r({"a": _pairs(1, 99)})["r"] == 0.7
    # words further apart than one word height are not neighbours: 201 px apart, no; 200 px, yes
    far = np.round([_rect(0, 0, 1000, 200), _rect(1201, 0, 2201, 200), _rect(2401, 0, 3401, 200)]).astype(np.int32)
    assert shrink.measure_r({"a": far})["pairs"] == 1


def test_r_is_chosen_by_the_pairs_newly_overlapping_beyond_the_first_r_and_folds_pool():
    # ten pairs overlapping by 600 px, cores overlapping at every r (2D <= 152 px): Terra's box errors, the floor
    floor = _pairs(10, 0, o=600)
    got = shrink.measure_r({"a": floor, "b": _pairs(2, 88)})
    assert got["pairs"] == 100 and got["floor"] == 10 and got["floor_share"] == pytest.approx(0.1)
    curve = {c["r"]: c for c in got["curve"]}
    assert [curve[r]["overlapping"] for r in shrink.R_GRID] == [10] * 5 + [12] * 4
    assert [curve[r]["new"] for r in shrink.R_GRID] == [0] * 5 + [2] * 4
    assert curve[0.55]["new_share"] == pytest.approx(0.02) and curve[0.3]["share"] == pytest.approx(0.1)
    # the floor is above 1% at every r, yet r is chosen by the new pairs alone: 2% from 0.55, 1% allowed
    assert got["r"] == 0.5
    assert shrink.measure_r({"a": floor, "b": _pairs(1, 89)})["r"] == 0.7
    # folds pooled: their pairs, floors and new overlaps summed, r from the sums (2 new of 200)
    other = shrink.measure_r({"a": _pairs(0, 100)})
    pooled = shrink.pool([got, other])
    assert pooled["pairs"] == 200 and pooled["floor"] == 10 and pooled["pooled"] == 2
    assert pooled["floor_share"] == pytest.approx(0.05)
    assert {c["r"]: c["new"] for c in pooled["curve"]}[0.55] == 2 and pooled["r"] == 0.7


def test_prepare_fills_each_core_with_its_own_rectangle():
    words = [((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0),          # R-4 | RH-2, touching
             ((160, 40, 180, 160), 90),                                  # a street name read upward, touching both
             ((40, 81, 160, 101), 30)]                                   # slanted text right under them
    sd = _sheet(words)
    (prepared,) = shrink.prepare([sd])
    # no shrink: each word's core is its own labelled rectangle, the rest of the sheet shared, the input left as it was
    assert np.array_equal(prepared.cores, sd.words) and prepared.cores.dtype == np.int32
    assert prepared.ink_bits is sd.ink_bits and sd.cores.shape == (0, 4, 2)
    # so the core targets are the words' own px, each tagged with its own word (the later one on a shared edge),
    # the touching words told apart only by their tags
    t = data.crop_targets(prepared, 100, 120, 200, 1.0, 0, words=True)          # the window's top-left px is (0, 20)
    assert np.array_equal(t["core"], t["text"])
    assert np.array_equal(t["tag_id"], data._words(sd, 0, 20, 200, 200))
    assert set(np.unique(t["tag_id"])) == {0, 1, 2, 3, 4}


# --- the targets -----------------------------------------------------------------------------------------------------

def test_geo_at_a_core_px_gives_the_distances_to_the_words_four_sides():
    sd, = shrink.prepare([_sheet([((100, 50, 200, 80), 0), ((230, 40, 260, 140), 90)])])
    t = data.crop_targets(sd, 65, 150, 64, 1.0, 0, words=True)        # the window's top-left px is (33, 118)
    assert t["geo"].dtype == np.float32 and t["geo"].shape == (5, 64, 64)
    assert t["core"].dtype == np.uint8 and t["tag_id"].dtype == np.int32
    # page px (150, 65): 15 px below the top edge, 15 above the bottom, 50 from the start and 50 from the end
    assert t["core"][32, 32] == 1 and t["tag_id"][32, 32] == 1
    assert np.allclose(t["geo"][:, 32, 32], [15, 15, 50, 50, 0])
    # page px (120, 60)
    assert np.allclose(t["geo"][:, 60 - 33, 120 - 118], [10, 20, 20, 80, 0])
    # off the cores: nothing
    assert t["core"][0, 0] == 0 and t["tag_id"][0, 0] == 0 and not t["geo"][:, 0, 0].any()
    # a word read bottom to top: its top is on the left, its start at the bottom
    up = data.crop_targets(sd, 90, 245, 64, 1.0, 0, words=True)      # top-left px (58, 213)
    assert up["tag_id"][120 - 58, 240 - 213] == 2
    assert np.allclose(up["geo"][:, 120 - 58, 240 - 213], [10, 20, 20, 80, 90])


def test_geo_scales_with_the_crop_and_turns_with_it():
    sd, = shrink.prepare([_sheet([((100, 50, 200, 80), 0)])])
    flat = data.crop_targets(sd, 65, 150, 64, 1.0, 0, words=True)
    big = data.crop_targets(sd, 65, 150, 128, 2.0, 0, words=True)        # the same 64 page px, twice the size
    assert np.allclose(big["geo"][:4, 64, 64], 2 * flat["geo"][:4, 32, 32]) and big["geo"][4, 64, 64] == 0
    assert np.array_equal(big["tag_id"][::2, ::2], flat["tag_id"])
    turned = data.crop_targets(sd, 65, 150, 64, 1.0, 1, words=True)      # a quarter turn counter-clockwise
    for k in ("core", "tag_id"):
        assert np.array_equal(turned[k], np.rot90(flat[k]))
    assert np.array_equal(turned["geo"][:4], np.rot90(flat["geo"][:4], axes=(1, 2)))
    assert np.array_equal(turned["geo"][4], np.where(np.rot90(flat["core"]) > 0, 90, 0))


def test_word_targets_need_the_cores_and_change_nothing_else():
    sd = _sheet([((100, 50, 200, 80), 0)])
    with pytest.raises(ValueError, match="shrink.prepare"):
        data.crop_targets(sd, 65, 150, 64, 1.0, 0, words=True)
    prepared, = shrink.prepare([sd])
    assert prepared.cores.dtype == np.int32 and prepared.cores.shape == (1, 4, 2)
    assert sd.cores.shape == (0, 4, 2)                                    # the default: none
    plain = data.crop_targets(sd, 65, 150, 64, 0.8, 3)
    for got in (data.crop_targets(prepared, 65, 150, 64, 0.8, 3),
                {k: v for k, v in data.crop_targets(prepared, 65, 150, 64, 0.8, 3, words=True).items()
                 if k not in ("core", "geo", "tag_id")}):
        assert sorted(got) == sorted(plain) == ["angle", "ink", "kind", "known", "text"]
        for k in plain:
            assert got[k].dtype == plain[k].dtype and np.array_equal(got[k], plain[k])


# --- the network and its loss ----------------------------------------------------------------------------------------

def test_the_word_heads_are_a_recipe_choice_off_by_default():
    assert train.Recipe().heads == "all" and train.CHOICES["heads"] == ("all", "text", "words")
    recipe = train.Recipe(heads="words", width=4, crop=32)
    assert recipe.name() == "crop=32-heads=words-width=4"
    out = train.build_net(recipe)(torch.zeros(2, 1, 32, 32))
    assert out.shape == (2, 16, 32, 32)
    half = train.build_net(dataclasses.replace(recipe, half_size=True))(torch.zeros(2, 1, 32, 32))
    assert half.shape == (2, 16, 32, 32)
    with pytest.raises(ValueError, match="composites"):
        train.Recipe(heads="words", composites=0.5)


def _word_batch():
    sd, = shrink.prepare([_sheet([((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0), ((160, 40, 180, 160), 90)])])
    return [data.crop_targets(sd, 90, 110, 64, 0.5, 0, words=True), data.crop_targets(sd, 100, 150, 64, 1.3, 1,
                                                                                         words=True)]


def test_word_tensors_number_each_crops_tags_from_one():
    crops = _word_batch()
    crops[1]["known"][:] = False                       # a crop held out: no tag, no core lesson
    core, geo, tag, known, groups = train.word_tensors(crops)
    assert core.dtype == geo.dtype == known.dtype == torch.float32 and tag.dtype == torch.int64
    assert geo.shape == (2, 5, 64, 64) and tag.shape == (2, 64, 64)
    assert set(tag[0].unique().tolist()) == {0, 1, 2, 3} and not tag[1].any()
    assert torch.equal(groups, torch.arange(1, 4))
    assert torch.equal(tag[0] > 0, core[0] > 0)


def test_hard_negatives_are_the_top_three_per_positive_ties_shared():
    target = torch.zeros(1, 4, 4)
    target[0, 0, :2] = 1                                # two positives
    known = torch.ones(1, 4, 4)
    known[0, 3, 3] = 0
    logit = torch.linspace(-3, 3, 16).reshape(1, 4, 4)        # the negatives' four highest: 2.6, 2.2, 1.8, 1.4
    logit[0, 2, 0] = logit[0, 2, 1] = logit[0, 2, 2] = 1.0     # three tied for the last two of the six places
    got = train._ohem_bce(logit, target, known)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    neg = bce[(target == 0) & (known > 0)]
    top = torch.sort(neg, descending=True).values[:6]
    want = (bce[target > 0].sum() + top.sum()) / 8
    assert torch.allclose(got, want)
    # no positive: no negative is read either
    assert train._ohem_bce(logit, torch.zeros_like(target), known) == 0


def test_word_loss_is_zero_for_a_perfect_answer_and_pushes_tags_apart():
    crops = _word_batch()
    core, geo, tag, known, groups = train.word_tensors(crops)
    out = torch.zeros(2, 16, 64, 64)
    out[:, model.CORE] = torch.where(core > 0, 30.0, -30.0)
    out[:, model.GEO] = torch.log(geo[:, :4].clamp(min=1e-3) / model.GEO_UNIT)
    rad = torch.deg2rad(geo[:, 4])
    out[:, 5], out[:, 6] = torch.cos(rad), torch.sin(rad)
    for c in range(4):                                   # each word's tag at 3 x its local number, on one axis
        out[:, model.TAG.start + c] = 3.0 * tag * (c == 0)
    loss, parts = train.word_loss(out, core, geo, tag, known, groups)
    names = dict(zip(train.WORD_PARTS, parts.tolist()))
    assert sorted(names) == sorted(("core", "dice", "iou", "direction", "pull", "push"))
    assert names["core"] < 1e-6 and names["dice"] < 1e-6 and names["iou"] < 1e-5 and names["direction"] < 1e-6
    assert names["pull"] == 0 and names["push"] == 0
    # every word's tag the same: each pair pushed by log(1 + 3^2)
    out[:, model.TAG] = 0
    _, parts = train.word_loss(out, core, geo, tag, known, groups)
    assert dict(zip(train.WORD_PARTS, parts.tolist()))["push"] == pytest.approx(np.log(10))
    # the total is the parts by their weights
    out = torch.randn(2, 16, 64, 64, generator=torch.Generator().manual_seed(1))
    loss, p = train.word_loss(out, core, geo, tag, known, groups)
    assert torch.allclose(loss, p[0] + p[1] + p[2] + 10 * p[3] + 0.25 * (p[4] + p[5]))


def test_a_word_heads_fit_trains_the_per_ink_head_by_todays_loss_and_repeats(monkeypatch):
    sheets = shrink.prepare([_sheet([((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0), ((160, 40, 180, 160), 90)]),
                             _sheet([((20, 20, 120, 50), 30), ((60, 150, 200, 175), 0)], sheet="t")])
    recipe = train.Recipe(heads="words", iterations=2, batch=3, crop=48, width=4)
    real, seen = train.loss_fn, []
    monkeypatch.setattr(train, "loss_fn", lambda out, t, k, a, w: seen.append(out.shape) or real(out, t, k, a, w))
    first = train.fit(sheets, recipe, "cpu", seed=0)
    again = train.fit(sheets, recipe, "cpu", seed=0)
    assert seen == [(3, 16, 48, 48)] * 4
    for (name, a), b in zip(first.state_dict().items(), again.state_dict().values()):
        assert torch.equal(a, b), name
    # the word loss reaches every word output
    net = train.build_net(recipe)
    crops = train.draw(sheets, train.centre_pools(sheets), recipe, np.random.default_rng(0))
    loss, _ = train.word_loss(net(train.batch_tensors(crops)[0]), *train.word_tensors(crops))
    loss.backward()
    assert (net.head.weight.grad.flatten(1).abs().sum(1)[model.CORE:] > 0).all()
