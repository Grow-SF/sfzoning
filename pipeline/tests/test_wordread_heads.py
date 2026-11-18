"""The detector's word heads (`train.Recipe.heads="words"`) and their targets (`data.crop_targets(..., words=True)`,
`wordread.shrink.prepare`): word cores (each word's own rectangle, no shrink), the sign-free geometry each core px
gives (its word's centre over its height, log sizes and axis), and word tags; and the record-only overlap measurement
of shrunk cores (`shrink.core`, `unclip`, `measure_r`)."""
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

def test_geo_is_the_offset_to_the_words_centre_over_its_height_its_log_sizes_and_its_axis():
    sd, = shrink.prepare([_sheet([((100, 50, 200, 80), 0), ((230, 40, 260, 140), 90)])])
    t = data.crop_targets(sd, 65, 150, 64, 1.0, 0, words=True)        # the window's top-left px is (33, 118)
    assert t["geo"].dtype == np.float32 and t["geo"].shape == (5, 64, 64)
    assert t["core"].dtype == np.uint8 and t["tag_id"].dtype == np.int32
    # word 1, 100 x 30 px, centre (150, 65): at page px (150, 65) no offset; at (120, 60) the centre is 30 px right
    # and 5 px down, over the height 30
    assert t["core"][32, 32] == 1 and t["tag_id"][32, 32] == 1
    assert np.allclose(t["geo"][:, 32, 32], [0, 0, np.log(100), np.log(30), 0])
    assert np.allclose(t["geo"][:, 60 - 33, 120 - 118], [1, 5 / 30, np.log(100), np.log(30), 0])
    # off the cores: nothing
    assert t["core"][0, 0] == 0 and t["tag_id"][0, 0] == 0 and not t["geo"][:, 0, 0].any()
    # word 2, read bottom to top: 100 long (up the page) and 30 high (across), centre (245, 90), axis 90; at page px
    # (240, 120) the centre is 5 px right and 30 up
    up = data.crop_targets(sd, 90, 245, 64, 1.0, 0, words=True)      # top-left px (58, 213)
    assert up["tag_id"][120 - 58, 240 - 213] == 2
    assert np.allclose(up["geo"][:, 120 - 58, 240 - 213], [5 / 30, -1, np.log(100), np.log(30), 90])


def _votes(t, word):
    """Where each core px of `word` puts its word's centre in the crop (x, y): the px plus its offset times the
    height."""
    rows, cols = np.nonzero(t["tag_id"] == word)
    height = np.exp(t["geo"][3][rows, cols])
    return np.c_[cols + t["geo"][0][rows, cols] * height, rows + t["geo"][1][rows, cols] * height]


AT_ANGLE = {0: (110, 95, 210, 145),       # 100 long, 50 high
            30: (112, 86, 209, 153),      # the axis box of a word about 101 long and 19 high read at 30 degrees
            90: (135, 70, 185, 170),      # read bottom to top: 100 long (up the page), 50 high (across)
            179: (110, 95, 210, 145), 181: (110, 95, 210, 145)}


@pytest.mark.parametrize("angle", sorted(AT_ANGLE))
def test_geo_at_any_angle_puts_every_core_px_on_the_words_centre_with_its_sizes_and_axis(angle):
    sd, = shrink.prepare([_sheet([(AT_ANGLE[angle], angle)])])
    rect = sd.words[0].astype(float)                    # its corners in order: the first edge its length
    length, height = np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[3] - rect[0])
    t = data.crop_targets(sd, 120, 160, 128, 1.0, 0, words=True)        # top-left px (56, 96): the whole word
    assert np.allclose(_votes(t, 1), rect.mean(0) - [96, 56], atol=1.0)
    on = t["core"] > 0
    assert np.allclose(np.exp(t["geo"][2][on]), length, atol=1.0)
    assert np.allclose(np.exp(t["geo"][3][on]), height, atol=1.0)
    assert np.allclose(t["geo"][4][on], sd.angles[0] % 180) and sd.angles[0] % 180 == angle % 180


def test_geo_is_continuous_across_180_degrees():
    # a word read at 179 degrees and one at 181 (-179) in the same box: the same geometry within their 2 degrees'
    # difference, and axes 179 and 1, two degrees apart on the axis circle (2t: 358 and 362 = 2)
    a, b = (data.crop_targets(shrink.prepare([_sheet([(AT_ANGLE[r], r)])])[0], 120, 160, 128, 1.0, 0, words=True)
            for r in (179, 181))
    both = (a["core"] > 0) & (b["core"] > 0)
    assert both.sum() > 4000
    assert np.abs(a["geo"][:4][:, both] - b["geo"][:4][:, both]).max() < 0.05
    assert set(np.unique(a["geo"][4][both])) == {179} and set(np.unique(b["geo"][4][both])) == {1}
    t = np.deg2rad(2 * np.array([179.0, 1.0]))
    assert np.hypot(np.cos(t[0]) - np.cos(t[1]), np.sin(t[0]) - np.sin(t[1])) < 0.08


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_geo_turns_as_a_vector_with_the_crop_and_its_axis_by_a_right_angle(turns):
    sd, = shrink.prepare([_sheet([(AT_ANGLE[30], 30)])])
    flat = data.crop_targets(sd, 120, 160, 128, 1.0, 0, words=True)
    turned = data.crop_targets(sd, 120, 160, 128, 1.0, turns, words=True)
    for k in ("core", "tag_id", "geo_known"):
        assert np.array_equal(turned[k], np.rot90(flat[k], turns))
    # every core px of the turned crop puts the centre where the turn takes it: (x, y) to (y, size - 1 - x) a turn
    x, y = _votes(flat, 1).mean(0)
    for _ in range(turns):
        x, y = y, 127 - x
    assert np.allclose(_votes(turned, 1), [x, y], atol=1e-3)
    on = turned["core"] > 0
    assert np.allclose(turned["geo"][2:4], np.rot90(flat["geo"][2:4], turns, axes=(1, 2)))     # the sizes stay
    assert np.allclose(turned["geo"][4][on], (30 + 90 * turns) % 180)
    # twice the scale: the sizes grow by log 2, the offsets (over the height) stay
    big = data.crop_targets(sd, 120, 160, 256, 2.0, 0, words=True)
    pick = flat["core"] > 0
    assert np.allclose(big["geo"][2:4][:, ::2, ::2][:, pick], flat["geo"][2:4][:, pick] + np.log(2))
    assert np.allclose(big["geo"][0:2][:, ::2, ::2][:, pick], flat["geo"][0:2][:, pick])


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
                 if k not in ("core", "geo", "tag_id", "geo_known")}):
        assert sorted(got) == sorted(plain) == ["angle", "ink", "kind", "known", "text"]
        for k in plain:
            assert got[k].dtype == plain[k].dtype and np.array_equal(got[k], plain[k])


def test_which_words_reach_a_held_out_area_is_found_once_per_sheet(monkeypatch):
    sd, = shrink.prepare([_sheet([((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0)])])
    calls = []
    monkeypatch.setattr(data, "_touch_held", lambda s, words: calls.append(len(words)) or np.array([True, False]))
    crops = [data.crop_targets(sd, 70, 100, 160, 1.0, turns, words=True) for turns in range(3)]   # both words whole
    assert calls == [2]                                   # every word, once, for the three crops
    t = crops[0]
    reach, free = t["tag_id"] == 1, t["tag_id"] == 2
    assert t["geo_known"].dtype == np.uint8 and not t["geo_known"][reach].any() and t["geo_known"][free].all()
    assert not t["geo"][:4][:, reach].any() and t["geo"][3][free].min() > 0
    assert not t["geo_known"][t["core"] == 0].any()
    # a sheet holding other areas is another SheetData: found again for it
    data.crop_targets(dataclasses.replace(sd, held=()), 70, 100, 160, 1.0, 0, words=True)
    assert calls == [2, 2]


@pytest.mark.parametrize("turns", [0, 1])
def test_a_word_the_crop_cuts_has_no_geometry_to_read_but_keeps_its_axis(turns):
    # the crop shows page columns and rows 68-131: word 1 ends on column 131 (whole), word 2 on column 132 (cut by
    # one px), word 3 is read upward and starts above row 68 (cut)
    sd, = shrink.prepare([_sheet([((80, 90, 131, 110), 0), ((80, 115, 132, 125), 0), ((120, 60, 130, 85), 90)])])
    t = data.crop_targets(sd, 100, 100, 64, 1.0, turns, words=True)
    whole, cut, up = (t["tag_id"] == k for k in (1, 2, 3))
    assert whole.any() and cut.any() and up.any()
    assert t["geo_known"][whole].all() and not t["geo_known"][cut].any() and not t["geo_known"][up].any()
    # the cut words keep their geometry (the crop hides it; it is no held-out word's) and their axis, read as before
    assert np.allclose(np.exp(t["geo"][2][cut]), 52) and np.allclose(t["geo"][4][cut], (0 + 90 * turns) % 180)
    assert np.allclose(t["geo"][4][up], (90 + 90 * turns) % 180)
    # a crop that shows them whole reads them: the same targets, now known
    big = data.crop_targets(sd, 100, 100, 160, 1.0, turns, words=True)
    for k in (1, 2, 3):
        assert big["geo_known"][big["tag_id"] == k].all()
    assert np.array_equal(big["geo_known"], big["core"])
    # at another scale the rule is the same, on the page px the crop shows
    small = data.crop_targets(sd, 100, 100, 128, 2.0, turns, words=True)        # the same 64 page px
    assert np.array_equal(small["geo_known"][::2, ::2], t["geo_known"])


# --- the network and its loss ----------------------------------------------------------------------------------------

def test_the_word_heads_are_a_recipe_choice_off_by_default():
    assert train.Recipe().heads == "all" and train.CHOICES["heads"] == ("all", "text", "words", "words_aff",
                                                                       "words_code")
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
    core, geo, tag, known, geo_known, groups = train.word_tensors(crops)
    assert core.dtype == geo.dtype == known.dtype == geo_known.dtype == torch.float32 and tag.dtype == torch.int64
    assert geo.shape == (2, 5, 64, 64) and tag.shape == (2, 64, 64)
    assert set(tag[0].unique().tolist()) == {0, 1, 2, 3} and not tag[1].any()
    assert torch.equal(groups, torch.arange(1, 4))
    assert torch.equal(tag[0] > 0, core[0] > 0)
    # nothing held: the geometry is known on the one word the first crop shows whole (its 2nd; the 1st and 3rd run
    # out of its page window, columns 46-173), on none of the second's (it shows no word whole)
    whole = torch.from_numpy(np.ascontiguousarray(crops[0]["tag_id"]) == 2)
    assert torch.equal(geo_known[0], core[0] * whole) and whole.any() and not geo_known[1].any()


def test_the_core_cross_entropy_weighs_each_class_by_the_inverse_of_its_weight():
    target = torch.zeros(1, 4, 4)
    target[0, 0, :2] = 1                                # two core px, thirteen others known, one not known
    known = torch.ones(1, 4, 4)
    known[0, 3, 3] = 0
    logit = torch.linspace(-3, 3, 16).reshape(1, 4, 4)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    got = train._balanced_bce(logit, target, known)
    assert torch.allclose(got, (bce[0, 0, :2].mean() + bce[(target == 0) & (known > 0)].mean()) / 2)
    # each class weighs the inverse of its weight: more background, or heavier background, changes nothing
    heavier = known * torch.where(target > 0, 1.0, 5.0)
    assert torch.allclose(train._balanced_bce(logit, target, heavier), got)
    # a batch with no core px: the background's mean alone
    none = torch.zeros_like(target)
    alone = F.binary_cross_entropy_with_logits(logit, none, reduction="none")[known > 0].mean()
    assert torch.allclose(train._balanced_bce(logit, none, known), alone)


def test_word_loss_is_zero_for_a_perfect_answer_and_its_tags_are_classified_among_the_crops_words():
    crops = _word_batch()
    core, geo, tag, known, geo_known, groups = train.word_tensors(crops)
    out = torch.zeros(2, 16, 64, 64)
    out[:, model.CORE] = torch.where(core > 0, 30.0, -30.0)
    out[:, model.GEO] = geo[:, :4]
    rad = torch.deg2rad(2 * geo[:, 4])                   # the axis as (cos 2t, sin 2t)
    out[:, 5], out[:, 6] = torch.cos(rad), torch.sin(rad)
    out[:, model.TAG.start] = 10.0 * tag                 # each word's tag far from the others' and the background's
    _, parts = train.word_loss(out, core, geo, tag, known, geo_known, groups)
    names = dict(zip(train.WORD_PARTS, parts.tolist()))
    assert sorted(names) == sorted(("core", "dice", "centre", "size", "axis", "tag"))
    assert all(v < 1e-5 for v in names.values()), names
    # every tag the same: each core px is any of its crop's classes alike, log(classes) (its words and background)
    out[:, model.TAG] = 0
    _, parts = train.word_loss(out, core, geo, tag, known, geo_known, groups)
    classes = [int(tag[c].max()) + int(((core[c] == 0) & (known[c] > 0)).any()) for c in range(2)]
    on = known * (tag > 0)
    want = sum(np.log(classes[c]) * float(on[c].sum()) for c in range(2)) / float(on.sum())
    assert dict(zip(train.WORD_PARTS, parts.tolist()))["tag"] == pytest.approx(want, rel=1e-6)
    # tags one apart: a warmer temperature makes the classes harder to tell apart
    out[:, model.TAG.start] = 1.0 * tag

    def tag_part(t):
        return train.word_loss(out, core, geo, tag, known, geo_known, groups, log_temp=torch.tensor(t))[1][-1]
    assert tag_part(0.0) < tag_part(1.0) < tag_part(2.0)


def test_the_word_terms_are_added_by_learned_log_variances():
    crops = _word_batch()
    targets = train.word_tensors(crops)
    out = torch.randn(2, 16, 64, 64, generator=torch.Generator().manual_seed(1))
    s = torch.tensor([0.5, -1.0, 0.0, 2.0, -0.3, 0.7])
    loss, p = train.word_loss(out, *targets, log_var=s)
    assert torch.allclose(loss, (torch.exp(-s) * p + s / 2).sum())
    assert torch.allclose(train.word_loss(out, *targets)[0], p.sum())       # at their start, 0: the plain sum
    # the module holds them, and the loss reaches each, the temperature and every word output
    crit, net = train.WordLoss(), train.build_net(train.Recipe(heads="words", width=4))
    loss, _ = crit(net(train.batch_tensors(crops)[0]), *targets)
    loss.backward()
    assert (crit.log_var.grad != 0).all() and crit.log_temp.grad != 0
    assert (net.head.weight.grad.flatten(1).abs().sum(1)[model.CORE:] > 0).all()


def test_word_geometry_decodes_the_centre_in_px_the_sizes_and_the_axis():
    sd, = shrink.prepare([_sheet([(AT_ANGLE[90], 90)])])
    t = data.crop_targets(sd, 120, 160, 128, 1.0, 1, words=True)       # a turned crop
    out = torch.zeros(16, 128, 128)
    out[model.GEO] = torch.from_numpy(t["geo"][:4].copy())
    twice = torch.deg2rad(2 * torch.from_numpy(t["geo"][4].copy()))
    out[model.AXIS] = 3.0 * torch.stack([torch.cos(twice), torch.sin(twice)])      # any length
    got = model.word_geometry(out).numpy()
    rows, cols = np.nonzero(t["core"])
    assert got.shape == (5, 128, 128)
    assert np.allclose(np.c_[cols + got[0][rows, cols], rows + got[1][rows, cols]], _votes(t, 1), atol=1e-4)
    assert np.allclose(got[2:4][:, rows, cols], np.exp(t["geo"][2:4][:, rows, cols]), rtol=1e-5)
    assert np.allclose(np.cos(np.deg2rad(2 * got[4][rows, cols])), 1, atol=1e-6)     # 90 turned: axis 0 (= 180)
    # a batch, and axes either side of 0 / 180 degrees
    axis = torch.tensor([179.0, 1.0, 90.0])
    batch = torch.zeros(3, 16, 1, 1)
    batch[:, 5, 0, 0], batch[:, 6, 0, 0] = torch.cos(torch.deg2rad(2 * axis)), torch.sin(torch.deg2rad(2 * axis))
    assert torch.allclose(model.word_geometry(batch)[:, 4, 0, 0], axis, atol=1e-3)


def _fixed_batch():
    """One 6 x 8 crop: word 1 (rows 1-2, cols 1-3, axis 0 deg, every px given dx 0.5, dy -0.25, length 3, height 2),
    word 2 (rows 1-3, cols 5-6, axis 90 deg, its geometry not known), px (5, 7) not known, the rest background; the
    outputs a fixed sine."""
    core, tag = torch.zeros(1, 6, 8), torch.zeros(1, 6, 8, dtype=torch.int64)
    core[0, 1:3, 1:4], tag[0, 1:3, 1:4] = 1, 1
    core[0, 1:4, 5:7], tag[0, 1:4, 5:7] = 1, 2
    known = torch.ones(1, 6, 8)
    known[0, 5, 7] = 0
    geo = torch.zeros(1, 5, 6, 8)
    geo[0, :4, 1:3, 1:4] = torch.tensor([0.5, -0.25, float(np.log(3)), float(np.log(2))])[:, None, None]
    geo[0, 4, 1:4, 5:7] = 90.0
    geo_known = torch.zeros(1, 6, 8)
    geo_known[0, 1:3, 1:4] = 1
    out = torch.sin(torch.arange(16 * 6 * 8, dtype=torch.float32) * 0.37).reshape(1, 16, 6, 8)
    return out, core, geo, tag, known, geo_known, torch.arange(1, 3)


def test_word_loss_fixed_values():
    # pinned from the implementation, and equal within 1e-7 to a float64 numpy reference written from the module's
    # formulas (balanced cross-entropy, Dice, the centre's and the size's mean r^2 / 2 over their two coordinates at
    # the px whose geometry is known, 1 - cos 2(t' - t) of the axis, the tag softmax at T = e^0.2; Kendall's sum)
    total, parts = train.word_loss(*_fixed_batch(), log_var=torch.tensor([0.1, -0.2, 0.3, 0.0, 0.5, -0.4]),
                                   log_temp=torch.tensor(0.2))
    assert parts.tolist() == pytest.approx(
        [0.794277489, 0.680517197, 0.302051008, 0.755115211, 0.914711177, 1.09964228], rel=1e-5)
    assert total.item() == pytest.approx(4.87403154, rel=1e-5)


def test_a_word_heads_fit_trains_the_per_ink_head_by_todays_loss_learns_its_weights_and_repeats(monkeypatch,
                                                                                                 tmp_path):
    sheets = shrink.prepare([_sheet([((40, 60, 100, 80), 0), ((100, 60, 160, 80), 0), ((160, 40, 180, 160), 90)]),
                             _sheet([((20, 20, 120, 50), 30), ((60, 150, 200, 175), 0)], sheet="t")])
    recipe = train.Recipe(heads="words", iterations=2, batch=3, crop=48, width=4)
    real, seen, given, drawn = train.loss_fn, [], [], []
    monkeypatch.setattr(train, "loss_fn", lambda out, t, k, a, w: seen.append(out.shape) or given.append(a)
                        or real(out, t, k, a, w))
    real_tensors = train.batch_tensors
    monkeypatch.setattr(train, "batch_tensors", lambda crops, ink_only: drawn.append(
        real_tensors(crops, ink_only)) or drawn[-1])
    first = train.fit(sheets, recipe, "cpu", seed=0)
    again = train.fit(sheets, recipe, "cpu", seed=0)
    assert seen == [(3, 16, 48, 48)] * 4
    # the per-ink terms read channels 5-6 as the axis: today's loss is given twice each px's reading angle
    assert len(drawn) == len(given) == 4 and all(torch.equal(a, 2 * b[3]) for a, b in zip(given, drawn))
    for (name, a), b in zip(first.state_dict().items(), again.state_dict().values()):
        assert torch.equal(a, b), name
    # the word terms' weights and the temperature were trained from their start (log-variance 0: weight 1), the same
    # both times, kept on the network and saved beside its weights
    learned = first.learned_word_loss
    assert learned == again.learned_word_loss and sorted(learned["weight"]) == sorted(train.WORD_PARTS)
    assert all(w != 1 for w in learned["weight"].values()) and learned["temperature"] != 1
    saved = torch.load(train.save_model(tmp_path / "m.pt", first, recipe, "cpu"), weights_only=False)
    assert saved["word_loss"] == learned
    plain = train.Recipe(width=4)
    assert "word_loss" not in torch.load(train.save_model(tmp_path / "plain.pt", train.build_net(plain), plain, "cpu"),
                                         weights_only=False)
