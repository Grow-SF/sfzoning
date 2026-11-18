"""The detector's same-word affinity (`train.Recipe.heads="words_aff"`, `model.AFF`): its targets
(`data.affinity_targets`, `data.crop_targets(..., affinity=True)`), exact under the quarter turns; its loss term
(`train.word_loss`, the balanced cross-entropy among the learned terms); its prediction (`predict.word_outputs`'s
`aff`); and the "words" heads left as they were."""
import contextlib

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, model, predict, train
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


def _sheet(words, size=(240, 320), sheet="s", weights=None) -> data.SheetData:
    """An in-memory sheet with its cores, no placement and nothing held out: a blank page with a stroke of ink along
    each word's middle, and Terra's `words` (axis box (x0, y0, x1, y1), reading angle)."""
    h, w = size
    gray = np.full((h, w), 255, np.uint8)
    rects = [data.word_rect(list(box), angle) for box, angle in words]
    for r in rects:
        a, b = (r[0] + r[3]) / 2, (r[1] + r[2]) / 2
        cv2.line(gray, tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b)), 0, 3)
    sd = data.SheetData(
        series=paths.SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.zeros(len(words), np.uint8), angles=np.array([data.reading_angle(r) for r in rects], np.int16),
        other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)),
        ft_per_px=float("nan"), **({} if weights is None else {"weights": np.array(weights, np.float32)}))
    return shrink.prepare([sd])[0]


TOUCHING = [((40, 60, 100, 80), 0), ((101, 60, 160, 80), 0),       # R-4 | RH-2: word 2 starts on the next column
            ((161, 40, 180, 160), 90),                              # a street name read upward, touching word 2
            ((40, 81, 160, 101), 30)]                               # slanted text right under them


def _reference(tag_id, known):
    """The affinity targets written px pair by px pair, as the module states them."""
    h, w = tag_id.shape
    right, down, taught = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), np.zeros((2, h, w), np.uint8)
    for r in range(h):
        for c in range(w):
            for k, (r2, c2) in enumerate(((r, c + 1), (r + 1, c))):
                if r2 >= h or c2 >= w:
                    continue
                a, b = int(tag_id[r, c]), int(tag_id[r2, c2])
                (right if k == 0 else down)[r, c] = a == b and a > 0
                taught[k, r, c] = bool(known[r, c]) and bool(known[r2, c2]) and (a > 0 or b > 0)
    return right, down, taught


def _pairs(t, size):
    """A crop's affinity targets as a set of (unordered px pair, target, taught)."""
    out = set()
    for k, (dr, dc) in enumerate(((0, 1), (1, 0))):
        target = t["aff_right"] if k == 0 else t["aff_down"]
        for r in range(size - dr):
            for c in range(size - dc):
                out.add((frozenset({(r, c), (r + dr, c + dc)}), int(target[r, c]), int(t["aff_known"][k, r, c])))
    return out


def _turned(pairs, turns, size):
    """The pairs of a crop turned `turns` quarter turns counter-clockwise (`np.rot90`): px (r, c) to (size - 1 - c, r)
    a turn."""
    def move(p):
        r, c = p
        for _ in range(turns):
            r, c = size - 1 - c, r
        return r, c
    return {(frozenset(move(p) for p in pair), target, taught) for pair, target, taught in pairs}


# --- the targets -----------------------------------------------------------------------------------------------------

def test_two_touching_words_are_apart_across_their_shared_edge_and_one_inside_each():
    sd = _sheet(TOUCHING)
    t = data.crop_targets(sd, 70, 100, 64, 1.0, 0, words=True, affinity=True)     # page window: rows 38-101, cols 68-131
    assert t["aff_right"].dtype == t["aff_down"].dtype == t["aff_known"].dtype == np.uint8
    assert t["aff_right"].shape == t["aff_down"].shape == (64, 64) and t["aff_known"].shape == (2, 64, 64)
    tag = t["tag_id"]
    r = 70 - 38                                         # a row through words 1 and 2
    c = 100 - 68                                        # word 1's last column: word 2 starts on the next
    assert tag[r, c] == 1 and tag[r, c + 1] == 2
    # across the shared edge: taught, not one word
    assert t["aff_known"][0, r, c] == 1 and t["aff_right"][r, c] == 0
    # inside each word, right and down: one word
    for word in (1, 2):
        rows, cols = np.nonzero(tag == word)
        right = (cols + 1 < 64) & (tag[rows, np.minimum(cols + 1, 63)] == word)
        down = (rows + 1 < 64) & (tag[np.minimum(rows + 1, 63), cols] == word)
        assert right.sum() > 100 and down.sum() > 100
        assert t["aff_right"][rows[right], cols[right]].all() and t["aff_known"][0][rows[right], cols[right]].all()
        assert t["aff_down"][rows[down], cols[down]].all() and t["aff_known"][1][rows[down], cols[down]].all()
    # a word's px beside the background: taught, apart
    above = 60 - 38 - 1                                 # the row above words 1 and 2
    assert tag[above, 12] == 0 and tag[above + 1, 12] == 1      # page column 80
    assert t["aff_known"][1, above, 12] == 1 and t["aff_down"][above, 12] == 0
    # the targets are exactly the module's rule, px pair by px pair
    for got, want in zip((t["aff_right"], t["aff_down"], t["aff_known"]), _reference(tag, t["known"])):
        assert np.array_equal(got, want)


def test_background_pairs_and_pairs_with_an_unknown_px_are_not_taught():
    tag = np.array([[0, 0, 1, 1],
                    [0, 0, 1, 2],
                    [0, 0, 0, 2]], np.int32)
    known = np.ones(tag.shape, bool)
    known[2, 3] = False                                  # e.g. held out
    got = data.affinity_targets(tag, known)
    # background beside background: not taught, either way
    assert got["aff_known"][0, 0, 0] == 0 and got["aff_known"][1, 0, 0] == 0 and got["aff_known"][0, 2, 1] == 0
    # background beside a word: taught, apart; a word beside itself: one; beside another word: apart
    assert got["aff_known"][0, 0, 1] == 1 and got["aff_right"][0, 1] == 0
    assert got["aff_known"][0, 0, 2] == 1 and got["aff_right"][0, 2] == 1
    assert got["aff_known"][0, 1, 2] == 1 and got["aff_right"][1, 2] == 0
    assert got["aff_known"][1, 0, 3] == 1 and got["aff_down"][0, 3] == 0
    # a pair with an unknown px is not taught, though one is a word's
    assert got["aff_known"][1, 1, 3] == 0 and got["aff_known"][0, 2, 2] == 0
    # the last column has no px to its right, the last row none below
    assert not got["aff_known"][0, :, -1].any() and not got["aff_known"][1, -1, :].any()
    for a, b in zip((got["aff_right"], got["aff_down"], got["aff_known"]), _reference(tag, known)):
        assert np.array_equal(a, b)
    # overlapping rectangles: the later word painted over the earlier, the pairs follow tag_id
    sd = _sheet([((40, 60, 110, 80), 0), ((100, 60, 160, 80), 0)])
    t = data.crop_targets(sd, 70, 100, 64, 1.0, 0, words=True, affinity=True)
    r, c = 70 - 38, 99 - 68                              # column 99 is word 1's, 100 the overlap: word 2's
    assert t["tag_id"][r, c] == 1 and t["tag_id"][r, c + 1] == 2 and t["aff_right"][r, c] == 0
    assert t["aff_right"][r, c + 1] == 1                # inside the overlap, word 2 on both


@pytest.mark.parametrize("scale", [1.0, 0.7, 1.3])
@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_targets_are_exact_under_every_turn(turns, scale):
    sd = _sheet(TOUCHING)
    size = 64
    flat = data.crop_targets(sd, 80, 110, size, scale, 0, words=True, affinity=True)
    t = data.crop_targets(sd, 80, 110, size, scale, turns, words=True, affinity=True)
    assert len(np.unique(t["tag_id"])) >= 4                              # several words, touching, in view
    # recomputed directly from the turned word ids and known px
    for got, want in zip((t["aff_right"], t["aff_down"], t["aff_known"]), _reference(t["tag_id"], t["known"])):
        assert np.array_equal(got, want)
    # and the unturned crop's pairs, each moved by the turn: a right pair becomes an up pair after one turn (a down
    # pair of the px above), a left pair after two, a down pair after three
    assert _pairs(t, size) == _turned(_pairs(flat, size), turns, size)
    if turns == 1:
        assert np.array_equal(t["aff_down"][:-1], np.rot90(flat["aff_right"][:, :-1], 1))
        assert np.array_equal(t["aff_known"][1][:-1], np.rot90(flat["aff_known"][0][:, :-1], 1))
    # the other keys are those without affinity
    plain = data.crop_targets(sd, 80, 110, size, scale, turns, words=True)
    assert sorted(set(t) - set(plain)) == ["aff_down", "aff_known", "aff_right"]
    for k in plain:
        assert np.array_equal(t[k], plain[k])


def test_affinity_needs_the_word_targets():
    with pytest.raises(ValueError, match="words"):
        data.crop_targets(_sheet(TOUCHING), 70, 100, 64, 1.0, 0, affinity=True)


# --- the loss --------------------------------------------------------------------------------------------------------

def _aff_batch(weights=None):
    sd = _sheet(TOUCHING, weights=weights)
    return [data.crop_targets(sd, 70, 100, 64, 1.0, 0, words=True, affinity=True),
            data.crop_targets(sd, 90, 140, 64, 0.8, 1, words=True, affinity=True)]


def _perfect(targets):
    """Outputs that answer every word term (as test_wordread_heads' perfect answer) and the affinity exactly."""
    core, geo, tag, known, geo_known, groups, aff, aff_weight = targets
    out = torch.zeros(core.shape[0], model.WORD_AFF_OUTPUTS, *core.shape[1:])
    out[:, model.CORE] = torch.where(core > 0, 30.0, -30.0)
    out[:, model.GEO] = geo[:, :4]
    rad = torch.deg2rad(2 * geo[:, 4])
    out[:, 5], out[:, 6] = torch.cos(rad), torch.sin(rad)
    out[:, model.TAG.start] = 10.0 * tag
    out[:, model.AFF] = torch.where(aff > 0, 30.0, -30.0)
    return out


def test_the_affinity_term_is_zero_when_right_and_rises_when_wrong():
    targets = train.word_tensors(_aff_batch(), affinity=True)
    assert len(targets) == 8
    aff, aff_weight = targets[-2:]
    assert aff.shape == aff_weight.shape == (2, 2, 64, 64) and aff.dtype == aff_weight.dtype == torch.float32
    out = _perfect(targets)
    _, parts = train.word_loss(out, *targets)
    names = dict(zip(train.AFF_PARTS, parts.tolist()))
    assert list(names) == ["core", "dice", "centre", "size", "axis", "tag", "affinity"]
    assert all(v < 1e-5 for v in names.values()), names
    # the affinity the wrong way round on the taught pairs: the term rises, the others stay
    wrong = out.clone()
    wrong[:, model.AFF] = -out[:, model.AFF]
    _, bad = train.word_loss(wrong, *targets)
    assert bad[-1] > 10 and torch.allclose(bad[:-1], parts[:-1])
    # only the words' borders wrong (the pairs across two words): it rises, by less
    cross = (aff == 0) & (aff_weight > 0)
    tags = targets[2]
    two_words = torch.zeros_like(cross)
    two_words[:, 0, :, :-1] = (tags[:, :, :-1] > 0) & (tags[:, :, 1:] > 0) & (tags[:, :, :-1] != tags[:, :, 1:])
    two_words[:, 1, :-1] = (tags[:, :-1] > 0) & (tags[:, 1:] > 0) & (tags[:, :-1] != tags[:, 1:])
    assert (two_words & cross).any()
    merged = out.clone()
    merged[:, model.AFF] = torch.where(two_words & cross, 30.0, out[:, model.AFF])
    _, joined = train.word_loss(merged, *targets)
    assert parts[-1] < joined[-1] < bad[-1]
    # untaught pairs (background beside background, the last column and row) teach nothing
    free = out.clone()
    free[:, model.AFF] = torch.where(aff_weight > 0, out[:, model.AFF], torch.randn_like(out[:, model.AFF]) * 50)
    assert torch.allclose(train.word_loss(free, *targets)[1], parts)


def test_the_affinity_term_is_the_balanced_cross_entropy_over_the_taught_pairs():
    targets = train.word_tensors(_aff_batch(), affinity=True)
    aff, aff_weight = targets[-2:]
    out = torch.randn(2, model.WORD_AFF_OUTPUTS, 64, 64, generator=torch.Generator().manual_seed(3))
    logit = out[:, model.AFF]
    bce = F.binary_cross_entropy_with_logits(logit, aff, reduction="none")
    taught = aff_weight > 0
    want = (bce[taught & (aff > 0)].mean() + bce[taught & (aff == 0)].mean()) / 2
    assert torch.allclose(train.word_loss(out, *targets)[1][-1], want)
    # added by its own learned log-variance, after the six word terms'
    s = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.7])
    loss, p = train.word_loss(out, *targets, log_var=s)
    assert torch.allclose(loss, (torch.exp(-s) * p + s / 2).sum())
    with pytest.raises(ValueError, match="log-variance"):
        train.word_loss(out, *targets, log_var=torch.zeros(6))
    with pytest.raises(ValueError, match="words_aff"):
        train.word_loss(out[:, :model.WORD_OUTPUTS], *targets)


def test_a_pair_weighs_the_chance_its_target_is_right():
    crops = _aff_batch(weights=[0.5, 0.8, 1.0, 0.25])
    _, weight = train.word_tensors(crops, affinity=True)[-2:]
    tag, known = crops[0]["tag_id"], crops[0]["aff_known"]
    r, c = 70 - 38, 100 - 68                             # word 1's last px, word 2's first to its right
    assert weight[0, 0, r, c] == pytest.approx(0.5 * 0.8)                # two words: both right
    assert weight[0, 0, r, c - 5] == pytest.approx(0.5)                  # inside word 1
    assert tag[21, 12] == 0 and tag[22, 12] == 1          # page rows 59 and 60, column 80
    assert weight[0, 1, 21, 12] == pytest.approx(0.5)               # word 1 beside the background
    assert torch.equal(weight[0] > 0, torch.from_numpy(known.astype(bool)))
    # without weights every taught pair weighs 1
    plain = train.word_tensors(_aff_batch(), affinity=True)[-1]
    assert torch.equal(plain, torch.from_numpy(np.stack([c["aff_known"] for c in _aff_batch()]).astype(np.float32)))


def test_a_crop_without_the_keys_gets_them_from_its_word_ids_and_known_px():
    crops = _aff_batch()
    bare = [{k: v for k, v in c.items() if not k.startswith("aff_")} for c in crops]
    bare[1]["known"] = bare[1]["known"].copy()
    bare[1]["known"][:, :20] = False                      # e.g. a scene's bank changing known after crop_targets
    got = train.word_tensors(bare, affinity=True)
    want = data.affinity_targets(bare[1]["tag_id"], bare[1]["known"])
    assert torch.equal(got[-1][1], torch.from_numpy(want["aff_known"].astype(np.float32)))
    assert torch.equal(got[-2][0], train.word_tensors(crops, affinity=True)[-2][0])


# --- the recipe, the network, the fit and the prediction -------------------------------------------------------------

def test_words_aff_is_a_recipe_choice_and_the_words_heads_are_unchanged(tmp_path):
    recipe = train.Recipe(heads="words_aff", width=4, crop=32)
    assert recipe.name() == "crop=32-heads=words_aff-width=4"
    with pytest.raises(ValueError, match="composites"):
        train.Recipe(heads="words_aff", composites=0.5)
    for half in (False, True):
        out = train.build_net(train.Recipe(heads="words_aff", width=4, half_size=half))(torch.zeros(2, 1, 32, 32))
        assert out.shape == (2, model.WORD_AFF_OUTPUTS, 32, 32) == (2, 18, 32, 32)
    # the words heads: 16 outputs, six learned terms, six targets, and a saved words model loads as it was
    words = train.Recipe(heads="words", width=4)
    torch.manual_seed(0)
    net = train.build_net(words)
    assert net.head.out_channels == model.WORD_OUTPUTS == 16
    assert train.WordLoss().parts == train.WORD_PARTS and train.WordLoss().log_var.shape == (6,)
    assert train.WordLoss(affinity=True).log_var.shape == (7,)
    crops = [{k: v for k, v in c.items() if not k.startswith("aff_")} for c in _aff_batch()]
    assert len(train.word_tensors(crops)) == 6
    loaded = predict.load_model(train.save_model(tmp_path / "words.pt", net, words, "cpu"), "cpu")
    assert loaded.head.out_channels == 16
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    ink = np.zeros((64, 64), bool)
    ink[20:30, 10:50] = True
    assert predict.word_outputs(loaded, ink, "cpu", tile=32).aff is None
    aff = train.Recipe(heads="words_aff", width=4)
    loaded = predict.load_model(train.save_model(tmp_path / "aff.pt", train.build_net(aff), aff, "cpu"), "cpu")
    assert loaded.head.out_channels == 18


@pytest.mark.parametrize("half", [False, True])
def test_word_outputs_stitch_the_affinity_as_one_pass_over_the_page(half):
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(width=4, half_size=half, heads="words_aff")).eval()
    ink = np.random.default_rng(5).random((150, 140)) < 0.2
    ink[60:75, 20:120] = True
    tile = 2 * net.multiple
    got = predict.word_outputs(net, ink, "cpu", tile=tile)
    with torch.no_grad():
        ph, pw = (-150) % net.multiple, (-140) % net.multiple
        raw = net(torch.from_numpy(np.pad(ink, ((0, ph), (0, pw))).astype(np.float32))[None, None])[0, :, :150, :140]
    assert got.aff.shape == (2, 150, 140) and got.aff.dtype == np.float32
    want = raw[model.AFF].numpy()
    assert np.allclose(got.aff[0, :, :-1], want[0, :, :-1], atol=1e-4)
    assert np.allclose(got.aff[1, :-1], want[1, :-1], atol=1e-4)
    assert (got.aff[0, :, -1] == predict.NOT_SCORED).all() and (got.aff[1, -1] == predict.NOT_SCORED).all()
    assert np.allclose(got.core, raw[model.CORE].numpy(), atol=1e-4)
    only = np.zeros(ink.shape, bool)
    only[:tile, :tile] = True
    some = predict.word_outputs(net, ink, "cpu", only=only, tile=tile)
    assert (some.aff[:, tile:, :] == predict.NOT_SCORED).all()


def test_a_words_aff_fit_learns_the_affinity_term_and_repeats(tmp_path):
    sheets = [_sheet(TOUCHING), _sheet([((20, 20, 120, 50), 30), ((60, 150, 200, 175), 0)], sheet="t")]
    recipe = train.Recipe(heads="words_aff", iterations=2, batch=3, crop=48, width=4)
    first = train.fit(sheets, recipe, "cpu", seed=0)
    again = train.fit(sheets, recipe, "cpu", seed=0)
    for (name, a), b in zip(first.state_dict().items(), again.state_dict().values()):
        assert torch.equal(a, b), name
    learned = first.learned_word_loss
    assert list(learned["weight"]) == list(train.AFF_PARTS) and learned["weight"]["affinity"] != 1
    saved = torch.load(train.save_model(tmp_path / "m.pt", first, recipe, "cpu"), weights_only=False)
    assert saved["word_loss"] == learned and saved["recipe"]["heads"] == "words_aff"
