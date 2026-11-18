"""Code spotting (`train.Recipe.heads="words_code"`, `model.CODE`): each Terra word's class (`wordread.codes`), the
code targets (`data.crop_targets(..., code=True)`, painted as the words are, turned with them, known where the sheet's
classes are), the class-balanced cross-entropy among the learned word terms (`train.word_loss`), the prediction
(`predict.word_outputs`'s `code` and `code_p`), the saved outputs, and the other heads left as they were."""
import contextlib

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, model, predict, train
from sfzmaps.solver.textread.read import CodeList
from sfzmaps.solver.wordread import codes, shrink


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


# 57 codes (the head's classes less the background and another text), the first with a second printed form
CODE_LIST = CodeList({**{f"C{i:02d}": f"C{i:02d}" for i in range(model.CODE_CLASSES - 2)}, "C00 NCD": "C00"})

WORDS = [((40, 60, 100, 80), 0, "c00 ncd"),       # C00 by its second form
         ((101, 60, 160, 80), 0, "7520"),         # a block number: another text, touching the first
         ((161, 40, 180, 160), 90, "C05"),        # a code read upward, touching the second
         ((80, 65, 130, 95), 0, "C07")]           # a code over the first two (painted later: on top)


def _sheet(words=WORDS, size=(240, 320), sheet="s", series=paths.SERIES) -> data.SheetData:
    """An in-memory sheet with its cores, no placement and nothing held out: a stroke of ink along each word."""
    h, w = size
    gray = np.full((h, w), 255, np.uint8)
    rects = [data.word_rect(list(box), angle) for box, angle, _ in words]
    for r in rects:
        a, b = (r[0] + r[3]) / 2, (r[1] + r[2]) / 2
        cv2.line(gray, tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b)), 0, 3)
    sd = data.SheetData(
        series=series, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.arange(len(words), dtype=np.uint8) % 4,
        angles=np.array([data.reading_angle(r) for r in rects], np.int16),
        other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)),
        ft_per_px=float("nan"))
    return shrink.prepare([sd])[0]


def _terra(words=WORDS):
    return [{"box": list(box), "angle_deg": angle, "text": text} for box, angle, text in words]


@pytest.fixture
def coded(monkeypatch):
    """The sheet's words' classes by `codes.prepare`, Terra's boxes the sheet's own."""
    monkeypatch.setattr(codes.boxes, "terra_boxes", lambda series, sheet, records=None: _terra())
    return codes.prepare([_sheet()], CODE_LIST, records=[])[0]


# --- the classes -----------------------------------------------------------------------------------------------------

def test_a_word_is_a_codes_when_its_string_is_one_of_its_forms():
    names = codes.classes(CODE_LIST)
    assert names[:3] == ["background", "other text", "C00"] and len(names) == model.CODE_CLASSES
    assert codes.word_class("c00   ncd", CODE_LIST) == codes.word_class("C00", CODE_LIST) == 2
    assert codes.word_class("C05", CODE_LIST) == 7
    assert codes.word_class("7520", CODE_LIST) == codes.word_class("C00-X", CODE_LIST) == codes.OTHER
    with pytest.raises(ValueError, match="classes"):
        codes.classes(CodeList({"RH-1": "RH-1"}))


def test_prepare_gives_the_2000_sheets_their_words_classes_and_leaves_the_others(coded, monkeypatch):
    assert coded.codes.tolist() == [2, 1, 7, 9] and coded.codes.dtype == np.uint8
    other = _sheet(series="1948")
    assert codes.prepare([other], CODE_LIST, records=[])[0] is other
    monkeypatch.setattr(codes.boxes, "terra_boxes", lambda series, sheet, records=None: _terra()[:3])
    with pytest.raises(ValueError, match="not the sheet's words"):
        codes.prepare([_sheet()], CODE_LIST, records=[])
    assert codes.counts([coded, other])[[1, 2, 7, 9]].tolist() == [1, 1, 1, 1]


# --- the targets -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_code_is_each_words_class_painted_as_its_kind_and_turned_with_it(coded, turns):
    t = data.crop_targets(coded, 90, 110, 128, 1.0, turns, code=True)
    kind_to_class = np.full(256, 255, np.uint8)
    kind_to_class[coded.kinds] = coded.codes                 # one kind a word here: the code follows the kind
    kind_to_class[data.NO_KIND] = 0
    assert np.array_equal(t["code"], kind_to_class[t["kind"]])
    assert set(np.unique(t["code"])) == {0, 1, 2, 7, 9} and t["code_known"].all()
    plain = data.crop_targets(coded, 90, 110, 128, 1.0, turns)
    assert plain.keys() == t.keys() - {"code", "code_known"}
    for k, v in plain.items():
        assert np.array_equal(v, t[k]) and v.dtype == t[k].dtype, k


def test_the_later_word_holds_the_px_where_rectangles_overlap(coded):
    full = data.crop_targets(coded, 120, 160, 240, 1.0, 0, code=True)    # page rows 0-239, columns 40-279
    first = data._paint(coded.words[:1], np.ones(1), 0, 40, 240, 240) > 0
    over = data._paint(coded.words[3:], np.ones(1), 0, 40, 240, 240) > 0
    assert (first & over).any() and (first & ~over).any()
    assert (full["code"][first & over] == 9).all() and (full["code"][first & ~over] == 2).all()


def test_line_px_inside_a_word_are_known_to_the_code_not_to_the_text(coded):
    lines = np.zeros((coded.height, coded.width), bool)
    lines[:, 70] = True                                          # a line through the first word
    sd = data.dataclasses.replace(coded, line_bits=np.packbits(lines, axis=1))
    t = data.crop_targets(sd, 90, 110, 128, 1.0, 0, labels="lines-unknown", code=True)
    inside = (t["code"] > 0) & np.zeros_like(t["known"])
    col = 70 - (110 - 64)
    inside[:, col] = t["code"][:, col] > 0
    assert inside.any() and not t["known"][inside].any() and t["code_known"][inside].all()


def test_a_sheet_with_no_classes_teaches_no_code():
    t = data.crop_targets(_sheet(series="1948"), 90, 110, 128, 1.0, 0, code=True)
    assert not t["code_known"].any() and not t["code"].any() and t["known"].any()
    with pytest.raises(ValueError, match="one class per word"):
        data.crop_targets(data.dataclasses.replace(_sheet(), codes=np.ones(2, np.uint8)), 90, 110, 64, 1.0, 0,
                          code=True)


# --- the loss --------------------------------------------------------------------------------------------------------

def _batch(sd):
    return [data.crop_targets(sd, 70, 100, 64, 1.0, 0, words=True, code=True),
            data.crop_targets(sd, 90, 140, 64, 0.8, 1, words=True, code=True)]


def _outputs(targets, right=True):
    core, geo, tag, known, geo_known, groups, code, code_weight = targets
    out = torch.zeros(core.shape[0], model.WORD_CODE_OUTPUTS, *core.shape[1:])
    out[:, model.CORE] = torch.where(core > 0, 30.0, -30.0)
    out[:, model.GEO] = geo[:, :4]
    rad = torch.deg2rad(2 * geo[:, 4])
    out[:, 5], out[:, 6] = torch.cos(rad), torch.sin(rad)
    out[:, model.TAG.start] = 10.0 * tag
    want = code if right else (code + 1) % model.CODE_CLASSES
    out[:, model.CODE] = 30.0 * F.one_hot(want, model.CODE_CLASSES).permute(0, 3, 1, 2).float()
    return out


def test_the_code_term_is_the_mean_over_each_class_averaged_over_the_classes(coded):
    targets = train.word_tensors(_batch(coded), code=True)
    assert len(targets) == 8
    code, weight = targets[-2:]
    assert code.dtype == torch.int64 and weight.dtype == torch.float32 and code.shape == weight.shape == (2, 64, 64)
    torch.manual_seed(0)
    out = torch.randn(2, model.WORD_CODE_OUTPUTS, 64, 64)
    _, parts = train.word_loss(out, *targets[:-2], code=code, code_weight=weight)
    ce = F.cross_entropy(out[:, model.CODE], code, reduction="none")
    known = torch.from_numpy(np.stack([c["code_known"] for c in _batch(coded)]))
    means = [ce[(code == c) & known].mean() for c in range(model.CODE_CLASSES) if ((code == c) & known).any()]
    assert len(means) >= 3 and torch.isclose(parts[-1], torch.stack(means).mean(), rtol=1e-5)
    assert torch.isclose(weight.sum(), torch.tensor(1.0)) and (weight[~known] == 0).all()


def test_the_code_term_is_zero_when_right_and_rises_when_wrong(coded):
    targets = train.word_tensors(_batch(coded), code=True)
    crit = train.WordLoss(code=True)
    assert crit.parts == train.CODE_PARTS == train.WORD_PARTS + ("code",) and crit.log_var.shape == (7,)
    _, right = crit(_outputs(targets), *targets)
    _, wrong = crit(_outputs(targets, right=False), *targets)
    assert right[-1] < 1e-6 and wrong[-1] > 10 and torch.allclose(right[:-1], wrong[:-1])
    with pytest.raises(ValueError, match="words_code"):
        train.word_loss(torch.zeros(2, model.WORD_OUTPUTS, 64, 64), *targets[:-2], code=targets[-2],
                        code_weight=targets[-1])
    with pytest.raises(ValueError, match="not both"):
        train.WordLoss(affinity=True, code=True)


def test_a_crop_without_code_targets_teaches_none(coded):
    crops = _batch(coded)
    bare = {k: v for k, v in crops[1].items() if not k.startswith("code")}
    code, weight = train.word_tensors([crops[0], bare], code=True)[-2:]
    assert (weight[1] == 0).all() and (code[1] == 0).all() and weight[0].sum() > 0


# --- the recipe, the network, the prediction -------------------------------------------------------------------------

def test_words_code_is_a_recipe_choice_and_the_other_heads_are_unchanged():
    recipe = train.Recipe(heads="words_code", width=4, crop=32)
    assert recipe.name() == "crop=32-heads=words_code-width=4"
    for half in (False, True):
        out = train.build_net(train.Recipe(heads="words_code", width=4, half_size=half))(torch.zeros(2, 1, 32, 32))
        assert out.shape == (2, model.WORD_CODE_OUTPUTS, 32, 32) == (2, 16 + model.CODE_CLASSES, 32, 32)
    assert model.HEAD_OUTPUTS["words"] == 16 and model.HEAD_OUTPUTS["words_aff"] == 18
    assert train.WordLoss().parts == train.WORD_PARTS and train.WordLoss(affinity=True).parts == train.AFF_PARTS
    assert train.Recipe().name() == "default" and train.Recipe(heads="words").name() == "heads=words"


@pytest.mark.parametrize("half", [False, True])
def test_word_outputs_give_each_px_its_likeliest_class_and_its_chance(half, tmp_path):
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(width=4, half_size=half, heads="words_code")).eval()
    ink = np.random.default_rng(5).random((150, 140)) < 0.2
    ink[60:75, 20:120] = True
    tile = 2 * net.multiple
    got = predict.word_outputs(net, ink, "cpu", tile=tile)
    with torch.no_grad():
        ph, pw = (-150) % net.multiple, (-140) % net.multiple
        raw = net(torch.from_numpy(np.pad(ink, ((0, ph), (0, pw))).astype(np.float32))[None, None])[0, :, :150, :140]
    p = torch.softmax(raw[model.CODE], 0)
    assert got.code.dtype == np.uint8 and got.code_p.dtype == np.float16 and got.aff is None
    agree = got.code == p.argmax(0).numpy()
    top2 = p.topk(2, 0).values
    assert (agree | ((top2[0] - top2[1]).numpy() < 1e-4)).all() and agree.mean() > 0.999
    assert np.allclose(got.code_p, p.max(0).values.numpy(), atol=2e-3)
    only = np.zeros(ink.shape, bool)
    only[:tile, :tile] = True
    some = predict.word_outputs(net, ink, "cpu", only=only, tile=tile)
    assert (some.code[tile:, :] == predict.NO_CODE).all() and (some.code_p[tile:, :] == 0).all()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
    import variant
    variant.save_word_outputs(tmp_path / "w.npz", got)
    back = variant.load_word_outputs(tmp_path / "w.npz")
    assert back.code.dtype == np.uint8 and np.array_equal(back.code, got.code) and back.aff is None
    assert np.array_equal(back.code_p, got.code_p.astype(np.float32))
    words = train.build_net(train.Recipe(width=4, half_size=half, heads="words")).eval()
    plain = predict.word_outputs(words, ink, "cpu", tile=tile)
    assert plain.code is None and plain.code_p is None


def test_a_words_code_fit_learns_the_code_term_and_repeats(coded, tmp_path):
    sheets = [coded, _sheet(series="1948", sheet="t")]
    recipe = train.Recipe(heads="words_code", iterations=2, batch=3, crop=48, width=4)
    first = train.fit(sheets, recipe, "cpu", seed=0)
    again = train.fit(sheets, recipe, "cpu", seed=0)
    for (name, a), b in zip(first.state_dict().items(), again.state_dict().values()):
        assert torch.equal(a, b), name
    learned = first.learned_word_loss
    assert list(learned["weight"]) == list(train.CODE_PARTS) and learned["weight"]["code"] != 1
    saved = torch.load(train.save_model(tmp_path / "m.pt", first, recipe, "cpu"), weights_only=False)
    assert saved["word_loss"] == learned and saved["recipe"]["heads"] == "words_code"
    assert predict.load_model(tmp_path / "m.pt", "cpu").head.out_channels == model.WORD_CODE_OUTPUTS
