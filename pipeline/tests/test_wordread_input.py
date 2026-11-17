"""The detector's input (`train.Recipe.input`): the 1-bit ink (the default, byte-identical to before) or the scan's grey
levels (`data.SheetData.grey_bits`, `data.crop_targets(..., grey=True)`), and the full-size network (`half_size` off)."""
import dataclasses

import cv2
import numpy as np
import pytest
import torch

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, model, train
from sfzmaps.solver.wordread import shrink

H, W = 240, 320
WORDS = [((100, 50, 200, 80), 0), ((230, 40, 260, 140), 90)]


def _page() -> np.ndarray:
    """A grey page with every level in it (a fixed pseudo-random one), so that a wrong px or a wrong rounding shows."""
    return np.random.default_rng(7).integers(0, 256, (H, W)).astype(np.uint8)


def _sheet(grey: bool, words=WORDS) -> data.SheetData:
    gray = _page()
    rects = [data.word_rect(box, angle) for box, angle in words]
    sd = data.SheetData(
        series=paths.SERIES, sheet="s", height=H, width=W, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.zeros(len(words), np.uint8), angles=np.array([data.reading_angle(r) for r in rects], np.int16),
        other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)),
        ft_per_px=float("nan"), **({"grey_bits": gray} if grey else {}))
    return shrink.prepare([sd])[0]


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)


# --- the defaults are today's ------------------------------------------------------------------------------------

def test_default_recipe_is_ink_and_unnamed():
    r = train.Recipe()
    assert r.input == "ink" and r.name() == "default"
    assert train.Recipe(input="grey").name() == "input=grey"
    assert dataclasses.asdict(r) == {
        "iterations": 8000, "max_lr": 2e-3, "pct_start": 0.3, "batch": 12, "crop": 320, "width": 24, "ink_only": "off",
        "half_size": False, "heads": "all", "weight_average": 0.0, "prefetch": False, "decoder": "full",
        "precision": "float32", "labels": "boxes", "composites": 0.0, "input": "ink"}


def test_recipe_input_is_ink_or_grey():
    with pytest.raises(ValueError, match="input is one of ink, grey"):
        train.Recipe(input="rgb")
    with pytest.raises(ValueError, match="composites must be 0 with input grey"):
        train.Recipe(input="grey", composites=0.5)


def test_a_sheet_holds_no_grey_by_default():
    assert _sheet(False).grey_bits.shape == (0, 0)
    assert "grey" not in data.crop_targets(_sheet(False), 120, 160, 64, 1.0, 0)


@pytest.mark.parametrize("words", [False, True])
@pytest.mark.parametrize("scale,turns", [(1.0, 0), (0.6, 1), (1.6, 3)])
def test_grey_adds_a_key_and_changes_no_target(words, scale, turns):
    # every other key, byte for byte: the ink, text, kind, angle, known and, with the word heads, core, geo (the axis
    # on 5-6, centre offset 8-9 and sizes 10-11 are made from it), geo_known, tag_id
    without = data.crop_targets(_sheet(False), 120, 160, 64, scale, turns, words=words)
    with_grey = data.crop_targets(_sheet(True), 120, 160, 64, scale, turns, words=words, grey=True)
    assert with_grey.pop("grey").shape == (64, 64)
    assert _same(with_grey, without)
    if words:
        assert {"core", "geo", "geo_known", "tag_id"} <= without.keys()


def test_a_grey_crop_is_the_page_over_255_at_the_crop():
    sd = _sheet(True)
    page = _page()
    got = data.crop_targets(sd, 120, 160, 64, 1.0, 0, grey=True)["grey"]
    want = page[120 - 32:120 + 32, 160 - 32:160 + 32].astype(np.float32) / 255
    assert got.dtype == np.float32 and np.array_equal(got, want)
    assert 0 <= got.min() and got.max() <= 1 and len(np.unique(got)) > 100       # grey levels, not ink


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_a_grey_crop_turns_with_the_ink(turns):
    sd = _sheet(True)
    a = data.crop_targets(sd, 120, 160, 64, 1.0, 0, grey=True)["grey"]
    b = data.crop_targets(sd, 120, 160, 64, 1.0, turns, grey=True)["grey"]
    assert np.array_equal(b, np.rot90(a, turns))


@pytest.mark.parametrize("scale", [0.6, 1.6])
def test_a_scaled_grey_crop_is_resized_bilinearly(scale):
    sd = _sheet(True)
    src = int(round(64 / scale))
    y0, x0 = 120 - src // 2, 160 - src // 2
    want = cv2.resize(_page()[y0:y0 + src, x0:x0 + src].astype(np.float32) / 255, (64, 64),
                      interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(data.crop_targets(sd, 120, 160, 64, scale, 0, grey=True)["grey"], want)


def test_a_crop_at_the_page_edge_is_moved_inside_like_the_ink():
    sd = _sheet(True)
    got = data.crop_targets(sd, 2, 3, 64, 1.0, 0, grey=True)
    assert np.array_equal(got["grey"], _page()[:64, :64].astype(np.float32) / 255)


def test_grey_without_the_sheets_grey_is_an_error():
    with pytest.raises(ValueError, match="no grey levels"):
        data.crop_targets(_sheet(False), 120, 160, 64, 1.0, 0, grey=True)


# --- the batch -------------------------------------------------------------------------------------------------------

def _draw(recipe, sd, seed=3):
    pools = train.centre_pools([sd])
    return train.draw([sd], pools, recipe, np.random.default_rng(seed))


def test_the_default_draw_and_tensors_are_the_inks():
    sd = _sheet(True)       # holding grey changes nothing for an ink recipe
    r = train.Recipe(batch=4, crop=64)
    crops = _draw(r, sd)
    assert all("grey" not in c for c in crops)
    x, *rest = train.batch_tensors(crops, r.ink_only, r.input)
    assert torch.equal(x, torch.from_numpy(np.stack([c["ink"] for c in crops]).astype(np.float32))[:, None])
    assert all(torch.equal(a, b) for a, b in zip(rest, train.batch_tensors(crops, r.ink_only)[1:]))


@pytest.mark.parametrize("heads", ["all", "words"])
def test_a_grey_batch_is_the_ink_batch_with_grey_in_place_of_the_input(heads):
    sd = _sheet(True)
    ink, grey = (train.Recipe(batch=4, crop=64, heads=heads, input=i) for i in ("ink", "grey"))
    a, b = _draw(ink, sd), _draw(grey, sd)         # the same crops: the same rng draws
    ta, tb = train.batch_tensors(a, ink.ink_only, ink.input), train.batch_tensors(b, grey.ink_only, grey.input)
    assert not torch.equal(ta[0], tb[0])
    assert torch.equal(tb[0], torch.from_numpy(np.stack([c["grey"] for c in b]))[:, None])
    assert all(torch.equal(p, q) for p, q in zip(ta[1:], tb[1:]))         # text, kind, angle and the loss weight
    if heads == "words":
        assert all(torch.equal(p, q) for p, q in zip(train.word_tensors(a), train.word_tensors(b)))


# --- the network at full size and at half -----------------------------------------------------------------------------

@pytest.mark.parametrize("half", [False, True])
def test_the_network_reads_a_grey_crop_at_full_size_and_at_half(half):
    r = train.Recipe(batch=2, crop=64, width=4, half_size=half, input="grey", heads="words")
    crops = _draw(r, _sheet(True))
    x, *_ = train.batch_tensors(crops, r.ink_only, r.input)
    net = train.build_net(r)
    out = net(x)
    assert x.shape == (2, 1, 64, 64) and out.shape == (2, model.WORD_OUTPUTS, 64, 64) and torch.isfinite(out).all()
    assert net.half_size is half


# --- the comparison runs ---------------------------------------------------------------------------------------------

def test_the_runs_change_one_setting_of_the_kept_recipe():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wordread"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
    import run_input
    kept = train.Recipe(heads="words", half_size=True)
    assert run_input.item_recipe(kept, "grey") == dataclasses.replace(kept, input="grey")
    assert run_input.item_recipe(kept, "full") == dataclasses.replace(kept, half_size=False)
    with pytest.raises(ValueError, match="item is one of"):
        run_input.item_recipe(kept, "rgb")
