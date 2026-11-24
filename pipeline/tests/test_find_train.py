"""The finder's training items (`find.data`) and its trainer (`find.train`): labels that survive a quarter turn, the
reserved ground left unlabelled on sheets and training crops, tiles shared fairly between eras, an answer left out,
the noise on the tiles, and the same bits from the same seed."""
import json
import logging
import math
from collections import Counter

import numpy as np
import pytest
import torch

from find_fixtures import Flat
from sfzmaps import config
from sfzmaps.find import data, handlabels, lotlabels, model, pages, train
from sfzmaps.find.cuts import BORDER, EDGE, Cuts

R = model.REACH


def _sources(tmp_path, monkeypatch):
    """Series `s` (hatched, no zoning borders) and `t` (lined, zoning borders from the table)."""
    f = tmp_path / "label_sources.csv"
    f.write_text("series,era,lots,zoning_borders\ns,hatched-grey,,none\nt,lined-1bit,year,zoning\n")
    monkeypatch.setattr(pages, "SOURCES", f)


def _crop_file(set_, series, sheet, box, **marks):
    d = config.INPUTS_DIR / "find-labels" / set_ / series
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sheet}__{'_'.join(str(v) for v in box)}.json"
    f.write_text(json.dumps({"box": list(box), "notes": "", "drawn_by": "",
                             **(marks or {"street": [list(box[:2])]})}))
    return f


def _cuts(shape, rng, n=8) -> Cuts:
    """`n` short cuts at random on an item of `shape`, across or down, of both answers and both statements; every
    "crosses" cut has a px of prior 1; the last cut is not used (weight 0)."""
    h, w = shape
    px, cut, prior = [], [], []
    for i in range(n):
        length = int(rng.integers(3, 8))
        r, c = int(rng.integers(0, h - 8)), int(rng.integers(0, w - 8))
        down = bool(rng.integers(2))
        k = np.arange(length)
        px.append((r + k) * w + c if down else r * w + c + k)
        cut.append(np.full(length, i, np.int32))
        p = rng.uniform(0.05, 1.0, length).astype(np.float32)
        p[length // 2] = 1.0
        prior.append(p)
    weight = np.ones(n, np.float32)
    weight[-1] = 0
    return Cuts(np.concatenate(px).astype(np.int64), np.concatenate(cut), np.concatenate(prior),
                (np.arange(n) % 2).astype(np.uint8), (np.arange(n) // 2 % 2).astype(bool), weight,
                np.full((n, 2), -1, np.int32), np.zeros(0, np.int64))


def _item(shape, rng, era="lined-1bit", name="x", cuts=True, set_="train") -> data.Labelled:
    """An item with labels of every kind at random: about a third of each answer's px unlabelled; with `cuts` a
    sheet's (cut statements, no set), else a hand crop of `set_`."""
    def lab():
        return rng.choice(np.array([0, 1, 255], np.uint8), size=shape)
    street = lab()
    street_w = np.where(street == 255, 0, rng.uniform(0.2, 1.0, shape)).astype(np.float32)
    return data.Labelled(name, era, rng.uniform(0, 1, shape).astype(np.float32), street, street_w, lab(), lab(),
                         _cuts(shape, rng) if cuts else None, set=None if cuts else set_)


def test_a_quarter_turn_keeps_the_loss():
    rng = np.random.default_rng(0)
    t = _item((24, 40), rng)
    z = torch.from_numpy(rng.normal(0, 2, (1, 3, 24, 40)))          # float64: the sums agree to rounding
    recipe = train.Recipe()
    want = float(train.batch_loss(z, [t], recipe))
    assert want > 0
    for k in (1, 2, 3):
        got = float(train.batch_loss(torch.rot90(z, k, (2, 3)), [data.turned(t, k)], recipe))
        assert got == pytest.approx(want, abs=1e-5), k
    # the cuts count: the same turn with the cuts left where they were gives another loss
    moved = data.turned(t, 1)
    stale = data.Labelled(moved.name, moved.era, moved.lum, moved.street, moved.street_w, moved.edge, moved.border,
                          t.cuts)
    assert float(train.batch_loss(torch.rot90(z, 1, (2, 3)), [stale], recipe)) != pytest.approx(want, abs=1e-3)


def _label_file(sid, shape, cuts: Cuts):
    f = lotlabels.sheet_path(sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    street = np.zeros(shape, np.uint8)
    street[:, 400:420] = 1
    np.savez(f, street=street, weight=np.ones(shape, np.float32), cuts_px=cuts.px, cuts_cut=cuts.cut,
             cuts_prior=cuts.prior, cuts_answer=cuts.answer, cuts_says=cuts.says, cuts_weight=cuts.weight,
             edge_none=np.ones(shape, bool), border_none=np.ones(shape, bool), sigma=np.float64(2.0),
             placement=np.array("warp"))


def test_reserved_ground_has_no_label(tmp_path, monkeypatch):
    _sources(tmp_path, monkeypatch)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    shape = (400, 500)
    monkeypatch.setattr(pages, "load_scan", lambda sid: np.ones(shape, np.float32))
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    w = shape[1]
    # three cuts across row 10: wholly outside the reserved ground, half in it, wholly in it
    runs = [np.arange(5, 15), np.arange(70, 90), np.arange(150, 160)]
    c = Cuts(np.concatenate([10 * w + r for r in runs]).astype(np.int64),
             np.repeat(np.arange(3, dtype=np.int32), [len(r) for r in runs]),
             np.ones(sum(len(r) for r in runs), np.float32), np.array([EDGE, BORDER, EDGE], np.uint8),
             np.array([True, False, True]), np.ones(3, np.float32), np.full((3, 2), -1, np.int32),
             np.zeros(0, np.int64))
    _label_file(("t", "A"), shape, c)
    _crop_file("choose", "t", "A", (200, 50, 20, 20))
    got = data.sheet_labelled(("t", "A"))
    held = np.zeros(shape, bool)
    held[0:70 + R, 200 - R:220 + R] = True                       # the crop's box grown by REACH, cut at the page
    assert (got.street[held] == 255).all() and (got.street_w[held] == 0).all()
    assert (got.edge[held] == 255).all() and (got.border[held] == 255).all()
    assert (got.street_w[~held] == 1).all() and (got.edge[~held] == 0).all() and (got.border[~held] == 0).all()
    assert np.array_equal(got.street[~held], np.where(np.arange(w) // 20 == 20, 1, 0)[None].repeat(400, 0)[~held])
    assert not held.ravel()[got.cuts.px].any()
    assert np.array_equal(got.cuts.px, 10 * w + runs[0]) and list(got.cuts.answer) == [EDGE]
    assert (got.era, got.name, got.hand) == ("lined-1bit", "t/A", False)


def _two_line_page():
    page = np.ones((400, 500), np.float32)
    page[110:112] = page[128:130] = 0           # a street between two lines, rows 112-127
    return page


def test_a_training_crop_on_reserved_ground_loses_that_ground(tmp_path, monkeypatch, caplog):
    _sources(tmp_path, monkeypatch)
    page = _two_line_page()
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    box = (100, 100, 60, 40)
    marks = {"street": [[110, 120]], "blocks": [[120, 104], [120, 135]]}
    _crop_file("train", "s", "A", box, **marks)
    _crop_file("test", "s", "A", (250, 100, 10, 40), street=[[255, 120]], blocks=[[255, 104], [255, 135]])
    # reserved: columns 250 - R = 128 on
    [crop] = handlabels.crops(("train",))
    want = handlabels.make_labels(handlabels.crop_scan(page, crop), crop)
    with caplog.at_level(logging.INFO, logger="sfzmaps.find.data"):
        got = data.crop_labelled(crop)
    # the crop grown by REACH where the page allows: rows 0-261, columns 0-281
    assert got.lum.shape == (140 + R, 160 + R) and np.array_equal(got.lum, page[:140 + R, :160 + R])
    inner = (slice(100, 140), slice(100, 160))
    keep, lost = (slice(None), slice(None, 28)), (slice(None), slice(28, None))   # crop columns 100-127, 128-159
    for a in (got.street, got.edge, got.border):
        outside = np.ones(a.shape, bool)
        outside[inner] = False
        assert (a[outside] == 255).all()
        assert (a[inner][lost] == 255).all()
    assert (got.street_w[inner][lost] == 0).all()
    assert np.array_equal(got.street[inner][keep], want.street[keep])
    assert np.array_equal(got.edge[inner][keep], want.edge[keep])
    assert (got.border[inner][keep] == 0).all()                  # series s draws no zoning border
    assert (got.street_w[inner][keep] == (want.street[keep] != 255)).all()
    assert got.left_out == pytest.approx(32 / 60)
    assert got.hand and got.set == "train" and got.name == "train/s/A__100_100_60_40" and got.era == "hatched-grey"
    [line] = [r.getMessage() for r in caplog.records if r.name == "sfzmaps.find.data"]
    assert "train/s/A__100_100_60_40" in line and "0.533" in line
    [test_crop] = handlabels.crops(("test",))
    held_out = data.crop_labelled(test_crop)                     # the reserved ground itself is never masked
    assert held_out.left_out == 0 and held_out.set == "test"
    assert np.array_equal(held_out.street[100:140, R:R + 10],
                          handlabels.make_labels(handlabels.crop_scan(page, test_crop), test_crop).street)


def test_each_era_gets_its_share_of_a_batch():
    rng = np.random.default_rng(1)
    items = [_item((64, 64), rng, era, era, cuts=False) for era in ("b", "a", "c", "b")]
    corner = items[1]                                             # era a: labelled only in a 4 x 4 corner patch
    for arr in (corner.street, corner.edge, corner.border):
        arr[:] = 255
    corner.street_w[:] = 0
    corner.street[:4, :4], corner.street_w[:4, :4] = 1, 1
    recipe = train.Recipe(batch=16, tile=32)
    tiles = train.Tiles(items, recipe).draw(np.random.default_rng(2), np.random.default_rng(3))
    assert Counter(t.era for t in tiles) == {"a": 6, "b": 5, "c": 5}
    for t in tiles:
        if t.era == "a":
            assert (t.street_w > 0).sum() == 16                  # every tile of era a holds its labelled px


def test_an_item_smaller_than_a_tile_is_named():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="small one"):
        train.Tiles([_item((64, 64), rng), _item((64, 20), rng, name="small one")], train.Recipe(tile=32))


def _fit_items():
    rng = np.random.default_rng(5)
    return [_item((48, 48), rng, "lined-1bit", "sheet"), _item((40, 40), rng, "hatched-grey", "crop", cuts=False)]


SMALL = dict(batch=2, tile=32, width=4)


def test_an_answer_left_out_gets_no_gradient():
    recipe = train.Recipe(steps=3, answers=("street", "edge"), **SMALL)
    torch.manual_seed(recipe.seed)                                # the network fit makes, before training
    before = model.FindNet(recipe.width).state_dict()
    after = train.fit(_fit_items(), recipe, "cpu").state_dict
    for k in ("head.weight", "head.bias"):
        assert torch.equal(after[k][model.BORDER], before[k][model.BORDER]), k
        assert not torch.equal(after[k][model.STREET], before[k][model.STREET]), k


def test_same_seed_same_bits_cpu():
    recipe = train.Recipe(steps=5, **SMALL)
    a, b = (train.fit(_fit_items(), recipe, "cpu") for _ in range(2))
    assert a.state_dict.keys() == b.state_dict.keys()
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in a.state_dict)
    assert a.curve == b.curve and a.step == 5


def test_the_kept_step_is_the_best_by_the_callback():
    values = iter([0.2, 0.9, 0.9, 0.1])
    seen = {}

    def score(step, state):
        seen[step] = {k: v.clone() for k, v in state.items()}
        return next(values)
    fitted = train.fit(_fit_items(), train.Recipe(steps=4, every=1, **SMALL), "cpu", on_checkpoint=score)
    assert sorted(seen) == [1, 2, 3, 4]
    assert fitted.step == 2
    assert all(torch.equal(fitted.state_dict[k], seen[2][k]) for k in seen[2])
    assert not all(torch.equal(seen[2][k], seen[3][k]) for k in seen[2])


def _drawn(recipe, seed=7):
    items = _fit_items()
    return train.Tiles(items, recipe).draw(np.random.default_rng(seed), np.random.default_rng([seed, 1]))


def test_noise_changes_the_picture_and_never_the_labels():
    noisy, clean = _drawn(train.Recipe(noise=1.0, **SMALL)), _drawn(train.Recipe(noise=0.0, **SMALL))
    for a, b in zip(noisy, clean):
        assert not np.array_equal(a.lum, b.lum)
        assert a.lum.dtype == np.float32 and a.lum.min() >= 0 and a.lum.max() <= 1
        for k in ("street", "street_w", "edge", "border"):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k
        assert (a.cuts is None) == (b.cuts is None)
        if a.cuts is not None:
            for k in ("px", "cut", "prior", "answer", "says", "weight"):
                assert np.array_equal(getattr(a.cuts, k), getattr(b.cuts, k)), k
    lum = clean[0].lum
    off = train.noised(lum, np.random.default_rng(0), train.Recipe(noise=0.0))
    assert off.dtype == lum.dtype and np.array_equal(off, lum)


def test_noise_is_drawn_per_tile(monkeypatch):
    rng = np.random.default_rng(0)
    flat = _item((64, 64), rng, "lined-1bit", "flat", cuts=False)
    flat.lum[:] = 0.5                                             # every tile the same picture
    recipe = train.Recipe(jitter=0.0, **SMALL)                    # and no jitter: both tiles reach the noise equal
    drawn = []
    real = train.noise_amounts

    def spy(rng, recipe):
        drawn.append(real(rng, recipe))
        return drawn[-1]
    monkeypatch.setattr(train, "noise_amounts", spy)

    def batch():
        return train.Tiles([flat], recipe).draw(np.random.default_rng(7), np.random.default_rng([7, 1]))
    first = batch()
    assert len(drawn) == 2 and drawn[0] != drawn[1]               # each tile draws its own amounts
    # their printing alone moves the two tiles' level apart, far past what grain and specks average to over a tile
    level = [0.5 - a[1] * a[2] for a in drawn]
    assert abs(first[0].lum.mean() - first[1].lum.mean()) == pytest.approx(abs(level[0] - level[1]), abs=0.02)
    assert abs(level[0] - level[1]) > 0.05
    again = batch()
    assert all(np.array_equal(a.lum, b.lum) for a, b in zip(first, again))


def test_a_held_out_crop_is_never_trained_on():
    rng = np.random.default_rng(0)
    for set_ in handlabels.HELD_OUT:
        items = [_item((48, 48), rng, cuts=False, name="ok"), _item((48, 48), rng, cuts=False, name="held",
                                                                     set_=set_)]
        with pytest.raises(ValueError, match=f"held: a {set_} crop is never trained on"):
            train.Tiles(items, train.Recipe(**SMALL))
        with pytest.raises(ValueError, match="held"):
            train.fit(items, train.Recipe(steps=1, **SMALL), "cpu")


def test_the_unbroken_terms_reach_only_the_hand_tiles_block_edge():
    rng = np.random.default_rng(3)
    sheet = _item((32, 32), rng)
    crop = _item((32, 32), rng, cuts=False)
    crop.edge[:] = 0
    crop.edge[10, 4:28] = 1                                       # a hand block edge across the crop
    crop.edge[:, :2] = 255
    z = torch.from_numpy(rng.normal(0, 1, (2, 3, 32, 32))).requires_grad_(True)
    loss = train.unbroken_loss(z, [sheet, crop], train.Recipe(unbroken=True, tile=32))
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    loss.backward()
    g = z.grad
    assert g[1, model.EDGE].abs().sum() > 0
    assert (g[0] == 0).all() and (g[1, model.STREET] == 0).all() and (g[1, model.BORDER] == 0).all()
    band = train.corridor(crop.edge)                              # the precision's target is wider than the line
    assert band[9:12, 4:28].all() and band.sum() > (crop.edge == 1).sum()


def test_log_json_holds_no_nan(tmp_path):
    fitted = train.Fitted({"w": torch.zeros(1)}, 3, [{"step": 3, "loss_per_statement": math.nan,
                                                     "chosen_by": math.inf}], "cpu")
    _, js = train.save_run(tmp_path, fitted, train.Recipe(), {"x": [-math.inf, 1.5]})
    text = js.read_text()
    assert "NaN" not in text and "Infinity" not in text
    d = json.loads(text)
    assert d["curve"][0]["loss_per_statement"] is None and d["curve"][0]["chosen_by"] is None
    assert d["x"] == [None, 1.5]


def test_a_used_cut_with_no_px_is_left_out_of_tiles():
    rng = np.random.default_rng(4)
    t = _item((40, 40), rng)
    c = t.cuts
    n = len(c.answer)                                             # one more used cut, with no px
    bare = Cuts(c.px, c.cut, c.prior, np.r_[c.answer, EDGE].astype(np.uint8), np.r_[c.says, True],
                np.r_[c.weight, 1].astype(np.float32), np.r_[c.ends, [[-1, -1]]].astype(np.int32), c.dropped)
    t2 = data.Labelled(t.name, t.era, t.lum, t.street, t.street_w, t.edge, t.border, bare)
    assert data.cuts_without_px(bare) == 1
    got, want = data.window(t2, 0, 0, 40), data.window(t, 0, 0, 40)
    assert len(got.cuts.answer) == len(want.cuts.answer) == n - 1           # the unused cut and the bare one gone
    assert np.array_equal(got.cuts.px, want.cuts.px)


def test_train_refuses_before_training_when_a_crop_cannot_be_read(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps.cli import app
    _sources(tmp_path, monkeypatch)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(pages, "load_scan", lambda sid: _two_line_page())
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))

    def no_fit(*a, **k):
        raise AssertionError("trained")
    monkeypatch.setattr(train, "fit", no_fit)
    _crop_file("train", "s", "A", (100, 100, 60, 40), street=[[110, 120]], blocks=[[120, 104], [120, 135]])
    _crop_file("train", "s", "B", (100, 100, 60, 40), street=[[110, 110]])         # on the line's ink
    _crop_file("train", "s", "C", (480, 100, 60, 40), street=[[490, 120]])         # off the page
    r = CliRunner().invoke(app, ["find", "train", "--run", "r", "--device", "cpu"])
    assert r.exit_code == 1, r.output
    assert "B__100_100_60_40.json" in r.output and "not on paper" in r.output
    assert "C__480_100_60_40.json" in r.output and "2 training crop(s) could not be read" in r.output
    assert not (tmp_path / "find" / "runs").exists()
    for bad in ("../r", "a/b", "..", "."):
        r = CliRunner().invoke(app, ["find", "train", "--run", bad, "--device", "cpu"])
        assert r.exit_code == 2 and "--run" in r.output, (bad, r.output)
