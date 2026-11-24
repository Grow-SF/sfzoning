"""The finder's training items (`find.data`) and its trainer (`find.train`): labels that survive a quarter turn, the
reserved ground left unlabelled on sheets and training crops, a sheet's block-edge cuts the only cuts trained on,
tiles shared fairly between eras, both answers trained, the noise on the tiles, and the same bits from the same
seed."""
import dataclasses
import json
import logging
import math
from collections import Counter

import numpy as np
import pytest
import torch

from find_fixtures import Flat, git_answers, invoke
from sfzmaps import config
from sfzmaps.find import data, handlabels, lotlabels, model, pages, train
from sfzmaps.find.cuts import Cuts

R = model.REACH


def _sources(tmp_path, monkeypatch):
    """Series `s` (hatched, no lots) and `t` (lined, labelled from the April 2000 table's lots)."""
    f = tmp_path / "label_sources.csv"
    f.write_text("series,era,lots\ns,hatched-grey,\nt,lined-1bit,table\n")
    monkeypatch.setattr(pages, "SOURCES", f)


def _crop_file(set_, series, sheet, box, **marks):
    d = config.INPUTS_DIR / "find-labels" / set_ / series
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sheet}__{'_'.join(str(v) for v in box)}.json"
    f.write_text(json.dumps({"box": list(box), "notes": "", "drawn_by": "",
                             **(marks or {"street": [list(box[:2])]})}))
    return f


def _cuts(shape, rng, n=8) -> Cuts:
    """`n` short block-edge cuts at random on an item of `shape`, across or down, each with a px of prior 1; the last
    cut is not used (weight 0)."""
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
    return Cuts(np.concatenate(px).astype(np.int64), np.concatenate(cut), np.concatenate(prior), weight,
                np.full(n, -1, np.int32), np.zeros(0, np.int64))


def _item(shape, rng, era="lined-1bit", name="x", cuts=True, set_="train") -> data.Labelled:
    """An item with labels of both answers at random: about a third of each answer's px unlabelled; with `cuts` a
    sheet's (cut statements, no set), else a hand crop of `set_`."""
    def lab():
        return rng.choice(np.array([0, 1, 255], np.uint8), size=shape)
    street = lab()
    street_w = np.where(street == 255, 0, rng.uniform(0.2, 1.0, shape)).astype(np.float32)
    return data.Labelled(name, era, rng.uniform(0, 1, shape).astype(np.float32), street, street_w, lab(),
                         _cuts(shape, rng) if cuts else None, set=None if cuts else set_)


def test_a_quarter_turn_keeps_the_loss():
    rng = np.random.default_rng(0)
    t = _item((24, 40), rng)
    z = torch.from_numpy(rng.normal(0, 2, (1, 2, 24, 40)))          # float64: the sums agree to rounding
    want = float(train.batch_loss(z, [t]))
    assert want > 0
    for k in (1, 2, 3):
        got = float(train.batch_loss(torch.rot90(z, k, (2, 3)), [data.turned(t, k)]))
        assert got == pytest.approx(want, abs=1e-5), k
    # the cuts count: the same turn with the cuts left where they were gives another loss
    moved = data.turned(t, 1)
    stale = data.Labelled(moved.name, moved.era, moved.lum, moved.street, moved.street_w, moved.edge, t.cuts)
    assert float(train.batch_loss(torch.rot90(z, 1, (2, 3)), [stale])) != pytest.approx(want, abs=1e-3)


def _label_file(sid, shape, cuts: Cuts, before: bool):
    """A label file of the block-edge cuts `cuts`; `before`: as made before the finder dropped the zoning border, with
    a zoning-border cut among them (cuts_answer 1) and a `border_none` label, neither of which is read."""
    f = lotlabels.sheet_path(sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    street = np.zeros(shape, np.uint8)
    street[:, 400:420] = 1
    arrays = dict(street=street, weight=np.ones(shape, np.float32), cuts_px=cuts.px, cuts_cut=cuts.cut,
                  cuts_prior=cuts.prior, cuts_weight=cuts.weight, edge_none=np.ones(shape, bool),
                  sigma=np.float64(2.0), placement=np.array("warp"))
    if before:
        n = len(cuts.weight)
        zb = 10 * shape[1] + np.arange(30, 40)                   # "a zoning border crosses", off the reserved ground
        arrays.update(cuts_px=np.r_[cuts.px, zb], cuts_cut=np.r_[cuts.cut, np.full(len(zb), n)].astype(np.int32),
                      cuts_prior=np.r_[cuts.prior, np.ones(len(zb), np.float32)],
                      cuts_weight=np.r_[cuts.weight, 1].astype(np.float32),
                      cuts_answer=np.r_[np.zeros(n), 1].astype(np.uint8), cuts_says=np.ones(n + 1, bool),
                      border_none=np.ones(shape, bool))
    np.savez(f, **arrays)


@pytest.mark.parametrize("before", [False, True])
def test_reserved_ground_has_no_label_and_only_block_edge_cuts_are_read(tmp_path, monkeypatch, before):
    _sources(tmp_path, monkeypatch)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    shape = (400, 500)
    monkeypatch.setattr(pages, "load_scan", lambda sid: np.ones(shape, np.float32))
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    w = shape[1]
    # three block-edge cuts across row 10: wholly outside the reserved ground, half in it, wholly in it
    runs = [np.arange(5, 15), np.arange(70, 90), np.arange(150, 160)]
    c = Cuts(np.concatenate([10 * w + r for r in runs]).astype(np.int64),
             np.repeat(np.arange(3, dtype=np.int32), [len(r) for r in runs]),
             np.ones(sum(len(r) for r in runs), np.float32), np.ones(3, np.float32), np.full(3, -1, np.int32),
             np.zeros(0, np.int64))
    _label_file(("t", "A"), shape, c, before)
    _crop_file("choose", "t", "A", (200, 50, 20, 20))
    got = data.sheet_labelled(("t", "A"))
    held = np.zeros(shape, bool)
    held[0:70 + R, 200 - R:220 + R] = True                       # the crop's box grown by REACH, cut at the page
    assert (got.street[held] == 255).all() and (got.street_w[held] == 0).all() and (got.edge[held] == 255).all()
    assert (got.street_w[~held] == 1).all() and (got.edge[~held] == 0).all()
    assert np.array_equal(got.street[~held], np.where(np.arange(w) // 20 == 20, 1, 0)[None].repeat(400, 0)[~held])
    assert not held.ravel()[got.cuts.px].any()
    assert np.array_equal(got.cuts.px, 10 * w + runs[0]) and list(got.cuts.cut) == [0] * 10
    assert len(got.cuts.weight) == 1                             # the zoning-border cut of before is not read
    assert (got.era, got.name, got.hand) == ("lined-1bit", "t/A", False)
    assert got.ft_per_px == pytest.approx(4.0) and got.sigma == 2.0        # the page's scale; the label file's sigma


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
    for a in (got.street, got.edge):
        outside = np.ones(a.shape, bool)
        outside[inner] = False
        assert (a[outside] == 255).all()
        assert (a[inner][lost] == 255).all()
    assert (got.street_w[inner][lost] == 0).all()
    assert np.array_equal(got.street[inner][keep], want.street[keep])
    assert np.array_equal(got.edge[inner][keep], want.edge[keep])
    assert (got.street_w[inner][keep] == (want.street[keep] != 255)).all()
    assert got.left_out == pytest.approx(32 / 60)
    assert got.hand and got.set == "train" and got.name == "train/s/A__100_100_60_40" and got.era == "hatched-grey"
    assert got.ft_per_px == pytest.approx(4.0) and math.isnan(got.sigma)    # the scale at its box; no cut, no sigma
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
    for arr in (corner.street, corner.edge):
        arr[:] = 255
    corner.street_w[:] = 0
    corner.street[:4, :4], corner.street_w[:4, :4] = 1, 1
    recipe = train.Recipe(batch=16, tile=32, resize=False)
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


SMALL = dict(batch=2, tile=32, width=4, resize=False)     # tiles at the items' own size (test_find_resize: resized)


def test_same_seed_same_bits_cpu():
    recipe = train.Recipe(steps=5, **SMALL)
    a, b = (train.fit(_fit_items(), recipe, "cpu") for _ in range(2))
    assert a.state_dict.keys() == b.state_dict.keys()
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in a.state_dict)
    assert a.curve == b.curve and a.step == 5


def test_the_last_step_is_kept_whatever_the_choosing_score():
    values = iter([0.9, 0.6, 0.3, 0.1])                          # the choosing score falls: step 1 scores highest
    seen = {}

    def score(step, state):
        seen[step] = {k: v.clone() for k, v in state.items()}
        return {"choose_found_share": next(values), "choose_panoptic_quality": step / 10}
    fitted = train.fit(_fit_items(), train.Recipe(steps=4, every=1, **SMALL), "cpu", on_checkpoint=score)
    assert sorted(seen) == [1, 2, 3, 4]
    assert fitted.step == 4
    assert all(torch.equal(fitted.state_dict[k], seen[4][k]) for k in seen[4])
    assert not all(torch.equal(seen[1][k], seen[4][k]) for k in seen[1])
    assert [r["choose_found_share"] for r in fitted.curve] == [0.9, 0.6, 0.3, 0.1]   # every checkpoint's, traced
    assert [r["choose_panoptic_quality"] for r in fitted.curve] == [0.1, 0.2, 0.3, 0.4]
    assert all("chosen_by" not in r for r in fitted.curve)


def test_the_seed_moves_the_batches_and_the_start_and_only_the_seed():
    items = _fit_items()
    draws = {}
    for seed in (0, 1):
        recipe = train.Recipe(seed=seed, **SMALL)
        draws[seed] = train.Tiles(items, recipe).draw(*train.generators(recipe))
        torch.manual_seed(recipe.seed)                           # as fit makes the network
        draws[seed, "net"] = model.FindNet(recipe.width).state_dict()
    assert any(not np.array_equal(a.lum, b.lum) for a, b in zip(draws[0], draws[1]))
    assert not all(torch.equal(draws[0, "net"][k], draws[1, "net"][k]) for k in draws[0, "net"])
    assert train.Recipe().seed == 0


def test_train_passes_the_seed_and_keeps_the_default(tmp_path, monkeypatch):
    git_answers(monkeypatch)
    _sources(tmp_path, monkeypatch)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(pages, "load_scan", lambda sid: _two_line_page())
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    monkeypatch.setattr(data, "print_sizes", lambda: ({"s": [4.0]}, [], {}))
    _crop_file("train", "s", "A", (100, 100, 60, 40), street=[[110, 120]], blocks=[[120, 104], [120, 135]])
    seen = []

    def fit(items, recipe, device, on_checkpoint=None, sizes=None):
        seen.append(recipe)
        return train.Fitted({}, recipe.steps, [], "cpu")
    monkeypatch.setattr(train, "fit", fit)
    for args, run in ((["--seed", "3"], "a"), ([], "b")):
        r = invoke(["find", "train", "--run", run, "--device", "cpu", "--no-trace", *args])
        assert r.exit_code == 0, r.output
    assert seen[0].seed == 3 and json.loads((tmp_path / "find" / "runs" / "a" / "log.json").read_text())[
        "recipe"]["seed"] == 3
    assert seen[1] == train.Recipe()                             # no --seed: the recipe of before
    r = invoke(["find", "train", "--run", "c", "--device", "cpu"])     # no choosing crop, traced
    assert r.exit_code == 1 and "no choose crop" in r.output and "--no-trace" in r.output, r.output


def _drawn(recipe, seed=7):
    items = _fit_items()
    return train.Tiles(items, recipe).draw(np.random.default_rng(seed), np.random.default_rng([seed, 1]))


def test_noise_changes_the_picture_and_never_the_labels():
    noisy, clean = _drawn(train.Recipe(noise=1.0, **SMALL)), _drawn(train.Recipe(noise=0.0, **SMALL))
    for a, b in zip(noisy, clean):
        assert not np.array_equal(a.lum, b.lum)
        assert a.lum.dtype == np.float32 and a.lum.min() >= 0 and a.lum.max() <= 1
        for k in ("street", "street_w", "edge"):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k
        assert (a.cuts is None) == (b.cuts is None)
        if a.cuts is not None:
            for k in ("px", "cut", "prior", "weight"):
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
    z = torch.from_numpy(rng.normal(0, 1, (2, 2, 32, 32))).requires_grad_(True)
    loss = train.unbroken_loss(z, [sheet, crop], train.Recipe(unbroken=True, tile=32))
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    loss.backward()
    g = z.grad
    assert g[1, model.EDGE].abs().sum() > 0
    assert (g[0] == 0).all() and (g[1, model.STREET] == 0).all()
    band = train.corridor(crop.edge)                              # the precision's target is wider than the line
    assert band[9:12, 4:28].all() and band.sum() > (crop.edge == 1).sum()


def test_log_json_holds_no_nan(tmp_path):
    fitted = train.Fitted({"w": torch.zeros(1)}, 3, [{"step": 3, "loss_per_statement": math.nan,
                                                     "choose_found_share": math.inf}], "cpu")
    _, js = train.save_run(tmp_path, fitted, train.Recipe(), {"x": [-math.inf, 1.5]})
    text = js.read_text()
    assert "NaN" not in text and "Infinity" not in text
    d = json.loads(text)
    assert d["curve"][0]["loss_per_statement"] is None and d["curve"][0]["choose_found_share"] is None
    assert d["x"] == [None, 1.5]


def test_a_used_cut_with_no_px_is_left_out_of_tiles():
    rng = np.random.default_rng(4)
    t = _item((40, 40), rng)
    c = t.cuts
    n = len(c.weight)                                             # one more used cut, with no px
    bare = Cuts(c.px, c.cut, c.prior, np.r_[c.weight, 1].astype(np.float32), np.r_[c.inside, -1].astype(np.int32),
                c.dropped)
    t2 = data.Labelled(t.name, t.era, t.lum, t.street, t.street_w, t.edge, bare)
    assert data.cuts_without_px(bare) == 1
    got, want = data.window(t2, 0, 0, 40), data.window(t, 0, 0, 40)
    assert len(got.cuts.weight) == len(want.cuts.weight) == n - 1           # the unused cut and the bare one gone
    assert np.array_equal(got.cuts.px, want.cuts.px)


def test_train_refuses_before_training_when_a_crop_cannot_be_read(tmp_path, monkeypatch):
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
    r = invoke(["find", "train", "--run", "r", "--device", "cpu"])
    assert r.exit_code == 1, r.output
    assert "B__100_100_60_40.json" in r.output and "not on paper" in r.output
    assert "C__480_100_60_40.json" in r.output and "2 training crop(s) could not be read" in r.output
    assert not (tmp_path / "find" / "runs").exists()
    for bad in ("../r", "a/b", "..", "."):
        r = invoke(["find", "train", "--run", bad, "--device", "cpu"])
        assert r.exit_code == 2 and "--run" in r.output, (bad, r.output)


def test_a_block_edge_cut_moves_only_the_block_edge_answer():
    shape = (16, 16)
    none = np.full(shape, data.NONE, np.uint8)
    c = Cuts(np.arange(40, 46, dtype=np.int64), np.zeros(6, np.int32), np.ones(6, np.float32), np.ones(1, np.float32),
             np.full(1, -1, np.int32), np.zeros(0, np.int64))
    t = data.Labelled("x", "lined-1bit", np.ones(shape, np.float32), none, np.zeros(shape, np.float32), none, c)
    z = torch.zeros((1, 2, *shape), dtype=torch.float64, requires_grad=True)
    train.batch_loss(z, [t]).backward()
    assert z.grad[0, model.EDGE].abs().sum() > 0 and (z.grad[0, model.STREET] == 0).all()
    assert train.statements([t]) == 1


def test_train_reports_a_series_with_no_source_row_and_a_refused_fit_without_a_traceback(tmp_path, monkeypatch):
    git_answers(monkeypatch)
    _sources(tmp_path, monkeypatch)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(pages, "load_scan", lambda sid: _two_line_page())
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    marks = dict(street=[[110, 120]], blocks=[[120, 104], [120, 135]])
    f = _crop_file("train", "u", "A", (100, 100, 60, 40), **marks)        # series u has no row
    r = invoke(["find", "train", "--run", "r", "--device", "cpu", "--no-trace"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit), r.output
    assert "series 'u' has no row in" in r.output and "1 training crop(s) could not be read" in r.output, r.output
    f.unlink()
    _crop_file("train", "s", "A", (100, 100, 60, 40), **marks)
    monkeypatch.setattr(data, "print_sizes", lambda: ({"s": [4.0]}, [], {}))   # no real sheet measured

    def refused(*a, **k):                   # as `train.fit` refuses, for example, an item smaller than a tile
        raise ValueError("s/A: 40 x 60 px is smaller than a 256 px tile")
    monkeypatch.setattr(train, "fit", refused)
    r = invoke(["find", "train", "--run", "r", "--device", "cpu", "--no-trace"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit), r.output
    assert "nothing trained: s/A: 40 x 60 px is smaller than a 256 px tile" in r.output, r.output
    assert not (tmp_path / "find" / "runs").exists()


@pytest.mark.parametrize("fail", ["exit", "missing", None])
def test_train_goes_on_without_the_git_state_when_git_cannot_tell(tmp_path, monkeypatch, fail):
    """git exiting 128 (no repository) or not installed: the run is trained and saved all the same, the commit and
    uncommitted_changes_in_src are null in log.json, and one line says why; with git working, the log is as before."""
    git_answers(monkeypatch, head="abc123", status=" M src/x.py", fail=fail)
    _sources(tmp_path, monkeypatch)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(pages, "load_scan", lambda sid: _two_line_page())
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    monkeypatch.setattr(data, "print_sizes", lambda: ({"s": [4.0]}, [], {}))
    monkeypatch.setattr(train, "fit", lambda items, recipe, device, on_checkpoint=None, sizes=None:
                        train.Fitted({}, recipe.steps, [], "cpu"))
    _crop_file("train", "s", "A", (100, 100, 60, 40), street=[[110, 120]], blocks=[[120, 104], [120, 135]])
    r = invoke(["find", "train", "--run", "r", "--device", "cpu", "--no-trace"])
    assert r.exit_code == 0, r.output
    run = tmp_path / "find" / "runs" / "r"
    log = json.loads((run / "log.json").read_text())
    saved = torch.load(run / "model.pt")
    assert saved["step"] == train.Recipe().steps and "commit" not in saved          # model.pt holds no git state
    said = [line for line in r.output.splitlines() if "git state was not recorded" in line]
    if fail is None:
        assert (log["commit"], log["uncommitted_changes_in_src"]) == ("abc123", True) and not said
    else:
        assert log["commit"] is None and log["uncommitted_changes_in_src"] is None
        why = "git is not installed" if fail == "missing" else "exited 128 (fatal: not a git repository"
        assert len(said) == 1 and why in said[0], r.output
