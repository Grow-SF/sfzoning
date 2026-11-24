"""Training tiles shown at the print sizes of the sheets the finder is used on: a square of the item resized to a tile
(`find.data.resized`), its block-edge cuts laid again at the new size (`find.cuts.relaid`), the target size drawn per
tile (`find.train.Tiles` with `Recipe.resize`), and the sizes measured from the sheets' placements
(`find.data.print_sizes`).  Synthetic items on flat placements."""
import dataclasses
import json
import math
from collections import Counter

import cv2
import numpy as np
import pytest
from scipy import ndimage

from find_fixtures import Flat
from sfzmaps.find import cuts, data, lotlabels, pages, train
from sfzmaps.find.cuts import Cuts

FT = 4.0                    # the made-up sheet's ft per px
SIGMA = 2.0                 # its cuts' sigma, px
SHAPE = (200, 200)
BAND = (124, 130)           # the street's columns (end exclusive): 6 px wide, printed sides 2 px either side
LINE_Y = np.arange(40, 161, 8)      # the rows at which a cut crosses the block edge
NORMAL = np.array([1.0, -0.25]) / math.hypot(1.0, 0.25)     # (x, y) across the block edge
EIGHT = np.ones((3, 3), bool)       # 8-connected


def line_x(y):
    """The block edge: a straight line, on whole columns at the cut rows."""
    return 140 + (np.asarray(y, float) - 100) / 4


def _laid(mid, d, sigma, shape):
    n = len(mid)
    return cuts._lay(np.asarray(mid, float), np.asarray(d, float), np.arange(n), sigma, cuts.reach_px(sigma), shape,
                     np.ones(shape, bool))


def _sheet_item(sigma=SIGMA, mid=None, d=None) -> data.Labelled:
    """A sheet item: a street 6 px wide down BAND, between two printed lines; the top 20 rows without a label (as
    reserved ground); block-edge cuts across the slanted line `line_x` at LINE_Y (or at `mid` along `d`), weights
    0.5 to 1, `inside` 100 on."""
    lum = np.ones(SHAPE, np.float32)
    lum[:, BAND[0] - 2:BAND[0]] = 0
    lum[:, BAND[1]:BAND[1] + 2] = 0
    street = np.zeros(SHAPE, np.uint8)
    street[:, BAND[0]:BAND[1]] = 1
    street[:20] = data.NONE
    street_w = np.where(street == data.NONE, 0, 1).astype(np.float32)
    edge = np.where(street == data.NONE, data.NONE, 0).astype(np.uint8)
    if mid is None:
        mid = np.c_[line_x(LINE_Y), LINE_Y]
        d = np.repeat(NORMAL[None], len(mid), 0)
    laid = _laid(mid, d, sigma, SHAPE)
    n = len(laid.weight)
    c = Cuts(laid.px, laid.cut, laid.prior, np.linspace(0.5, 1.0, n).astype(np.float32),
             (100 + np.arange(n)).astype(np.int32), np.zeros(0, np.int64))
    return data.Labelled("t/A", "lined-1bit", lum, street, street_w, edge, c, ft_per_px=FT, sigma=sigma)


def _crop_item(shape=SHAPE, ft=FT, era="hatched-grey", name="train/s/A") -> data.Labelled:
    rng = np.random.default_rng(3)
    street = rng.choice(np.array([0, 1, 255], np.uint8), size=shape)
    edge = rng.choice(np.array([0, 1, 255], np.uint8), size=shape)
    return data.Labelled(name, era, rng.uniform(0, 1, shape).astype(np.float32), street,
                         (street != 255).astype(np.float32), edge, None, set="train", ft_per_px=ft)


def _rc(c: Cuts, width: int, k: int):
    r, q = np.divmod(c.px[c.cut == k], width)
    return r, q, c.prior[c.cut == k]


def _same(a: data.Labelled, b: data.Labelled) -> None:
    for k in ("lum", "street", "street_w", "edge"):
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), k
    assert (a.cuts is None) == (b.cuts is None)
    if a.cuts is not None:
        for k in ("px", "cut", "prior", "weight", "inside"):
            x, y = getattr(a.cuts, k), getattr(b.cuts, k)
            assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), k


# --- one square resized ------------------------------------------------------------------------------------------

def test_factor_one_is_the_window():
    it = _sheet_item()
    _same(data.resized(it, 84, 120, 64, 64), data.window(it, 84, 120, 64))


@pytest.mark.parametrize("y0, x0, side", [(84, 120, 32), (36, 70, 128)])      # factor 2, factor one half
def test_the_print_and_its_labels_take_the_new_size_together(y0, x0, side):
    size = 64
    f = size / side
    it = _sheet_item()
    t = data.resized(it, y0, x0, side, size)
    for k in ("lum", "street", "street_w", "edge"):
        assert getattr(t, k).shape == (size, size), k
    # the street band is f times as wide, on every row, and sits between the printed lines
    on = t.street == 1
    assert (on.sum(axis=1) == round(6 * f)).all()
    cols = np.flatnonzero(on[0])
    assert np.array_equal(cols, np.arange(cols[0], cols[0] + round(6 * f)))
    assert t.lum[:, cols].mean() > 0.7 and t.lum[:, cols[0] - 1].mean() < 0.5 and t.lum[:, cols[-1] + 1].mean() < 0.5
    assert set(np.unique(t.street)) <= {0, 1} and (t.street_w[on] == 1).all()
    # every cut wholly inside the square is laid again; its centre on the moved line, its px distinct and touching,
    # its prior that of sigma times f (to the 1% its direction, read off its px, is known to); its weight and inside
    # carried over
    src = data.window(it, y0, x0, side).cuts
    got = t.cuts
    assert len(got.weight) == len(src.weight) > 0
    sigma, reach, m = SIGMA * f, cuts.reach_px(SIGMA * f), np.abs(NORMAL).max()
    for k in range(len(got.weight)):
        r, q, prior = _rc(got, size, k)
        assert len(set(zip(r, q))) == len(r) == 2 * reach + 1
        order = np.argsort(q)
        assert (np.diff(q[order]) == 1).all() and (np.abs(np.diff(r[order])) <= 1).all()
        c = int(np.argmax(prior))
        step = np.abs(q - q[c])                                    # steps along the long axis (x)
        off = step > 0
        assert prior[c] == 1 and (prior[off] < 1).all()
        implied = (step[off] / m) / np.sqrt(-2 * np.log(prior[off].astype(float)))
        np.testing.assert_allclose(implied, sigma, rtol=0.01)
        xs, ys = (q[c] + 0.5) / f - 0.5 + x0, (r[c] + 0.5) / f - 0.5 + y0         # back on the item
        assert abs((xs - line_x(ys)) * NORMAL[0] * f) <= 1.0                       # tile px from the moved line
        # the source cut this one is: the one whose centre lies nearest
        best, near = None, math.inf
        for j in range(len(src.weight)):
            sr, sq, sp = _rc(src, side, j)
            i = int(np.argmax(sp))
            dist = math.hypot(sq[i] + x0 - xs, sr[i] + y0 - ys)
            if dist < near:
                best, near = j, dist
        assert near <= 1.0 / f
        assert got.weight[k] == src.weight[best] and got.inside[k] == src.inside[best]
    assert len(set(got.inside)) == len(got.inside)


def test_a_cut_that_would_leave_the_tile_is_dropped():
    sigma = 1.6                     # reach 7 px; at factor one half, sigma 0.8 and reach 4 px
    it = _sheet_item(sigma, mid=[(7, 60), (64, 60)], d=[(1, 0), (1, 0)])
    src = data.window(it, 0, 0, 128).cuts
    assert len(src.weight) == 2                                       # both wholly inside the square
    got = data.resized(it, 0, 0, 128, 64).cuts
    assert len(got.weight) == 1 and got.weight[0] == it.cuts.weight[1]
    r, q, prior = _rc(got, 64, 0)
    c = int(np.argmax(prior))
    assert abs(q[c] - (64.5 / 2 - 0.5)) <= 0.5 and abs(r[c] - (60.5 / 2 - 0.5)) <= 0.5


@pytest.mark.parametrize("side", [128, 100])
def test_a_one_px_block_edge_stays_unbroken_when_shrunk(side):
    it = _crop_item()
    edge = np.zeros(SHAPE, np.uint8)
    cv2.line(edge, (20, 10), (75, 190), 1, 1)                         # one px wide, 8-connected
    edge[60, 90] = data.NONE                                          # one px without a label, off the line
    it = dataclasses.replace(it, edge=edge)
    size = 64
    t = data.resized(it, 0, 0, side, size)
    line = t.edge == 1
    rows = min(190, side - 1) - 10 + 1                                # the line's rows inside the square
    assert ndimage.label(line, EIGHT)[1] == 1 and line.sum() >= 0.9 * rows * size / side
    near = edge[np.ix_(*(((2 * np.arange(size) + 1) * side) // (2 * size),) * 2)]
    assert ndimage.label(near == 1, EIGHT)[1] > 1                    # a lookup under each centre would break it
    assert t.edge[int(60 * size / side), int(90 * size / side)] == data.NONE     # no label stays no label
    assert set(np.unique(t.edge)) == {0, 1, data.NONE}


# --- the tiles ---------------------------------------------------------------------------------------------------

def _old_draw(tiles: train.Tiles, rng, noise_rng) -> list[data.Labelled]:
    """A batch as `Tiles.draw` made it before tiles were resized: centre, window, quarter turn, jitter, noise."""
    out, side, j = [], tiles.recipe.tile, tiles.recipe.jitter
    for era, n in zip(tiles.eras, tiles.shares()):
        for _ in range(n):
            i, r, c = tiles.pools[era].centre(rng)
            it = tiles.items[i]
            h, w = it.lum.shape
            y0, x0 = min(max(r - side // 2, 0), h - side), min(max(c - side // 2, 0), w - side)
            t = data.turned(data.window(it, y0, x0, side, tiles.boxes[i]), int(rng.integers(4)))
            lo, hi = rng.uniform(0.0, j), rng.uniform(1.0 - j, 1.0)
            lum = (lo + (hi - lo) * t.lum).astype(np.float32)
            out.append(dataclasses.replace(t, lum=train.noised(lum, noise_rng, tiles.recipe)))
    return out


RECIPE = train.Recipe(batch=6, tile=64)


def _draws(recipe, sizes, steps=3, seed=7):
    tiles = train.Tiles([_sheet_item(), _crop_item()], recipe, sizes)
    rng, noise_rng = np.random.default_rng(seed), np.random.default_rng([seed, 1])
    return tiles, [tiles.draw(rng, noise_rng) for _ in range(steps)]


def test_without_resize_the_tiles_are_those_of_before():
    recipe = dataclasses.replace(RECIPE, resize=False)
    tiles, got = _draws(recipe, None)
    rng, noise_rng = np.random.default_rng(7), np.random.default_rng([7, 1])
    for batch in got:
        for a, b in zip(batch, _old_draw(tiles, rng, noise_rng), strict=True):
            _same(a, b)
    assert tiles.factors == []


def test_every_factor_one_gives_the_tiles_of_before():
    tiles, got = _draws(RECIPE, {"t": [FT]})                    # every target is the items' own size
    rng, noise_rng = np.random.default_rng(7), np.random.default_rng([7, 1])
    for batch in got:
        for a, b in zip(batch, _old_draw(tiles, rng, noise_rng), strict=True):
            _same(a, b)
    assert tiles.factors == [1.0] * 18


def test_with_resize_the_same_seed_gives_the_same_bytes():
    sizes = {"a": [2.0, 2.4], "b": [FT, 8.0]}
    (ta, a), (tb, b) = _draws(RECIPE, sizes), _draws(RECIPE, sizes)
    for x, y in zip(a, b, strict=True):
        for s, t in zip(x, y, strict=True):
            _same(s, t)
    assert ta.factors == tb.factors and len(set(ta.factors)) > 1


def test_a_square_bigger_than_the_item_takes_the_nearest_factor_that_fits():
    it = _crop_item((100, 100))
    recipe = dataclasses.replace(RECIPE, batch=4)
    tiles = train.Tiles([it], recipe, {"a": [2 * FT]})          # factor one half: a 128 px square, the item 100
    batch = tiles.draw(np.random.default_rng(0), np.random.default_rng(1))
    want = data.resized(it, 0, 0, 100, 64)
    for t in batch:
        assert t.street.shape == (64, 64)
        assert any(np.array_equal(t.street, np.rot90(want.street, k)) for k in range(4))
    assert tiles.factors == [0.64] * 4 and tiles.moved == 4
    assert tiles.factor_summary() == {"count": 4, "least": 0.64, "median": 0.64, "most": 0.64, "moved_to_fit": 4}


def test_the_target_is_drawn_by_series_then_sheet():
    sizes = {"a": [2.0], "b": [3.0, 3.1, 3.2, 3.3, 3.4]}       # one sheet against five
    tiles = train.Tiles([_crop_item()], RECIPE, sizes)
    n = 6000
    got = [tiles.target() for _ in range(n)]
    by_series = Counter(s for s, _ in got)
    # each count lies within four binomial standard deviations of its share (the generator is fixed: this is exact)
    assert abs(by_series["a"] - n / 2) < 4 * math.sqrt(n / 4)
    by_sheet = Counter(v for s, v in got if s == "b")
    assert set(by_sheet) == set(sizes["b"])
    nb = by_series["b"]
    assert all(abs(c - nb / 5) < 4 * math.sqrt(nb * 0.2 * 0.8) for c in by_sheet.values())
    assert all(v == 2.0 for s, v in got if s == "a")


def test_resizing_refuses_an_item_with_no_scale_and_needs_the_sizes():
    with pytest.raises(ValueError, match="measured print sizes"):
        train.Tiles([_crop_item()], RECIPE, None)
    with pytest.raises(ValueError, match="nameless: no ft per px"):
        train.Tiles([_crop_item(ft=math.nan, name="nameless")], RECIPE, {"a": [FT]})
    train.Tiles([_crop_item(ft=math.nan)], dataclasses.replace(RECIPE, resize=False), None)


# --- the measured sizes --------------------------------------------------------------------------------------------

def test_the_sizes_are_each_placed_sheets_and_a_series_with_none_is_named(tmp_path, monkeypatch):
    f = tmp_path / "label_sources.csv"
    f.write_text("series,era,lots\na,hatched-grey,\nb,lined-grey,\nc,lined-grey,\n")
    monkeypatch.setattr(pages, "SOURCES", f)
    found = {"a": ["a1"], "b": ["b1", "b2", "b3"], "c": ["c1"]}
    monkeypatch.setattr(pages, "sheets", lambda series=None: [(series, h) for h in found[series]])
    scale = {("a", "a1"): 2.0, ("b", "b1"): 3.0, ("b", "b2"): 3.5}         # b3 and c1 have no placement

    def placement(sid, which="warp"):
        if sid not in scale:
            raise FileNotFoundError(f"{sid}: no transform.json")
        return Flat(scale[sid])
    monkeypatch.setattr(lotlabels, "placement", placement)
    monkeypatch.setattr(lotlabels, "page_shape", lambda sid: (300, 200))
    monkeypatch.setattr(lotlabels, "placement_check", lambda sid: {"passed": True, "reason": None})
    sizes, left_out, failed = data.print_sizes()
    assert sizes == {"a": [pytest.approx(2.0)], "b": [pytest.approx(3.0), pytest.approx(3.5)]} and left_out == ["c"]
    assert failed == {}
    assert data.size_summary(sizes) == {"a": {"count": 1, "least": 2.0, "median": 2.0, "most": 2.0},
                                        "b": {"count": 2, "least": 3.0, "median": 3.25, "most": 3.5}}
    tiles = train.Tiles([_crop_item()], RECIPE, sizes)
    assert {tiles.target()[0] for _ in range(200)} == {"a", "b"}


def test_a_sheet_whose_placement_fails_the_warps_check_gives_no_size(tmp_path, monkeypatch):
    f = tmp_path / "label_sources.csv"
    f.write_text("series,era,lots\na,hatched-grey,\nb,lined-grey,\nd,lined-grey,\n")
    monkeypatch.setattr(pages, "SOURCES", f)
    found = {"a": ["a1", "a2"], "b": ["b1", "b2"], "d": ["d1"]}
    monkeypatch.setattr(pages, "sheets", lambda series=None: [(series, h) for h in found[series]])
    scale = {("a", "a1"): 2.0, ("a", "a2"): math.nan, ("b", "b1"): 3.0, ("b", "b2"): 12.0, ("d", "d1"): 5.0}
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(scale[sid]))
    monkeypatch.setattr(lotlabels, "page_shape", lambda sid: (300, 200))
    verdict = {("b", "b2"): {"passed": False, "reason": "misplaced"},       # the warp's own check
               ("d", "d1"): {"passed": False, "reason": "too_few_matches"}}
    monkeypatch.setattr(lotlabels, "placement_check", lambda sid: verdict.get(sid, {"passed": True, "reason": None}))
    sizes, left_out, failed = data.print_sizes()
    assert sizes == {"a": [pytest.approx(2.0)], "b": [pytest.approx(3.0)]}
    assert left_out == ["d"] and failed == {"a/a2": "measured size not finite (nan ft per px)", "b/b2": "misplaced",
                                            "d/d1": "too_few_matches"}
    tiles = train.Tiles([_crop_item()], RECIPE, sizes)
    drawn = Counter(tiles.target() for _ in range(400))
    assert set(drawn) == {("a", 2.0), ("b", 3.0)}                         # b only at b1's size; d never


def test_the_run_log_holds_the_sizes_and_the_factors(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import config
    from sfzmaps.cli import app
    f = tmp_path / "label_sources.csv"
    f.write_text("series,era,lots\ns,hatched-grey,\n")
    monkeypatch.setattr(pages, "SOURCES", f)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    page = np.ones((400, 500), np.float32)
    page[110:112] = page[128:130] = 0                           # a street between two lines, rows 112-127
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    box = (100, 100, 60, 40)
    d = config.INPUTS_DIR / "find-labels" / "train" / "s"
    d.mkdir(parents=True)
    (d / "A__100_100_60_40.json").write_text(json.dumps({"box": list(box), "notes": "", "drawn_by": "",
                                                        "street": [[110, 120]], "blocks": [[120, 104], [120, 135]]}))
    monkeypatch.setattr(data, "print_sizes", lambda: ({"s": [2.0, 3.0]}, ["t"], {"s/A": "misplaced"}))
    seen = {}

    def fit(items, recipe, device, on_checkpoint=None, sizes=None):
        seen.update(recipe=recipe, sizes=sizes, ft=[it.ft_per_px for it in items])
        return train.Fitted({}, 1, [], "cpu", 0, {"count": 2, "least": 1.0, "median": 1.5, "most": 2.0,
                                                  "moved_to_fit": 0})
    monkeypatch.setattr(train, "fit", fit)
    r = CliRunner().invoke(app, ["find", "train", "--run", "r", "--device", "cpu", "--no-chooser"])
    assert r.exit_code == 0, r.output
    log = json.loads((tmp_path / "find" / "runs" / "r" / "log.json").read_text())
    assert log["print_sizes"] == {"s": {"count": 2, "least": 2.0, "median": 2.5, "most": 3.0}}
    assert log["print_sizes_left_out"] == ["t"] and log["factors"]["median"] == 1.5
    assert log["print_sizes_sheets_left_out"] == {"count": 1, "sheets": {"s/A": "misplaced"}}
    assert log["training_items_on_sheets_left_out"] == ["train/s/A__100_100_60_40"]
    assert seen["recipe"].resize and seen["sizes"] == {"s": [2.0, 3.0]} and seen["ft"] == [pytest.approx(4.0)]
    r = CliRunner().invoke(app, ["find", "train", "--run", "q", "--device", "cpu", "--no-chooser", "--no-resize"])
    assert r.exit_code == 0, r.output
    assert not seen["recipe"].resize and seen["sizes"] is None
    log = json.loads((tmp_path / "find" / "runs" / "q" / "log.json").read_text())
    assert log["recipe"]["resize"] is False and "print_sizes" not in log
