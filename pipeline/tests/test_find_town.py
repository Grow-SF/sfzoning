"""Made-up towns (`sfzmaps.find.town`): their sizes measured on real sheets, their zoning borders where their lots'
codes say, and the design's made-up case: from cut statements alone the network learns which ink is the zoning
border, though block edges cross the same cuts whether the zoning changes or not."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from find_fixtures import Flat
from sfzmaps.find import cuts, model, town

MEASURES = Path(__file__).parents[1] / "inputs" / "find" / "town-measures.json"
# A grid px a page px, as the town is drawn.  Offset a quarter px so that no cut px falls on a half px, where rounding
# puts two of a cut's px on one and skips the next (with no offset every cut across a street does).
FLAT = Flat(4.0, town.OFFSET, town.OFFSET)


@pytest.fixture(scope="module")
def m():
    return town.measures(MEASURES)


def test_measures_file_is_complete():
    found = json.loads(MEASURES.read_text())
    for key in town.SIZES:
        q = found[key]
        assert len(q) == 101 and all(isinstance(v, (int, float)) for v in q)
        assert all(a <= b for a, b in zip(q, q[1:]))
    assert isinstance(found["spread_px"], float) and found["spread_px"] > 0
    assert found["command"].startswith("sfzmaps find measure-town")
    assert len(found["commit"]) == 40 and int(found["commit"], 16) >= 0


def _hits(t: town.Town, c: cuts.Cuts) -> np.ndarray:
    """Bool per cut: a `border_line` px is on it."""
    return np.bincount(c.cut, t.border_line.ravel()[c.px], minlength=len(c.answer)) > 0


def test_town_borders_follow_the_codes(m):
    rng = np.random.default_rng(0)
    crosses = missed = none = 0
    kinds = set()
    for _ in range(20):
        t = town.draw(rng, m)
        c = cuts.border_cuts(t.lot, t.codes, FLAT, t.lum.shape, np.ones(t.lum.shape, bool), m["spread_px"])
        hit = _hits(t, c)
        assert not hit[~c.says].any()
        crosses += int(c.says.sum())
        missed += int((c.says & ~hit).sum())        # the placement error pushed the line past the cut's reach
        none += int((~c.says).sum())
        kinds |= set(t.kinds)
        assert ((t.lum == 0) == (t.block_line | t.border_line)).all()
    assert crosses > 0 and none > 0
    assert missed / crosses < 1 / 20
    assert kinds == set(town.KINDS)


def test_measuring_a_town_gives_back_its_sizes():
    """A town drawn with one size each, labelled as `lotlabels.sheet_labels` labels a sheet, measures back to those
    sizes (the medians: a cut at a street crossing or under a zoning border beside a block line measures longer)."""
    sizes = {"street_width": 18, "block_side": 60, "block_line_width": 3, "zoning_border_width": 7}
    m = {k: [float(v)] * 101 for k, v in sizes.items()} | {"spread_px": 0.9}
    t = town.draw(np.random.default_rng(1), m)
    found = town.sizes(town.labels(t, m["spread_px"]), t.lum)
    for k, v in sizes.items():
        assert np.median(found[k]) == v, k


@pytest.mark.xfail(strict=True, reason=(
    "the zoning-border answer lights the zoning-border ink and not the block lines (mean chance 0.137 against 0.009 on "
    "the fresh town; no 'none' cut lit) but no px passes one half in 400 steps: a 'crosses' statement over a line "
    "several px wide is met by several px of moderate chance (median brightest px of a 'crosses' cut 0.20, median "
    "chance of its statement 0.79), and the chances rise slowly after (0.29 at 800 steps, 0.42 at 1600, when the "
    "test's three checks hold); an owner decision (task-6 report)"))
def test_the_border_answer_learns_which_ink_is_the_border(m):
    """The design's made-up case.  Test settings: CPU, FindNet(8), tiles of 64 px, batches of 8, 400 steps, seed 0,
    16 towns to learn from and a fresh one to read."""
    torch.use_deterministic_algorithms(True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(0)
        towns = [town.draw(rng, m) for _ in range(16)]
        fresh = town.draw(rng, m)
        t0 = time.time()
        net = town.fit(towns, m["spread_px"], width=8, tile=64, batch=8, steps=400, seed=0)
        chance = town.chances(net, fresh.lum)[model.BORDER]
        took = time.time() - t0
    finally:
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(threads)
    c = cuts.border_cuts(fresh.lot, fresh.codes, FLAT, fresh.lum.shape, np.ones(fresh.lum.shape, bool),
                         m["spread_px"])
    lit = np.bincount(c.cut, chance.ravel()[c.px] > 0.5, minlength=len(c.answer)) > 0
    on_border = chance[fresh.border_line].mean()
    on_block_only = chance[fresh.block_line & ~fresh.border_line].mean()
    print(f"\n'crosses' lit {lit[c.says].mean():.3f}, 'none' lit {lit[~c.says].mean():.3f}; zoning-border chance on "
          f"zoning-border ink {on_border:.3f}, on block-line-only ink {on_block_only:.3f}; {took:.1f} s")
    assert lit[c.says].mean() > 0.5
    assert lit[~c.says].mean() < 0.5
    assert on_border > on_block_only
