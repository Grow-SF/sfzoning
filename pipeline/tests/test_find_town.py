"""Made-up towns (`sfzmaps.find.town`): their sizes measured on real sheets, their zoning borders where their lots'
codes say, and the design's made-up case: from cut statements alone the network learns which ink is the zoning
border, though block edges cross the same cuts whether the zoning changes or not."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from find_fixtures import Flat
from sfzmaps.find import cuts, losses, model, town

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
    # a town draws both widths at one quantile: nowhere is the zoning border lighter than the block line, and at the
    # median it is heavier (the two lowest quantiles tie, as measured)
    line, heavy = found["block_line_width"], found["zoning_border_width"]
    assert all(z >= b for b, z in zip(line, heavy)) and heavy[50] > line[50]
    assert found["command"].startswith("sfzmaps find measure-town")
    assert len(found["commit"]) == 40


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
    sizes: every street width and block side exactly (no cut between two lots of one block counts as a street); every
    line at least as wide as drawn and most of them exactly (a cut where two lines meet, or under a zoning border
    beside a block line, measures the lines together)."""
    sizes = {"street_width": 18, "block_side": 60, "block_line_width": 3, "zoning_border_width": 7}
    m = {k: [float(v)] * 101 for k, v in sizes.items()} | {"spread_px": 0.9}
    t = town.draw(np.random.default_rng(1), m)
    found = town.sizes(town.labels(t, m["spread_px"]), t.lum)
    for k in ("street_width", "block_side"):
        assert len(found[k]) and (found[k] == sizes[k]).all(), (k, np.unique(found[k]))
    for k in ("block_line_width", "zoning_border_width"):
        values, counts = np.unique(found[k], return_counts=True)
        assert values[0] == sizes[k] and values[np.argmax(counts)] == sizes[k], (k, values, counts)


def test_the_border_answer_learns_which_ink_is_the_border(m):
    """The design's made-up case.  Test settings: CPU, FindNet(8), tiles of 64 px, batches of 8, 400 steps, seed 0;
    16 towns to learn from and 20 fresh ones to read, all drawn in turn from np.random.default_rng(0).  The network
    agrees with a cut when it gives the cut's own statement a chance over one half; judged on the fresh towns' cuts
    pooled."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(0)
        towns = [town.draw(rng, m) for _ in range(16)]
        fresh = [town.draw(rng, m) for _ in range(20)]
        net = town.fit(towns, m["spread_px"], width=8, tile=64, batch=8, steps=400, seed=0)
        chance = [town.chances(net, t.lum)[model.BORDER] for t in fresh]
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)
    agreed, says, on_border, on_block_only = [], [], [], []
    for t, p in zip(fresh, chance):
        c = cuts.border_cuts(t.lot, t.codes, FLAT, t.lum.shape, np.ones(t.lum.shape, bool), m["spread_px"])
        logit = torch.logit(torch.from_numpy(p.ravel()).double(), eps=1e-12)
        got = losses.statement_chance(logit, torch.from_numpy(c.px), torch.from_numpy(c.cut).long(),
                                      torch.from_numpy(c.prior), torch.from_numpy(c.says))
        agreed.append(got.numpy() > 0.5)
        says.append(c.says)
        on_border.append(p[t.border_line])
        on_block_only.append(p[t.block_line & ~t.border_line])
    agreed, says = np.concatenate(agreed), np.concatenate(says)
    crosses, none = agreed[says].mean(), agreed[~says].mean()
    border, block_only = np.concatenate(on_border).mean(), np.concatenate(on_block_only).mean()
    found = (f"'crosses' cuts agreed with {crosses:.3f}, 'none' cuts {none:.3f}; mean zoning-border chance on "
             f"zoning-border ink {border:.3f}, on block-line-only ink {block_only:.3f}")
    assert crosses > 0.5 and none > 0.5 and border > block_only, found
