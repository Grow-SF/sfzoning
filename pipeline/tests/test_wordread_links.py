"""learned-links (scripts/wordread/links.py): every 8-neighbour pair of core px once, the pairs' classes by Terra's
word ids, the pair logit the same whichever px is first, and the words as the components of the kept links."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wordread"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import links  # noqa: E402


def test_every_8_neighbour_pair_of_on_px_comes_once():
    rng = np.random.default_rng(3)
    on = rng.random((9, 11)) < 0.5
    ya, xa, yb, xb, d = links.pairs_of(on)
    got = {frozenset({(a, b), (c, e)}) for a, b, c, e in zip(ya, xa, yb, xb)}
    want = set()
    for y, x in zip(*np.nonzero(on)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v, u = y + dy, x + dx
                if (dy or dx) and 0 <= v < 9 and 0 <= u < 11 and on[v, u]:
                    want.add(frozenset({(y, x), (v, u)}))
    assert got == want and len(ya) == len(want)
    steps = np.array(links.STEPS)[d]
    assert np.array_equal(yb - ya, steps[:, 0]) and np.array_equal(xb - xa, steps[:, 1])


def test_a_pair_is_same_in_one_word_different_across_a_boundary_and_untaught_off_the_words():
    wid = np.array([[1, 1, 2, 0, 0]])
    ya, xa, yb, xb = np.zeros(4, int), np.array([0, 1, 2, 3]), np.zeros(4, int), np.array([1, 2, 3, 4])
    assert links.labels_of(wid, ya, xa, yb, xb).tolist() == [1, 0, 0, -1]


def test_the_pair_logit_is_the_same_whichever_px_comes_first():
    torch.manual_seed(0)
    net = links._net(torch)
    fa, fb = torch.randn(50, 16), torch.randn(50, 16)
    d = torch.randint(0, 4, (50,))
    ab = links.pair_logit(net, fa, fb, d, torch)
    ba = links.pair_logit(net, fb, fa, d, torch)          # the same pairs, b first: the direction from b to a is
    one = lambda x, y, k: net(torch.cat([x, y, x - y, x * y, torch.eye(8)[k]], 1))[:, 0]  # noqa: E731
    assert torch.allclose(ab, (one(fa, fb, d) + one(fb, fa, d + 4)) / 2)
    assert torch.allclose(ab, (one(fb, fa, d + 4) + one(fa, fb, d)) / 2) and not torch.allclose(ab, ba)


def test_held_out_cells_are_whole_and_fixed():
    ys, xs = np.array([0, 10, 511, 512, 2000]), np.array([0, 20, 511, 0, 3000])
    a, b = links.held_cell(4, ys, xs), links.held_cell(4, ys, xs)
    assert np.array_equal(a, b) and a[0] == a[1] == a[2]
    share = links.held_cell(1, *np.divmod(np.arange(0, 512 * 400, 512), 20000)).mean()
    assert 0.1 < share < 0.3
