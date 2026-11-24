"""Whether a block is drawn on a sheet (`sfzmaps.find.drawn`): a two-part binomial mixture over the sheet's blocks of
how many of each block's edge cuts cross ink, fitted by EM."""
import numpy as np

from sfzmaps.find import drawn


def test_drawn_and_blank_blocks_separate():
    n = np.r_[np.arange(20) + 10, np.arange(10) + 10]
    k = np.r_[n[:20], np.zeros(10, int)]
    p, fit = drawn.drawn_chance(k, n)
    assert (p[:20] > 0.99).all() and (p[20:] < 0.01).all()
    assert fit["groups"] == 2 and fit["q_drawn"] > fit["q_blank"]


def test_the_fit_does_not_depend_on_its_start():
    rng = np.random.default_rng(7)
    n = rng.integers(5, 60, 300)
    is_drawn = np.arange(300) < 200
    k = rng.binomial(n, np.where(is_drawn, 0.9, 0.05))
    fits = [drawn.drawn_chance(k, n, start)[1] for start in ((0.5, 0.75, 0.25), (0.2, 0.6, 0.4), (0.9, 0.99, 0.01))]
    for f in fits[1:]:
        for key in ("pi", "q_drawn", "q_blank"):
            assert abs(f[key] - fits[0][key]) < 1e-6
    p, fit = drawn.drawn_chance(k, n)
    assert abs(fit["q_drawn"] - 0.9) < 0.02 and abs(fit["q_blank"] - 0.05) < 0.02
    assert (p[is_drawn] > 0.5).mean() > 0.99 and (p[~is_drawn] < 0.5).mean() > 0.99


def test_one_kind_of_block_is_all_drawn():
    n = np.arange(30) + 5
    p, fit = drawn.drawn_chance(n.copy(), n)
    assert (p == 1).all() and fit["groups"] == 1 and fit["pi"] == 1


def test_a_block_with_no_cut_on_the_page_counts_for_nothing():
    n = np.r_[np.arange(20) + 10, np.arange(10) + 10, 0, 0]
    k = np.r_[n[:20], np.zeros(12, int)]
    p, fit = drawn.drawn_chance(k, n)
    assert (p[-2:] == 0).all()
    assert fit == drawn.drawn_chance(k[:-2], n[:-2])[1]          # and does not move the fit
    p, fit = drawn.drawn_chance(np.zeros(3, int), np.zeros(3, int))
    assert (p == 0).all() and fit["groups"] == 0
