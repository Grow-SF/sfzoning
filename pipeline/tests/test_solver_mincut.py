"""The solver's minimum cut, against every partition of small random graphs."""
import itertools

import numpy as np
import pytest

from sfzmaps.solver.mincut import min_cut


def cut_cost(side, src, snk, ea, eb, cab, cba):
    """cost of a cut: the source capacity of nodes on the sink side, the sink capacity of nodes on the source side,
    and each edge's capacity from the source side to the sink side"""
    side = np.asarray(side, bool)
    return (src[side].sum() + snk[~side].sum() + cab[~side[ea] & side[eb]].sum() + cba[~side[eb] & side[ea]].sum())


def brute_force(n, src, snk, ea, eb, cab, cba):
    return min(cut_cost(np.array(s, bool), src, snk, ea, eb, cab, cba)
               for s in itertools.product([False, True], repeat=n))


def random_graph(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 11))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    pick = rng.choice(len(pairs), size=int(rng.integers(1, min(len(pairs), 25)) + 1, ), replace=False)
    ea = np.array([pairs[i][0] for i in pick])
    eb = np.array([pairs[i][1] for i in pick])
    src = rng.exponential(3.0, n) * (rng.random(n) < 0.6)
    snk = rng.exponential(3.0, n) * (rng.random(n) < 0.6)
    cab = rng.exponential(2.0, len(ea))
    cba = rng.exponential(2.0, len(ea)) * (rng.random(len(ea)) < 0.5)
    return n, src, snk, ea, eb, cab, cba


@pytest.mark.parametrize("seed", range(60))
def test_cut_is_a_minimum_within_the_bound(seed):
    n, src, snk, ea, eb, cab, cba = random_graph(seed)
    side, bound = min_cut(n, src, snk, ea, eb, cab, cba)
    got = cut_cost(side, src, snk, ea, eb, cab, cba)
    best = brute_force(n, src, snk, ea, eb, cab, cba)
    assert best - 1e-9 <= got <= best + bound
    assert 0 < bound < 1e-3


def test_edges_are_one_way_unless_given_both():
    # node 0 is pulled to the source side, node 1 to the sink side; the edge costs only from source side to sink side
    src = np.array([5.0, 0.0])
    snk = np.array([0.0, 7.0])
    one = np.array([0]), np.array([1])
    side, _ = min_cut(2, src, snk, *one, np.array([20.0]))
    assert side.tolist() == [True, True]           # cutting 0 -> 1 costs 20: both go to the sink side, node 0 paying its 5
    side, _ = min_cut(2, src, snk, *one[::-1], np.array([20.0]))
    assert side.tolist() == [False, True]          # 1 -> 0 is never cut with 0 source side, 1 sink side: free
    side, _ = min_cut(2, src, snk, *one, np.array([20.0]), np.array([20.0]))
    assert side.tolist() == [True, True]


def test_terminal_only_nodes():
    src = np.array([4.0, 0.0, 0.0])
    snk = np.array([0.0, 3.0, 0.0])
    side, _ = min_cut(3, src, snk, np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    assert not side[0] and side[1]                 # pulled by source capacity: source side; by sink capacity: sink side


def test_deterministic_and_huge_capacities_stay_in_range():
    n, src, snk, ea, eb, cab, cba = random_graph(7)
    big = 1e9
    args = (n, src * big, snk * big, ea, eb, cab * big, cba * big)
    a1 = min_cut(*args)
    a2 = min_cut(*args)
    assert np.array_equal(a1[0], a2[0]) and a1[1] == a2[1]
    assert cut_cost(a1[0], *args[1:]) <= brute_force(*args) + a1[1]


def test_a_node_nothing_pulls_stays_on_the_source_side():
    # node 0 is pulled to the source side, node 2 to the sink side.  Node 1 has only a zero edge, node 3 only an
    # edge 2 -> 3 (paid when 2 is on the source side, which it is not): both cost nothing on either side and stay
    # on the source side, so the expansion move never switches a lot nothing pulls.  Node 4's edge 4 -> 2 is paid
    # when 4 is on the source side: it goes with 2
    side, _ = min_cut(5, np.array([2.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0, 0.0, 0.0]),
                      np.array([0, 2, 4]), np.array([1, 3, 2]), np.array([0.0, 4.0, 1.0]))
    assert side.tolist() == [False, False, True, False, True]
    # a tie between the two sides goes to the source side too: node 1 pays 3 on either side
    side, _ = min_cut(2, np.array([3.0, 0.0]), np.array([0.0, 3.0]), np.array([0]), np.array([1]), np.array([0.0]))
    assert side.tolist() == [False, True]
    side, _ = min_cut(1, np.array([3.0]), np.array([3.0]), np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    assert side.tolist() == [False]


@pytest.mark.parametrize("bad", ["negative", "nan", "inf", "index", "negative_index", "lengths"])
def test_bad_capacities_or_indices_are_refused(bad):
    n, src, snk, ea, eb, cab, cba = random_graph(3)
    src, cab = src.copy(), cab.copy()
    if bad == "negative":
        cab[0] = -1e-3
    elif bad == "nan":
        src[0] = np.nan
    elif bad == "inf":
        cab[0] = np.inf
    elif bad == "index":
        eb = eb.copy()
        eb[0] = n
    elif bad == "negative_index":
        ea = ea.copy()
        ea[0] = -1
    else:
        cab = cab[:-1]
    with pytest.raises(ValueError):
        min_cut(n, src, snk, ea, eb, cab, cba)
