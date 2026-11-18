"""The search's compound fusion move (search.compound_fusion_moves) and its roof duality (qpbo.py): against brute
force on small random binary energies, never raising the energy, and the two synthetic cases it was built for."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sfzmaps.solver import energy as E
from sfzmaps.solver import qpbo, search


# --- roof duality against brute force ----------------------------------------------------------------------------

def binary_terms(seed, submodular):
    """a random binary energy on at most 10 variables: unaries c0, c1 and pairs (pa, pb) with costs e00 e01 e10 e11"""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 11))
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35] or [(0, 1)])
    pa, pb = pairs[:, 0], pairs[:, 1]
    c0, c1 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    e = rng.exponential(1.0, (len(pa), 4))
    if submodular:                                       # e00 + e11 <= e01 + e10
        e[:, 3] = np.minimum(e[:, 3], e[:, 1] + e[:, 2] - e[:, 0])
    return n, (c0, c1, pa, pb, e[:, 0], e[:, 1], e[:, 2], e[:, 3])


def every_labelling(n, terms):
    S = np.array(list(itertools.product([False, True], repeat=n)), bool)
    return S, np.array([qpbo.binary_energy(*terms, s) for s in S])


@pytest.mark.parametrize("seed", range(40))
def test_roof_duality_labels_every_variable_of_a_submodular_energy_at_a_minimum(seed):
    n, terms = binary_terms(seed, submodular=True)
    s, labelled = qpbo.roof_dual(*terms)
    _, Es = every_labelling(n, terms)
    assert labelled.all()
    assert qpbo.binary_energy(*terms, s) == pytest.approx(Es.min(), abs=1e-6)


@pytest.mark.parametrize("seed", range(60))
def test_roof_duality_is_persistent_and_an_autarky_on_any_energy(seed):
    n, terms = binary_terms(1000 + seed, submodular=seed % 3 == 0)
    s, labelled = qpbo.roof_dual(*terms)
    S, Es = every_labelling(n, terms)
    # persistency: some minimiser agrees with the labelled variables
    agree = np.all(S[:, labelled] == s[labelled], axis=1)
    assert Es[agree].min() == pytest.approx(Es.min(), abs=1e-6)
    # autarky: the labelled variables put into the all-zero labelling never raise its energy; unlabelled come back 0
    assert not s[~labelled].any()
    assert qpbo.binary_energy(*terms, s) <= Es[0] + 1e-6


@pytest.mark.parametrize("seed", range(30))
def test_the_improve_step_never_raises_the_energy_and_keeps_the_labelled_variables(seed):
    n, terms = binary_terms(2000 + seed, submodular=False)
    s, labelled = qpbo.roof_dual(*terms)
    t = qpbo.improve(*terms, s, labelled)
    assert qpbo.binary_energy(*terms, t) <= qpbo.binary_energy(*terms, s) + 1e-9
    assert np.array_equal(t[labelled], s[labelled])


def test_a_frustrated_cycle_left_open_is_taken_whole_by_the_improve_step():
    # three variables in a triangle, each pulled to 1 by 1, every pair paying 4 when its two agree: roof duality
    # decides none of them, and the open piece is better all 1 (12 - 3) than all 0 (12)
    c0, c1 = np.zeros(3), np.full(3, -1.0)
    pa, pb = np.array([0, 1, 0]), np.array([1, 2, 2])
    terms = (c0, c1, pa, pb, np.full(3, 4.0), np.zeros(3), np.zeros(3), np.full(3, 4.0))
    s, labelled = qpbo.roof_dual(*terms)
    assert not labelled.any() and not s.any()
    t = qpbo.improve(*terms, s, labelled)
    assert t.all() and qpbo.binary_energy(*terms, t) == pytest.approx(qpbo.binary_energy(*terms, s) - 3.0)


# --- the fusion and the move never raise the energy --------------------------------------------------------------

def small_problem(seed, n=12, K=4):
    """a random energy on a grid of lots: label costs U (lot x code, min 0 per lot), the grid's pairs and odds"""
    rng = np.random.default_rng(seed)
    w = 4
    h = n // w
    idx = np.arange(w * h).reshape(h, w)
    a = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    b = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    U = rng.exponential(1.0, (w * h, K)) * (rng.random((w * h, K)) < 0.5)
    U -= U.min(1, keepdims=True)
    odds = rng.normal(0, 2.5, len(a))
    return U, a, b, odds


def brute_fusion(U, a, b, odds, same, x, y):
    var = np.flatnonzero(x != y)
    best = E.energy(U, a, b, odds, x, same)
    for s in itertools.product([False, True], repeat=len(var)):
        q = x.copy()
        q[var[np.array(s, bool)]] = y[var[np.array(s, bool)]]
        best = min(best, E.energy(U, a, b, odds, q, same))
    return best


@pytest.mark.parametrize("seed", range(40))
def test_a_fusion_never_raises_the_energy_and_reaches_the_best_mix_when_every_lot_is_decided(seed):
    U, a, b, odds = small_problem(seed)
    rng = np.random.default_rng(seed)
    K = U.shape[1]
    x, y = rng.integers(0, K, len(U)), rng.integers(0, K, len(U))
    q, decided = search.fusion_move(U, a, b, odds, 1.0, x, y)
    e_x, e_q = E.energy(U, a, b, odds, x, 1.0), E.energy(U, a, b, odds, q, 1.0)
    assert e_q <= e_x
    assert np.all((q == x) | (q == y))                   # every lot keeps its code or takes the proposal's
    if decided:
        assert e_q == pytest.approx(brute_fusion(U, a, b, odds, 1.0, x, y), abs=1e-6)


@pytest.mark.parametrize("seed", range(30))
def test_the_compound_fusion_move_never_raises_the_energy(seed):
    U, a, b, odds = small_problem(100 + seed, n=20, K=5)
    rng = np.random.default_rng(seed)
    for p in (U.argmin(1), rng.integers(0, U.shape[1], len(U))):
        log = []
        q = search.compound_fusion_moves(U, a, b, odds, 1.0, p, log=log)
        assert E.energy(U, a, b, odds, q, 1.0) <= E.energy(U, a, b, odds, p, 1.0) + 1e-12
        assert log and log[-1][0] == "compound fusion"
        assert log[-1][3] == pytest.approx(E.energy(U, a, b, odds, q, 1.0))


def test_the_compound_proposal_offers_the_favoured_code_and_each_holder_its_next_code():
    # three lots in a row: lot 0 (region A, code 0, its labels favour 1) | a line | lots 1-2 (region B, code 1,
    # joined with no line).  The proposal gives A code 1 and each of B's lots its own next code but 1 (lot 1: 2,
    # lot 2: 0), even one a neighbour holds
    U = np.array([[5.0, 0.0, 9.0], [4.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    a, b, odds = np.array([0, 1]), np.array([1, 2]), np.array([3.0, -3.0])
    y = search.compound_proposal(U, a, b, odds, np.array([0, 1, 1]))
    assert y.tolist() == [1, 2, 0]


# --- the two synthetic cases: the basin, every number measured on the map --------------------------------------
#
# Ruling R10: each case starts from the stage's wrong answer x and tests the basin.  Cause: x is a local minimum of
# the stage's own moves (solve.polish, the solve_joint loop with the option off), which leave the incident wrong, as
# on the map (0% right).  Fix: with the option on, the loop reaches the right answer y.
#   (a) the chain: #1/#34 sw, #7 se, #9/#44 se: the district D takes r and the strip's tip T takes s together;
#   (b) the expansion-bound shape: #4/#23 se, #14 sw: X takes al (the bound overcharges it on the map).
# #18 (nw) has shape (b) but the compound fusion does not fix it on the map (exact expansion moves do; not adopted),
# so it is no case of this move.
#
# The fixtures are written by scripts/incidents/search_synthetic_measure.py (--fixture) from the stage's solve:
# search_lab.problem (each quarter's energy rebuilt from its held-out inputs), x = final there, y = the census's
# lower answer.  The incident's neighbourhood, in blocks: one per transition of the changed lots (x code -> y code),
# one per x-region touching them (its lots touching them and the lots carrying its labels), and a region whose labels
# trigger the proposal on the map, whole (#4's: #7's district).  search_synthetic_costs.npz: those map lots' cost rows
# U (labels, placement and tie-break, as solve.costs makes them); the map's borders among them with their odds in the
# quarter's fit and their spread over the four quarter fits; each lot's x-region; and every border from those lots to
# the rest of the city (the lot, the code across, the odds and their spread).  search_synthetic_measured.json: the
# code names and each block's codes and lots; the incident's blocks (D and T, or X, with lots in the solve's own
# quarter: #7's tip is scored from sw, under #1).
#
# A draw: the lots and who borders whom as on the map, each border's odds drawn around its own by its own spread
# (normal; spread=False keeps the map's own odds); the rest of the city held at x, so each border to it is an exact
# term of its lot's costs; and lots of one x-region joined only through the rest of the city tied by a border of
# odds 0 (no cost either way: region moves and the proposal see the map's regions).  On the map's own odds a draw's
# energy differs between x and y exactly as the map's does.

FIXTURE = Path(__file__).parent / "fixtures" / "search_synthetic_measured.json"
COSTS = FIXTURE.with_name("search_synthetic_costs.npz")


def measured():
    return json.loads(FIXTURE.read_text())


class Draw:
    """one synthetic city: the map lots of the incident's neighbourhood (their cost rows U, with the borders to the
    rest of the city in them), the map's borders among them with odds drawn by each border's own spread, and each
    lot's code in the stage's answer (x) and the right one (y)"""

    def __init__(self, case, incident, seed, spread=True):
        m = measured()[case][incident]
        z = np.load(COSTS)
        rng = np.random.default_rng(seed)
        self.U = z[f"{case}-{incident}-U"].astype(float)
        block = z[f"{case}-{incident}-block"].astype(int)
        pairs = z[f"{case}-{incident}-pairs"].astype(int)
        code = {n: i for i, n in enumerate(m["names"])}
        self.x = np.array([code[m["blocks"][k]["x"]] for k in block])
        self.y = np.array([code[m["blocks"][k]["y"]] for k in block])
        self.lots = [np.flatnonzero(block == k) for k in range(len(m["blocks"]))]
        own, sd = z[f"{case}-{incident}-odds"].astype(float), z[f"{case}-{incident}-odds_sd"].astype(float) * spread
        a, b, odds = [pairs[:, 0]], [pairs[:, 1]], [own + sd * rng.standard_normal(len(own))]
        # the map's x-regions: lots of one region joined in the excerpt only through the rest of the city get a tie
        # of odds 0 (no cost either way; it joins them, as region moves and the proposal see the map's regions)
        region = z[f"{case}-{incident}-region"].astype(int)
        no_line = (region[pairs[:, 0]] == region[pairs[:, 1]]) & (own <= 0)
        n = len(region)
        _, piece = connected_components(coo_matrix((np.ones(int(no_line.sum())), (pairs[no_line, 0],
                                                                                 pairs[no_line, 1])), shape=(n, n)),
                                        directed=False)
        for r_ in np.unique(region):
            heads = [np.flatnonzero((region == r_) & (piece == p_))[0] for p_ in np.unique(piece[region == r_])]
            if len(heads) > 1:
                a.append(np.array(heads[1:]))
                b.append(np.full(len(heads) - 1, heads[0]))
                odds.append(np.zeros(len(heads) - 1))
        self.a, self.b, self.odds = np.concatenate(a), np.concatenate(b), np.concatenate(odds)
        # the borders to the rest of the city, held at x: each its lot's cost of differing (no line) or sharing (a
        # line) with the code across, its odds drawn around its own by its own spread
        row, across = z[f"{case}-{incident}-edge_row"].astype(int), z[f"{case}-{incident}-edge_code"].astype(int)
        e = z[f"{case}-{incident}-edge_odds"] + z[f"{case}-{incident}-edge_sd"] * spread * rng.standard_normal(len(row))
        np.add.at(self.U, row, np.maximum(-e, 0.0)[:, None] * (np.arange(self.U.shape[1]) != across[:, None])
                  + np.maximum(e, 0.0)[:, None] * (np.arange(self.U.shape[1]) == across[:, None]))
        self.incident = list(m["incident"].values())


def basin(case, incident, seed, spread=True):
    """one draw of the case: each incident block's share right after the stage's loop from x with the option off
    and on, and the energies of x, y (every changed lot at its y code, the map's lower answer) and the two ends"""
    from sfzmaps.solver.solve import polish
    d = Draw(case, incident, seed, spread)
    off = polish(d.U, d.a, d.b, d.odds, 1.0, d.x.copy())
    on = polish(d.U, d.a, d.b, d.odds, 1.0, d.x.copy(), compound_fusion=True)
    right = {k: (float(np.mean(off[d.lots[k]] == d.y[d.lots[k]])), float(np.mean(on[d.lots[k]] == d.y[d.lots[k]])))
             for k in d.incident}
    e = {name: E.energy(d.U, d.a, d.b, d.odds, p, 1.0) for name, p in (("x", d.x), ("y", d.y), ("off", off),
                                                                           ("on", on))}
    return right, e


CASES = [("chain", "1"), ("chain", "7"), ("chain", "9"), ("bound", "4"), ("bound", "14")]
DRAWS = range(10)


@pytest.mark.parametrize("case,incident", CASES)
def test_on_the_maps_own_odds_the_stage_moves_keep_x_and_the_compound_fusion_move_reaches_y(case, incident):
    """the cause and the fix alone, as on the map: from x the stage's loop leaves every incident block 0% right, and
    the loop with the compound fusion move puts all of it right, at a lower energy; y is lower than x by the map's
    own amount"""
    right, e = basin(case, incident, 0, spread=False)
    for k, (off, on) in right.items():
        assert off == 0.0, f"block {k}: the stage's moves left x ({off:.0%} right)"
        assert on == 1.0, f"block {k}: the compound fusion reached {on:.0%} right"
    assert e["on"] < e["off"]
    assert e["y"] - e["x"] == pytest.approx(measured()[case][incident]["map"]["energy_change"], abs=0.01)


@pytest.mark.parametrize("case,incident", CASES)
def test_within_the_measured_spread_of_the_odds_the_move_puts_the_incident_right_every_time(case, incident):
    """ten draws, each border's odds drawn by its own measured spread.  An incident block comes out right when most
    of its lots get its code (measures.py's rule for a district).  In every draw the compound fusion move puts every
    incident block right, below x and no higher than the stage's moves end; the stage's moves keep x (every incident
    block 0% right) in most
    draws: the map's outcome is the sampler's usual one (#4's basin rests on a 0.6 margin on one lot, which the
    spread can tip)"""
    held = 0
    for seed in DRAWS:
        right, e = basin(case, incident, seed)
        for k, (off, on) in right.items():
            assert on > 0.5, f"draw {seed}, block {k}: the compound fusion put {on:.0%} right"
        assert e["on"] < e["x"] and e["on"] <= e["off"]
        held += all(off == 0.0 for off, _ in right.values())
    assert held > len(DRAWS) / 2, f"the stage's moves kept x in {held} of {len(DRAWS)} draws"
