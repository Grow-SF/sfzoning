import geopandas as gpd
import numpy as np
import pytest
import shapely

from sfzmaps.solver import grid, lotsize, paths, streetside, strokes

NW, NE, SW, SE = range(4)


def _box(x0, y0, x1, y1):
    return shapely.box(x0, y0, x1, y1)


def test_streetside_finds_the_far_lot_across_a_street():
    # one east-west street along y = 0; lot A north of it, lot B south of it, lot C north of it and nearer to the
    # print than B, so the nearest lot other than A is on A's side
    lots = np.array([_box(-50, 30, 50, 130),       # 0: A
                     _box(-50, -130, 50, -30),     # 1: B, 50 ft from the print
                     _box(40, 30, 140, 130),       # 2: C, 41 ft from the print, A's side
                     _box(5000, 30, 5100, 130)])   # 3: far away, north
    streets = gpd.GeoDataFrame(
        {"st_type": ["ST", "HWY", "RAMP", "ST", "ST", "ST"],
         "streetname": ["MAIN", "HWY 101", "OFF", "I-280", "HWY 1 ACCESS", "HWY"]},
        geometry=[shapely.LineString([(-500, 0), (500, 0)]), shapely.LineString([(0, 9000), (10, 9000)]),
                  shapely.LineString([(0, 9100), (10, 9100)]), shapely.LineString([(0, 9200), (10, 9200)]),
                  shapely.LineString([(0, 9300), (10, 9300)]), shapely.LineString([(0, 9400), (10, 9400)])])
    np.testing.assert_array_equal(streetside.freeways(streets), [False, True, True, True, True, False])
    xy = np.array([[0.0, 20.0],     # in the street, nearest A
                   [0.0, 60.0],     # inside A: its side is the print's own side of the centreline
                   [0.0, -20.0],    # in the street, nearest B: the far lot is A (C is farther than A here)
                   [0.0, 20.0],     # no lot under it
                   [5050, 50.0]])   # inside lot 3; nothing south of the street within 3,200 ft
    under = np.array([0, 0, 1, -1, 3])
    far, nearest = streetside.far_side(xy, under, lots, streets.geometry.values)
    assert far.tolist() == [1, 1, 0, -1, -1]
    assert nearest.tolist() == [0, 0, 0, 0, 0]
    # the same street drawn the other way round: the same sides
    flipped = streets.geometry.values.copy()
    flipped[0] = shapely.LineString([(500, 0), (-500, 0)])
    assert streetside.far_side(xy, under, lots, flipped)[0].tolist() == [1, 1, 0, -1, -1]


def test_streetside_top_reading_is_the_most_trusted_first():
    names = np.array(["NC-1", "RH-1", "RH-2"])
    top = streetside.top_codes(names, print_index=np.array([0, 0, 1, 1, 3]), code_index=np.array([0, 1, 2, 1, 0]),
                               trust=np.array([0.4, 0.6, 0.5, 0.5, 0.1]), prints=4)
    assert top.tolist() == ["RH-1", "RH-2", "", "NC-1"]


def _rate_set():
    """Eight prints in the street, each with lot A (`under`) and lot B (`far`); lots 0-3 in nw, 4-7 in ne."""
    codes = np.array(["RH-1", "RH-1", "RH-2", "RH-2", "NC-1", "NC-1", "M-1", "M-1"])
    quarter = np.array([NW] * 4 + [NE] * 4, np.int8)
    known = np.ones(8, bool)
    under = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    far = np.array([4, 2, 1, -1, 0, 6, 5, 3])
    top = np.array(["RH-1", "RH-2", "RH-2", "RH-2", "NC-1", "M-1", "M-1", ""])
    street = np.array([True, True, True, True, True, True, False, True])
    freeway = np.array([False, False, False, True, False, False, False, False])
    return top, under, far, street, freeway, codes, known, quarter


def test_streetside_rates_exclude_quarter_q():
    top, under, far, street, freeway, codes, known, quarter = _rate_set()
    r = streetside.rates(top, under, far, street, freeway, codes, known, quarter, "ne")
    # held out ne: side A counts the prints whose lot A is outside ne (0, 1, 2 ordinary; 3 a freeway); side B the
    # prints whose lot B is outside ne (1, 2 and 4 ordinary; 3 has no lot B; 7 has no top reading)
    assert r == {"rA_st": pytest.approx(2 / 3), "rB_st": pytest.approx(1 / 3), "rA_fw": 1.0, "rB_fw": 0.0}
    # the table's codes of ne's lots change nothing in the ne rates, and change the nw ones
    recoded = np.where(quarter == NE, "P", codes)
    assert streetside.rates(top, under, far, street, freeway, recoded, known, quarter, "ne") == r
    nw = streetside.rates(top, under, far, street, freeway, codes, known, quarter, "nw")
    assert streetside.rates(top, under, far, street, freeway, recoded, known, quarter, "nw") != nw
    assert nw == {"rA_st": 0.5, "rB_st": 0.5, "rA_fw": 0.0, "rB_fw": 0.0}
    # lots with no known code count on neither side
    unknown = known & (codes != "RH-2")
    assert streetside.rates(top, under, far, street, freeway, codes, unknown, quarter, "ne") == {
        "rA_st": 0.5, "rB_st": 0.0, "rA_fw": 0.0, "rB_fw": 0.0}


def test_lotsize_pmi_shape_and_smoothing():
    edges = lotsize.LOG_AREA_EDGES
    assert len(edges) == 37 and edges[0] == 5.0 and edges[-1] == 14.0
    nb = len(edges) - 1
    # code 0: every lot at log area 8.1 (bin 12); code 1: one lot in every bin; code 2: lots only in quarter ne
    area = np.exp(np.r_[np.full(40, 8.1), edges[:-1] + 0.1, np.full(10, 8.1)])
    bins = lotsize.size_bins(area)
    assert bins[0] == 12 and bins[40:40 + nb].tolist() == list(range(nb))
    assert lotsize.size_bins(np.array([1.0, 1e9])).tolist() == [0, nb - 1]
    code = np.r_[np.zeros(40, int), np.ones(nb, int), np.full(10, 2)]
    quarter = np.r_[np.tile([NW, SW, SE, NW], 10), np.tile([NW, SW, SE], 12), np.full(10, NE)].astype(np.int8)
    pmi = lotsize.pmi_by_quarter(bins, code, quarter, 3)
    assert pmi.shape == (4, 3, nb)
    ne = pmi[NE]
    # all of code 0's lots in one bin: positive there, and in its neighbours (each bin takes half of each
    # neighbour's count), most in its own bin; negative in the far bins
    assert ne[0, 12] > ne[0, 11] > 0 and ne[0, 12] > ne[0, 13] > 0 and (ne[0, :10] < 0).all()
    # code 2 has no training lots when ne is held out: smoothed to the overall sizes, it carries no evidence
    np.testing.assert_allclose(ne[2], 0, atol=1e-12)
    assert pmi[NW, 2, 12] > 0
    # a lot in ne changes nothing in the ne fit: ne's lots given another size and code
    moved = lotsize.pmi_by_quarter(np.where(quarter == NE, 30, bins), np.where(quarter == NE, 0, code), quarter, 3)
    np.testing.assert_array_equal(moved[NE], ne)
    assert not np.array_equal(moved[NW], pmi[NW])
    # no wraparound: the smallest and largest bins blend only inward (one lot of code 0 in the smallest bin, one
    # of code 1 in every bin)
    one = lotsize.pmi_by_quarter(np.r_[0, np.arange(nb)], np.r_[0, np.ones(nb, int)],
                                 np.r_[NW, np.tile([NW, SW], nb // 2)].astype(np.int8), 2)[NE]
    assert one[0, 0] > one[0, 1] > 0 and one[0, nb - 1] < 0
    # by hand: the ne fold's counts with the half-neighbour blend, then α = nb pseudo-lots spread as the overall
    held = quarter != NE
    counts = np.zeros((3, nb))
    np.add.at(counts, (code[held], bins[held]), 1.0)
    padded = np.pad(counts, ((0, 0), (1, 1)))
    blended = counts + 0.5 * (padded[:, :-2] + padded[:, 2:])
    overall = blended.sum(0) / blended.sum()
    want = np.log((blended + nb * overall) / (blended.sum(1, keepdims=True) + nb)) - np.log(overall)
    np.testing.assert_allclose(ne, want, atol=1e-12)


def _ink_L(stub):
    """A drawn L 5 px wide (vertical arm rows 10-49 at cols 10-14, horizontal arm rows 45-49 to col 54), its
    vertical arm carried on `stub` px past the corner."""
    ink = np.zeros((70, 70), bool)
    ink[10:50, 10:15] = True
    ink[45:50, 10:55] = True
    ink[50:50 + stub, 10:15] = True
    return ink


def test_strokes_of_an_L_and_a_spur():
    # a plain L: one stroke round the corner (a corner is not a node: only pixels of degree other than 2 are)
    one = strokes.strokes(_ink_L(0))
    assert len(one["length"]) == 1 and not one["spur"][0]
    # carried on 3 px past the corner: a junction there, so the L's arms are two strokes and the stub a third
    s = strokes.strokes(_ink_L(3))
    assert len(s["length"]) == 3 and len(s["off"]) == 4 and s["off"][-1] == len(s["xy"]) == len(s["w"])
    arms, stub = np.flatnonzero(~s["spur"]), np.flatnonzero(s["spur"])
    assert len(arms) == 2 and len(stub) == 1
    # the stub is a dead end shorter than its median width
    k = stub[0]
    assert 1 in (s["dega"][k], s["degb"][k]) and s["length"][k] < s["wmed"][k]
    # the arms run from their free ends (rows 11 and col 53) to the junction (row 47, col 12): 36 and 41 px long;
    # widths near the drawn 5 px (20 ft; twice the distance to the paper at the centre line, so up to a px wider)
    assert sorted(np.round(s["length"][arms] / grid.FT_PER_PX).tolist()) == pytest.approx([36, 41], abs=1)
    assert ((s["wmed"][arms] >= 20) & (s["wmed"][arms] <= 24)).all()
    assert (s["dega"][arms] == 3).sum() + (s["degb"][arms] == 3).sum() == 2
    # positions in map ft: a stroke's points are grid px centres, on the drawn ink, and its ends include the three
    # free ends (col, row): the top of the vertical arm, the right end of the horizontal arm and the stub's end
    cr = grid.to_grid(s["xy"]) - 0.5
    assert _ink_L(3)[np.round(cr[:, 1]).astype(int), np.round(cr[:, 0]).astype(int)].all()
    ends = cr[np.r_[s["off"][:-1], s["off"][1:] - 1]]
    for free in ([13, 11], [53, 46], [11, 51]):
        assert np.abs(ends - free).max(1).min() <= 1, free
    # carried on 6 px past the corner, the stub is longer than it is wide: not a spur
    longer = strokes.strokes(_ink_L(6))
    assert len(longer["length"]) == 3 and not longer["spur"].any()
    # a ring: one closed stroke, its two ends one node, not a spur
    ring = np.zeros((40, 40), bool)
    yy, xx = np.mgrid[0:40, 0:40]
    ring[(np.hypot(yy - 20, xx - 20) <= 12) & (np.hypot(yy - 20, xx - 20) >= 9)] = True
    r = strokes.strokes(ring)
    assert len(r["length"]) == 1 and r["a"][0] == r["b"][0] and not r["spur"][0]
    assert r["length"][0] == pytest.approx(2 * np.pi * 10.5 * grid.FT_PER_PX, rel=0.15)


def test_labels_stage_depends_on_its_code_and_reads():
    from sfzmaps import build
    st = build.stages()["solver-labels"]
    deps = build.stage_deps(st)
    for f in ("trust", "streetside", "lotsize", "strokes", "labels_stage"):
        assert f"src/sfzmaps/solver/{f}.py" in deps
    for d in ("data/solver/prints", "data/solver/lots", "data/solver/page", "data/ref"):
        assert d in deps
    # the prints' code (and the model client it imports) is not this stage's: it reads only prints.parquet
    assert not any(d.endswith(("solver/prints.py", "/llm.py", "/cli.py", "/build.py")) for d in deps)
    assert st.upstream == ("solver-prints",) and build.stage_outs(st) == ["data/solver/labels"]
    assert build.stage_commands("solver-labels") == [["solver", "labels"]]
    assert paths.LABELS_DIR.name == "labels"
