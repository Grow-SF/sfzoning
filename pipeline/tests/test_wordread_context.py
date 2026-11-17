"""Context packs and the reader's posterior with them (`wordread.context`), on a small synthetic pack."""
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from sfzmaps.solver.wordread import context as C

CODES = {"RH-1": "RH-1", "RH-2": "RH-2", "RH-3": "RH-3", "NC-1": "NC-1"}
STREET_ONLY = np.array([0.0, 0.0, 1.0, 0.0])       # kind_p: a street name for sure (zoning, block, street, other)
CODE_ONLY = np.array([1.0, 0.0, 0.0, 0.0])
BLOCK_ONLY = np.array([0.0, 1.0, 0.0, 0.0])


def _spreads():
    return C.Spreads(street_ft=np.arange(0.0, 55.0, 5.0), street_deg=np.arange(0.0, 11.0, 1.0),
                     block_ft=np.r_[np.zeros(18), 10.0, 20.0], street_away=0.1, block_away=0.1)


# the synthetic pack's own numbers: A (one block's convex hull) 200 x 200 ft, L (two centrelines) 2000 ft, N 1 block
AREA, LENGTH, AWAY = 200.0 * 200.0, 2000.0, 0.1


def _street_odds(sp, d, deg):
    return (1 - AWAY) / AWAY * sp.street_density(d) * sp.angle_density(deg) * 90 * AREA / (2 * LENGTH)


def _pack(year=1980):
    streets = gpd.GeoDataFrame({"name": ["MISSION ST", "03RD ST"], "street": ["MISSION", "03RD"], "type": ["ST", "ST"],
                                "active": [True, True]},
                               geometry=[shapely.LineString([(0, 0), (1000, 0)]),
                                         shapely.LineString([(0, 400), (0, 1400)])], crs=C.CRS)
    blocks = gpd.GeoDataFrame({"number": ["0767"]}, geometry=[shapely.box(100, 100, 300, 300)], crs=C.CRS)
    return C.Pack(year=year, streets=streets, blocks=blocks, codes=dict(CODES), spreads={"all": _spreads()})


def _ink(**strings):
    """log p(s | ink) from linear probabilities."""
    return {s.replace("_", "-"): math.log(p) for s, p in strings.items()}


def _p(post):
    return {k: math.exp(v) for k, v in post.items()}


# --- the five brief tests -------------------------------------------------------------------------------------------

def test_a_string_off_the_list_with_clearly_stronger_ink_wins():
    pr = C.prior(_pack(), {"x": 500.0, "y": 2000.0, "axis": 0.0}, CODE_ONLY)
    assert set(pr) == set(CODES) | {C.ESCAPE}
    ink = {**{c: math.log(1e-6) for c in CODES}, "MISSION": math.log(0.95)}
    post = C.posterior(ink, pr, w=1.0, e=0.05)
    assert max(post, key=post.get) == "MISSION"
    # the same ink on a listed string loses nothing to the list
    weak = {**{c: math.log(0.2) for c in CODES}, "MISSION": math.log(0.1)}
    post = C.posterior(weak, pr, w=1.0, e=0.05)
    assert max(post, key=post.get) in CODES


def test_without_a_pack_the_posterior_is_the_ink_alone():
    ink = _ink(RH_1=0.5, RH_2=0.2, MISSION=0.1)
    alone = C.posterior(ink, None, w=1.0, e=0.05)
    assert set(alone) == {"RH-1", "RH-2", "MISSION", C.OTHER}
    for s, lp in ink.items():
        assert alone[s] == pytest.approx(lp, abs=1e-12)
    assert math.exp(alone[C.OTHER]) == pytest.approx(0.2, abs=1e-12)
    # a pack's prior with every mass on the escape is the same thing
    assert C.posterior(ink, C.prior(None, {"x": 0, "y": 0, "axis": 0}, CODE_ONLY), 1.0, 0.05) == pytest.approx(alone)


def test_a_box_with_no_street_within_range_falls_back_to_the_escape_mass():
    pack = _pack()
    assert 0 < pack.reach(pack.spreads["all"])[0] < 45000
    far = C.prior(pack, {"x": 500.0, "y": 50000.0, "axis": 0.0}, STREET_ONLY)
    assert far == {C.ESCAPE: 0.0}
    ink = _ink(MISSION=0.4, ST=0.3)
    alone = _p(C.posterior(ink, None, 1.0, 0.05))
    post = C.posterior(ink, far, w=1.3, e=0.02)
    assert _p(post) == pytest.approx(alone)
    assert all(v > -math.inf for v in post.values())
    # a street listed with next to no mass is read by the ink alone: the pack never pushes a string below it
    faint = {"MISSION": math.log(1e-9), C.ESCAPE: math.log(1 - 1e-9)}
    assert _p(C.posterior(ink, faint, w=1.3, e=0.02)) == pytest.approx(alone, rel=1e-6)
    # a kind with no list at all (`other`) goes to the escape too
    assert C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 0.0}, np.array([0, 0, 0, 1.0])) == {C.ESCAPE: 0.0}


def test_build_sf_pack_2000_refuses_lot_codes():
    with pytest.raises(ValueError, match="answer key"):
        C.build_sf_pack(2000, lot_codes=True)
    with pytest.raises(ValueError, match="lot codes"):
        C.Pack(year=2000, streets=_pack().streets, blocks=_pack().blocks.assign(zoning=["RH-1"]), codes=dict(CODES))


def _word(truth="RH-2", block="1000", quarter=1, ink=None, prior=None):
    pr = prior or {"RH-1": math.log(0.3), "RH-2": math.log(0.3), C.ESCAPE: math.log(0.4)}
    return {"truth": truth, "ink": ink or _ink(RH_1=0.45, RH_2=0.45), "prior": pr, "block": block,
            "quarter": quarter}


def test_fit_prior_never_sees_an_answered_block_word():
    words = pd.DataFrame([_word(block="1000"), _word(block="0767")])
    with pytest.raises(ValueError, match="answered block"):
        C.fit_prior(words, held_blocks={"0767"})
    with pytest.raises(ValueError, match="answered block"):
        C.fit_prior(pd.DataFrame([_word()]).assign(touch_blocks=[True]), held_blocks=set())
    with pytest.raises(ValueError, match="block"):
        C.fit_prior(pd.DataFrame([_word()]).drop(columns="block"), held_blocks=set())
    with pytest.raises(ValueError, match="held-out quarter"):
        C.fit_prior(words.iloc[:1], held_blocks=set(), quarter=1)


# --- the prior's parts ------------------------------------------------------------------------------------------------

def test_a_box_along_a_near_street_gets_its_names_and_across_it_does_not():
    pack, sp = _pack(), _spreads()
    along = C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 0.0}, STREET_ONLY)
    assert {"MISSION", "ST", "ST.", "MISSION ST", "MISSION ST."} <= set(along)
    # Bayes: MISSION ST 10 ft off and parallel; 03RD ST 634 ft off and square to the box
    o_mission, o_third = _street_odds(sp, 10.0, 0.0), _street_odds(sp, math.hypot(500, 390), 90.0)
    assert math.exp(along[C.ESCAPE]) == pytest.approx(1 / (1 + o_mission + o_third))
    assert math.exp(along["MISSION"]) == pytest.approx(o_mission / (1 + o_mission + o_third) / 5)
    assert 5 * math.exp(along["MISSION"]) > 0.9
    across = C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 90.0}, STREET_ONLY)
    assert math.exp(across[C.ESCAPE]) > 0.999


def test_numbered_streets_carry_their_spelled_forms():
    forms = C.street_forms("03RD", "ST")
    assert {"3RD", "THIRD", "ST", "ST.", "3RD ST", "THIRD ST."} <= set(forms)
    assert "03RD" not in forms
    assert {"TWENTY-THIRD", "23RD"} <= set(C.street_forms("23RD", "AVE"))
    assert set(C.street_forms("SAN JOSE", "AVE")) >= {"SAN", "JOSE", "SAN JOSE", "SAN JOSE AVE."}
    assert C.street_forms("UNNAMED 161", "") == []


def test_the_block_holding_the_box_centre_gets_its_number():
    pack = _pack()
    inside = C.prior(pack, {"x": 200.0, "y": 200.0, "axis": 0.0}, BLOCK_ONLY)
    assert set(inside) == {"767", C.ESCAPE}         # printed without its leading zero, as the maps print it
    odds = (1 - AWAY) / AWAY * AREA * 0.9 / AREA    # (1 - m_b) / m_b x A / N x p_in / area
    assert math.exp(inside["767"]) == pytest.approx(odds / (1 + odds))
    assert math.exp(inside[C.ESCAPE]) == pytest.approx(1 / (1 + odds))
    near = C.prior(pack, {"x": 200.0, "y": 315.0, "axis": 0.0}, BLOCK_ONLY)     # 15 ft outside it
    odds = (1 - AWAY) / AWAY * AREA * 0.1 * pack.spreads["all"].block_out_density(15.0) / 800.0
    assert math.exp(near["767"]) == pytest.approx(odds / (1 + odds))
    assert C.prior(pack, {"x": 200.0, "y": 900.0, "axis": 0.0}, BLOCK_ONLY) == {C.ESCAPE: 0.0}


def test_codes_are_flat_over_the_era_list_and_kinds_mix():
    pack = _pack()
    pr = C.prior(pack, {"x": 200.0, "y": 200.0, "axis": 0.0}, np.array([0.5, 0.5, 0.0, 0.0]))
    for c in CODES:
        assert math.exp(pr[c]) == pytest.approx(0.5 / len(CODES))
    assert math.exp(pr["767"]) == pytest.approx(0.5 * 8.1 / 9.1)
    assert sum(math.exp(v) for v in pr.values()) == pytest.approx(1.0)


def test_a_codes_mass_goes_to_its_forms_by_their_measured_weights(tmp_path):
    base = _pack()
    codes = {"BROADWAY": "BROADWAY", "BROADWAY NCD": "BROADWAY", "RH-1": "RH-1", "RH-2": "RH-2", "RH 2": "RH-2"}
    pack = C.Pack(year=1980, streets=base.streets, blocks=base.blocks, codes=codes, spreads=base.spreads,
                  form_weights={"BROADWAY": 0.25, "BROADWAY NCD": 0.75, "RH-2": 1.0})     # RH 2 unmeasured: flat
    pr = C.prior(pack, {"x": 0.0, "y": 2000.0, "axis": 0.0}, CODE_ONLY)
    assert {s: math.exp(v) for s, v in pr.items() if s != C.ESCAPE} == \
        pytest.approx({"BROADWAY": 1 / 12, "BROADWAY NCD": 0.25, "RH-1": 1 / 3, "RH-2": 1 / 6, "RH 2": 1 / 6})
    pack.save(tmp_path / "p.parquet")
    assert C.Pack.load(tmp_path / "p.parquet").form_weights == pack.form_weights


def test_posterior_needs_ink_for_every_listed_string_and_a_positive_escape():
    pr = C.prior(_pack(), {"x": 0.0, "y": 0.0, "axis": 0.0}, CODE_ONLY)
    with pytest.raises(ValueError, match="ink"):
        C.posterior(_ink(RH_1=0.5), pr, 1.0, 0.05)
    with pytest.raises(ValueError, match="escape"):
        C.posterior({c: math.log(0.2) for c in CODES}, pr, 1.0, 0.0)


def test_fit_prior_weighs_a_list_that_helps_and_escapes_one_that_does_not():
    rng = np.random.default_rng([0, 1])
    helps = []
    for k in range(200):
        truth = "RH-2"
        # ink can't tell RH-1 from RH-2; the prior can
        helps.append(_word(truth=truth, block=f"b{k}", prior={"RH-1": math.log(0.05), "RH-2": math.log(0.9),
                                                               C.ESCAPE: math.log(0.05)},
                           ink=_ink(RH_1=0.45 + 0.01 * rng.standard_normal(), RH_2=0.45)))
    got = C.fit_prior(pd.DataFrame(helps), held_blocks=set())
    assert got["w"] > 0.5 and got["loglik"] > got["loglik_ink_only"]
    misleads = [_word(truth="MISSION", block=f"b{k}", prior={"RH-1": math.log(0.5), "RH-2": math.log(0.5),
                                                              C.ESCAPE: -math.inf},
                      ink=_ink(RH_1=0.3, RH_2=0.3, MISSION=0.3)) for k in range(50)]
    got = C.fit_prior(pd.DataFrame(misleads), held_blocks=set())
    assert got["e"] > 0.5 or got["w"] < 0.1


def test_the_fit_scores_each_truth_as_the_posterior_does():
    pr = {"RH-1": math.log(0.2), "RH-2": math.log(0.5), C.ESCAPE: math.log(0.3)}
    ink = _ink(RH_1=0.3, RH_2=0.1, MISSION=0.4)
    words = pd.DataFrame([_word(truth="RH-2", ink=ink, prior=pr), _word(truth="MISSION", ink=ink, prior=pr)])
    for w, e in ((0.0, 0.5), (0.7, 0.1), (2.0, 0.9)):
        post = C.posterior(ink, pr, w, e)
        np.testing.assert_allclose(C.truth_loglik(words, w, e), [post["RH-2"], post["MISSION"]], rtol=1e-12)


def test_the_spreads_are_measured_on_words_that_name_what_they_sit_by():
    pack = _pack()
    pack = C.Pack(year=1980, streets=pack.streets, codes={},
                  blocks=gpd.GeoDataFrame({"number": ["0767", "0768"]}, geometry=[shapely.box(100, 100, 300, 300),
                                          shapely.box(400, 100, 600, 300)], crs=C.CRS))
    words = pd.DataFrame([
        {"text": "MISSION", "kind": "street name", "x": 500.0, "y": 5.0, "axis": 2.0},     # by its street
        {"text": "MISSION", "kind": "street name", "x": 5.0, "y": 900.0, "axis": 90.0},    # by 03RD ST: away
        {"text": "ELM", "kind": "street name", "x": 500.0, "y": 5.0, "axis": 0.0},         # no such street: away
        {"text": "767", "kind": "block number", "x": 200.0, "y": 200.0, "axis": 0.0},      # inside its block
        {"text": "767", "kind": "block number", "x": 500.0, "y": 200.0, "axis": 0.0},      # inside 768: away
        {"text": "767", "kind": "block number", "x": 200.0, "y": 330.0, "axis": 0.0},      # 30 ft outside its own
    ]).assign(block="", quarter=0)
    sp, off = C.measure_spreads(pack, words, held_blocks=set())
    assert list(off["found"]) == [True, True, False, True, True, True]
    assert list(off["here"]) == [True, False, False, True, False, True]
    np.testing.assert_allclose(sp.street_ft, [5.0])
    np.testing.assert_allclose(sp.street_deg, [2.0])
    np.testing.assert_allclose(sp.block_ft, [0.0, 30.0])
    assert sp.street_away == pytest.approx(2 / 3) and sp.block_away == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="answered block"):
        C.measure_spreads(pack, words.assign(block="0767"), held_blocks={"0767"})


def test_a_pack_round_trips_through_its_file(tmp_path):
    pack = _pack()
    pack.spreads["ne"] = _spreads()
    f = tmp_path / "pack.parquet"
    pack.save(f)
    back = C.Pack.load(f)
    assert back.year == pack.year and back.codes == pack.codes
    assert list(back.streets["name"]) == list(pack.streets["name"])
    assert back.blocks.geometry.iloc[0].equals(pack.blocks.geometry.iloc[0])
    assert set(back.spreads) == {"all", "ne"}
    np.testing.assert_array_equal(back.spreads["all"].block_ft, pack.spreads["all"].block_ft)
    box = {"x": 500.0, "y": 10.0, "axis": 0.0}
    assert C.prior(back, box, STREET_ONLY) == C.prior(pack, box, STREET_ONLY)


def test_box_on_map_turns_page_corners_into_a_map_centre_and_axis():
    class Flip:     # page px -> map ft: y up, 2 ft a px
        def to_world(self, p):
            p = np.asarray(p, float)
            return np.c_[2 * p[..., 0], -2 * p[..., 1]]
    corners = np.array([[0, 0], [10, -10], [12, -8], [2, 2]], float)   # reading up and to the right on the page
    got = C.box_on_map(corners, Flip())
    assert got["axis"] == pytest.approx(45.0)             # the page's y runs down and the map's up: still 45 degrees
    assert (got["x"], got["y"]) == pytest.approx((12.0, 8.0))


def test_the_ink_is_ctc_mixed_over_the_two_reading_directions():
    from sfzmaps.solver.textread import ctc
    rng = np.random.default_rng([0, 2])
    reads = [ctc.log_softmax(rng.standard_normal((12, 47))) for _ in range(2)]
    strings = ["RH-1", "MISSION", "", "7", "SAN FRANCISCO"]
    one, other = (ctc.string_log_probs(r, strings) for r in reads)
    np.testing.assert_allclose(list(C.ink_logp(reads, 0.0, strings).values()), one, rtol=1e-12)
    np.testing.assert_allclose(list(C.ink_logp(reads, 1.0, strings).values()), other, rtol=1e-12)
    mixed = C.ink_logp(reads, 0.25, strings)
    np.testing.assert_allclose(np.exp(list(mixed.values())), 0.75 * np.exp(one) + 0.25 * np.exp(other), rtol=1e-12)
    assert mixed["SAN FRANCISCO"] == -math.inf     # thirteen letters can't fit in twelve frames


def test_the_kinds_and_the_alphabet_are_the_detectors_and_the_readers():
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.textread import model
    assert C.KIND_ORDER == tuple(sorted(data.KINDS, key=data.KINDS.get))
    assert C.VOCAB == model.VOCAB


# --- the real pack (skips without pipeline/data) ----------------------------------------------------------------------

def test_the_built_sf_2000_pack_holds_no_lot_codes():
    f = C.PACKS_DIR / "sf-2000.parquet"
    if not f.is_file():
        pytest.skip("no built pack (pipeline/data)")
    pack = C.Pack.load(f)
    assert pack.year == 2000
    assert set(pack.blocks.columns) == {"number", "geometry"}
    assert len(pack.streets) > 10000 and len(pack.blocks) > 3000 and len(set(pack.codes.values())) > 30
