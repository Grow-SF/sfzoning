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
W1 = {C.CODE: 1.0, C.BLOCK: 1.0, C.STREET: 1.0}

# the synthetic pack's own numbers: A (one block's convex hull) 200 x 200 ft, L (two centrelines) 2000 ft, N 1 block
AREA, LENGTH, AWAY, CODE_AWAY = 200.0 * 200.0, 2000.0, 0.1, 0.05


def _spreads(**kw):
    base = dict(street_ft=np.arange(0.0, 55.0, 5.0), street_deg=np.arange(0.0, 11.0, 1.0), street_r=np.ones(11),
                street_h=4.0, street_deg_h=1.5, street_away=AWAY, block_ft=np.r_[np.zeros(18), 10.0, 20.0],
                block_r=np.ones(20), block_h=3.0, block_in=0.9, block_away=AWAY, code_away=CODE_AWAY,
                form_weights={})
    return C.Spreads(**{**base, **kw})


def _pack(year=1980, codes=None):
    streets = gpd.GeoDataFrame({"name": ["MISSION ST", "03RD ST"], "street": ["MISSION", "03RD"], "type": ["ST", "ST"],
                                "active": [True, True]},
                               geometry=[shapely.LineString([(0, 0), (1000, 0)]),
                                         shapely.LineString([(0, 400), (0, 1400)])], crs=C.CRS)
    blocks = gpd.GeoDataFrame({"number": ["0767"]}, geometry=[shapely.box(100, 100, 300, 300)], crs=C.CRS)
    return C.Pack(year=year, streets=streets, blocks=blocks, codes=dict(codes or CODES), spreads={"all": _spreads()})


def _ink(**strings):
    """log p(s | ink) from linear probabilities."""
    return {s.replace("_", "-"): math.log(p) for s, p in strings.items()}


def _p(post):
    return {k: math.exp(v) for k, v in post.items()}


def _street_odds(sp, d, deg):
    return (1 - AWAY) / AWAY * sp.street_density(d) * sp.angle_density(deg) * 90 * AREA / (2 * LENGTH)


def _lift(odds, w, c=1.0):
    """A string's lift from one item it prints with P 1, its kind's share c of the escape weight."""
    return 1.0 + c * ((1.0 + odds) ** w - 1.0)


# --- the five brief tests -------------------------------------------------------------------------------------------

def test_a_string_off_the_list_with_clearly_stronger_ink_wins():
    pack = _pack()
    pr = C.prior(pack, {"x": 500.0, "y": 2000.0, "axis": 0.0}, CODE_ONLY, pack.spreads["all"])
    assert pr.strings == sorted(CODES)
    ink = {**{c: math.log(1e-6) for c in CODES}, "MISSION": math.log(0.95)}
    for w in (1.0, 5.0):
        post = C.posterior(ink, pr, {C.CODE: w})
        assert max(post, key=post.get) == "MISSION"
    # the same ink on a listed string loses nothing to the list
    weak = {**{c: math.log(0.2) for c in CODES}, "MISSION": math.log(0.1)}
    post = C.posterior(weak, pr, {C.CODE: 1.0})
    assert max(post, key=post.get) in CODES


def test_without_a_pack_the_posterior_is_the_ink_alone():
    ink = _ink(RH_1=0.5, RH_2=0.2, MISSION=0.1)
    alone = C.posterior(ink, None, {})
    assert set(alone) == {"RH-1", "RH-2", "MISSION", C.OTHER}
    for s, lp in ink.items():
        assert alone[s] == pytest.approx(lp, abs=1e-12)
    assert math.exp(alone[C.OTHER]) == pytest.approx(0.2, abs=1e-12)
    assert C.prior(None, {"x": 0, "y": 0, "axis": 0}, CODE_ONLY, None) is None
    # a pack with every weight 0 is the same thing
    pack = _pack()
    pr = C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 0.0}, np.array([0.5, 0, 0.5, 0]), pack.spreads["all"])
    full = {**{s: math.log(1e-3) for s in pr.strings}, **ink}
    assert C.posterior(full, pr, {C.CODE: 0.0, C.STREET: 0.0}) == pytest.approx(C.posterior(full, None, {}))


def test_a_box_with_no_street_within_range_falls_back_to_the_escape_mass():
    pack = _pack()
    sp = pack.spreads["all"]
    street_reach, _ = pack.reach(sp)
    assert 0 < street_reach < sp.street_density.top + 1
    far = C.prior(pack, {"x": 500.0, "y": 5000.0, "axis": 0.0}, STREET_ONLY, sp)
    assert far.items == {} and far.escape[2] == 1.0 and far.strings == []
    ink = _ink(MISSION=0.4, ST=0.3)
    post = C.posterior(ink, far, {C.STREET: 5.0})
    assert _p(post) == pytest.approx(_p(C.posterior(ink, None, {})))
    assert all(v > -math.inf for v in post.values())
    # a kind with no list at all (`other`) lifts nothing either
    other = C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 0.0}, np.array([0, 0, 0, 1.0]), sp)
    assert C.log_lift(other, ["MISSION", "RH-1"], W1).tolist() == [0.0, 0.0]


def test_build_sf_pack_2000_refuses_lot_codes():
    with pytest.raises(ValueError, match="answer key"):
        C.build_sf_pack(2000, lot_codes=True)
    with pytest.raises(ValueError, match="lot codes"):
        C.Pack(year=2000, streets=_pack().streets, blocks=_pack().blocks.assign(zoning=["RH-1"]), codes=dict(CODES))


def _word(truth="RH-2", block="1000", quarter=1, **touch):
    return {"truth": truth, "block": block, "quarter": quarter, "touch_blocks": False,
            **{f"touch_{q}": touch.get(q, False) for q in ("nw", "ne", "sw", "se")}}


def test_fit_prior_and_the_spreads_never_see_an_answered_or_held_out_word():
    words = pd.DataFrame([_word(block="1000"), _word(block="0767")])
    with pytest.raises(ValueError, match="answered block"):
        C.fit_prior(words, held_blocks={"0767"})
    with pytest.raises(ValueError, match="answered block"):
        C.fit_prior(pd.DataFrame([_word()]).assign(touch_blocks=[True]), held_blocks=set())
    with pytest.raises(ValueError, match="block"):
        C.fit_prior(pd.DataFrame([_word()]).drop(columns="block"), held_blocks=set())
    with pytest.raises(ValueError, match="held-out quarter"):
        C.fit_prior(words.iloc[:1], held_blocks=set(), quarter=1)
    with pytest.raises(ValueError, match="touch the held-out quarter"):
        C.fit_prior(pd.DataFrame([_word(quarter=0, se=True)]), held_blocks=set(), quarter=3)
    with pytest.raises(ValueError, match="answered block"):
        C.measure_spreads(_pack(), pd.DataFrame([{**_word(block="0767"), "text": "767", "kind": C.BLOCK, "x": 0.0,
                                                  "y": 0.0, "axis": 0.0}]), held_blocks={"0767"})


# --- the prior's parts ------------------------------------------------------------------------------------------------

def test_a_box_along_a_near_street_gets_its_names_by_bayes_and_across_it_does_not():
    pack, sp = _pack(), _spreads()
    along = C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 0.0}, STREET_ONLY, sp)
    assert {"MISSION", "ST", "ST.", "MISSION ST", "MISSION ST."} <= set(along.strings)
    o_mission = float(_street_odds(sp, 10.0, 0.0))
    o_third = float(_street_odds(sp, math.hypot(500, 390), 90.0))
    assert o_third == 0.0                                    # square to the box and far: every kernel is exactly 0
    assert along.escape[2] == pytest.approx(1 / (1 + o_mission))
    assert along.mass(C.STREET)["MISSION"] == pytest.approx(o_mission / (1 + o_mission) / 5)
    across = C.prior(pack, {"x": 500.0, "y": 10.0, "axis": 90.0}, STREET_ONLY, sp)
    assert across.items == {} and across.escape[2] == 1.0


def test_numbered_streets_carry_their_spelled_forms():
    forms = C.street_forms("03RD", "ST")
    assert {"3RD", "THIRD", "ST", "ST.", "3RD ST", "THIRD ST."} <= set(forms)
    assert "03RD" not in forms
    assert {"TWENTY-THIRD", "23RD"} <= set(C.street_forms("23RD", "AVE"))
    assert set(C.street_forms("SAN JOSE", "AVE")) >= {"SAN", "JOSE", "SAN JOSE", "SAN JOSE AVE."}
    assert C.street_forms("UNNAMED 161", "") == []


def test_the_block_holding_the_box_centre_gets_its_number_by_bayes():
    pack, sp = _pack(), _spreads()
    inside = C.prior(pack, {"x": 200.0, "y": 200.0, "axis": 0.0}, BLOCK_ONLY, sp)
    assert inside.strings == ["767"]                  # printed without its leading zero, as the maps print it
    odds = (1 - AWAY) / AWAY * (AREA / 1) * 0.9 / AREA     # (1 - m_b) / m_b x A / N x p_in / area
    assert inside.items[C.BLOCK][0][0] == pytest.approx(odds)
    assert inside.escape[1] == pytest.approx(1 / (1 + odds))
    near = C.prior(pack, {"x": 200.0, "y": 315.0, "axis": 0.0}, BLOCK_ONLY, sp)     # 15 ft outside it
    odds = (1 - AWAY) / AWAY * AREA * 0.1 * float(sp.block_out_density(15.0)) / (800.0 + 2 * math.pi * 15.0)
    assert near.items[C.BLOCK][0][0] == pytest.approx(odds)
    assert C.prior(pack, {"x": 200.0, "y": 900.0, "axis": 0.0}, BLOCK_ONLY, sp).items == {}


def test_codes_are_flat_over_the_era_list_with_their_measured_escape():
    pack, sp = _pack(), _spreads()
    pr = C.prior(pack, {"x": 200.0, "y": 200.0, "axis": 0.0}, np.array([0.5, 0.5, 0.0, 0.0]), sp)
    odds = (1 - CODE_AWAY) / (len(CODES) * CODE_AWAY)
    assert [o for o, _ in pr.items[C.CODE]] == pytest.approx([odds] * len(CODES))
    assert pr.escape[0] == pytest.approx(CODE_AWAY)
    for c in CODES:
        assert pr.mass(C.CODE)[c] == pytest.approx((1 - CODE_AWAY) / len(CODES))


def test_the_weight_tempers_a_code_whole_before_its_forms_share_it():
    codes = {"BROADWAY": "BROADWAY", "BROADWAY NCD": "BROADWAY", "NORTH BEACH": "NORTH BEACH",
             "NORTH BEACH NCD": "NORTH BEACH", "NO BEACH": "NORTH BEACH", "N. BEACH": "NORTH BEACH", "RH-1": "RH-1"}
    pack = _pack(codes=codes)
    sp = _spreads(form_weights={"BROADWAY": 1.0, "BROADWAY NCD": 3.0, "NORTH BEACH": 1.0, "NORTH BEACH NCD": 1.0,
                                "NO BEACH": 1.0, "N. BEACH": 1.0, "RH-1": 5.0})
    pr = C.prior(pack, {"x": 500.0, "y": 2000.0, "axis": 0.0}, CODE_ONLY, sp)
    forms = list(codes)
    for w in (1.0, 3.0, 6.0):
        lifts = dict(zip(forms, np.exp(C.log_lift(pr, forms, {C.CODE: w}))))
        one = lifts["RH-1"] - 1.0
        assert sum(lifts[f] - 1.0 for f in ("NORTH BEACH", "NORTH BEACH NCD", "NO BEACH", "N. BEACH")) == \
            pytest.approx(one)                        # four forms share one code's lift: never (1/4)^w each
        assert lifts["BROADWAY NCD"] - 1.0 == pytest.approx(0.75 * one)
        assert lifts["BROADWAY"] - 1.0 == pytest.approx(0.25 * one)


def test_posterior_needs_ink_for_every_listed_string():
    pack = _pack()
    pr = C.prior(pack, {"x": 0.0, "y": 2000.0, "axis": 0.0}, CODE_ONLY, pack.spreads["all"])
    with pytest.raises(ValueError, match="ink"):
        C.posterior(_ink(RH_1=0.5), pr, {C.CODE: 1.0})
    with pytest.raises(KeyError):
        C.posterior({c: math.log(0.2) for c in CODES}, pr, {})
    with pytest.raises(ValueError, match="needs the fold"):
        C.prior(pack, {"x": 0.0, "y": 0.0, "axis": 0.0}, CODE_ONLY, None)


def test_a_listed_string_is_lifted_and_an_unlisted_one_is_not():
    pack = _pack()
    pr = C.prior(pack, {"x": 200.0, "y": 200.0, "axis": 0.0}, np.array([0.5, 0.5, 0, 0]), pack.spreads["all"])
    lift = dict(zip(["767", "RH-1", "MISSION"], C.log_lift(pr, ["767", "RH-1", "MISSION"], W1)))
    assert lift["767"] > lift["RH-1"] > 0 and lift["MISSION"] == 0.0
    c = pr.kind_p * pr.escape / (pr.kind_p * pr.escape).sum()
    assert math.exp(lift["RH-1"]) == pytest.approx(_lift((1 - CODE_AWAY) / (4 * CODE_AWAY), 1.0, c[0]))


# --- the ink ----------------------------------------------------------------------------------------------------------

def test_the_ink_is_ctc_at_its_temperature_mixed_over_the_two_reading_directions():
    import torch

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
    hot = ctc.string_log_probs(torch.log_softmax(reads[0] * 0.5, -1), strings)
    np.testing.assert_allclose(list(C.ink_logp(reads, 0.0, strings, tau=0.5).values()), hot, rtol=1e-12)
    assert C.ink_logp(reads, 0.0, ["7"], tau=0.5)["7"] != pytest.approx(one[3])


# --- measuring --------------------------------------------------------------------------------------------------------

def test_em_splits_words_that_name_what_they_sit_by_from_far_misreads():
    rng = np.random.default_rng([0, 3])
    pack = _pack()
    pack = C.Pack(year=1980, streets=pack.streets, codes={"RH-1": "RH-1", "RH-2": "RH-2", "RH 2": "RH-2"},
                  blocks=gpd.GeoDataFrame({"number": ["0767", "0768"]}, geometry=[shapely.box(100, 100, 300, 300),
                                          shapely.box(400, 100, 600, 300)], crs=C.CRS))
    near = [{"text": "MISSION", "kind": C.STREET, "x": 100.0 + 8 * i, "y": float(rng.exponential(4.0)),
             "axis": float(rng.normal(0, 2.0)) % 180} for i in range(80)]
    far = [{"text": "MISSION", "kind": C.STREET, "x": float(rng.uniform(-3000, 3000)),
            "y": float(rng.uniform(2000, 9000)), "axis": float(rng.uniform(0, 180))} for _ in range(12)]
    unnamed = [{"text": "ELM", "kind": C.STREET, "x": 500.0, "y": 5.0, "axis": 0.0}] * 4
    inside = [{"text": "767", "kind": C.BLOCK, "x": 200.0, "y": 200.0, "axis": 0.0}] * 30
    edge = [{"text": "767", "kind": C.BLOCK, "x": 200.0, "y": 300.0 + float(rng.exponential(3.0)), "axis": 0.0}
            for _ in range(8)]                       # just outside the block they name
    misread = [{"text": "768", "kind": C.BLOCK, "x": 200.0 + 900 * k, "y": 3000.0 + 1700 * k, "axis": 0.0}
               for k in range(1, 7)]
    codes = [{"text": t, "kind": C.CODE, "x": 0.0, "y": 0.0, "axis": 0.0}
             for t in ["RH-1"] * 6 + ["RH-2"] * 3 + ["RH 2"] + ["XYZ"] * 2]
    words = pd.DataFrame(near + far + unnamed + inside + edge + misread + codes).assign(block="", quarter=0)
    sp, off = C.measure_spreads(pack, words, held_blocks=set())
    r = off["r"].to_numpy()
    assert (r[:80] > 0.9).mean() >= 0.95 and (r[80:92] < 1e-6).all() and (r[92:96] == 0).all()   # an outlying
    #                                                                       near draw may be doubted, never a far one
    assert sp.street_away == pytest.approx(1 - r[:92].sum() / 96)
    assert (r[96:126] == 1.0).all() and (r[126:140] == 0.0).all()      # outside the block they name: away
    assert sp.block_in == 1.0 and sp.block_away == pytest.approx(1 - 30 / 44)
    assert sp.code_away == pytest.approx(2 / 12)
    assert sp.form_weights == pytest.approx({"RH-1": 1.0, "RH-2": 0.75, "RH 2": 0.25})
    assert sp.street_h > 0 and sp.street_deg_h > 0


def test_a_kernel_density_is_exactly_zero_past_its_top_and_its_bandwidth_is_the_likeliest():
    x = np.array([1.0, 2.0, 2.5, 4.0, 7.0])
    d = C.Density(x, np.ones(5), 1.0, 0.0, None)
    assert d(d.top + 1e-6) == 0.0 and d(d.top - 5.0) > 0
    assert C.Density(x, np.zeros(5), 1.0)(2.0) == 0.0
    h = C.loo_bandwidth(x, np.ones(5), 0.0, None)
    ll = [float(np.log(C.loo_density(x, np.ones(5), v, 0.0, None)).sum()) for v in (0.8 * h, h, 1.25 * h)]
    assert ll[1] >= ll[0] and ll[1] >= ll[2]


# --- fitting ----------------------------------------------------------------------------------------------------------

def _fit_words(n=24, seed=4):
    """Words whose ink can't tell RH-1 from RH-2 but whose place lists RH-2 above it (a street prior around it)."""
    from sfzmaps.solver.textread import ctc, model
    rng = np.random.default_rng([seed, 5])
    pack = _pack()
    sp = pack.spreads["all"]
    rows = []
    for i in range(n):
        reads = []
        for _ in range(2):
            z = rng.standard_normal((10, model.CLASSES)) * 0.3
            for t, ch in enumerate("RH-2"):
                z[2 * t, model.encode(ch)[0]] += 3.0
            z[6, model.encode("1")[0]] += 2.6           # the last letter: 2 or 1, nearly as likely
            reads.append(ctc.log_softmax(z))
        pr = C.prior(pack, {"x": 500.0, "y": 2000.0, "axis": 0.0}, CODE_ONLY,
                     _spreads(form_weights={"RH-2": 1.0}, code_away=0.2) if i % 2 == 0 else sp)
        rows.append({**_word(truth="RH-2", block=f"b{i}", quarter=0), "reads": reads, "q": 0.1, "prior": pr})
    return pd.DataFrame(rows)


def test_the_fit_scores_each_truth_as_the_posterior_does():
    words = _fit_words(4)
    for tau, w in ((1.0, {C.CODE: 0.0}), (0.7, {C.CODE: 1.3}), (1.4, {C.CODE: 3.0})):
        got = C.truth_loglik(words, tau, w)
        for k, r in enumerate(words.itertuples(index=False)):
            names = sorted(set(r.prior.strings) | {r.truth})
            post = C.posterior(C.ink_logp(r.reads, r.q, names, tau), r.prior, w)
            assert got[k] == pytest.approx(post[r.truth], rel=1e-10, abs=1e-10)


def test_fit_prior_finds_tau_and_a_weight_no_worse_than_the_ink_alone():
    got = C.fit_prior(_fit_words(), held_blocks=set(), quarter=3)
    assert got["tau"] > 0 and set(got["w"]) == {C.CODE}
    assert got["loglik"] >= got["loglik_ink_at_tau"] - 1e-9 and got["loglik"] >= got["loglik_ink_as_read"] - 1e-9
    assert got["w"][C.CODE] > 0.1 and got["tau_converged"]


def test_a_truth_its_box_cannot_emit_is_left_out_of_the_fit_and_counted():
    assert C.ctc_fits("AB", 2) and not C.ctc_fits("AA", 2) and C.ctc_fits("AA", 3) and C.ctc_fits("", 0)
    words = _fit_words(6)
    long = words.iloc[[0]].assign(truth="SAN FRANCISCO MUNICIPAL CODE", block="b99", weight=0.2)
    both = pd.concat([words.assign(weight=1.0), long], ignore_index=True)
    assert C.readable(both).tolist() == [True] * 6 + [False]
    got = C.fit_prior(both, held_blocks=set(), quarter=3)
    assert got["impossible"] == {"pairs": 1, "weight": pytest.approx(0.2)} and got["n"] == 6
    assert math.isfinite(got["loglik"])


def test_a_pairs_weight_counts_like_that_many_copies_and_one_boxs_rows_share_its_ink():
    words = _fit_words(3)
    w = {C.CODE: 1.7}
    twice = pd.concat([words, words.iloc[[0]]], ignore_index=True)
    weighted = words.assign(weight=[2.0, 1.0, 1.0])
    a, b = C._Words(twice), C._Words(weighted)
    assert float((a.weight * a.loglik(0.8, w)).sum()) == pytest.approx(float((b.weight * b.loglik(0.8, w)).sum()))
    # two Terra words held by one box: one read, each row its own truth
    pair = pd.concat([words.iloc[[0]], words.iloc[[0]].assign(truth="RH-1")], ignore_index=True).assign(box=[7, 7])
    got = C.truth_loglik(pair, 0.8, w)
    alone = [C.truth_loglik(pair.iloc[[k]].drop(columns="box"), 0.8, w)[0] for k in range(2)]
    assert got == pytest.approx(alone)


def test_weighted_em_ignores_a_pair_of_weight_zero():
    rng = np.random.default_rng([0, 6])
    d = np.r_[rng.exponential(4.0, 60), rng.uniform(1000, 9000, 10)]
    deg = np.r_[np.abs(rng.normal(0, 2.0, 60)), rng.uniform(0, 90, 10)]
    feats = [(d, 0.0, None), (deg, 0.0, 90.0)]
    base = C.em_split(feats, d, True)
    extra_d, extra_deg = np.r_[d, 3000.0], np.r_[deg, 1.0]
    zero = C.em_split([(extra_d, 0.0, None), (extra_deg, 0.0, 90.0)], extra_d, True, np.r_[np.ones(70), 0.0])
    assert zero["ll"] == pytest.approx(base["ll"]) and zero["h"] == pytest.approx(base["h"])


def test_a_temperature_on_the_combined_posterior_undoes_overconfidence():
    rng = np.random.default_rng([0, 7])
    rows = []
    for _ in range(400):
        right = rng.random() < 0.7               # the top hypothesis is right 70% of the time, posterior says 99%
        lp = np.log(np.array([0.99, 0.005, 0.005]))
        rows.append((lp, 0 if right else int(rng.integers(1, 3))))
    got = C.fit_temperature(rows)
    assert got["temperature"] > 1.5 and got["nll"] < got["nll_at_1"]
    p_top = math.exp(C.tempered(rows[0][0], got["temperature"])[0])
    assert p_top == pytest.approx(0.7, abs=0.05)
    assert C.tempered(rows[0][0], 1.0) == pytest.approx(rows[0][0])


# --- the pack file ----------------------------------------------------------------------------------------------------

def test_a_pack_round_trips_through_its_file(tmp_path):
    pack = _pack()
    pack.spreads["ne"] = _spreads(form_weights={"RH-1": 1.0, "RH-2": 0.5})
    f = tmp_path / "pack.parquet"
    pack.save(f)
    back = C.Pack.load(f)
    assert back.year == pack.year and back.codes == pack.codes
    assert list(back.streets["name"]) == list(pack.streets["name"])
    assert back.blocks.geometry.iloc[0].equals(pack.blocks.geometry.iloc[0])
    assert set(back.spreads) == {"all", "ne"}
    assert back.spreads["ne"].form_weights == {"RH-1": 1.0, "RH-2": 0.5}
    np.testing.assert_array_equal(back.spreads["all"].block_ft, pack.spreads["all"].block_ft)
    for k in C.SPREAD_SCALARS:
        assert getattr(back.spreads["ne"], k) == getattr(pack.spreads["ne"], k)
    box = {"x": 500.0, "y": 10.0, "axis": 0.0}
    a, b = C.prior(back, box, STREET_ONLY, back.spreads["all"]), C.prior(pack, box, STREET_ONLY, pack.spreads["all"])
    assert a.strings == b.strings and a.escape.tolist() == pytest.approx(b.escape.tolist())


def test_box_on_map_turns_page_corners_into_a_map_centre_and_axis():
    class Flip:     # page px -> map ft: y up, 2 ft a px
        def to_world(self, p):
            p = np.asarray(p, float)
            return np.c_[2 * p[..., 0], -2 * p[..., 1]]
    corners = np.array([[0, 0], [10, -10], [12, -8], [2, 2]], float)   # reading up and to the right on the page
    got = C.box_on_map(corners, Flip())
    assert got["axis"] == pytest.approx(45.0)             # the page's y runs down and the map's up: still 45 degrees
    assert (got["x"], got["y"]) == pytest.approx((12.0, 8.0))


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
    try:
        pack = C.Pack.load(f)
    except (KeyError, IndexError, TypeError) as err:
        pytest.skip(f"the built pack is of an older layout: {err}")
    assert pack.year == 2000
    assert set(pack.blocks.columns) == {"number", "geometry"}
    assert len(pack.streets) > 10000 and len(pack.blocks) > 3000 and len(set(pack.codes.values())) > 30
