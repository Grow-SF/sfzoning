"""Box choice by the reader (`wordread.choose`), on synthetic pages and a stub reader."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver.wordread import choose as C

WIDTHS = {"zoning": 8.0, "block": 6.0, "street": 7.0, "other": 7.0}
KINDS = ("zoning", "block", "street", "other")


def make_box(x0, x1, y0=20.0, y1=30.0, axis=0.0, kind="zoning"):
    """An axis-0 box row as boxes.parquet has it (trimmed corners, edge coordinates)."""
    row = {"axis": axis, "t_length": x1 - x0, "t_height": y1 - y0, **{f"p_{k}": 0.0 for k in KINDS}}
    row[f"p_{kind}"] = 1.0
    for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        row[f"t{i}x"], row[f"t{i}y"] = x, y
    return row


def page_with(*spans, shape=(60, 200)):
    """Ink: a solid bar over rows 22..28 across each (x0, x1) span."""
    ink = np.zeros(shape, bool)
    for a, b in spans:
        ink[22:28, a:b] = True
    return ink


def by_kind(cands):
    return dict(cands)


def feat(cands, name):
    c = next(c for c in cands if c.name == name)
    return c.along, c.across


def read_by_span(table, default=-30.0):
    """A stub reader: the log posterior of the best string for a word, by (rounded left, rounded right)."""
    def read(corners, box):
        key = (round(float(corners[:, 0].min())), round(float(corners[:, 0].max())))
        s, lp = table.get(key, ("x", default))
        return {s: lp, "none": math.log(1e-3), "other": math.log(1e-3)}
    return read


# --- candidates -------------------------------------------------------------------------------------------------

def test_candidates_start_with_as_found_and_have_every_kind():
    box = make_box(10, 90)
    c = C.candidates(box, page_with((10, 40), (60, 90)), [make_box(100, 130)], WIDTHS)
    assert [k for k, _ in c] == ["as_found", "split_gap", "join_nearest", "trim_start", "trim_end"]
    assert by_kind(c)["as_found"].shape == (1, 4, 2)


def test_split_is_at_the_widest_gap_and_gives_two_halves():
    box = make_box(10, 100)
    ink = page_with((10, 30), (33, 40), (70, 100))      # gaps: 3 px and 30 px
    halves = by_kind(C.candidates(box, ink, [], WIDTHS))["split_gap"]
    assert halves.shape == (2, 4, 2)
    assert halves[0][:, 0].max() == pytest.approx(40) and halves[1][:, 0].min() == pytest.approx(70)
    assert halves[0][:, 0].min() == pytest.approx(10) and halves[1][:, 0].max() == pytest.approx(100)


def test_no_split_without_an_interior_gap():
    c = by_kind(C.candidates(make_box(10, 60), page_with((10, 60)), [], WIDTHS))
    assert "split_gap" not in c


def test_split_follows_a_slanted_axis():
    a = math.radians(30)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c0 = np.array([100.0, 100.0])
    ink = np.zeros((200, 200), bool)
    for p in list(range(-40, -10)) + list(range(10, 40)):
        for q in range(-3, 4):
            x, y = c0 + p * u + q * v
            ink[int(y), int(x)] = True
    box = {"axis": 30.0, "t_length": 80.0, "t_height": 8.0, **{f"p_{k}": 0.0 for k in KINDS}, "p_zoning": 1.0}
    for i, (p, q) in enumerate([(-40, -4), (40, -4), (40, 4), (-40, 4)]):
        box[f"t{i}x"], box[f"t{i}y"] = c0 + p * u + q * v
    halves = by_kind(C.candidates(box, ink, [], WIDTHS))["split_gap"]
    mids = [((h - c0) @ u).mean() for h in halves]
    assert mids[0] < -10 < 10 < mids[1]


def test_join_takes_the_nearest_box_however_far_and_its_distance_is_the_gap():
    box = make_box(10, 50)                      # height 10
    near, far, off = make_box(56, 90), make_box(70, 100), make_box(54, 90, 40, 50)
    cands = C.candidates(box, page_with((10, 50)), [far, near, off], WIDTHS)
    c = by_kind(cands)
    assert c["join_nearest"][0][:, 0].min() == pytest.approx(10)
    assert c["join_nearest"][0][:, 0].max() == pytest.approx(90)
    assert feat(cands, "join_nearest") == pytest.approx((6 / 8, 0))
    # no window: a box 450 px away is joined when it is the only one
    assert feat(C.candidates(box, page_with((10, 50)), [make_box(500, 540)], WIDTHS), "join_nearest")[0] \
        == pytest.approx(450 / 8)


def test_trim_cuts_the_kinds_median_character_width_from_either_end():
    c = by_kind(C.candidates(make_box(10, 90, kind="block"), page_with((10, 90)), [], WIDTHS))
    assert c["trim_start"][0][:, 0].min() == pytest.approx(16) and c["trim_start"][0][:, 0].max() == pytest.approx(90)
    assert c["trim_end"][0][:, 0].max() == pytest.approx(84) and c["trim_end"][0][:, 0].min() == pytest.approx(10)


def test_no_trim_of_a_box_not_longer_than_one_character():
    c = by_kind(C.candidates(make_box(10, 18), page_with((10, 18)), [], WIDTHS))
    assert "trim_start" not in c and "trim_end" not in c


# --- choose -----------------------------------------------------------------------------------------------------

def test_a_box_holding_two_words_is_split_when_the_halves_read_better():
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [], WIDTHS)
    read = read_by_span({(10, 100): ("R-1 R-2", math.log(0.05)), (10, 40): ("R-1", math.log(0.97)),
                         (70, 100): ("R-2", math.log(0.96))})
    kind, info = C.choose(box, cands, read)
    assert kind == "split_gap"
    assert info["words"] == ["R-1", "R-2"]


def test_a_word_cut_in_two_is_joined():
    box = make_box(10, 50)
    cands = C.candidates(box, page_with((10, 50)), [make_box(56, 90)], WIDTHS)
    read = read_by_span({(10, 50): ("R", math.log(0.1)), (10, 90): ("R-3", math.log(0.95))})
    assert C.choose(box, cands, read)[0] == "join_nearest"


def test_a_word_is_kept_as_found_when_trimming_reads_worse():
    box = make_box(10, 90)
    cands = C.candidates(box, page_with((10, 90)), [], WIDTHS)
    read = read_by_span({(10, 90): ("M-1", math.log(0.9))}, default=math.log(0.2))
    assert C.choose(box, cands, read)[0] == "as_found"


def test_a_trimmed_end_wins_when_it_reads_better():
    box = make_box(10, 90)
    cands = C.candidates(box, page_with((10, 90)), [], WIDTHS)
    read = read_by_span({(10, 90): ("M-1X", math.log(0.1)), (10, 82): ("M-1", math.log(0.9))}, default=math.log(0.05))
    assert C.choose(box, cands, read)[0] == "trim_end"


def test_equal_reads_keep_as_found():
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [make_box(106, 130)], WIDTHS)
    assert len(cands) == 5
    flat = lambda corners, b: {"R-1": math.log(0.5), "none": math.log(0.25), "other": math.log(0.25)}   # noqa: E731
    assert C.choose(box, cands, flat)[0] == "as_found"
    assert C.choose(box, [c for c in cands if c.name != "split_gap"], flat)[0] == "as_found"


def test_a_two_word_candidate_scores_the_sum_of_its_words_log_posteriors():
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [], WIDTHS)
    read = read_by_span({(10, 100): ("A B", math.log(0.5)), (10, 40): ("A", math.log(0.8)),
                         (70, 100): ("B", math.log(0.8))})
    _, info = C.choose(box, cands, read)
    assert info["scores"]["split_gap"] == pytest.approx(2 * math.log(0.8))
    assert info["scores"]["as_found"] == pytest.approx(math.log(0.5))


def test_a_split_does_not_win_a_tie_of_word_posteriors():
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [], WIDTHS)
    read = lambda corners, b: {"A": math.log(0.9), "none": math.log(0.05), "other": math.log(0.05)}   # noqa: E731
    assert C.choose(box, cands, read)[0] == "as_found"


def test_the_prior_renormalises_each_words_posterior():
    box = make_box(10, 50)
    cands = [c for c in C.candidates(box, page_with((10, 50)), [], WIDTHS) if c.name in ("as_found", "trim_end")]

    def read(corners, b):
        return {"P": math.log(0.6), "Q": math.log(0.3), "none": math.log(0.05), "other": math.log(0.05)}

    assert C.choose(box, cands, read)[0] == "as_found"
    prior = lambda s, b: -20.0 if s == "P" else 0.0   # noqa: E731
    _, info = C.choose(box, cands, read, prior)
    want = math.log(0.3) - math.log(0.3 + 0.05 + 0.05 + 0.6 * math.exp(-20))
    assert info["scores"]["as_found"] == pytest.approx(want, abs=1e-9)
    assert info["words"] == ["Q"]


def test_candidates_need_the_widths():
    with pytest.raises(ValueError):
        C.candidates(make_box(10, 90), page_with((10, 90)), [], None)


def test_a_box_of_unknown_kind_has_no_trims():
    box = make_box(10, 90)
    for k in KINDS:
        box[f"p_{k}"] = 0.0
    names = [k for k, _ in C.candidates(box, page_with((10, 90)), [], WIDTHS)]
    assert names == ["as_found"]


def test_join_leaves_out_the_box_itself_and_an_overlapping_neighbour():
    box = make_box(10, 50)
    names = lambda ns: [k for k, _ in C.candidates(box, page_with((10, 50)), ns, WIDTHS)]   # noqa: E731
    assert "join_nearest" not in names([dict(box)])
    assert "join_nearest" not in names([make_box(40, 80)])
    assert "join_nearest" in names([make_box(40, 80), make_box(56, 90)])


def test_a_vertical_box_splits_and_joins_along_its_axis():
    def vbox(y0, y1, x=50.0):
        row = {"axis": 90.0, "t_length": y1 - y0, "t_height": 10.0, **{f"p_{k}": 0.0 for k in KINDS}, "p_zoning": 1.0}
        for i, (cx, cy) in enumerate([(x + 5, y0), (x + 5, y1), (x - 5, y1), (x - 5, y0)]):
            row[f"t{i}x"], row[f"t{i}y"] = cx, cy
        return row

    ink = np.zeros((200, 100), bool)
    ink[20:50, 46:54] = True
    ink[80:110, 46:54] = True
    cands = C.candidates(vbox(20, 110), ink, [vbox(116, 150)], WIDTHS)
    c = by_kind(cands)
    halves = c["split_gap"]
    assert halves[0][:, 1].min() == pytest.approx(20) and halves[0][:, 1].max() == pytest.approx(50)
    assert halves[1][:, 1].min() == pytest.approx(80) and halves[1][:, 1].max() == pytest.approx(110)
    assert feat(cands, "split_gap")[0] == pytest.approx(30 / 8)
    assert c["join_nearest"][0][:, 1].max() == pytest.approx(150)
    assert feat(cands, "join_nearest") == pytest.approx((6 / 8, 0))
    assert c["trim_start"][0][:, 1].min() == pytest.approx(28)


# --- the fitted choice --------------------------------------------------------------------------------------------

def synthetic_rows(n=400, seed=0, ink_scores=True):
    """Boxes with an as_found and a split_gap; the split is right with a chance rising with the gap (in character
    widths), and the reads' scores say nothing about which."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n):
        gap = rng.uniform(0.05, 3.0)
        split = rng.random() < 1 / (1 + math.exp(-3 * (gap - 1.5)))
        a, s = rng.normal(-0.5, 0.3), rng.normal(-0.7, 0.3)
        out += [{"box": b, "cand": "as_found", "score": a, "gap": 0.0, "across": 0.0, "weight": float(not split)},
                {"box": b, "cand": "split_gap", "score": s, "gap": gap, "across": 0.0, "weight": float(split)}]
    return pd.DataFrame(out).assign(held_out=False, answered=False)


def test_the_fit_recovers_a_positive_gap_weight_when_splits_are_right_only_with_a_large_gap():
    p = C.fit_choice(synthetic_rows())
    assert p["gamma"]["split_gap"] > 1.5 and p["alpha"]["split_gap"] < -2
    assert p["gamma"]["as_found"] == 0 and p["alpha"]["as_found"] == 0


def test_the_chosen_version_flips_with_the_gap():
    params = C.fit_choice(synthetic_rows())
    read = read_by_span({(10, 100): ("A B", math.log(0.9))}, default=math.log(0.8))
    pick = lambda gap: C.choose(   # noqa: E731
        make_box(10, 100), C.candidates(make_box(10, 100), page_with((10, 40), (40 + gap, 100)), [], WIDTHS),
        read, params=params)
    big, small = pick(26), pick(1)
    assert big[0] == "split_gap" and big[1]["prob"] > 0.5
    assert small[0] == "as_found"


def test_a_merged_box_reading_other_with_confidence_loses_to_a_split_with_a_large_gap():
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [], WIDTHS)
    merged = {"other": math.log(0.95), "none": math.log(0.025), "A": math.log(0.025)}
    half = {"R-1": math.log(0.9), "other": math.log(0.05), "none": math.log(0.05)}
    read = lambda corners, b: merged if corners[:, 0].max() - corners[:, 0].min() > 80 else half   # noqa: E731
    assert C.choose(box, cands, read)[0] == "as_found"           # the raw sum: the confident "other" wins
    kind, info = C.choose(box, cands, read, params=C.fit_choice(synthetic_rows()))
    assert kind == "split_gap" and info["prob"] > 0.5


def test_a_one_pixel_letter_gap_does_not_split():
    box = make_box(10, 100)
    ink = page_with((10, 50), (51, 100))
    cands = C.candidates(box, ink, [], WIDTHS)
    assert feat(cands, "split_gap")[0] == pytest.approx(1 / 8)
    half = {"R-1": math.log(0.95), "other": math.log(0.025), "none": math.log(0.025)}
    whole = {"R-1": math.log(0.5), "other": math.log(0.4), "none": math.log(0.1)}
    read = lambda corners, b: whole if corners[:, 0].max() - corners[:, 0].min() > 80 else half   # noqa: E731
    assert C.choose(box, cands, read)[0] == "split_gap"           # raw: two clear halves beat the whole's doubt
    kind, _ = C.choose(box, cands, read, params=C.fit_choice(synthetic_rows()))
    assert kind == "as_found"


def test_choice_features_are_one_row_per_version_with_the_gap_in_character_widths():
    box = make_box(10, 100, kind="block")
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [], WIDTHS)
    read = lambda corners, b: {"A": math.log(0.9), "none": math.log(0.05), "other": math.log(0.05)}   # noqa: E731
    f = C.choice_features(box, cands, read)
    assert list(f["cand"]) == [k for k, _ in cands]
    assert f.loc[f["cand"] == "split_gap", "gap"].item() == pytest.approx(30 / 6.0)
    assert (f.loc[f["cand"] != "split_gap", "gap"] == 0).all()
    assert f.loc[f["cand"] == "split_gap", "score"].item() == pytest.approx(2 * math.log(0.9))


def ink_page(*spans):
    ink = np.zeros((60, 200), bool)
    for a, b in spans:
        ink[22:28, a:b] = True
    return ink


def word(x0, x1):
    return (x0, 20, x1, 30)


def test_a_short_box_beside_a_different_word_labels_as_found_not_the_join():
    box = make_box(10, 40)                                   # Terra's word runs 10 to 50: our box is short
    ink = ink_page((10, 50), (56, 90))
    cands = C.candidates(box, ink, [make_box(56, 90)], WIDTHS)
    w = C.label_weights(cands, [word(10, 50), word(56, 90)], ink)
    assert w["as_found"] == pytest.approx(0.75, abs=0.02)
    assert w["join_nearest"] < 0.3 and w["as_found"] > 2 * w["join_nearest"]
    assert max(w, key=w.get) == "as_found"


def test_a_word_split_across_two_boxes_labels_the_join():
    box = make_box(10, 50)
    ink = ink_page((10, 50), (56, 90))
    cands = C.candidates(box, ink, [make_box(56, 90)], WIDTHS)
    w = C.label_weights(cands, [word(10, 90)], ink)
    assert w["join_nearest"] == pytest.approx(1.0) and w["as_found"] < 0.6
    assert max(w, key=w.get) == "join_nearest"


def test_a_box_holding_two_words_labels_the_split():
    box = make_box(10, 100)
    ink = ink_page((10, 40), (70, 100))
    cands = C.candidates(box, ink, [], WIDTHS)
    w = C.label_weights(cands, [word(10, 40), word(70, 100)], ink)
    assert w["split_gap"] == pytest.approx(1.0) and w["as_found"] == pytest.approx(0.0)


def test_a_box_with_no_word_in_it_weighs_nothing_in_every_version():
    box = make_box(10, 50)
    ink = ink_page((10, 50))
    cands = C.candidates(box, ink, [], WIDTHS)
    assert all(v == 0 for v in C.label_weights(cands, [word(150, 190)], ink).values())
    assert all(v == 0 for v in C.label_weights(cands, [], ink).values())


def test_label_weights_take_a_slanted_word_as_its_rotated_rectangle():
    a = math.radians(45)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c0 = np.array([100.0, 100.0])
    corners = np.array([c0 + p * u + q * v for p, q in ((-40, -3), (40, -3), (40, 3), (-40, 3))])
    ink = np.zeros((200, 200), bool)
    for p in range(-40, 40):
        for q in range(-2, 3):
            x, y = c0 + p * u + q * v
            ink[int(y), int(x)] = True
    box = {"axis": 45.0, "t_length": 80.0, "t_height": 6.0, **{f"p_{k}": 0.0 for k in KINDS}, "p_zoning": 1.0}
    for i, (x, y) in enumerate(corners):
        box[f"t{i}x"], box[f"t{i}y"] = x, y
    cands = C.candidates(box, ink, [], WIDTHS)
    w = C.label_weights(cands, [corners], ink)
    assert w["as_found"] == pytest.approx(1.0, abs=0.02)


def test_the_soft_target_fit_recovers_known_coefficients():
    true = {"beta": 0.8, "alpha": {"split_gap": -1.5, "trim_end": -0.7}, "gamma": {"split_gap": 1.2}}
    rng = np.random.default_rng(3)
    rows = []
    for b in range(600):
        gap = rng.uniform(0.0, 3.0)
        sc = {"as_found": rng.normal(-1, 0.8), "split_gap": rng.normal(-1.5, 0.8), "trim_end": rng.normal(-1.2, 0.8)}
        gp = {"as_found": 0.0, "split_gap": gap, "trim_end": 0.0}
        u = {n: true["beta"] * sc[n] + true["alpha"].get(n, 0.0) + true["gamma"].get(n, 0.0) * gp[n] for n in sc}
        z = sum(math.exp(x) for x in u.values())
        for n in sc:
            rows.append({"box": b, "cand": n, "score": sc[n], "gap": gp[n], "across": 0.0,
                         "weight": math.exp(u[n]) / z})
    p = C.fit_choice(pd.DataFrame(rows).assign(held_out=False, answered=False))
    assert p["beta"] == pytest.approx(0.8, abs=0.02)
    assert p["alpha"]["split_gap"] == pytest.approx(-1.5, abs=0.03) and p["alpha"]["trim_end"] == pytest.approx(-0.7,
                                                                                                             abs=0.03)
    assert p["gamma"]["split_gap"] == pytest.approx(1.2, abs=0.03)


def test_a_box_whose_versions_all_weigh_nothing_adds_nothing_to_the_fit():
    rows = synthetic_rows(60)
    base = C.fit_choice(rows)
    dead = rows.copy()
    dead["box"] = dead["box"] + 1000
    dead["weight"] = 1e-9
    more = C.fit_choice(pd.concat([rows, dead], ignore_index=True))
    assert more["gamma"]["split_gap"] == pytest.approx(base["gamma"]["split_gap"], abs=1e-3)


def test_fit_refuses_held_out_and_answered_rows():
    rows = synthetic_rows(20)
    for flag in ("held_out", "answered"):
        with pytest.raises(ValueError):
            C.fit_choice(rows.assign(**{flag: [False] * (len(rows) - 1) + [True]}))
        with pytest.raises(ValueError):           # a missing column is refused, not assumed False
            C.fit_choice(rows.drop(columns=flag))
    assert C.fit_choice(rows)["boxes"] == 20


def test_a_separated_fit_stays_finite_because_beta_has_the_ridge():
    rows = synthetic_rows(60)
    rows["score"] = np.where(rows["weight"] > 0, 0.0, -50.0)            # the score alone separates the labels
    p = C.fit_choice(rows)
    assert np.isfinite(p["beta"]) and abs(p["beta"]) < 1e3


def test_a_candidate_pickles_and_its_features_go_to_a_table():
    import pickle
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [make_box(106, 130)], WIDTHS)
    back = pickle.loads(pickle.dumps(cands))
    assert [(c.name, c.along, c.across) for c in back] == [(c.name, c.along, c.across) for c in cands]
    read = lambda corners, b: {"A": math.log(0.9), "none": math.log(0.05), "other": math.log(0.05)}   # noqa: E731
    f = C.choice_features(box, back, read)
    assert pd.DataFrame(f).to_parquet() and (f["gap"] > 0).any()


def test_a_next_line_box_has_lower_join_odds_than_a_same_line_box_at_the_same_distance():
    rng = np.random.default_rng(1)
    rows = []
    for b in range(500):          # joins are right only on the same line (across 0); distances are drawn alike
        d = rng.uniform(0.2, 3.0)
        same = rng.random() < 0.5
        rows += [{"box": 2 * b, "cand": "as_found", "score": rng.normal(-0.5, 0.2), "gap": 0.0, "across": 0.0,
                  "weight": float(not same)},
                 {"box": 2 * b, "cand": "join_nearest", "score": rng.normal(-0.5, 0.2), "gap": d if same else 0.0,
                  "across": 0.0 if same else d / 2, "weight": float(same)}]
    params = C.fit_choice(pd.DataFrame(rows).assign(held_out=False, answered=False))
    assert params["gamma_across"]["join_nearest"] < params["gamma"]["join_nearest"]
    box = make_box(10, 50)
    read = lambda corners, b: {"A": math.log(0.8), "none": math.log(0.1), "other": math.log(0.1)}   # noqa: E731
    ink = page_with((10, 50))
    same_line = C.candidates(box, ink, [make_box(56, 90)], WIDTHS)               # 6 px along, 0 across
    next_line = C.candidates(box, ink, [make_box(10, 50, 36, 46)], WIDTHS)       # 0 along, 6 px across
    p_same = C.choose(box, same_line, read, params=params)[1]["probs"]["join_nearest"]
    p_next = C.choose(box, next_line, read, params=params)[1]["probs"]["join_nearest"]
    assert p_next < p_same


def test_choose_needs_as_found_first():
    with pytest.raises(ValueError):
        C.choose(make_box(0, 10), [], lambda c, b: {"none": 0.0})


# --- the measured character widths --------------------------------------------------------------------------------

def test_char_widths_are_the_median_length_over_string_length_per_kind_outside_answered_blocks():
    rows = pd.DataFrame({
        "word": [0, 1, 2, 3, 4, 5, 6], "kind": ["zoning code"] * 3 + ["block number"] * 2 + ["street name", "none"],
        "text": ["R-1", "R-2", "R-3", "0274", "0275", "MAIN", ""], "length": [24, 30, 21, 24, 40, 28, 5.0],
        "quarter": [0, 0, 0, 1, 1, 2, 3], "touch_blocks": [False, False, True, False, False, False, False]})
    w = C.measure_char_widths(rows)
    assert w["all"]["zoning"] == pytest.approx(np.median([8.0, 10.0]))      # the third word touches the blocks
    assert w["all"]["block"] == pytest.approx(np.median([6.0, 10.0]))
    assert w["all"]["street"] == pytest.approx(7.0)
    assert "other" not in w["all"]
    # quarter ne (1) held out: its block words are not used
    assert w["held_out"]["ne"]["zoning"] == pytest.approx(np.median([8.0, 10.0]))
    assert "block" not in w["held_out"]["ne"]


def test_load_char_widths_reads_the_file_for_a_held_out_quarter(tmp_path):
    f = tmp_path / "char_width.json"
    f.write_text(json.dumps({"all": {"zoning": 8.0, "block": 6.0}, "held_out": {"ne": {"zoning": 9.0}}}))
    assert C.load_char_widths(f)["zoning"] == 8.0
    got = C.load_char_widths(f, held_out="ne")
    assert got["zoning"] == 9.0 and got["block"] == 6.0       # a kind with no words left falls back to the pool
