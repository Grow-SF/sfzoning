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


# --- the fitted quality --------------------------------------------------------------------------------------------

def rows_frame(rows):
    return pd.DataFrame(rows).assign(held_out=False, answered=False)


def synthetic_rows(n=400, seed=0):
    """Boxes with an as_found and a split_gap; the split holds one word in each half with a chance rising with the gap
    (in character widths), and the reads' scores say nothing about which."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n):
        gap = rng.uniform(0.05, 3.0)
        split = rng.random() < 1 / (1 + math.exp(-3 * (gap - 1.5)))
        a, s0, s1 = rng.normal(-0.5, 0.3), rng.normal(-0.35, 0.2), rng.normal(-0.35, 0.2)
        out += [{"box": b, "cand": "as_found", "score": a, "word_scores": [a], "gap": 0.0, "across": 0.0,
                 "kind": "zoning", "weight": float(not split)},
                {"box": b, "cand": "split_gap", "score": s0 + s1, "word_scores": [s0, s1], "gap": gap, "across": 0.0,
                 "kind": "zoning", "weight": float(split)}]
    return rows_frame(out)


def as_found_and_split(box, ink):
    """The versions the synthetic rows fit (as found and the split): the fit has seen no trim."""
    return [c for c in C.candidates(box, ink, [], WIDTHS) if c.name in ("as_found", "split_gap")]


def test_the_fit_recovers_a_positive_gap_weight_when_splits_are_right_only_with_a_large_gap():
    p = C.fit_quality(synthetic_rows())
    assert p["gamma"]["split_gap"] > 0.5
    assert p["gamma"]["as_found"] == 0 and p["alpha"]["as_found"] == 0
    assert p["converged"]


def test_the_chosen_version_flips_with_the_gap():
    params = C.fit_quality(synthetic_rows())
    read = read_by_span({(10, 100): ("A B", math.log(0.9))}, default=math.log(0.8))
    pick = lambda gap: C.choose(   # noqa: E731
        make_box(10, 100), as_found_and_split(make_box(10, 100), page_with((10, 40), (40 + gap, 100))), read,
        params=params)
    big, small = pick(26), pick(1)
    assert big[0] == "split_gap" and big[1]["prob"] > 0.5
    assert small[0] == "as_found"


def test_a_merged_box_reading_other_with_confidence_loses_to_a_split_with_a_large_gap():
    box = make_box(10, 100)
    cands = as_found_and_split(box, page_with((10, 40), (70, 100)))
    merged = {"other": math.log(0.95), "none": math.log(0.025), "A": math.log(0.025)}
    half = {"R-1": math.log(0.9), "other": math.log(0.05), "none": math.log(0.05)}
    read = lambda corners, b: merged if corners[:, 0].max() - corners[:, 0].min() > 80 else half   # noqa: E731
    assert C.choose(box, cands, read)[0] == "as_found"           # the raw sum: the confident "other" wins
    kind, info = C.choose(box, cands, read, params=C.fit_quality(synthetic_rows()))
    assert kind == "split_gap" and info["prob"] > 0.5


def test_a_one_pixel_letter_gap_does_not_split():
    box = make_box(10, 100)
    ink = page_with((10, 50), (51, 100))
    cands = as_found_and_split(box, ink)
    assert feat(cands, "split_gap")[0] == pytest.approx(1 / 8)
    half = {"R-1": math.log(0.95), "other": math.log(0.025), "none": math.log(0.025)}
    whole = {"R-1": math.log(0.5), "other": math.log(0.4), "none": math.log(0.1)}
    read = lambda corners, b: whole if corners[:, 0].max() - corners[:, 0].min() > 80 else half   # noqa: E731
    assert C.choose(box, cands, read)[0] == "split_gap"           # raw: two clear halves beat the whole's doubt
    kind, _ = C.choose(box, cands, read, params=C.fit_quality(synthetic_rows()))
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
    assert list(f.loc[f["cand"] == "split_gap", "word_scores"].item()) == pytest.approx([math.log(0.9)] * 2)
    assert (f["kind"] == "block").all() and "word_count1" not in f
    with_count = C.choice_features(box, cands, read, count_fn=lambda corners, b: -0.5)
    assert [len(x) for x in with_count["word_count1"]] == [len(c.rects) for c in cands]


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


def _sig(x):
    return 1 / (1 + math.exp(-x))


def eta_of(p, cand, along, across, kind, s, c1=None):
    """One word rectangle's linear predictor under the params `p` (written out here, as the module documents it)."""
    out = (p["intercept"] + p["alpha"].get(cand, 0.0) + p["gamma"].get(cand, 0.0) * along
           + p["gamma_across"].get(cand, 0.0) * across + p["kind"].get(kind, 0.0) + p["beta"] * s)
    return out + (p["count1"] * c1 if c1 is not None else 0.0)


TRUE = {"intercept": 0.4, "beta": 0.9, "count1": 0.6,
        "alpha": {"split_gap": -0.6, "join_nearest": -1.1, "trim_start": -0.3, "trim_end": -0.5},
        "gamma": {"split_gap": 0.7, "join_nearest": -0.8}, "gamma_across": {"join_nearest": -1.3},
        "kind": {"block": 0.5, "street": -0.4, "other": -0.9}}


def test_the_quality_fit_recovers_known_coefficients_and_a_split_is_the_product_of_its_halves():
    rng = np.random.default_rng(3)
    rows = []
    for b in range(700):
        kind = str(rng.choice(["zoning", "block", "street", "other"]))
        for cand in ("as_found", "split_gap", "join_nearest", "trim_start", "trim_end"):
            k = 2 if cand == "split_gap" else 1
            ws, c1 = rng.normal(-1.0, 1.0, k), rng.normal(-1.0, 1.0, k)
            along = float(rng.uniform(0, 3)) if cand in ("split_gap", "join_nearest") else 0.0
            across = float(rng.uniform(0, 2)) if cand == "join_nearest" else 0.0
            w = math.prod(_sig(eta_of(TRUE, cand, along, across, kind, s, c)) for s, c in zip(ws, c1))
            rows.append({"box": b, "cand": cand, "score": float(ws.sum()), "word_scores": list(ws),
                         "word_count1": list(c1), "gap": along, "across": across, "kind": kind, "weight": w})
    p = C.fit_quality(rows_frame(rows))
    assert p["converged"] and p["boxes"] == 700 and p["versions"] == 3500
    for k in ("intercept", "beta", "count1"):
        assert p[k] == pytest.approx(TRUE[k], abs=0.02), k
    for group in ("alpha", "gamma", "gamma_across", "kind"):
        for name, v in TRUE[group].items():
            assert p[group][name] == pytest.approx(v, abs=0.02), (group, name)
    assert p["alpha"]["as_found"] == 0 and p["kind"]["zoning"] == 0


def test_without_the_box_models_column_the_fit_has_no_count_weight_and_choose_needs_count_fn_with_one():
    rows = synthetic_rows(60)
    assert C.fit_quality(rows)["count1"] is None
    params = C.fit_quality(rows.assign(word_count1=[[-1.0] * len(w) for w in rows["word_scores"]]))
    assert params["count1"] is not None
    box = make_box(10, 100)
    read = lambda corners, b: {"A": math.log(0.9), "none": math.log(0.05), "other": math.log(0.05)}   # noqa: E731
    with pytest.raises(ValueError):
        C.choose(box, as_found_and_split(box, page_with((10, 40), (70, 100))), read, params=params)


def hand_params(**alpha):
    """Quality params set by hand (a test fixture): the read's score alone, with offsets per version."""
    return {"intercept": 0.0, "beta": 1.0, "count1": None, "alpha": {"as_found": 0.0, **alpha},
            "gamma": {}, "gamma_across": {}, "kind": {}}


def test_a_lone_join_of_equal_quality_does_not_beat_as_found_and_its_near_copies():
    box = make_box(10, 50)
    cands = C.candidates(box, page_with((10, 50)), [make_box(56, 90)], WIDTHS)
    assert [c.name for c in cands] == ["as_found", "join_nearest", "trim_start", "trim_end"]
    read = read_by_span({(10, 50): ("R-1", math.log(0.8)), (10, 90): ("R-1 C", math.log(0.8))},
                        default=math.log(0.75))                           # the trims: near-copies, a little worse
    kind, info = C.choose(box, cands, read, params=hand_params())
    assert kind == "as_found"                                                # equal quality: as found
    assert info["quality"]["join_nearest"] == pytest.approx(info["quality"]["as_found"])
    assert C.choose(box, cands, read, params=hand_params(join_nearest=-0.1))[0] == "as_found"     # lower: as found


def test_near_copies_of_as_found_do_not_split_its_quality_against_a_lone_join():
    rng = np.random.default_rng(5)
    rows = []
    for b in range(1000):             # 55% of the boxes as found is right (the trims nearly), 45% the join; reads mute
        right = rng.random() < 0.55
        w = {"as_found": 1.0, "join_nearest": 0.0, "trim_start": 0.9, "trim_end": 0.9} if right else \
            {"as_found": 0.0, "join_nearest": 1.0, "trim_start": 0.0, "trim_end": 0.0}
        for cand, wt in w.items():
            s = float(rng.normal(-0.5, 0.3))
            rows.append({"box": b, "cand": cand, "score": s, "word_scores": [s], "kind": "zoning", "across": 0.0,
                         "gap": 0.75 if cand == "join_nearest" else 0.0, "weight": wt})
    rows = rows_frame(rows)
    share = (rows["weight"] / rows.groupby("box")["weight"].transform("sum")).groupby(rows["cand"]).mean()
    assert share["join_nearest"] > share["as_found"]          # by shares of each box, the join would win: the defect
    params = C.fit_quality(rows)
    box = make_box(10, 50)
    cands = C.candidates(box, page_with((10, 50)), [make_box(56, 90)], WIDTHS)       # the join at 6 px: 0.75 widths
    read = lambda corners, b: {"A": math.log(0.6), "none": math.log(0.2), "other": math.log(0.2)}   # noqa: E731
    kind, info = C.choose(box, cands, read, params=params)
    q = info["quality"]
    assert kind == "as_found"
    assert q["as_found"] == pytest.approx(0.55, abs=0.05) and q["join_nearest"] == pytest.approx(0.45, abs=0.05)
    _, alone = C.choose(box, cands[:2], read, params=params)       # each version's quality is its own
    assert alone["quality"]["as_found"] == pytest.approx(q["as_found"])
    assert alone["quality"]["join_nearest"] == pytest.approx(q["join_nearest"])


def _spans_reads(rng, table):
    """Stub reads by span: "good" a clear read, "bad" a doubtful one, "mid" between (drawn per box)."""
    draw = {"good": (0.7, 0.99), "mid": (0.3, 0.7), "bad": (0.05, 0.3)}
    return {span: ("w", math.log(rng.uniform(*draw[how]))) for span, how in table.items()}


def _span(corners):
    return round(float(corners[:, 0].min())), round(float(corners[:, 0].max()))


def _labelled_rows(cases, n, seed, count=False):
    """Fit rows for boxes drawn from `cases` {name: (box, neighbours, ink, terra words, reads, counts)}: each version's
    features by the stub reads (and the stub box model's log P(count = 1) when `count`), its weight by ink."""
    rng = np.random.default_rng(seed)
    out, names = [], sorted(cases)
    for b in range(n):
        box, near, ink, words, reads, counts = cases[names[b % len(names)]]
        cands = C.candidates(box, ink, near, WIDTHS)
        read = read_by_span(_spans_reads(rng, reads))
        cnt = _spans_reads(rng, counts)
        count_fn = (lambda corners, bx, cnt=cnt: cnt[_span(corners)][1]) if count else None
        f = C.choice_features(box, cands, read, count_fn=count_fn)
        w = C.label_weights(cands, words, ink)
        out.append(f.assign(box=b, weight=[w[c] for c in f["cand"]]))
    return pd.concat(out, ignore_index=True).assign(held_out=False, answered=False)


def test_a_split_wins_when_both_halves_are_single_words_and_its_quality_is_the_product_of_its_halves():
    box, ink = make_box(10, 100), ink_page((10, 40), (70, 100))
    two = (box, [], ink, [word(10, 40), word(70, 100)],
           {(10, 100): "bad", (10, 40): "good", (70, 100): "good", (18, 100): "bad", (10, 92): "bad"},
           {(10, 100): "bad", (10, 40): "good", (70, 100): "good", (18, 100): "bad", (10, 92): "bad"})
    one = (box, [], ink, [word(10, 100)],                     # one letter-spaced word
           {(10, 100): "good", (10, 40): "bad", (70, 100): "bad", (18, 100): "mid", (10, 92): "mid"},
           {(10, 100): "good", (10, 40): "bad", (70, 100): "bad", (18, 100): "mid", (10, 92): "mid"})
    params = C.fit_quality(_labelled_rows({"two": two, "one": one}, 300, 7, count=True))
    assert params["beta"] > 0 and params["count1"] > 0
    cands = C.candidates(box, ink, [], WIDTHS)
    clear, doubt = math.log(0.9), math.log(0.15)
    for case, want in (("two", "split_gap"), ("one", "as_found")):
        good = {"two": {(10, 40), (70, 100)}, "one": {(10, 100)}}[case]
        read = lambda c, b, good=good: {"w": clear if _span(c) in good else doubt, "none": -9.0,   # noqa: E731
                                        "other": -9.0}
        count_fn = lambda c, b, good=good: clear if _span(c) in good else doubt   # noqa: E731
        kind, info = C.choose(box, cands, read, params=params, count_fn=count_fn)
        assert kind == want, case
    halves = next(c for c in cands if c.name == "split_gap")
    _, info = C.choose(box, cands, lambda c, b: {"w": clear, "none": -9.0, "other": -9.0}, params=params,
                       count_fn=lambda c, b: clear)
    half = _sig(eta_of(params, "split_gap", halves.along, 0.0, "zoning", clear, clear))
    assert info["quality"]["split_gap"] == pytest.approx(half * half)


def test_a_join_wins_only_for_a_word_split_across_two_boxes():
    box, near, ink = make_box(10, 50), [make_box(56, 90)], ink_page((10, 50), (56, 90))
    cut = (box, near, ink, [word(10, 90)],                      # one word, cut in two by our boxes
           {(10, 50): "bad", (10, 90): "good", (18, 50): "bad", (10, 42): "bad"}, {})
    two = (box, near, ink, [word(10, 50), word(56, 90)],        # two words, side by side
           {(10, 50): "good", (10, 90): "bad", (18, 50): "mid", (10, 42): "mid"}, {})
    params = C.fit_quality(_labelled_rows({"cut": cut, "two": two}, 300, 11))
    cands = C.candidates(box, ink, near, WIDTHS)
    read_as = lambda whole: read_by_span(   # noqa: E731
        {(10, 50): ("w", math.log(0.15 if whole else 0.9)), (10, 90): ("w", math.log(0.9 if whole else 0.15))},
        default=math.log(0.4))
    assert C.choose(box, cands, read_as(True), params=params)[0] == "join_nearest"
    assert C.choose(box, cands, read_as(False), params=params)[0] == "as_found"


def test_fit_refuses_held_out_and_answered_rows():
    rows = synthetic_rows(20)
    for flag in ("held_out", "answered"):
        with pytest.raises(ValueError):
            C.fit_quality(rows.assign(**{flag: [False] * (len(rows) - 1) + [True]}))
        with pytest.raises(ValueError):           # a missing column is refused, not assumed False
            C.fit_quality(rows.drop(columns=flag))
    for col in ("weight", "word_scores", "kind"):
        with pytest.raises(ValueError):
            C.fit_quality(rows.drop(columns=col))
    assert C.fit_quality(rows)["boxes"] == 20


def test_a_separated_fit_stays_finite_because_beta_has_the_ridge():
    rows = synthetic_rows(60)
    sep = np.where(rows["weight"] > 0, 0.0, -50.0)                      # the score alone separates the labels
    rows["word_scores"] = [[v / len(w)] * len(w) for v, w in zip(sep, rows["word_scores"])]
    rows["score"] = sep
    p = C.fit_quality(rows)
    assert np.isfinite(p["beta"]) and abs(p["beta"]) < 1e3


def test_a_candidate_pickles_and_its_features_go_to_a_table():
    import io
    import pickle
    box = make_box(10, 100)
    cands = C.candidates(box, page_with((10, 40), (70, 100)), [make_box(106, 130)], WIDTHS)
    back = pickle.loads(pickle.dumps(cands))
    assert [(c.name, c.along, c.across) for c in back] == [(c.name, c.along, c.across) for c in cands]
    read = lambda corners, b: {"A": math.log(0.9), "none": math.log(0.05), "other": math.log(0.05)}   # noqa: E731
    f = C.choice_features(box, back, read)
    again = pd.read_parquet(io.BytesIO(f.to_parquet()))
    assert (again["gap"] > 0).any() and [len(w) for w in again["word_scores"]] == [len(c.rects) for c in cands]
    assert C.fit_quality(pd.concat([again.assign(box=0, weight=0.5)] * 3, ignore_index=True)
                         .assign(held_out=False, answered=False))["versions"] == 3 * len(cands)


def test_a_next_line_box_has_lower_join_quality_than_a_same_line_box_at_the_same_distance():
    rng = np.random.default_rng(1)
    rows = []
    for b in range(500):          # joins are right only on the same line (across 0); distances are drawn alike
        d = rng.uniform(0.2, 3.0)
        same = rng.random() < 0.5
        a, j = rng.normal(-0.5, 0.2), rng.normal(-0.5, 0.2)
        rows += [{"box": b, "cand": "as_found", "score": a, "word_scores": [a], "gap": 0.0, "across": 0.0,
                  "kind": "zoning", "weight": float(not same)},
                 {"box": b, "cand": "join_nearest", "score": j, "word_scores": [j], "gap": d if same else 0.0,
                  "across": 0.0 if same else d / 2, "kind": "zoning", "weight": float(same)}]
    params = C.fit_quality(rows_frame(rows))
    assert params["gamma_across"]["join_nearest"] < params["gamma"]["join_nearest"]
    box = make_box(10, 50)
    read = lambda corners, b: {"A": math.log(0.8), "none": math.log(0.1), "other": math.log(0.1)}   # noqa: E731
    ink = page_with((10, 50))
    same_line = C.candidates(box, ink, [make_box(56, 90)], WIDTHS)               # 6 px along, 0 across
    next_line = C.candidates(box, ink, [make_box(10, 50, 36, 46)], WIDTHS)       # 0 along, 6 px across
    q_same = C.choose(box, same_line, read, params=params)[1]["quality"]["join_nearest"]
    q_next = C.choose(box, next_line, read, params=params)[1]["quality"]["join_nearest"]
    assert q_next < q_same


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
