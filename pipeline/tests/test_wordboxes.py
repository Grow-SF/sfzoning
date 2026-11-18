"""The word boxes (`solver/wordboxes.py`) on small synthetic chance and ink pages."""
import cv2
import numpy as np
import pandas as pd

from sfzmaps.solver import paths, wordboxes as W


def _page(h=120, w=200):
    return np.zeros((h, w), np.uint8), np.zeros((h, w), bool)


def _bar(ink, x0, y0, x1, y1):
    ink[y0:y1, x0:x1] = True


def _group(chance, ink, level=0.5, kind_p=None, min_px=20):
    return W.group(chance, ink, W.ink_gate(ink), level, kind_p, min_px)


def _spread(ink, value=255):
    """The detector's chance: `value` on the ink and within 3 px of it (it spreads past the ink)."""
    return np.where(W.ink_gate(ink, 3), value, 0).astype(np.uint8)


def test_two_words_are_two_boxes():
    chance, ink = _page()
    _bar(ink, 10, 20, 60, 30)
    _bar(ink, 100, 20, 150, 30)          # 40 px of paper between them: more than twice the ink radius
    G = _group(_spread(ink), ink)
    assert len(G["g"]) == 2
    assert sorted(np.round(G["g"].mean(1)[:, 0])) == [35, 125]


def test_one_word_whose_gaps_are_under_the_ink_radius_is_one_box():
    chance, ink = _page()
    _bar(ink, 10, 20, 30, 30)
    _bar(ink, 34, 20, 54, 30)            # a 4 px gap, as between letters
    assert len(_group(_spread(ink), ink)["g"]) == 1


def test_axis_is_modulo_180_counter_clockwise_from_left_to_right():
    for angle in (0.0, 30.0, 90.0, 150.0):
        chance, ink = _page(200, 200)
        r = np.deg2rad(angle)
        for t in np.linspace(-40, 40, 400):
            x, y = 100 + t * np.cos(r), 100 - t * np.sin(r)        # y grows downwards: counter-clockwise is up
            ink[int(round(y)) - 2:int(round(y)) + 3, int(round(x)) - 2:int(round(x)) + 3] = True
        G = _group(_spread(ink), ink)
        assert len(G["g"]) == 1
        assert 0 <= G["axis"][0] < 180
        assert W.axis_error(G["axis"][0], angle) < 3, (angle, G["axis"][0])
        assert G["length"][0] > G["height"][0]


def test_trimmed_box_is_the_ink_and_the_grouped_box_runs_past_it():
    chance, ink = _page()
    _bar(ink, 50, 40, 110, 52)            # 60 x 12 ink px
    G = _group(_spread(ink), ink)
    assert abs(G["t_length"][0] - 60) < 1.5 and abs(G["t_height"][0] - 12) < 1.5
    assert G["length"][0] > 64 and G["height"][0] > 16
    assert G["t"].shape == (1, 4, 2)
    assert (G["t"].min(1) >= G["g"].min(1) - 1e-6).all() and (G["t"].max(1) <= G["g"].max(1) + 1e-6).all()


def test_specks_under_the_ink_count_are_dropped_and_blank_paper_makes_no_box():
    chance, ink = _page()
    _bar(ink, 10, 10, 13, 13)             # 9 ink px
    chance[60:90, 60:140] = 255            # chance on blank paper, no ink within the radius
    assert len(_group(chance, ink)["g"]) == 0
    _bar(ink, 100, 20, 140, 30)
    assert len(_group(_spread(ink), ink)["g"]) == 1


def test_level_is_a_threshold_on_the_chance():
    chance, ink = _page()
    _bar(ink, 50, 40, 110, 52)
    c = _spread(ink, 200)                  # chance 0.78 everywhere it is called text
    assert len(_group(c, ink, level=0.7)["g"]) == 1
    assert len(_group(c, ink, level=0.9)["g"]) == 0


def test_kind_and_chance_are_means_over_the_boxes_ink():
    chance, ink = _page()
    _bar(ink, 50, 40, 110, 52)
    p = np.zeros((4, *ink.shape), np.float16)
    p[0] = 0.75
    p[2] = 0.25
    p[0][~ink] = 0.0                       # off the ink the detector gives nothing: it must not dilute the mean
    G = _group(_spread(ink, 230), ink, kind_p=p)
    assert np.allclose(G["kind"][0], [0.75, 0, 0.25, 0], atol=1e-3)
    assert abs(G["mean"][0] - 230 / 255) < 1e-6
    assert G["ink_px"][0] == 60 * 12


def test_a_box_found_by_a_word_at_its_own_angle():
    word = np.array([[[0, 0], [40, 0], [40, 10], [0, 10]]], float)
    same = cv2.boxPoints(((20.0, 5.0), (40.0, 10.0), 0.0))[None]
    best, err = W.best_axis_match(word, np.array([0.0]), same, np.array([180.0]))
    assert best[0] > 0.99 and err[0] < 1e-9                  # 180 degrees is 0 modulo 180
    far = word + [100, 0]
    best, err = W.best_axis_match(word, np.array([0.0]), far, np.array([0.0]))
    assert best[0] == 0 and np.isnan(err[0])
    half = word + [20, 0]
    best, _ = W.best_axis_match(word, np.array([0.0]), half, np.array([90.0]))
    assert abs(best[0] - 1 / 3) < 1e-6


def _words(quarter, best):
    return pd.DataFrame({"quarter": quarter, "best": best, "err": 0.0, "tbest": best})


def test_a_quarters_level_is_chosen_on_the_other_three_quarters_only():
    names = {W.level_name(lv): lv for lv in W.LEVELS}
    low, high = W.level_name(W.LEVELS[0]), W.level_name(W.LEVELS[-1])
    words = {n: _words([0, 1, 2, 3] * 5, [0.5] * 20) for n in names}
    # level `low` finds quarter 0's words only; `high` finds every other quarter's words only
    words[low] = _words([0, 1, 2, 3] * 5, [0.9 if q == 0 else 0.0 for q in [0, 1, 2, 3] * 5])
    words[high] = _words([0, 1, 2, 3] * 5, [0.0 if q == 0 else 0.9 for q in [0, 1, 2, 3] * 5])
    chosen = W.choose_levels(words)
    assert chosen["level"]["nw"] == W.LEVELS[-1]             # judged on ne, sw, se: `high` finds them
    assert chosen["recall_on_others"]["ne"][high] == (10 / 15)       # nw's words are not found by `high`
    # quarter 0's own words never decide its level
    assert chosen["recall_on_others"]["nw"][low] == 0.0
    assert chosen["recall_on_others"]["ne"][low] == (5 / 15)


def test_ties_go_to_the_higher_level():
    words = {W.level_name(lv): _words([0, 1, 2, 3], [0.5] * 4) for lv in W.LEVELS}
    assert set(W.choose_levels(words)["level"].values()) == {W.LEVELS[-1]}


def test_each_box_is_kept_at_its_quarters_level():
    def boxes(level, quarters):
        return pd.DataFrame({"level": level, "quarter": np.array(quarters, np.int8), "i": range(len(quarters))})
    lo, hi = W.LEVELS[0], W.LEVELS[-1]
    per = {W.level_name(lv): boxes(lv, [0, 1, 2, 3]) for lv in W.LEVELS}
    kept = W.keep_chosen(per, {"nw": lo, "ne": hi, "sw": lo, "se": hi})
    assert sorted(zip(kept["quarter"], kept["level"])) == [(0, lo), (1, hi), (2, lo), (3, hi)]


def test_quarter_measures_count_a_quarters_own_words_at_its_level():
    lv = {q: W.LEVELS[0] for q in paths.QUARTERS}
    words = {W.level_name(l): _words([0, 0, 1, 2], [0.9, 0.0, 0.9, 0.9]) for l in W.LEVELS}
    m = W.quarter_measures(words, lv)
    assert m["nw"]["words"] == 2 and m["nw"]["recall"] == 0.5
    assert m["se"]["words"] == 0 and m["se"]["recall"] is None
