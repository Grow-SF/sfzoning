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


def test_join_takes_the_nearest_box_within_one_word_height_along_the_axis():
    box = make_box(10, 50)                      # height 10
    near, far, off = make_box(56, 90), make_box(70, 100), make_box(54, 90, 40, 50)
    c = by_kind(C.candidates(box, page_with((10, 50)), [far, near, off], WIDTHS))
    assert c["join_nearest"][0][:, 0].min() == pytest.approx(10)
    assert c["join_nearest"][0][:, 0].max() == pytest.approx(90)
    assert "join_nearest" not in by_kind(C.candidates(box, page_with((10, 50)), [far, off], WIDTHS))


def test_trim_cuts_the_kinds_median_character_width_from_either_end():
    c = by_kind(C.candidates(make_box(10, 90, kind="block"), page_with((10, 90)), [], WIDTHS))
    assert c["trim_start"][0][:, 0].min() == pytest.approx(16) and c["trim_start"][0][:, 0].max() == pytest.approx(90)
    assert c["trim_end"][0][:, 0].max() == pytest.approx(84) and c["trim_end"][0][:, 0].min() == pytest.approx(10)


def test_no_trim_of_a_box_not_longer_than_two_characters():
    c = by_kind(C.candidates(make_box(10, 24), page_with((10, 24)), [], WIDTHS))
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
    assert C.choose(box, [c for c in cands if c[0] != "split_gap"], flat)[0] == "as_found"


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
    cands = [(k, c) for k, c in C.candidates(box, page_with((10, 50)), [], WIDTHS) if k in ("as_found", "trim_end")]

    def read(corners, b):
        return {"P": math.log(0.6), "Q": math.log(0.3), "none": math.log(0.05), "other": math.log(0.05)}

    assert C.choose(box, cands, read)[0] == "as_found"
    prior = lambda s, b: -20.0 if s == "P" else 0.0   # noqa: E731
    _, info = C.choose(box, cands, read, prior)
    want = math.log(0.3) - math.log(0.3 + 0.05 + 0.05 + 0.6 * math.exp(-20))
    assert info["scores"]["as_found"] == pytest.approx(want, abs=1e-9)
    assert info["words"] == ["Q"]


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
