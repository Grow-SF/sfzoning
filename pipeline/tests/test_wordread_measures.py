"""The end-to-end measures (`wordread.measures`) on small synthetic boxes, reads and answers."""
import pandas as pd

from sfzmaps.solver.wordread import measures


def _rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _boxes(*rects, quarter=0):
    rows = [{"sheet": "s", "quarter": quarter, **dict(zip(measures.G_COLS, _rect(*r)))} for r in rects]
    return pd.DataFrame(rows)


def _terra(*rects, quarter=0):
    rows = [{"sheet": "s", "text": "R-1", "kind": "zoning code", "quarter": quarter,
             **dict(zip(measures.C_COLS, _rect(*r)))} for r in rects]
    return pd.DataFrame(rows)


def _read(code, p=0.9, top=None):
    other = {c: (1 - p) for c in ("A", "B") if c != code}
    return {"closed": {code: p, **other}, "top": top or code}


def _answers(*points, truth="A"):
    return pd.DataFrame([{"sheet": "s", "point_x": x, "point_y": y, "truth": truth, "quarter": 0, "block": f"b{k}"}
                         for k, (x, y) in enumerate(points)])


BOXES_ONE = _boxes((0, 0, 100, 10))
READS_ONE = {0: _read("A")}
ANSWERS_TWO_ONE_UNBOXED = _answers((50, 5), (5000, 5000))

BOXES_OVERLAP = _boxes((0, 0, 100, 10), (20, 0, 120, 10))        # the point (60, 5) is in both; the first's centre is nearer
READS_OVERLAP_SECOND_RIGHT = {0: _read("B", 0.6), 1: _read("A", 0.95)}
ANSWERS_ONE = _answers((60, 5))

TERRA_ONE = _terra((0, 0, 100, 10))
BOXES_SPLIT = _boxes((0, 0, 45, 10), (55, 0, 100, 10))
READS_SPLIT = {0: _read("A"), 1: _read("A")}

TERRA_TWO = _terra((0, 0, 100, 10), (100, 0, 200, 10))
BOXES_MERGED = _boxes((0, 0, 150, 10))
READS_MERGED = {0: _read("A")}

ARM = pd.DataFrame({"right": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], "block": ["a", "a", "b", "b", "c", "c"]})


def test_a_label_with_no_box_counts_wrong():
    r = measures.headline(BOXES_ONE, READS_ONE, ANSWERS_TWO_ONE_UNBOXED, ("ne",))
    assert r["n"] == 2 and r["right"] == 1 and r["with_box"] == 1


def test_a_point_in_two_boxes_goes_to_the_higher_posterior_box():
    r = measures.headline(BOXES_OVERLAP, READS_OVERLAP_SECOND_RIGHT, ANSWERS_ONE, ("ne",))
    assert r["right"] == 1


def test_a_terra_word_split_over_two_boxes_is_missed_with_reason_split():
    b = measures.box_numbers(BOXES_SPLIT, TERRA_ONE, READS_SPLIT)
    assert b[0.5]["missed"] == 1 and b[0.5]["missed_reasons"]["split"] == 1


def test_a_merged_box_matches_one_word_and_misses_the_other():
    b = measures.box_numbers(BOXES_MERGED, TERRA_TWO, READS_MERGED)
    assert b[0.5]["matched"] == 1 and b[0.5]["missed_reasons"]["merged"] == 1


def test_paired_labels_of_identical_arms_is_zero():
    p = measures.paired_labels(ARM, ARM, boot=50)
    assert p["diff"] == 0 and p["lo"] == 0 == p["hi"]


def test_a_reason_function_replaces_the_geometric_reasons():
    b = measures.box_numbers(BOXES_SPLIT, TERRA_ONE, READS_SPLIT, reason_fn=lambda t, ti, bj, iou, best: "below_level")
    assert b[0.5]["missed_reasons"] == {"below_level": 1}


def test_unmatched_boxes_are_counted_by_what_they_read_as():
    boxes = _boxes((0, 0, 100, 10), (300, 0, 400, 10), (500, 0, 600, 10))
    reads = {0: _read("A"), 1: _read("A"), 2: _read("A", top="none")}
    b = measures.box_numbers(boxes, TERRA_ONE, reads)
    assert b[0.5]["matched"] == 1 and b[0.5]["ours_unmatched_text"] == 1 and b[0.5]["ours_unmatched_none"] == 1


def test_a_box_at_iou_point_four_matches_only_at_the_looser_level():
    b = measures.box_numbers(_boxes((0, 0, 40, 10)), TERRA_ONE, {0: _read("A")})
    assert b[0.5]["matched"] == 0 and b[0.3]["matched"] == 1


def test_read_numbers_count_same_and_different_codes_on_matched_boxes():
    terra = _terra((0, 0, 100, 10), (200, 0, 300, 10))
    terra.loc[1, "text"] = "B"
    boxes = _boxes((0, 0, 100, 10), (200, 0, 300, 10))
    r = measures.read_numbers(boxes, terra, {0: _read("R-1"), 1: _read("A")})
    assert r["same_as_terra"] == 1 and r["differs"] == 1


def test_compare_is_now_minus_before():
    d = measures.compare({"a": {"n": 5, "x": "s"}, "b": 1.5}, {"a": {"n": 3}, "b": 1.0, "c": 9})
    assert d == {"a": {"n": 2}, "b": 0.5}
