"""The line rules of the text detector's measures, checked on exact-truth cases."""
import cv2
import numpy as np

from sfzmaps.solver.codereader import labels as codereader_labels
from sfzmaps.solver.textnet import linerules


def dilate(mask, r):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(mask.view(np.uint8), k) > 0


def test_the_copy_matches_codereaders_run_rule():
    rng = np.random.default_rng(0)
    ink = rng.random((120, 160)) < 0.2
    ink[60, :] = True
    assert (linerules.run_lines(ink, 40) == codereader_labels.run_lines(ink, 40)).all()
    assert linerules.run_px(3.98) == codereader_labels.run_px(3.98)


def test_a_line_through_a_word_is_line_and_the_letters_are_not():
    c = linerules.truth_case("line through the middle", "RM-2", cap_px=27, seed=0)
    found = linerules.line_px(c["ink"], ft_per_px=3.98)
    line = c["line"] & ~c["glyph"]
    assert (found & line).sum() / line.sum() >= 0.95
    far = c["glyph"] & ~dilate(c["line"], 1)
    assert (found & far).sum() / far.sum() <= 0.02


def test_rule_accuracy_reports_every_case_and_text():
    acc = linerules.rule_accuracy(linerules.line_px, texts=("RM-2", "P"), caps=(27,), seeds=range(1))
    assert set(acc) == set(linerules.TRUTH_CASES)
    assert all({"line_recall", "glyph_false_line"} <= set(v["RM-2"]) for v in acc.values())


def test_every_case_has_line_ink_and_glyph_ink_and_is_seeded():
    for name in linerules.TRUTH_CASES:
        a = linerules.truth_case(name, "NC-3", cap_px=20, seed=1)
        b = linerules.truth_case(name, "NC-3", cap_px=20, seed=1)
        assert a["line"].sum() > 0 and a["glyph"].sum() > 0
        assert (a["ink"] == (a["glyph"] | a["line"])).all()
        assert (a["ink"] == b["ink"]).all()


def test_chain_lines_follow_a_bent_line_but_not_short_pieces():
    ink = np.zeros((200, 200), np.uint8)
    cv2.line(ink, (0, 50), (60, 50), 1, 3)       # two 60 px runs meeting at a corner
    cv2.line(ink, (60, 50), (60, 110), 1, 3)
    cv2.line(ink, (120, 150), (135, 150), 1, 3)  # a 15 px piece
    got = linerules.chain_lines(ink.view(bool), 80, 40)
    assert got[50, 30] and got[80, 60] and not got[150, 128]
