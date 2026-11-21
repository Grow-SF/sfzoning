import numpy as np
import pytest

from sfzmaps.frame import scores


def test_sheet_scores_counts_kept_and_cut_ink():
    inside = np.zeros((10, 10), bool); inside[2:8, 2:8] = True
    outside = ~inside; outside[2, 2:8] = True; outside[0, 0] = False      # cut a map row; keep one frame px
    lum = np.zeros((10, 10), np.float32)
    cls = np.where(inside, 0, 1).astype(np.uint8)
    p = np.stack([1 - outside, outside, 0 * outside]).astype(float)
    s = scores.sheet_scores(p, outside, cls, lum)
    assert s["map_cut"] == 6 and s["frame_kept"] == 1 and s["cost"] == 3 * 1 + 6


def test_removed_block_left_in_counts_as_frame_kept():
    cls = np.zeros((4, 4), np.uint8); cls[0] = 2
    p = np.stack([np.ones((4, 4)), 0 * np.ones((4, 4)), 0 * np.ones((4, 4))])
    s = scores.sheet_scores(p, np.zeros((4, 4), bool), cls, np.zeros((4, 4)))
    assert s["frame_kept"] == 4 and s["frame_kept_share"] == 1.0 and s["map_cut"] == 0


def test_shares_are_nan_for_a_class_with_no_ink():
    cls = np.zeros((4, 4), np.uint8)
    p = np.stack([np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))])
    s = scores.sheet_scores(p, np.zeros((4, 4), bool), cls, np.zeros((4, 4)))
    assert np.isnan(s["frame_kept_share"]) and s["map_cut_share"] == 0.0


def test_log_loss_of_a_constant_quarter():
    cls = np.ones((4, 4), np.uint8); cls[:2] = 0
    q = np.full((4, 4), 0.25)
    p = np.stack([1 - q, q, 0 * q])
    s = scores.sheet_scores(p, np.zeros((4, 4), bool), cls, np.zeros((4, 4)))
    assert s["log_loss"] == pytest.approx(-(np.log(0.75) + np.log(0.25)) / 2)


def test_log_loss_weighs_by_ink():
    cls = np.zeros((2, 2), np.uint8); cls[1] = 1
    p = np.stack([np.full((2, 2), 0.5), np.full((2, 2), 0.5), np.zeros((2, 2))])
    lum = np.array([[1.0, 1.0], [0.0, 0.0]])             # only the class-1 row has ink
    assert scores.sheet_scores(p, np.zeros((2, 2), bool), cls, lum)["log_loss"] == pytest.approx(np.log(2))


def test_hd95_of_a_shifted_square():
    inside = np.zeros((60, 60), bool); inside[10:50, 10:50] = True
    outside = np.ones((60, 60), bool); outside[13:53, 10:50] = False
    cls = np.where(inside, 0, 1).astype(np.uint8)
    p = np.zeros((3, 60, 60))
    assert scores.sheet_scores(p, outside, cls, np.zeros((60, 60)))["hd95_px"] == pytest.approx(3, abs=0.5)


def test_hd95_is_nan_without_a_boundary():
    cls = np.zeros((8, 8), np.uint8)
    s = scores.sheet_scores(np.zeros((3, 8, 8)), np.ones((8, 8), bool), cls, np.zeros((8, 8)))
    assert np.isnan(s["hd95_px"])


def test_bootstrap_mean_drops_nans():
    m, lo, hi = scores.bootstrap_mean([1.0, np.nan, 1.0, 1.0])
    assert (m, lo, hi) == (1.0, 1.0, 1.0)


def test_paired_difference_sign_and_interval():
    a = {("s", str(i)): 1.0 + 0.1 * i for i in range(20)}; b = {k: v - 0.5 for k, v in a.items()}
    m, lo, hi = scores.paired_difference(a, b)
    assert m == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
