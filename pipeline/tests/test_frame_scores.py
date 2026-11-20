import numpy as np
import pytest

from sfzmaps.frame.scores import bands, edge_distance, log_loss, sheet_scores


def test_perfect_prediction():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1
    lum = np.full(mask.shape, 0.0, np.float32)
    s = sheet_scores(mask.astype(np.float32), mask, lum)
    assert s["log_loss_all_ink"] < 1e-5 and s["frame_called_map"] < 1e-5 and s["edge_hd95_px"] == 0


def test_confident_mistake_costs_more_than_unsure():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 1
    lum = np.zeros(mask.shape, np.float32)
    wrong, unsure = mask.astype(np.float32), mask.astype(np.float32)
    wrong[5, 5:45] = 0.99
    unsure[5, 5:45] = 0.5
    fb, _ = bands(mask)
    w = 1 - lum
    assert (
        log_loss(wrong, mask, w, fb)
        > log_loss(unsure, mask, w, fb)
        > log_loss(mask.astype(np.float32), mask, w, fb)
    )


def test_edge_distance_of_a_shift():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1
    p = np.zeros((100, 100), np.float32)
    p[23:83, 23:83] = 1  # a diagonal shift: no edge stays put, so the median is unambiguous
    assert edge_distance(p, mask)["median"] == pytest.approx(3, abs=1)


def test_edge_distance_empty_boundary_is_inf():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 1
    d = edge_distance(np.full(mask.shape, 0.2, np.float32), mask)
    assert d["median"] == float("inf") and d["hd95"] == float("inf")


def test_paper_does_not_count():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1
    p = mask.astype(np.float32)
    p[5:10, :] = 1.0  # wrong: a strip of frame called map
    lum = np.zeros(mask.shape, np.float32)
    lum[5:10, :] = 1.0  # but it is paper
    s = sheet_scores(p, mask, lum)
    assert s["frame_called_map"] == 0
    lum[5:10, :] = 0.0
    assert sheet_scores(p, mask, lum)["frame_called_map"] > 0
