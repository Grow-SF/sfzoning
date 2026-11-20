import numpy as np
import pytest

from sfzmaps.frame.data import map_mask
from sfzmaps.frame.scores import BAND_PX, UNSURE_PX, bands, edge_distance, log_loss, sheet_scores, unsure


def _square(n=100, lo=30, hi=70):
    mask = np.zeros((n, n), np.uint8)
    mask[lo:hi, lo:hi] = 1
    return mask


def test_unsure_px_is_the_label_precision():
    assert UNSURE_PX == 8


def test_unsure_strip_is_within_unsure_px_on_both_sides():
    """Inside, the outline px is 1 px from it and px 8 is the last in the strip; outside likewise."""
    mask = _square()
    u = unsure(mask)
    assert u.dtype == bool and u.shape == mask.shape
    row = u[50]
    assert row[30:38].all() and not row[38]           # inside: columns 30..37 are 1..8 px from the outline
    assert row[22:30].all() and not row[21]           # outside: columns 22..29 are 8..1 px from it
    assert not u[50, 45:55].any() and not u[50, :15].any()
    assert u.sum() == (56 * 56 - 24 * 24) - 4 * (8 * 8 - int((np.hypot(*np.mgrid[1:9, 1:9]) <= 8).sum()))


def test_unsure_px_scales():
    mask = _square()
    assert unsure(mask, 1)[50, 29:31].all() and not unsure(mask, 1)[50, [28, 31]].any()


def test_a_holes_outline_gets_the_strip():
    square = [(10, 10), (89, 10), (89, 89), (10, 89)]
    hole = [(40, 40), (59, 40), (59, 59), (40, 59)]
    mask = map_mask([square], (100, 100), holes=[hole])
    u = unsure(mask)
    assert u[50, 39] and u[50, 32] and not u[50, 31]   # map side of the hole's outline
    assert u[50, 40] and u[50, 47] and not u[50, 50]   # inside the hole


def test_no_outline_no_strip():
    assert not unsure(np.ones((20, 30), np.uint8)).any()
    assert not unsure(np.zeros((20, 30), np.uint8)).any()


def test_bands_start_beyond_the_strip():
    mask = _square(200, 70, 130)
    fb, mb = bands(mask)
    u = unsure(mask)
    assert not (fb & u).any() and not (mb & u).any()
    assert fb[100, 70 - UNSURE_PX - 1] and not fb[100, 70 - UNSURE_PX]
    assert fb[100, 70 - BAND_PX] and not fb[100, 70 - BAND_PX - 1]
    assert mb[100, 70 + UNSURE_PX] and not mb[100, 70 + UNSURE_PX - 1]


def test_scores_ignore_the_strip():
    """A confident mistake inside the strip changes no score but the edge distance."""
    mask = _square()
    lum = np.zeros(mask.shape, np.float32)
    good = np.where(mask == 1, 0.99, 0.01).astype(np.float32)
    bad = good.copy()
    u = unsure(mask)
    bad[u] = 1.0 - bad[u]
    a, b = sheet_scores(good, mask, lum), sheet_scores(bad, mask, lum)
    for k in ("log_loss_frame_band", "log_loss_map_band", "log_loss_all_ink", "frame_called_map",
              "map_called_frame"):
        assert a[k] == pytest.approx(b[k], abs=1e-12), k
    bad[50, 50] = 0.01                                  # outside the strip it counts
    assert sheet_scores(bad, mask, lum)["log_loss_all_ink"] > a["log_loss_all_ink"]


def test_perfect_prediction():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1
    lum = np.full(mask.shape, 0.0, np.float32)
    s = sheet_scores(mask.astype(np.float32), mask, lum)
    assert s["log_loss_all_ink"] < 1e-5 and s["frame_called_map"] < 1e-5 and s["edge_hd95_px"] == 0


def test_confident_mistake_costs_more_than_unsure():
    mask = np.zeros((50, 50), np.uint8)
    mask[20:40, 20:40] = 1          # row 5 is 15 px from the outline: in the frame band, beyond the uncertain strip
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
