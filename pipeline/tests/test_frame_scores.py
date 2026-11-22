import math

import numpy as np
import pytest

from sfzmaps.frame import data, scores

N = 40
MAP = [[(5, 5), (34, 5), (34, 34), (5, 34)]]          # the map: px 5..34, outlines included
BOX = [(20, 20), (29, 20), (29, 29), (20, 29)]        # a labelled box inside it: px 20..29


def _page() -> np.ndarray:
    lum = np.ones((N, N), np.float32)
    lum[2, :] = 0.0                                    # a frame rule: 40 px of class-1 ink
    lum[22:28, 22:28] = 0.0                            # the box's ink: 36 px
    return lum


def _outside() -> np.ndarray:
    return data.map_mask(MAP, (N, N)) == 0


def test_split_scores_separates_frame_from_boxes():
    lum = _page()
    removed = np.zeros((N, N), bool)
    removed[20:30, 20:25] = True                       # half the box's ink columns (22..24 of 22..27)
    s = scores.split_scores(_outside(), removed, MAP, [BOX], lum)
    assert s["frame_ink"] == 40 and s["frame_left"] == 0
    assert s["box_ink"] == 36 and s["box_left"] == 18
    (box,) = s["boxes"]
    assert box["deleted_share"] == 0.5 and box["kind"] == "rect"
    assert box["size"] == pytest.approx([9, 9])
    assert s["map_ink"] == 0 and s["map_cut_frame"] == 0 and s["map_cut_boxes"] == 0


def test_map_cut_is_split_by_what_cut_it():
    lum = np.ones((N, N), np.float32)
    lum[8, 10:20] = 0.0                                # 10 px of map ink
    lum[15, 10:14] = 0.0                               # 4 more
    outside = _outside()
    outside[8, 10:15] = True                           # the frame cuts 5 of the first
    removed = np.zeros((N, N), bool)
    removed[15, 10:14] = True                          # a fitted box cuts the 4 ...
    outside[15, 10:12] = True                          # ... 2 of them also outside: they count under boxes only
    s = scores.split_scores(outside, removed, MAP, [], lum)
    assert s["map_ink"] == 14 and s["map_cut_frame"] == 5 and s["map_cut_boxes"] == 4
    shares = scores.pooled_split([s])
    assert shares["map_cut_frame_share"] == pytest.approx(5 / 14) and shares["map_cut_boxes_share"] == pytest.approx(4 / 14)


def test_sheet_without_boxes_is_scored():
    lum = _page()
    lum[22:28, 22:28] = 1.0                            # no box ink either
    lum[10, 10:20] = 0.0                               # some map ink, kept
    plain = scores.split_scores(_outside(), np.zeros((N, N), bool), MAP, [], lum)
    assert plain["box_ink"] == 0 and plain["box_left"] == 0 and plain["boxes"] == []
    one = scores.pooled_split([plain])
    assert one["frame_left_share"] == 0 and one["map_cut_frame_share"] == 0 and one["map_cut_boxes_share"] == 0
    assert math.isnan(one["box_left_share"]) and scores.percent(one["box_left_share"]) == "n/a"
    boxed = scores.split_scores(_outside(), np.zeros((N, N), bool), MAP, [BOX], _page())   # box all left in
    pooled = scores.pooled_split([plain, boxed])
    assert all(math.isfinite(v) for v in pooled.values())
    assert pooled["box_left_share"] == 1.0
    assert pooled["frame_left_share"] == 0.0           # summed ink over summed ink: 0 of 80


def test_pooled_split_sums_ink_not_shares():
    small = {"frame_ink": 10.0, "frame_left": 10.0, "box_ink": 0.0, "box_left": 0.0, "map_ink": 100.0,
             "map_cut_frame": 0.0, "map_cut_boxes": 0.0, "boxes": []}
    big = {**small, "frame_ink": 990.0, "frame_left": 0.0, "map_ink": 900.0, "map_cut_frame": 10.0}
    pooled = scores.pooled_split([small, big])
    assert pooled["frame_left_share"] == pytest.approx(0.01)        # 10 of 1000, not the mean of 1 and 0
    assert pooled["map_cut_frame_share"] == pytest.approx(0.01)


def test_frameless_sheet_whose_only_deletions_are_boxes_is_scored():
    lum = _page()
    lum[2, :] = 1.0                                    # no frame rule: the map is the whole page
    whole = [[(0, 0), (N - 1, 0), (N - 1, N - 1), (0, N - 1)]]
    removed = np.zeros((N, N), bool)
    removed[20:30, 20:30] = True                       # the box, all deleted
    s = scores.split_scores(np.zeros((N, N), bool), removed, whole, [BOX], lum)
    assert s["frame_ink"] == 0 and s["box_ink"] == 36 and s["box_left"] == 0 and s["boxes"][0]["deleted_share"] == 1
    one = scores.pooled_split([s])
    assert math.isnan(one["frame_left_share"]) and one["box_left_share"] == 0
    assert scores.pooled_split([s, scores.split_scores(_outside(), np.zeros((N, N), bool), MAP, [], _page())])[
        "frame_left_share"] == 0
