"""The counts a read is judged by (`find.counts`): each printed block found, joined, cut up or lost (exactly one of
them), open land left out, the depth fitted on the choosing crops, how far found block edges lie from the hand ones,
and the zoning-border statements agreed with, each read by its own statement's chance."""
from pathlib import Path

import numpy as np
import pytest

from sfzmaps.find import counts, handlabels
from sfzmaps.find.cuts import BORDER, EDGE, Cuts
from sfzmaps.find.handlabels import CropLabels


def _truth(blocks: dict[int, tuple[slice, slice]], shape, kinds=None) -> CropLabels:
    area = np.zeros(shape, np.int32)
    for i, (r, c) in blocks.items():
        area[r, c] = i
    kinds = kinds or {i: "block" for i in blocks}
    street = (area == 0).astype(np.uint8)
    return CropLabels(street, area, kinds, np.zeros(shape, np.uint8), [])


SHAPE = (60, 80)
C = (slice(5, 25), slice(5, 35))        # found: 600 px, its area also holds B
B = (slice(5, 25), slice(37, 42))       # joined: 100 px, all in C's area, which is mostly C
D = (slice(30, 50), slice(5, 35))       # cut up: three areas of 200 px
E = (slice(30, 50), slice(45, 61))      # lost: no area
O = (slice(52, 58), slice(40, 70))      # open land


def _found() -> np.ndarray:
    found = np.zeros(SHAPE, np.int32)
    found[5:25, 5:42] = 1                       # C, the street between, and B
    for k, c0 in enumerate((5, 15, 25)):
        found[30:50, c0:c0 + 10] = 2 + k
    found[52:58, 5:31] = 5                      # on hand street: an extra area
    found[52:58, 40:70] = 6                     # on the open land
    return found


def test_each_block_is_exactly_one_thing():
    truth = _truth({1: C, 2: B, 3: D, 4: E}, SHAPE)
    got = counts.blocks(_found(), truth)
    assert counts.block_status(_found(), truth) == {1: "found", 2: "joined", 3: "cut_up", 4: "lost"}
    assert (got["found"], got["joined"], got["cut_up"], got["lost"]) == (1, 1, 1, 1)
    assert got["found"] + got["joined"] + got["cut_up"] + got["lost"] == got["blocks"] == 4
    assert got["extra"] == 2                    # the area on hand street, and the one on O, street in this truth


def test_open_land_is_not_counted():
    with_open = _truth({1: C, 2: B, 3: D, 4: E, 5: O}, SHAPE, {1: "block", 2: "block", 3: "block", 4: "block",
                                                               5: "open"})
    without = _truth({1: C, 2: B, 3: D, 4: E}, SHAPE)
    without.street[O] = handlabels.NONE         # the same ground with no label at all
    got = counts.blocks(_found(), with_open)
    assert got == counts.blocks(_found(), without)
    assert got["blocks"] == 4 and got["extra"] == 1
    assert 5 not in counts.block_status(_found(), with_open)


def test_found_share_is_the_mean_over_eras():
    by_era = {"a": [{"found": 1, "blocks": 1}, {"found": 0, "blocks": 3}], "b": [{"found": 3, "blocks": 3}]}
    assert counts.found_share(by_era) == pytest.approx((1 / 4 + 1) / 2)
    assert counts.era_shares(by_era) == {"a": 0.25, "b": 1.0}


H, W, RIDGE = 40, 60, 30


def test_fit_depth_takes_the_smallest_best():
    # the faint-stretch case (a column of block edge at 0.9 dipping to 0.6 between two blocks), the left block made of
    # three parts parted by two low bumps of street chance 25/255, floors 10/255, 12.5/255 and 12.5/255: below 12.5/255
    # it is cut up; the right block's floor 50.5/255 keeps it apart up to 0.6 - 50.5/255 = 102.5/255
    street = np.ones((H, W))
    street[2:-2, 2:RIDGE] = 12.5 / 255
    street[2:-2, 2:11] = 10 / 255
    street[2:-2, [11, 20]] = 25 / 255
    street[2:-2, RIDGE + 1:-2] = 50.5 / 255
    street[2:-2, RIDGE] = 0.0
    edge = np.zeros((H, W))
    edge[2:-2, RIDGE] = 0.9
    edge[18:21, RIDGE] = 0.6
    border = np.zeros((H, W))
    read = (np.stack([street, edge, border]).astype(np.float32), np.ones((H, W), bool))
    truth = _truth({1: (slice(2, H - 2), slice(2, RIDGE)), 2: (slice(2, H - 2), slice(RIDGE + 1, W - 2))}, (H, W))
    depth, share = counts.fit_depth([read], [truth], ["lined-grey"])
    assert depth == 13 / 255
    assert share == 1.0


def test_edge_feet_measure_found_blocks_only():
    shape = (60, 50)
    truth = _truth({1: (slice(10, 30), slice(10, 40)), 2: (slice(40, 55), slice(5, 45))}, shape)
    edge = np.zeros(shape, np.uint8)
    for i in (1, 2):
        a = truth.area == i
        inner = np.zeros_like(a)
        inner[1:-1, 1:-1] = a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:]
        edge[a & ~inner] = 1
    truth = CropLabels(truth.street, truth.area, truth.kinds, edge, [])
    found = np.zeros(shape, np.int32)
    found[8:32, 8:42] = 1                       # the first block grown 2 px every way; the second lost
    got = counts.edge_feet(found, truth, 3.0)
    assert counts.block_status(found, truth) == {1: "found", 2: "lost"}
    assert got["median"] == 6.0 and got["p90"] == 6.0
    assert got["px"] == int(edge[truth.area == 1].sum())


def _cuts(lines, width) -> Cuts:
    """Cuts along row 0 of an item `width` px wide: each (start column, px, answer, says, weight)."""
    px, cut = [], []
    for i, (c0, n, *_rest) in enumerate(lines):
        px.append(np.arange(c0, c0 + n))
        cut.append(np.full(n, i, np.int32))
    m = len(lines)
    return Cuts(np.concatenate(px).astype(np.int64), np.concatenate(cut), np.ones(sum(len(p) for p in px), np.float32),
                np.array([a for *_x, a, _s, _w in lines], np.uint8), np.array([s for *_x, s, _w in lines], bool),
                np.array([w for *_x, w in lines], np.float32), np.full((m, 2), -1, np.int32), np.zeros(0, np.int64))


def test_border_counts_are_per_cut():
    chance = np.zeros((1, 40), np.float32)
    chance[0, 0:5] = 0.3            # "crosses": 1 - 0.7^5 = 0.83, agreed though no px is over one half
    chance[0, 6:9] = 0.1            # "crosses": 1 - 0.9^3 = 0.27, not agreed
    chance[0, 10:13] = 0.3          # "none": 0.7^3 = 0.34, not agreed though no px is over one half
    chance[0, 14:16] = 0.1          # "none": 0.9^2 = 0.81, agreed
    chance[0, 17:19] = (0.0, 1.0)   # "none" with a px at 1 as stored: not agreed (and no NaN)
    chance[0, 20:23] = 0.9          # "crosses", but weight 0: not counted
    chance[0, 24:27] = 0.9          # a block-edge cut: not counted
    cuts = _cuts([(0, 5, BORDER, True, 1.0), (6, 3, BORDER, True, 1.0), (10, 3, BORDER, False, 1.0),
                  (14, 2, BORDER, False, 1.0), (17, 2, BORDER, False, 1.0), (20, 3, BORDER, True, 0.0),
                  (24, 3, EDGE, True, 1.0)], 40)
    assert counts.borders(chance, cuts) == {"crosses": 2, "crosses_agreed": 1, "none": 3, "none_agreed": 1}


def test_cuts_in_a_box_are_those_wholly_inside_it():
    width = 20
    # cut 0 inside the box (rows 2-5, columns 3-9), cut 1 half out of it, cut 2 a zoning-border cut unused
    px = np.array([3 * width + 4, 3 * width + 5, 4 * width + 9, 4 * width + 10, 5 * width + 6], np.int64)
    cuts = Cuts(px, np.array([0, 0, 1, 1, 2], np.int32), np.ones(5, np.float32), np.array([BORDER] * 3, np.uint8),
                np.array([True, True, False]), np.array([1, 1, 0], np.float32), np.full((3, 2), -1, np.int32),
                np.zeros(0, np.int64))
    got = counts.cuts_in_box(cuts, width, (3, 2, 7, 4))
    assert list(got.cut) == [0, 0] and list(got.px) == [1 * 7 + 1, 1 * 7 + 2] and len(got.answer) == 1


def _crop(set_: str) -> handlabels.Crop:
    return handlabels.Crop(set_, "s", "p1", (0, 0, 10, 10), ((1, 1),), (), (), (), (), Path("s/p1__0_0_10_10.json"))


def test_the_chooser_takes_choosing_crops_only():
    with pytest.raises(ValueError, match="test"):
        counts.chooser([_crop("choose"), _crop("test")], "cpu")
    with pytest.raises(ValueError, match="no choose crop"):
        counts.chooser([], "cpu")


# --- the street and block boundary, whole blocks, the block edge on zoning-border cuts --------------------------------

def test_ground_counts_street_and_area_px():
    truth = _truth({1: (slice(0, 10), slice(0, 20))}, (10, 40))     # area on columns 0-19, street on 20-39
    truth.street[0, 39] = handlabels.NONE                           # a px with no label counts as neither
    everything_street = np.ones((10, 40), np.float32)
    assert counts.ground(everything_street, truth) == {"street_px": 199, "street_called_street": 199,
                                                       "area_px": 200, "area_called_not_street": 0}
    half = np.zeros((10, 40), np.float32)
    half[:, 10:30] = 0.6                                            # columns 10-29 called street
    got = counts.ground(half, truth)
    assert got == {"street_px": 199, "street_called_street": 100, "area_px": 200, "area_called_not_street": 100}
    summed = counts.ground_summed([got, counts.ground(everything_street, truth)])
    assert summed["street_share"] == pytest.approx(299 / 398) and summed["area_share"] == pytest.approx(100 / 400)


def _edge_truth(shape, rows) -> CropLabels:
    """No areas; hand block-boundary px (edge 1) along the given rows, every column."""
    edge = np.zeros(shape, np.uint8)
    edge[list(rows)] = 1
    return CropLabels(np.ones(shape, np.uint8), np.zeros(shape, np.int32), {}, edge, [])


def test_boundary_feet_from_each_hand_boundary_px_to_the_nearest_model_edge():
    shape = (30, 10)
    truth = _edge_truth(shape, (5, 15))
    chance = np.zeros(shape, np.float32)
    chance[7] = 0.9                         # the model's edge: 2 px from row 5, 8 px from row 15
    got = counts.boundary_feet(chance, truth, 2.5)
    assert got["px"] == 20 and got["no_model_edge"] is False
    assert got["median"] == pytest.approx(12.5) and got["p90"] == pytest.approx(20.0)
    assert got["p99"] == pytest.approx(20.0)
    nothing = counts.boundary_feet(np.full(shape, 0.4, np.float32), truth, 2.5)      # no px over one half
    assert nothing == {"px": 20, "no_model_edge": True, "median": None, "p90": None, "p99": None}


def test_boundary_feet_pool_every_px_of_an_era_together():
    shape = (30, 10)
    one, two = _edge_truth(shape, (5,)), _edge_truth(shape, (5, 6))
    near, far = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    near[5], far[9] = 1.0, 1.0              # 10 px at 0; 10 px at 4 and 10 at 3
    ds = [counts.boundary_distances(near, one, 1.0), counts.boundary_distances(far, two, 1.0),
          counts.boundary_distances(np.zeros(shape, np.float32), one, 1.0)]
    assert ds[2] is None
    got = counts.boundary_feet_pooled(ds, [10, 20, 10])
    assert (got["px"], got["px_measured"], got["crops_with_no_model_edge"]) == (40, 30, 1)
    assert got["median"] == pytest.approx(3.0) and got["p90"] == pytest.approx(4.0)       # the medians' mean: 1.75


def test_a_block_split_60_40_is_found_but_not_whole():
    shape = (20, 30)
    truth = _truth({1: (slice(5, 15), slice(5, 15)), 2: (slice(5, 15), slice(18, 28))}, shape)
    found = np.zeros(shape, np.int32)
    found[5:15, 5:11], found[5:15, 11:15] = 1, 2         # block 1: 60 px in area 1, 40 in area 2
    found[5:15, 18:28] = 3                               # block 2: exactly area 3
    assert counts.block_status(found, truth) == {1: "found", 2: "found"}
    got = counts.whole(found, truth)
    assert (got["blocks"], got["whole"], got["in_two_or_more"]) == (2, 1, 1)
    assert got["per_block"] == [{"block": 1, "held": 0.6, "whole": False, "pieces": 2},
                                {"block": 2, "held": 1.0, "whole": True, "pieces": 1}]
    summed = counts.whole_summed([got, got])
    assert summed["whole_share"] == 0.5 and summed["median_held"] == pytest.approx(0.8)


def test_everything_one_area_holds_no_block_whole():
    shape = (20, 30)
    truth = _truth({1: (slice(5, 15), slice(5, 15)), 2: (slice(5, 15), slice(18, 28))}, shape)
    got = counts.whole(np.ones(shape, np.int32), truth)                 # each block is a sixth of the one area
    assert (got["blocks"], got["whole"], got["in_two_or_more"]) == (2, 0, 0)
    assert [b["held"] for b in got["per_block"]] == [1.0, 1.0]
    lost = counts.whole(np.zeros(shape, np.int32), truth)
    assert [b["held"] for b in lost["per_block"]] == [0.0, 0.0] and lost["whole"] == 0


def test_the_block_edge_on_zoning_border_cuts_inside_a_block_and_across_a_street():
    width = 40
    street = np.zeros((1, width), np.uint8)
    street[0, 10:14] = 1                    # the lots' street, columns 10-13
    edge = np.zeros((1, width), np.float32)
    edge[0, 2] = 0.9                        # an edge crosses cut 0 (inside a block)
    edge[0, 11] = 0.9                       # and cut 2 (across the street)
    cuts = _cuts([(0, 5, BORDER, True, 1.0),            # inside a block, edge crosses
                  (5, 4, BORDER, True, 1.0),            # inside a block, no edge
                  (9, 6, BORDER, True, 1.0),            # across the street, edge crosses
                  (15, 5, BORDER, True, 0.0),           # not used
                  (20, 5, BORDER, False, 1.0),          # "none crosses": not counted
                  (25, 5, EDGE, True, 1.0)], width)     # a block-edge cut: not counted
    edge[0, 16] = edge[0, 21] = edge[0, 26] = 0.9
    assert counts.edge_on_zoning_border_cuts(edge, cuts, street) == {
        "inside_a_block": 2, "inside_a_block_edge_crosses": 1, "across_a_street": 1,
        "across_a_street_edge_crosses": 1}


def test_count_names_files_by_series_and_leaves_an_answer_not_trained_null(tmp_path, monkeypatch):
    import json

    import cv2
    import torch

    from find_fixtures import Flat
    from sfzmaps import config
    from sfzmaps.find import look, lotlabels, model, pages
    src = tmp_path / "label_sources.csv"
    src.write_text("series,era,lots,zoning_borders\nt,lined-1bit,year,zoning\n")
    monkeypatch.setattr(pages, "SOURCES", src)
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    page = np.ones((300, 400), np.float32)
    page[110:112] = page[128:130] = 0            # a street between two lines, rows 112-127
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    monkeypatch.setattr(pages, "load_keep", lambda sid, shape: np.ones(shape, bool))
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))

    def crop(set_, x0):
        d = config.INPUTS_DIR / "find-labels" / set_ / "t"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"A__{x0}_100_60_40.json"
        f.write_text(json.dumps({"box": [x0, 100, 60, 40], "street": [[x0 + 10, 120]],
                                 "blocks": [[x0 + 20, 104], [x0 + 20, 135]]}))
        return handlabels.read_crop(f, set_)
    choose, test = crop("choose", 20), crop("test", 200)
    street = np.zeros(page.shape, np.uint8)
    street[112:128] = 1
    rows = np.arange(105, 135)                   # one "a zoning border crosses" cut down column 230, across the street
    f = lotlabels.sheet_path(("t", "A"))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, street=street, weight=np.ones(page.shape, np.float32), cuts_px=(rows * 400 + 230).astype(np.int64),
             cuts_cut=np.zeros(len(rows), np.int32), cuts_prior=np.ones(len(rows), np.float32),
             cuts_answer=np.array([BORDER], np.uint8), cuts_says=np.array([True]),
             cuts_weight=np.ones(1, np.float32), edge_none=np.ones(page.shape, bool),
             border_none=np.ones(page.shape, bool), sigma=np.float64(2.0), placement=np.array("warp"))
    torch.manual_seed(0)
    state = model.FindNet(4).state_dict()
    full = counts.count(state, [choose], [test], "cpu", tmp_path / "a")
    part = counts.count(state, [choose], [test], "cpu", tmp_path / "b", ("street", "edge"))
    row, part_row = full["test"]["crops"][0], part["test"]["crops"][0]
    assert row["read"] == "reads/test/t__A__200_100_60_40.npz" and (tmp_path / "a" / row["read"]).is_file()
    assert row["picture"] == "look/test/t__A__200_100_60_40.png"
    assert full["not_trained"] == {} and row["zoning_borders"]["crosses"] == 1
    assert full["test"]["zoning_borders"]["pooled"]["crosses"] == 1
    assert part_row["zoning_borders"] is None and part["test"]["zoning_borders"] is None
    assert list(part["not_trained"]) == ["border"]
    assert part_row["edge_on_zoning_border_cuts"]["across_a_street"] == 1       # a block-edge measure: still counted
    for k in ("ground", "boundary_feet", "whole", "edge_on_zoning_border_cuts"):
        assert set(full["test"][k]) >= {"eras", "pooled"} and "lined-1bit" in full["test"][k]["eras"], k
    with_border, without = (cv2.imread(str(tmp_path / r / row["picture"])) for r in ("a", "b"))
    assert with_border.shape[1] - without.shape[1] == 60 + look.GAP          # the zoning-border panel left out
    with pytest.raises(ValueError, match="block-edge|edge answer"):
        counts.count(state, [choose], [test], "cpu", tmp_path / "c", ("street", "border"))


def test_the_zoning_border_panel_is_scaled_to_its_own_largest_chance():
    from sfzmaps.find import look
    h, w = 30, 40
    chances = np.zeros((3, h, w), np.float32)
    chances[0, 3, 4] = 0.2                      # street: grey on the 0 to 1 scale
    chances[2, 5, 6], chances[2, 7, 8] = 0.2, 0.1   # zoning border: black at its largest, mid-grey at half of it
    truth = _truth({}, (h, w))
    img = look.read_picture(np.ones((h, w), np.float32), chances, np.zeros((h, w), np.int32), truth, {}, "t")
    panel = [img[-h:, k * (w + look.GAP):k * (w + look.GAP) + w, 0] for k in range(5)]
    assert panel[1][3, 4] == 204 and panel[1].min() == 204
    assert panel[3][5, 6] == 0 and panel[3][7, 8] == 128 and (np.delete(panel[3].ravel(), [5 * w + 6, 7 * w + 8])
                                                              == 255).all()
