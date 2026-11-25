"""The counts a read is judged by (`find.counts`): each printed block found, joined, cut up or lost (exactly one of
them), open land left out, how far found block edges lie from the hand ones, the street and block ground, whole
blocks, and a run's counts, the choosing crops' measured as the test crops' are, with nothing about zoning borders."""
from pathlib import Path

import numpy as np
import pytest

from find_fixtures import invoke
from sfzmaps.find import counts, handlabels
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


def _panoptic_truth(shape):
    """Two key blocks, rows 5-14 and 25-34 by columns 5-54 (500 px each); the rest street; the last 5 columns with no
    label."""
    t = _truth({1: (slice(5, 15), slice(5, 55)), 2: (slice(25, 35), slice(5, 55))}, shape)
    t.street[:, 55:] = handlabels.NONE
    return t


def test_panoptic_parts_on_known_answers():
    shape = (40, 60)
    truth = _panoptic_truth(shape)
    same = (truth.area == 1).astype(np.int32)
    assert counts.panoptic_parts(same, truth) == {"matched": 1, "overlap_sum": 1.0, "found_unmatched": 0,
                                                  "key_unmatched": 1}
    split = np.zeros(shape, np.int32)
    split[5:15, 5:35], split[5:15, 35:55] = 1, 2          # block 1 in a 60% piece and a 40% piece
    split[25:35, 5:55] = 3
    got = counts.panoptic_parts(split, truth)
    assert got["matched"] == 2 and got["overlap_sum"] == pytest.approx(1.6)
    assert (got["found_unmatched"], got["key_unmatched"]) == (1, 0)
    joined = np.zeros(shape, np.int32)
    joined[5:35, 5:55] = 1                                 # one found area over both blocks and the street between
    assert counts.panoptic_parts(joined, truth) == {"matched": 0, "overlap_sum": 0.0, "found_unmatched": 1,
                                                    "key_unmatched": 2}
    stray = (truth.area > 0).astype(np.int32) * np.where(np.arange(40)[:, None] < 20, 1, 2)
    stray[18:22, 10:20] = 3                                # a stray found area on street ground
    stray[0:3, 56:60] = 4                                  # a found area on px with no label: counts for nothing
    got = counts.panoptic_parts(stray, truth)
    assert got == {"matched": 2, "overlap_sum": 2.0, "found_unmatched": 1, "key_unmatched": 0}
    spill = (truth.area > 0).astype(np.int32) * np.where(np.arange(40)[:, None] < 20, 1, 2)
    spill[5:15, 55:60] = 1                                 # block 1's found area runs on over unlabelled px
    assert counts.panoptic_parts(spill, truth)["overlap_sum"] == 2.0
    summed = counts.panoptic_summed([got, counts.panoptic_parts(joined, truth)])
    assert summed["quality"] == pytest.approx(2.0 / (2 + 2 / 2 + 2 / 2))
    assert counts.panoptic_summed([counts.panoptic_parts(np.zeros(shape, np.int32), _truth({}, shape))])[
        "quality"] is None


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


def _crop(set_: str) -> handlabels.Crop:
    return handlabels.Crop(set_, "s", "p1", (0, 0, 10, 10), ((1, 1),), (), (), (), (), Path("s/p1__0_0_10_10.json"))


def test_the_trace_scorer_takes_choosing_crops_only():
    with pytest.raises(ValueError, match="test"):
        counts.trace_scorer([_crop("choose"), _crop("test")], "cpu")
    with pytest.raises(ValueError, match="no choose crop"):
        counts.trace_scorer([], "cpu")


# --- the street and block boundary, whole blocks, a run's counts --------------------------------------------------

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


def test_model_edge_feet_is_the_other_half():
    shape = (40, 20)
    truth = _edge_truth(shape, (10,))                       # the answer key: one boundary along row 10
    on = np.zeros(shape, np.float32)
    on[10] = 0.9                                            # the model's edge exactly on it
    got = counts.model_edge_feet(on, truth, 2.0)
    assert (got["px"], got["median"], got["p90"], got["p99"]) == (20, 0.0, 0.0, 0.0)
    assert counts.boundary_feet(on, truth, 2.0)["p99"] == 0.0
    stray = on.copy()
    stray[30] = 0.9                                         # plus a stray straight line 20 px (40 ft) away
    got = counts.model_edge_feet(stray, truth, 2.0)
    assert got["px"] == 40 and got["median"] == pytest.approx(20.0) and got["p90"] == pytest.approx(40.0)
    assert got["p99"] == pytest.approx(40.0)
    assert counts.boundary_feet(stray, truth, 2.0) == counts.boundary_feet(on, truth, 2.0)     # the first half is blind
    band = np.zeros(shape, np.float32)
    band[4:17] = 0.9                                        # a thick band called edge round the thin key line
    first, other = counts.boundary_feet(band, truth, 2.0), counts.model_edge_feet(band, truth, 2.0)
    assert first["p99"] == 0.0
    assert other["median"] == pytest.approx(6.0) and other["p99"] == pytest.approx(12.0)
    # the ground without a label is not counted: a model edge there is no claim about the answer key
    unlabelled = _edge_truth(shape, (10,))
    unlabelled.edge[25:] = handlabels.NONE
    assert counts.model_edge_feet(stray, unlabelled, 2.0)["p99"] == 0.0


def test_model_edge_feet_says_plainly_when_there_is_nothing_to_measure():
    shape = (30, 10)
    truth, empty = _edge_truth(shape, (5,)), _edge_truth(shape, ())
    none_called = counts.model_edge_feet(np.full(shape, 0.4, np.float32), truth, 1.0)
    assert none_called == {"px": 0, "no_model_edge": True, "no_answer_key_boundary": False, "median": None,
                           "p90": None, "p99": None}
    chance = np.zeros(shape, np.float32)
    chance[7] = 0.9
    no_key = counts.model_edge_feet(chance, empty, 1.0)
    assert no_key == {"px": 10, "no_model_edge": False, "no_answer_key_boundary": True, "median": None, "p90": None,
                      "p99": None}
    ds = [counts.model_edge_distances(chance, truth, 1.0), counts.model_edge_distances(chance, empty, 1.0),
          counts.model_edge_distances(np.zeros(shape, np.float32), truth, 1.0)]
    assert ds[1] is None and ds[2] is None
    got = counts.model_edge_feet_pooled(ds, [10, 10, 0])
    assert (got["px"], got["px_measured"], got["crops_not_measured"], got["median"]) == (20, 10, 2, 2.0)


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


def test_count_names_files_by_series_and_holds_nothing_about_zoning_borders(tmp_path, monkeypatch):
    import json

    import cv2
    import torch

    from find_fixtures import Flat
    from sfzmaps import config
    from sfzmaps.find import look, lotlabels, model, pages
    src = tmp_path / "label_sources.csv"
    src.write_text("series,era,lots\nt,lined-1bit,year\n")
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
    torch.manual_seed(0)
    net = model.FindNet(4)
    out = counts.count(net, [choose], [test], "cpu", tmp_path / "a")
    row = out["test"]["crops"][0]
    assert row["read"] == "reads/test/t__A__200_100_60_40.npz" and (tmp_path / "a" / row["read"]).is_file()
    assert row["picture"] == "look/test/t__A__200_100_60_40.png"
    # the choosing crops are measured exactly as the test crops are, under `choose`
    assert set(out) == {"choose", "test"}
    assert set(out["choose"]) == set(out["test"])
    for k in ("ground", "boundary_feet", "model_edge_feet", "whole", "panoptic"):
        for s in ("choose", "test"):
            assert set(out[s][k]) >= {"eras", "pooled"} and "lined-1bit" in out[s][k]["eras"], (s, k)
    mine = out["choose"]["crops"][0]
    assert mine["name"] == "choose/t/A__20_100_60_40" and set(mine) == set(row)
    assert mine["read"] == "reads/choose/t__A__20_100_60_40.npz" and (tmp_path / "a" / mine["picture"]).is_file()
    assert out["choose"]["pooled"]["blocks"] == 2 and out["choose"]["ground"]["pooled"]["street_px"] > 0
    text = json.dumps(out)
    assert "border" not in text and "not_trained" not in text and "answers" not in text and "depth" not in text
    picture = cv2.imread(str(tmp_path / "a" / row["picture"]))
    assert picture.shape[1] == 4 * 60 + 3 * look.GAP                     # the scan, street, block edge, the areas
    # the training trace scores the same plain pieces the count does
    traced = counts.trace_scorer([choose], "cpu", test=[test])(1, net.state_dict())
    assert set(traced) == {"choose_found_share", "choose_panoptic_quality"}
    assert traced["choose_panoptic_quality"] == out["choose"]["panoptic"]["mean_over_eras"]
    assert traced["choose_found_share"] == out["choose"]["found_share"]


def test_a_read_picture_has_four_panels():
    from sfzmaps.find import look
    h, w = 30, 40
    chances = np.zeros((2, h, w), np.float32)
    chances[0, 3, 4] = 0.2                      # street: grey on the 0 to 1 scale
    chances[1, 5, 6] = 1.0                      # block edge: black
    truth = _truth({}, (h, w))
    img = look.read_picture(np.ones((h, w), np.float32), chances, np.zeros((h, w), np.int32), truth, {}, "t")
    assert img.shape[1] == 4 * w + 3 * look.GAP
    panel = [img[-h:, k * (w + look.GAP):k * (w + look.GAP) + w, 0] for k in range(4)]
    assert panel[1][3, 4] == 204 and panel[1].min() == 204
    assert panel[2][5, 6] == 0 and (np.delete(panel[2].ravel(), 5 * w + 6) == 255).all()


def test_count_refuses_a_model_saved_with_three_answers_in_one_line(tmp_path, monkeypatch):
    import torch
    from sfzmaps.find import model, pages
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    old = model.FindNet(4)
    old.head = torch.nn.Conv2d(4, 3, 1)                     # street, block edge and zoning border, as saved before
    (tmp_path / "find" / "runs" / "r").mkdir(parents=True)
    torch.save({"state_dict": old.state_dict(), "step": 1, "recipe": {"answers": ["street", "edge", "border"]},
                "device": "cpu"}, tmp_path / "find" / "runs" / "r" / "model.pt")
    r = invoke(["find", "count", "--run", "r", "--device", "cpu"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit), r.output
    assert len(r.output.strip().splitlines()) == 1 and "predates the two-answer finder" in r.output, r.output


def test_a_crop_is_read_exactly_as_its_sheet_is(tmp_path, monkeypatch):
    """A scoring crop's read (`prepare`, `read_one`) gives its px the numbers the whole sheet's read gives them, for
    crops whose box grown by model.REACH would put the window's corner off the page's 16 px grid."""
    import json

    import torch

    from sfzmaps import config
    from sfzmaps.find import model, pages, read
    rng = np.random.default_rng(4)
    page = np.ones((400, 500), np.float32)
    page[110:112] = page[128:130] = 0                        # a street between two lines, rows 112-127
    for _ in range(30):
        y, x = rng.integers(150, 390), rng.integers(0, 490)
        page[y:y + 2, x:x + rng.integers(5, 60)] = 0
    keep = np.ones(page.shape, bool)
    keep[:, :6] = False
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    monkeypatch.setattr(pages, "load_keep", lambda sid, shape: keep)
    src = tmp_path / "label_sources.csv"
    src.write_text("series,era,lots\nt,lined-1bit,year\n")
    monkeypatch.setattr(pages, "SOURCES", src)
    torch.manual_seed(0)
    net = model.FindNet(4).eval()
    with torch.no_grad():                                    # answers that vary with the print, as a trained net's do
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.mul_(3)
    whole = read.stored(read.read_page(net, page, keep, "cpu"))
    d = config.INPUTS_DIR / "find-labels" / "test" / "t"
    d.mkdir(parents=True)
    for x0, y0 in ((131, 100), (203, 97), (5, 101)):         # 131 - 122 = 9, 203 - 122 = 81: off the grid
        f = d / f"A__{x0}_{y0}_60_40.json"
        f.write_text(json.dumps({"box": [x0, y0, 60, 40], "street": [[x0 + 10, 120]],
                                 "blocks": [[x0 + 20, y0 + 4], [x0 + 20, y0 + 35]]}))
        [p] = counts.prepare([handlabels.read_crop(f, "test")])
        got, k = counts.read_one(net, p, "cpu")
        np.testing.assert_allclose(got, whole[:, y0:y0 + 40, x0:x0 + 60], atol=1.5 / 255, rtol=0, err_msg=f.name)
        assert np.abs(got - whole[:, y0:y0 + 40, x0:x0 + 60]).mean() < 1e-4
        assert np.array_equal(k, keep[y0:y0 + 40, x0:x0 + 60])
