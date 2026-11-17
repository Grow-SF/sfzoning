import contextlib
import dataclasses
import gzip
import hashlib
import json
import os
import platform
import shutil
import threading

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfzmaps import config
from sfzmaps.solver import grid, paths, runs
from sfzmaps.solver.textnet import (boxes, data, model, predict, reader_stage, score_stage, scores, stage, train,
                                    trial)
from sfzmaps.transform import SheetTransform

SHEET = "2000-digital-zoning-p01"
SIZE = 640
WORD_A = [100, 100, 180, 130]     # a zoning code read left to right
WORD_B = [450, 80, 470, 200]      # a street name read bottom to top
OTHER_NEAR_A = [160, 110, 230, 125]   # another pass's box, half on word A
OTHER_ALONE = [100, 400, 160, 430]    # another pass's box on no Terra box


@contextlib.contextmanager
def torch_settings_kept():
    """Put back what `train.settle_torch` and the seeds change for the whole process (torch's threads, its
    deterministic mode and its random state) when the body ends."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng)


@pytest.fixture(autouse=True)
def _torch_as_it_was():
    """Every test here leaves torch as it found it, for the tests that run after it in the same process."""
    with torch_settings_kept():
        yield


def test_torch_settings_come_back_after_settle_torch():
    with torch_settings_kept():
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(False)
        with torch_settings_kept():
            train.settle_torch()
            torch.manual_seed(123)
            assert torch.get_num_threads() == train.THREADS and torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 2 and not torch.are_deterministic_algorithms_enabled()


def _page() -> np.ndarray:
    """The synthetic page: white paper, word A's letters and word B's (drawn turned a quarter), a stray mark."""
    pg = np.full((SIZE, SIZE), 255, np.uint8)
    cv2.putText(pg, "RH-1", (102, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    turned = np.full((20, 120), 255, np.uint8)
    cv2.putText(turned, "ELM ST", (2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    pg[80:200, 450:470] = np.rot90(turned)           # reads bottom to top
    pg[410:420, 110:150] = 0                          # ink under the other pass's lone box
    cv2.line(pg, (0, 300), (639, 300), 0, 2)          # a printed line: ink that is not text
    return pg


def test_targets_mark_boxes_and_unknowns(maps):
    sd, = data.load_sheets([paths.SERIES], hold_out_validation=False)
    t = data.crop_targets(sd, SIZE // 2, SIZE // 2, SIZE, 1.0, 0)        # the whole page
    assert t["ink"].shape == (SIZE, SIZE) and t["ink"][300, 10] and not t["ink"][5, 5]
    # Terra's boxes are text, with their kind and reading angle: above the line, words A and B, edges included
    assert t["text"][100:131, 100:181].all() and t["text"][80:201, 450:471].all()
    assert t["text"][:300].sum() == 81 * 31 + 21 * 121
    assert (t["kind"][105, 105], t["kind"][100, 455]) == (data.KINDS["zoning code"], data.KINDS["street name"])
    assert t["kind"][50, 50] == data.NO_KIND
    assert (t["angle"][115, 140], t["angle"][150, 460]) == (0, 90)
    # unknown (not known): other passes' boxes and discs at Sol's points, minus every Terra box
    assert not t["known"][115, 200] and t["known"][115, 170]       # the other box beyond word A, and inside it
    assert not t["known"][415, 130] and not t["known"][500, 529] and t["known"][500, 532]
    assert not t["known"][140, 175] and t["known"][115, 150]       # Sol's point in A: its disc is unknown outside A
    assert t["known"][600, 50]


def test_model_output_shape():
    net = model.TextNet()
    out = net(torch.zeros(2, 1, 64, 96))
    assert out.shape == (2, model.OUTPUTS, 64, 96) and model.OUTPUTS == 7
    assert sum(isinstance(m, torch.nn.Conv2d) for m in net.modules()) == 2 * (5 + 4) + 1
    # the light decoder: one convolution in each block up
    light = model.TextNet(decoder="light")
    assert light(torch.zeros(2, 1, 64, 96)).shape == (2, model.OUTPUTS, 64, 96)
    assert sum(isinstance(m, torch.nn.Conv2d) for m in light.modules()) == 2 * 5 + 4 + 1


SMALL = train.Recipe(batch=4, crop=64)        # small batches, so CPU tests stay quick; every other setting today's
TINY = train.Recipe(iterations=2, batch=2, crop=64, width=8)      # a few seconds' training, for the stages' tests


def test_three_iterations_twice_give_identical_weights_on_cpu(maps):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    recipe = dataclasses.replace(SMALL, iterations=3)
    first = train.fit(sheets, recipe, device="cpu", seed=0).state_dict()
    second = train.fit(sheets, recipe, device="cpu", seed=0).state_dict()
    torch.manual_seed(0)
    untrained = model.TextNet().state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not all(torch.equal(first[k], untrained[k]) for k in first)


def test_crop_turns_the_reading_angle_with_the_picture(maps):
    sd, = data.load_sheets([paths.SERIES], hold_out_validation=False)
    flat = data.crop_targets(sd, 140, 460, 64, 1.0, 0)          # on word B, read bottom to top
    turned = data.crop_targets(sd, 140, 460, 64, 1.0, 1)
    assert np.array_equal(turned["ink"], np.rot90(flat["ink"]))
    on = turned["text"] > 0
    assert on.any() and (turned["angle"][on] == -180).all()      # 90 + 90 = 180, read in [-180, 180)
    assert (flat["angle"][flat["text"] > 0] == 90).all()


def _loss_and_grads(net, parts, ink_only="off"):
    """The loss of one crop (a dict) or a batch of them (a list), and every gradient, with `ink_only`'s weights."""
    x, t, k, a, w = train.to_tensors(parts if isinstance(parts, list) else [parts], "cpu", ink_only)
    net.zero_grad()
    loss, _ = train.loss_fn(net(x), t, k, a, w)
    loss.backward()
    return loss.item(), [p.grad.clone() for p in net.parameters()]


def _boolean_indexed_loss(out, t, k, a, w):
    """The loss as it was before it became masked sums: kind and angle computed on the known text px only, picked out
    by boolean indexing (whose backward mps can't run deterministically)."""
    lt = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    m = (t > 0) & (w > 0)
    lk = F.cross_entropy(out[:, 1:5].permute(0, 2, 3, 1)[m], k[m])
    cs = out[:, 5:7].permute(0, 2, 3, 1)[m]
    rad = torch.deg2rad(a[m])
    la = (1 - (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)).mean()
    return lt + train.KIND_WEIGHT * lk + train.ANGLE_WEIGHT * la


def test_masked_loss_equals_the_boolean_indexed_loss_even_where_the_direction_output_is_zero():
    g = torch.Generator().manual_seed(0)
    n, h, w_ = 3, 12, 10
    t = (torch.rand(n, h, w_, generator=g) < 0.4).double()
    w = (torch.rand(n, h, w_, generator=g) < 0.7).double()
    k = torch.where(t > 0, torch.randint(0, 4, (n, h, w_), generator=g), torch.full((n, h, w_), data.NO_KIND))
    a = torch.where(t > 0, torch.randint(-180, 180, (n, h, w_), generator=g).double(), torch.zeros(n, h, w_).double())
    out = torch.randn(n, model.OUTPUTS, h, w_, generator=g, dtype=torch.float64)
    off = ~((t > 0) & (w > 0))
    out[:, 5:7] = torch.where(off[:, None], 0.0, out[:, 5:7])     # (cos, sin) exactly (0, 0) off the known text
    grads = []
    for fn in (lambda o: train.loss_fn(o, t, k, a, w)[0], lambda o: _boolean_indexed_loss(o, t, k, a, w)):
        o = out.clone().requires_grad_(True)
        loss = fn(o)
        loss.backward()
        grads.append((loss.detach(), o.grad))
    (masked, g_masked), (indexed, g_indexed) = grads
    assert torch.isfinite(masked) and torch.isfinite(g_masked).all()
    assert torch.allclose(masked, indexed, rtol=1e-12, atol=0)
    assert torch.allclose(g_masked, g_indexed, rtol=1e-10, atol=1e-15)


def test_fit_stops_at_the_first_log_whose_loss_is_not_finite(maps, monkeypatch):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    monkeypatch.setattr(train, "LOG_EVERY", 2)
    real, seen = train.loss_fn, []

    def loss_fn(*args):
        loss, parts = real(*args)
        seen.append(parts)
        return (loss * float("nan"), parts * float("nan")) if len(seen) == 3 else (loss, parts)

    monkeypatch.setattr(train, "loss_fn", loss_fn)
    # the losses stay on the device until the log after step 4, which finds the third not finite
    with pytest.raises(FloatingPointError, match="iterations 3-4"):
        train.fit(sheets, dataclasses.replace(SMALL, iterations=6, batch=2, width=8), device="cpu", seed=0)
    assert len(seen) == 4


# --- the training data of many maps, held out by an area -------------------------------------------------------------

OLD, OLD_SHEET = "1960", "3"          # a sheet of an older map, at twice the 2000 sheets' px per foot
NEW_CORNER = (319.5, 319.5)           # page px of VALIDATION_AREA's north-west corner on the synthetic 2000 page
OLD_CORNER = (199.5, 449.5)           # ... and on the synthetic 1960 page
WORD_C = [300, 310, 360, 330]         # a block number across the validation area's west edge (2000 page)
WORD_D = [380, 400, 460, 450]         # slanted "other" text inside the area (2000 page)
WORD_E = [440, 430, 520, 470]         # a street name over part of word D, drawn after it (2000 page)
OLD_WORDS = [("R-1-D", [200, 150, 300, 190], "zoning code", 0),
             ("MAIN ST", [400, 300, 480, 380], "street name", -45),
             ("2", [180, 470, 230, 500], "other", 0)]          # across the area's west and north edges (1960 page)


def _placed_at_area(ft_per_px: float, corner) -> dict:
    """An affine placement, `ft_per_px` ft per page px, north up, with VALIDATION_AREA's north-west corner at page
    px `corner` (col, row): the page px east of and below it are inside the area."""
    ax0, _, _, ay1 = data.VALIDATION_AREA
    cx, cy = corner
    gcps = [{"px": px, "py": py, "wx": ax0 + ft_per_px * (px - cx), "wy": ay1 - ft_per_px * (py - cy),
             "source": "manual"} for px, py in [(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)]]
    return {"smoothing": 1.0, "gcps": gcps}


def _record(series, sheet, page_box, items, window="0,0,640,640"):
    """One saved answer of `text-on-every-sheet`: Terra's items in one tile (`page_box`) of a sheet's page."""
    return {"series": series, "sheet": sheet, "page_box": list(page_box), "answer": {"items": items},
            "script": {"args": ["gpt-6.1-terra", series, sheet, window, "<out>"]}}


def _word(text, box, kind="zoning code", angle=0, tile=(0, 0)):
    """A Terra item as saved, its box (page px) given in its tile's px."""
    x0, y0, x1, y1 = box
    return {"angle_deg": angle, "direction": "left-to-right" if angle == 0 else "slanted", "kind": kind,
            "text": text, "x0": x0 - tile[0], "x1": x1 - tile[0], "y0": y0 - tile[1], "y1": y1 - tile[1]}


def _write_gz(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _old_page() -> np.ndarray:
    pg = np.full((SIZE, SIZE), 255, np.uint8)
    cv2.putText(pg, "R-1-D", (202, 186), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(pg, (400, 380), (480, 300), 0, 3)
    cv2.putText(pg, "2", (190, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.line(pg, (0, 560), (639, 560), 0, 2)
    return pg


@pytest.fixture
def maps(tmp_path, monkeypatch):
    """Two synthetic sheets laid out as `load_sheets` reads them: the 2000 sheet (4 ft per px) and a 1960 sheet (2 ft
    per px), each with its page, Sol's labels and its corrected placement; Terra's saved answers for both, in tiles;
    one other pass's boxes (of the 2000 sheet, and of a sheet named like the 1960 one)."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SAVED_ANSWERS", tmp_path / "answers")
    monkeypatch.setattr(paths, "TEXT_SHEETS", {paths.SERIES: (SHEET,), OLD: (OLD_SHEET,)})
    new_page = _page()
    cv2.putText(new_page, "12", (305, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.line(new_page, (385, 440), (455, 405), 0, 3)
    cv2.putText(new_page, "OAK", (445, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    layout = {
        (paths.SERIES, SHEET): (new_page, _placed_at_area(4.0, NEW_CORNER), [(175, 115), (500, 500), (330, 300)]),
        (OLD, OLD_SHEET): (_old_page(), _placed_at_area(2.0, OLD_CORNER), [(250, 170), (420, 520)]),
    }
    for (series, sheet), (pg, placed, points) in layout.items():
        work = tmp_path / "work" / series / sheet
        work.mkdir(parents=True)
        cv2.imwrite(str(work / "page.png"), pg)
        labels = [{"code": "X", "kind": "use", "x": float(x), "y": float(y)} for x, y in points]
        (work / "district_labels.json").write_text(json.dumps({"labels": labels}))
        warp = tmp_path / "data" / "warp" / series / sheet
        warp.mkdir(parents=True)
        (warp / "transform.json").write_text(json.dumps(placed))
    tile = {(tx, ty): [tx, ty, tx + 320, ty + 320] for tx in (0, 320) for ty in (0, 320)}   # four tiles of 320 px
    terra = [
        _record(paths.SERIES, SHEET, tile[0, 0], [_word("RH-1", WORD_A)]),
        _record(paths.SERIES, SHEET, tile[320, 0], [_word("ELM ST", WORD_B, "street name", 90, (320, 0))]),
        _record(paths.SERIES, SHEET, tile[0, 320], [_word("12", WORD_C, "block number", 0, (0, 320))]),
        _record(paths.SERIES, SHEET, tile[320, 320], [_word("OAK", WORD_E, "street name", 0, (320, 320)),
                                                      _word("X", WORD_D, "other", 30, (320, 320))]),
        _record(OLD, OLD_SHEET, tile[0, 0], [_word(t, b, k, a) for t, b, k, a in OLD_WORDS]),
    ]
    _write_gz(tmp_path / "answers" / "text-on-every-sheet.jsonl.gz", terra)
    _write_gz(tmp_path / "answers" / data.OTHER_PASSES / "some-pass.jsonl.gz",
              [{"sheet": s, "text": "x", "kind": "other", "box": b, "angle_deg": 0}
               for s, b in ((SHEET, OTHER_NEAR_A), (SHEET, OTHER_ALONE), (OLD_SHEET, [300, 300, 340, 340]))])
    return tmp_path


def _reference_page(series: str, sheet: str, hold_out: bool) -> dict[str, np.ndarray]:
    """The whole page's targets drawn the plain way, for comparison: Terra's boxes (`boxes.terra_boxes`), each
    word's rectangle (`data.word_rect`) filled in order on one page-sized raster per target (padded, so no shape is cut
    at the page's edge); the other passes' rectangles (2000 only) and discs at Sol's points, minus text, unknown; the
    px whose map point (`data.page_world`) is in VALIDATION_AREA held out."""
    gray = cv2.imread(str(config.WORK_DIR / series / sheet / "page.png"), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    pad = 400
    text = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    kind = np.full(text.shape, data.NO_KIND, np.uint8)
    angle = np.zeros(text.shape, np.int16)
    other = np.zeros(text.shape, np.uint8)

    def fill(raster, rect, value):
        cv2.fillPoly(raster, [(np.round(rect).astype(np.int32) + pad).reshape(-1, 1, 2)], value)

    for b in boxes.terra_boxes(series, sheet):
        rect = data.word_rect(b["box"], float(b["angle_deg"]))
        fill(text, rect, 1)
        fill(kind, rect, data.KINDS[b["kind"]])
        fill(angle, rect, data.reading_angle(rect))
    if series == paths.SERIES:
        for b in data.other_boxes(sheet):
            fill(other, data.word_rect(b["box"], 0.0), 1)
    for lb in json.loads((config.WORK_DIR / series / sheet / "district_labels.json").read_text())["labels"]:
        cv2.circle(other, (int(round(lb["x"])) + pad, int(round(lb["y"])) + pad), data.READ_POINT_RADIUS, 1, -1)
    cut = (slice(pad, pad + h), slice(pad, pad + w))
    text, kind, angle, other = text[cut], kind[cut], angle[cut], other[cut]
    folder = config.DATA_DIR / "warp" / series / sheet
    x, y = data.page_world(SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder),
                           (h, w))
    ax0, ay0, ax1, ay1 = data.VALIDATION_AREA
    held = (x >= ax0) & (x < ax1) & (y >= ay0) & (y < ay1) & hold_out
    return {"ink": gray < grid.INK_BELOW, "text": text, "kind": kind, "angle": angle,
            "known": ~((other > 0) & (text == 0)) & ~held}


def _reference_crop(page: dict[str, np.ndarray], y, x, size, scale, turns) -> dict[str, np.ndarray]:
    """A crop of the whole page's targets: `size` / `scale` px of page centred on (y, x), moved inside the page,
    resized to size x size (nearest px), turned `turns` quarter turns, the reading angle with it."""
    h, w = page["ink"].shape
    src = int(round(size / scale))
    y0, x0 = int(np.clip(y - src // 2, 0, h - src)), int(np.clip(x - src // 2, 0, w - src))
    out = {}
    for k, v in page.items():
        v = v[y0:y0 + src, x0:x0 + src]
        if src != size:
            v = cv2.resize(v.astype(np.uint8) if v.dtype == bool else v, (size, size),
                           interpolation=cv2.INTER_NEAREST).astype(v.dtype)
        out[k] = v
    out["angle"] = np.where(out["text"] > 0, (out["angle"].astype(np.int32) + 90 * turns + 180) % 360 - 180,
                            0).astype(np.int16)
    return {k: np.rot90(v, turns) for k, v in out.items()}


def test_terra_boxes_decode_for_another_series():
    records = [
        _record("1960", "3", [0, 0, 768, 768], [_word("R-1-D", [700, 300, 760, 330])], window="0,0,5257,6600"),
        _record("1960", "3", [600, 0, 1368, 768], [_word("R-1-D", [700, 300, 760, 330], tile=(600, 0)),   # read again,
                                                   _word("MAIN ST", [800, 400, 900, 420], "street name",  # farther in
                                                         tile=(600, 0))], window="0,0,5257,6600"),
        _record("1964", "3", [0, 0, 768, 768], [_word("C-2", [10, 10, 50, 30])]),   # another map's sheet "3"
        _record("1960", "4", [0, 0, 768, 768], [_word("R-2", [10, 10, 50, 30])]),
    ]
    got = boxes.terra_boxes("1960", "3", records)
    # page px: each item moved by its tile's corner; the word read twice is kept once, from the tile it is farther
    # inside (edges 100 and 8 px); farthest from its tile's edge first
    assert [(b["text"], b["box"], b["tile"]) for b in got] == [("MAIN ST", [800, 400, 900, 420], [600, 0]),
                                                                ("R-1-D", [700, 300, 760, 330], [600, 0])]
    assert [b["text"] for b in boxes.terra_boxes("1964", "3", records)] == ["C-2"]
    assert boxes.terra_boxes("1960", "5", records) == []


def test_validation_area_pixels_carry_no_loss_even_inside_a_crop_centred_outside(maps):
    new, = data.load_sheets([paths.SERIES], hold_out_validation=True)
    kept, = data.load_sheets([paths.SERIES], hold_out_validation=False)
    # a crop centred outside the area (rows and columns 268-331 of the page) whose bottom-right corner is in it
    # (rows and columns 320 on): word C crosses into that corner
    c = data.crop_targets(new, 300, 300, 64, 1.0, 0)
    whole = data.crop_targets(kept, 300, 300, 64, 1.0, 0)
    inside = np.zeros((64, 64), bool)
    inside[52:, 52:] = True
    assert c["text"][inside].any() and c["ink"][inside].any()
    assert not c["known"][inside].any() and whole["known"][inside].all()
    assert np.array_equal(c["known"][~inside], whole["known"][~inside]) and c["known"][~inside].any()
    # so they carry no loss: changing their targets changes neither the loss nor any gradient
    torch.manual_seed(0)
    net = model.TextNet()
    loss, grads = _loss_and_grads(net, c)
    flipped = {**c, "text": np.where(inside, 1 - c["text"], c["text"]), "kind": np.where(inside, 2, c["kind"]),
               "angle": np.where(inside, 45, c["angle"])}
    loss2, grads2 = _loss_and_grads(net, flipped)
    assert loss2 == loss and all(torch.equal(g, g2) for g, g2 in zip(grads, grads2))
    # and no crop centres on them, on text or on ink; without the hold-out some do
    for sd, some_inside in ((new, False), (kept, True)):
        on_text, on_ink = data.centres(sd, 4)
        assert len(on_text) and len(on_ink)
        for pool in (on_text, on_ink):
            assert ((pool[:, 0] >= 320) & (pool[:, 1] >= 320)).any() == some_inside
            assert (pool % 4 == 0).all()


def test_crop_targets_equal_full_page_drawing(maps):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    assert [(s.series, s.sheet) for s in sheets] == [(paths.SERIES, SHEET), (OLD, OLD_SHEET)]
    centres = [(115, 140), (140, 460), (320, 330), (430, 450), (5, 630), (639, 0), (330, 300), (470, 200)]
    for sd in sheets:
        page = _reference_page(sd.series, sd.sheet, hold_out=True)
        assert not page["known"].all() and page["text"].any()
        for i, (y, x) in enumerate(centres):
            for size, scale in ((64, 1.0), (64, 0.6), (96, 1.6), (40, 0.37)):
                turns = (i + size) % 4
                got = data.crop_targets(sd, y, x, size, scale, turns)
                want = _reference_crop(page, y, x, size, scale, turns)
                assert sorted(got) == sorted(want)
                for k in want:
                    assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), (sd.series, y, x, k)


def test_sheet_data_holds_no_full_page_target_rasters(maps):
    for sd in data.load_sheets([paths.SERIES, OLD], hold_out_validation=True):
        fields = {f.name: getattr(sd, f.name) for f in dataclasses.fields(sd)}
        # the areas held out are map areas (one shared by every sheet), no raster of the page
        assert fields.pop("held") == (data.VALIDATION,)
        # no line px until the label rule "lines-unknown" asks for them (`zones.prepare`): an empty array, no bytes
        lines = fields.pop("line_bits")
        assert lines.dtype == np.uint8 and lines.shape == (0, 0)
        # every other field is an array or a plain value: nothing is held where its bytes would not be counted
        assert all(isinstance(v, (np.ndarray, str, int, float, bool)) for v in fields.values())
        arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
        assert all(v.base is None or v.base.nbytes == v.nbytes for v in arrays.values())
        ink = arrays["ink_bits"].nbytes
        assert ink <= SIZE * SIZE                                  # at most one byte per page px
        box_bytes = sum(v.nbytes for k, v in arrays.items() if k in data.BOX_FIELDS)
        assert sum(v.nbytes for v in arrays.values()) <= 1.2 * ink + box_bytes
        assert np.array_equal(data.page_ink(sd), _reference_page(sd.series, sd.sheet, True)["ink"])


def test_each_map_is_read_at_its_own_scale_with_its_own_unknowns(maps, monkeypatch):
    new, old = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    assert new.ft_per_px == pytest.approx(4.0) and old.ft_per_px == pytest.approx(2.0)
    # the other passes are of the 2000 sheets only: their lone box is unknown on the 2000 page; on the 1960 page,
    # the box an other pass gives a sheet named "3" is not
    assert len(new.other) == 2 and len(old.other) == 0
    assert not data.crop_targets(new, 415, 130, 64, 1.0, 0)["known"][32, 32]
    assert data.crop_targets(old, 320, 320, 64, 1.0, 0)["known"].all()
    # sheets named in the code but not in the saved answers stop the read
    monkeypatch.setattr(paths, "TEXT_SHEETS", {paths.SERIES: (SHEET, "2000-digital-zoning-p02"), OLD: (OLD_SHEET,)})
    with pytest.raises(ValueError, match="2000-digital-zoning-p02"):
        data.load_sheets([paths.SERIES], hold_out_validation=True)


def test_text_sheet_lists_match_the_saved_answers(tmp_path, monkeypatch):
    real = dict(paths.TEXT_SHEETS)
    monkeypatch.setattr(paths, "SAVED_ANSWERS", tmp_path)
    tiles = [("1960", s) for s in ("10", "2", "1", "2", "10")] + [
        ("1980-1990-zoning", "1980-1990-zoning-p02"), ("1980-1990-zoning", "1980-1990-zoning-p01")]
    _write_gz(tmp_path / "text-on-every-sheet.jsonl.gz", [_record(s, n, [0, 0, 768, 768], []) for s, n in tiles])
    # each map's sheets Terra read, each once, numbers in a name ordered as numbers
    want = {"1960": ("1", "2", "10"), "1980-1990-zoning": ("1980-1990-zoning-p01", "1980-1990-zoning-p02")}
    assert boxes.text_sheets() == want
    monkeypatch.setattr(paths, "TEXT_SHEETS", want)
    boxes.check_text_sheets()
    for wrong, named in (({**want, "1960": ("1", "2")}, "10"),                        # a sheet missing
                         ({**want, "1960": ("1", "10", "2")}, "1960"),                # out of order
                         ({**want, "1964": ("20180102105639444-p02",)}, "1964"),      # a map Terra never read
                         ({"1960": want["1960"]}, "1980-1990-zoning")):               # a map left out
        monkeypatch.setattr(paths, "TEXT_SHEETS", wrong)
        with pytest.raises(ValueError, match=named):
            boxes.check_text_sheets()
    # the code's own lists: every training and test map, each sheet once, in the same order
    assert list(real) == [*paths.TEXT_TRAIN_SERIES, *paths.TEXT_TEST_SERIES]
    assert not set(paths.TEXT_TRAIN_SERIES) & set(paths.TEXT_TEST_SERIES)
    assert all(list(v) == sorted(set(v), key=boxes.sheet_order) for v in real.values())


# --- the training recipe and its options -----------------------------------------------------------------------------

TODAYS_PARAMETERS = 2_758_567     # the detector's weights before the recipe (model.TextNet of 78c74c1)


def _todays_loss(out, t, k, a, w):
    """The loss as it was before the recipe (train.loss_fn of 78c74c1), copied: binary cross-entropy on text over the
    px of weight `w`, plus 0.5 x the kind's cross-entropy and 0.5 x (1 - cosine) of the reading direction, both on
    weighted text px."""
    lt = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    m = ((t > 0) & (w > 0)).float()
    n = m.sum().clamp(min=1)
    lk = (F.cross_entropy(out[:, 1:5], torch.where(m > 0, k, torch.zeros_like(k)), reduction="none") * m).sum() / n
    cs = out[:, 5:7]
    rad = torch.deg2rad(a)
    cosine = (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)
    la = ((1 - cosine) * m).sum() / n
    return lt + 0.5 * lk + 0.5 * la


def _fixed_batch(sheets, size=64) -> list[dict[str, np.ndarray]]:
    """Four crops of the two synthetic sheets: on word A, across the validation area's corner (word C), on the 1960
    zoning code and on its slanted street, at three scales and every turn."""
    new, old = sheets
    return [data.crop_targets(new, 115, 140, size, 1.0, 0), data.crop_targets(new, 300, 300, size, 1.0, 1),
            data.crop_targets(old, 170, 250, size, 0.8, 2), data.crop_targets(old, 340, 440, size, 1.3, 3)]


def _flip(crops, where) -> list[dict[str, np.ndarray]]:
    """The crops with their text, kind and reading-angle targets changed at the px `where` (one mask per crop)."""
    return [{**c, "text": np.where(m, 1 - c["text"], c["text"]).astype(np.uint8),
             "kind": np.where(m, 2, c["kind"]).astype(np.uint8),
             "angle": np.where(m, 45, c["angle"]).astype(np.int16)} for c, m in zip(crops, where)]


def test_default_recipe_trains_as_today(maps):
    # today's settings: 8,000 steps of 12 crops of 320 px, a one-cycle peak of 2e-3 reached 30% in, width 24, the
    # loss on every known px, the network at full size with all seven outputs, no average, no prefetch, every px of a
    # word box taught as text (the label rule "boxes"), no composites
    assert dataclasses.asdict(train.Recipe()) == {
        "iterations": 8000, "max_lr": 2e-3, "pct_start": 0.3, "batch": 12, "crop": 320, "width": 24, "ink_only": "off",
        "half_size": False, "heads": "all", "weight_average": 0.0, "prefetch": False, "decoder": "full",
        "precision": "float32", "labels": "boxes", "composites": 0.0}
    torch.manual_seed(0)
    net = train.build_net(train.Recipe())
    assert sum(p.numel() for p in net.parameters()) == TODAYS_PARAMETERS
    # on a fixed batch, the loss is today's: the weight is the known px, and the loss of the outputs is the same
    crops = _fixed_batch(data.load_sheets([paths.SERIES, OLD], hold_out_validation=True))
    x, t, k, a, w = train.to_tensors(crops, "cpu", train.Recipe().ink_only)
    known = torch.from_numpy(np.stack([c["known"] for c in crops]).astype(np.float32))
    assert torch.equal(w, known) and (t * w).sum() > 0 and (w == 0).any()
    out = net(x)
    assert out.shape == (4, 7, 64, 64)
    assert torch.equal(train.loss_fn(out, t, k, a, w)[0], _todays_loss(out, t, k, a, known))


def test_half_size_fold_is_lossless_and_shapes_round_trip(monkeypatch):
    x = torch.arange(2 * 6 * 10, dtype=torch.float32).reshape(2, 1, 6, 10)
    folded = model.fold(x)
    assert folded.shape == (2, 4, 3, 5)
    assert torch.equal(folded.flatten().sort().values, x.flatten())      # every px kept once: nothing lost
    assert torch.equal(model.unfold(folded), x)
    # the half-size network runs on the folded ink, at half the px each way, and gives every px its outputs
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(half_size=True, width=8)).eval()
    seen = []
    net.down[0].register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape)))
    assert net(torch.zeros(2, 1, 64, 96)).shape == (2, model.OUTPUTS, 64, 96) and seen == [(2, 4, 32, 48)]
    # crops are even: an odd crop can't be folded, so the recipe refuses it
    with pytest.raises(ValueError, match="even"):
        train.Recipe(half_size=True, crop=321)
    assert predict.TILE % 2 == 0 and predict.PAD % 2 == 0
    # an odd page: padded with paper to the network's multiple, scored, and cut back to the page
    ink = np.zeros((75, 101), bool)
    ink[10:20, 10:40] = True
    ink[74, :] = True
    ink[:, 100] = True
    got = predict.chance(net, ink, "cpu")
    assert got.shape == ink.shape and got.dtype == np.uint8
    padded = np.zeros((96, 128), np.float32)
    padded[:75, :101] = ink
    with torch.no_grad():
        want = torch.sigmoid(net(torch.from_numpy(padded)[None, None])[0, 0, :75, :101]).numpy()
    assert np.array_equal(got, np.round(255 * want).astype(np.uint8))
    # and in tiles smaller than the page, whose windows at the page's odd edge are odd before padding
    monkeypatch.setattr(predict, "TILE", 32)
    monkeypatch.setattr(predict, "PAD", 8)
    tiled = predict.chance(net, ink, "cpu")
    assert tiled.shape == ink.shape and tiled[74].all() and tiled[:, 100].all()


def test_ink_only_loss_ignores_paper(maps):
    crops = _fixed_batch(data.load_sheets([paths.SERIES, OLD], hold_out_validation=True))
    ink = [c["ink"] for c in crops]
    near = [cv2.dilate(i.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for i in ink]   # ink and the px next to it
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(width=8))
    for ink_only, paper in (("ink", [~i for i in ink]), ("ink+1px", [~n for n in near])):
        loss, grads = _loss_and_grads(net, crops, ink_only)
        assert loss > 0
        loss2, grads2 = _loss_and_grads(net, _flip(crops, paper), ink_only)
        assert loss2 == loss and all(torch.equal(g, g2) for g, g2 in zip(grads, grads2))
    # the px next to the ink count with "ink+1px" only; with "off", paper counts too
    ring = [n & ~i for n, i in zip(near, ink)]
    assert _loss_and_grads(net, _flip(crops, ring), "ink")[0] == _loss_and_grads(net, crops, "ink")[0]
    assert _loss_and_grads(net, _flip(crops, ring), "ink+1px")[0] != _loss_and_grads(net, crops, "ink+1px")[0]
    assert _loss_and_grads(net, _flip(crops, [~i for i in ink]), "off")[0] != _loss_and_grads(net, crops, "off")[0]


def test_prefetch_gives_bit_identical_weights(maps, monkeypatch):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    real, drawn_on = train.draw, []
    monkeypatch.setattr(train, "draw", lambda *args: drawn_on.append(threading.current_thread()) or real(*args))
    recipe = dataclasses.replace(SMALL, iterations=5, width=8)
    off = train.fit(sheets, recipe, device="cpu", seed=0).state_dict()
    on = train.fit(sheets, dataclasses.replace(recipe, prefetch=True), device="cpu", seed=0).state_dict()
    assert off.keys() == on.keys() and all(torch.equal(off[k], on[k]) for k in off)
    # the batches with prefetch were drawn in another thread, one per step and no more
    main = threading.main_thread()
    assert len(drawn_on) == 10 and all(t is main for t in drawn_on[:5]) and not any(t is main for t in drawn_on[5:])


def test_text_head_only_predicts_text(maps):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    recipe = dataclasses.replace(SMALL, heads="text", width=8, iterations=2)
    torch.manual_seed(0)
    net = train.build_net(recipe)
    crops = _fixed_batch(sheets)
    x, t, k, a, w = train.to_tensors(crops, "cpu")
    out = net(x)
    assert out.shape == (4, 1, 64, 64)
    # the loss is the text's alone
    loss, parts = train.loss_fn(out, t, k, a, w)
    text = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum()
    assert torch.equal(loss, text) and torch.equal(parts, torch.stack([text, torch.tensor(0.0), torch.tensor(0.0)]))
    # so the kind and the reading angle teach it nothing
    other = [{**c, "kind": np.where(c["text"] > 0, 3 - c["kind"] % 4, c["kind"]).astype(np.uint8),
              "angle": np.where(c["text"] > 0, 90, 0).astype(np.int16)} for c in crops]
    loss1, grads1 = _loss_and_grads(net, crops)
    loss2, grads2 = _loss_and_grads(net, other)
    assert loss1 == loss2 and all(torch.equal(g, g2) for g, g2 in zip(grads1, grads2))
    # it trains and scores pages
    fitted = train.fit(sheets, recipe, device="cpu", seed=0).eval()
    assert predict.chance(fitted, data.page_ink(sheets[0]), "cpu").shape == (SIZE, SIZE)


def test_weight_average_is_the_ema_of_weights(maps, monkeypatch):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    decay, steps = 0.6, 4
    recipe = dataclasses.replace(SMALL, iterations=steps, width=8, weight_average=decay)
    after, real = [], train.WeightAverage.update

    def update(self, net):
        after.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        return real(self, net)

    monkeypatch.setattr(train.WeightAverage, "update", update)
    got = train.fit(sheets, recipe, device="cpu", seed=0).state_dict()
    plain = train.fit(sheets, dataclasses.replace(recipe, weight_average=0.0), device="cpu", seed=0).state_dict()
    # the average is taken after every step, beside the training: the weights it averages are the plain recipe's
    assert len(after) == steps and all(torch.equal(after[-1][k], plain[k]) for k in plain)
    # the model is their exponential moving average, each step weighted as the decay gives and the shares scaled to
    # sum to 1 (so the untrained weights count for nothing); every float of the state, the batch norms' statistics too
    shares = [(1 - decay) * decay ** (steps - 1 - s) / (1 - decay ** steps) for s in range(steps)]
    assert sum(shares) == pytest.approx(1.0)
    for key, v in got.items():
        if v.is_floating_point():
            want = sum(share * after[s][key] for s, share in enumerate(shares))
            assert torch.allclose(v, want, rtol=1e-5, atol=1e-7), key
        else:
            assert torch.equal(v, plain[key]), key
    assert not all(torch.equal(got[k], plain[k]) for k in plain)


def test_recipe_round_trips_through_the_saved_model(maps, tmp_path):
    # the short id: the settings that differ from today's, by name; a switch on as its name alone
    assert train.Recipe().name() == "default"
    assert train.Recipe(iterations=2000, half_size=True, ink_only="ink").name() == (
        "half_size-ink_only=ink-iterations=2000")
    # but not prefetch, which never changes the weights (the same batches, drawn sooner)
    assert train.Recipe(prefetch=True).name() == "default"
    assert train.Recipe(half_size=True, prefetch=True).name() == "half_size"
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    recipe = train.Recipe(iterations=2, batch=2, crop=64, width=8, half_size=True, heads="text", decoder="light",
                          ink_only="ink+1px", weight_average=0.5, max_lr=6e-3, pct_start=0.1)
    assert recipe.name() == ("batch=2-crop=64-decoder=light-half_size-heads=text-ink_only=ink+1px-iterations=2-"
                             "max_lr=0.006-pct_start=0.1-weight_average=0.5-width=8")
    net = train.fit(sheets, recipe, device="cpu", seed=0).eval()
    f = train.save_model(tmp_path / "text" / "model.pt", net, recipe, "cpu", 0)
    saved = torch.load(f)
    assert train.Recipe(**saved["recipe"]) == recipe
    assert (saved["iterations"], saved["device"], saved["seed"]) == (2, "cpu", 0)
    # prediction builds the network the recipe names, and scores as the trained one does
    loaded = predict.load_model(f, "cpu")
    ink = data.page_ink(sheets[1])
    assert np.array_equal(predict.chance(loaded, ink, "cpu"), predict.chance(net, ink, "cpu"))
    # a model saved without its recipe is refused, not guessed
    torch.save({k: v for k, v in saved.items() if k != "recipe"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="recipe"):
        predict.load_model(tmp_path / "old.pt", "cpu")


# --- the scores ------------------------------------------------------------------------------------------------------

AREA_CORNER = (319.5, 319.5)      # page px of VALIDATION_AREA's north-west corner: rows and columns 320 on are in it
EDGE_WORDS = [([300, 400, 380, 430], "zoning code", 0),    # P: across the area's west edge, its centre (340, 415) in it
              ([400, 290, 460, 340], "street name", 0),    # Q: across its north edge, its centre (430, 315) outside
              ([500, 500, 560, 530], "block number", 0),   # R: inside
              ([50, 50, 120, 80], "other", 0)]             # S: outside
EDGE_POINT = (600, 450)           # one of Sol's points, in the area


def _in_memory_sheet(gray, ft_per_px=4.0, corner=AREA_CORNER, words=(), points=(), sheet=SHEET) -> data.SheetData:
    """A 2000 SheetData made in memory: the page `gray`, placed north up at `ft_per_px` ft per page px with
    VALIDATION_AREA's north-west corner at page px `corner` (col, row), so the px east of and below it are in the
    area; Terra's `words` (axis box, kind, reading angle) and Sol's `points` (x, y)."""
    h, w = gray.shape
    ax0, _, _, ay1 = data.VALIDATION_AREA
    step = data.PLACEMENT_STEP
    gx, gy = np.meshgrid(np.arange(0, w - 1 + step, step, dtype=float), np.arange(0, h - 1 + step, step, dtype=float))
    world = np.stack([ax0 + ft_per_px * (gx - corner[0]), ay1 - ft_per_px * (gy - corner[1])], -1)
    rects = [data.word_rect(box, angle) for box, _, angle in words]
    return data.SheetData(
        series=paths.SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.array([data.KINDS[k] for _, k, _ in words], np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.array(points, np.int32).reshape(-1, 2), world=world, ft_per_px=float(ft_per_px))


def _box_mask(shape, box) -> np.ndarray:
    """The page px (bool) of an axis box (x0, y0, x1, y1), its edges included: the px a word read left to right
    fills."""
    x0, y0, x1, y1 = box
    m = np.zeros(shape, bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def _u8(a) -> np.ndarray:
    return np.asarray(a).astype(np.uint8)


def _score(sd, chance, where) -> dict:
    return scores.score_sheets({(sd.series, sd.sheet): chance}, where, [sd])


def _drawing(ft_per_px: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One drawing in feet (1,600 ft square) at `ft_per_px` ft per page px: four straight bars apart (from x0 to x1,
    centred on y, of a width; ft): a long thin line, a short thin one, a long heavy one (a district line's 40 ft) and
    a long solid fill (64 ft); and a speck 8 ft below the long thin line (a bar 4 ft long).  Returns the page and the
    px of each."""
    lines = {"long thin": (100, 420, 200, 8), "short thin": (100, 260, 500, 8), "long heavy": (100, 420, 800, 40),
             "solid": (100, 420, 1200, 64), "speck": (300, 304, 214, 4)}
    n = int(round(1600 / ft_per_px))
    gray = np.full((n, n), 255, np.uint8)
    px = {}
    for name, (x0, x1, y, width) in lines.items():
        m = np.zeros((n, n), bool)
        r0 = int(round((y - width / 2) / ft_per_px))
        m[r0:r0 + int(round(width / ft_per_px)), int(round(x0 / ft_per_px)):int(round(x1 / ft_per_px))] = True
        gray[m] = 0
        px[name] = m
    return gray, px


def test_long_line_measure_uses_feet_per_sheet():
    # the same drawing at 2 and at 4 ft per page px: the same lines are long lines (at least LONG_LINE_FT long, at most
    # THIN_LINE_FT wide, each in the sheet's own feet), the thin and the heavy one, nearly all of their ink; the short
    # line and the solid fill are not
    for ft in (2.0, 4.0):
        gray, lines = _drawing(ft)
        found = scores.long_lines(gray < grid.INK_BELOW, ft)
        assert found[lines["long thin"]].mean() >= 0.9 and found[lines["long heavy"]].mean() >= 0.8, ft
        assert not found[lines["short thin"]].any() and not found[lines["solid"]].any(), ft
        # a speck beside the line, whose centre line is no stroke, is not the line's
        assert lines["speck"].any() and not found[lines["speck"]].any(), ft
        assert not (found & (gray >= grid.INK_BELOW)).any()
    # read at the grid's 4 ft per px instead, the 2 ft drawing's short line would be 320 ft long and its heavy line
    # 80 ft wide: the sheet's scale decides both
    gray, lines = _drawing(2.0)
    wrong = scores.long_lines(gray < grid.INK_BELOW, grid.FT_PER_PX)
    assert wrong[lines["short thin"]].mean() >= 0.9 and not wrong[lines["long heavy"]].any()
    # the scores read each sheet's own scale: the same long-line ink, in square feet, at both
    area_ft2 = []
    for ft in (2.0, 4.0):
        gray, _ = _drawing(ft)
        sd = _in_memory_sheet(gray, ft_per_px=ft)
        long_ink = _score(sd, np.zeros(gray.shape, np.uint8), "whole")["all"]["long_lines"]["ink_px"]
        area_ft2.append(long_ink * ft * ft)
    assert area_ft2[0] > 0 and area_ft2[0] == pytest.approx(area_ft2[1], rel=0.1)


def _edge_page() -> np.ndarray:
    """The page of EDGE_WORDS: each word's ink a block inside its box; a thin line 1,040 ft long whose last 160 ft
    are in the area; ink at Sol's point."""
    gray = np.full((SIZE, SIZE), 255, np.uint8)
    for (x0, y0, x1, y1), _, _ in EDGE_WORDS:
        gray[y0 + 3:y1 - 2, x0 + 2:x1 - 1] = 0
    gray[598:600, 100:360] = 0
    gray[440:460, 590:610] = 0
    return gray


def _edge_masks(gray) -> dict[str, np.ndarray]:
    """The edge page's px, drawn independently: ink, the area (rows and columns 320 on), text (the words' boxes),
    unknown (within READ_POINT_RADIUS of Sol's point, in no box)."""
    ink = gray < grid.INK_BELOW
    area = np.zeros(gray.shape, bool)
    area[320:, 320:] = True
    text = np.zeros(gray.shape, bool)
    for box, _, _ in EDGE_WORDS:
        text |= _box_mask(gray.shape, box)
    disc = np.zeros(gray.shape, np.uint8)
    cv2.circle(disc, EDGE_POINT, data.READ_POINT_RADIUS, 1, -1)
    return {"ink": ink, "area": area, "text": text, "unknown": (disc > 0) & ~text}


def test_validation_scores_use_only_validation_pixels():
    gray = _edge_page()
    sd = _in_memory_sheet(gray, words=EDGE_WORDS, points=[EDGE_POINT])
    m = _edge_masks(gray)
    rng = np.random.default_rng(0)
    chance = rng.integers(0, 256, gray.shape, dtype=np.uint8)
    v = _score(sd, chance, "validation")
    # the ink px scored are the known ink px in the area, and no others
    scored = m["ink"] & m["area"] & ~m["unknown"]
    assert m["unknown"].any() and (m["ink"] & m["unknown"]).any()
    assert v["all"]["ink_px"] == scored.sum() and v["all"]["text_ink_px"] == (scored & m["text"]).sum()
    assert v["series"][paths.SERIES]["sheets"] == 1 and v["where"] == "validation"
    # a chance changed anywhere outside the area changes nothing; changed on the ink inside, the scores change
    assert _score(sd, np.where(m["area"], chance, rng.integers(0, 256, gray.shape, dtype=np.uint8)),
                  "validation") == v
    assert _score(sd, _u8(np.where(scored, 255 - chance, chance)), "validation") != v
    # the long line is long by its whole length (its 160 ft in the area alone would be short): its ink in the area
    # is the long-line ink
    line = np.zeros(gray.shape, bool)
    line[598:600, 320:360] = True
    assert 0.9 * line.sum() <= v["all"]["long_lines"]["ink_px"] <= line.sum()
    assert not scores.long_lines(m["ink"] & m["area"], sd.ft_per_px)[line].any()
    # on the whole sheet every known ink px counts, outside the area too
    w = _score(sd, chance, "whole")
    assert w["all"]["ink_px"] == (m["ink"] & ~m["unknown"]).sum() and w["where"] == "whole"
    assert w["all"]["long_lines"]["ink_px"] > v["all"]["long_lines"]["ink_px"]
    assert _score(sd, _u8(np.where(m["area"], chance, 0)), "whole") != w
    # validation needs a chance for every sheet with px in the area, and only those; whole needs every sheet's
    away = _in_memory_sheet(gray, corner=(5000.0, 5000.0), sheet="2000-digital-zoning-p02")
    both = [sd, away]
    key = (sd.series, sd.sheet)
    assert scores.score_sheets({key: chance}, "validation", both) == v
    with pytest.raises(ValueError, match="2000-digital-zoning-p02"):
        scores.score_sheets({key: chance}, "whole", both)
    with pytest.raises(ValueError, match=SHEET):
        scores.score_sheets({(away.series, away.sheet): chance}, "validation", both)
    with pytest.raises(ValueError, match="elsewhere"):
        scores.score_sheets({key: chance, (paths.SERIES, "elsewhere"): chance}, "validation", both)
    with pytest.raises(ValueError, match="shape"):
        _score(sd, chance[:-1], "validation")
    with pytest.raises(ValueError, match="where"):
        _score(sd, chance, "train")


def test_a_word_across_the_validation_edge_counts_by_its_centre_and_only_its_ink_inside():
    # in validation a word is counted when its centre px is in the area, and judged on its ink in the area alone (no
    # px the detector trained on counts); a word whose centre is outside is not counted, though its ink inside counts
    # as text ink. On the whole sheet every word counts, judged on all its ink
    gray = _edge_page()
    sd = _in_memory_sheet(gray, words=EDGE_WORDS, points=[EDGE_POINT])
    m = _edge_masks(gray)
    p_ink, q_ink, r_ink = (_box_mask(gray.shape, box) & m["ink"] for box, _, _ in EDGE_WORDS[:3])
    assert (p_ink & ~m["area"]).any() and (q_ink & m["area"]).any() and (q_ink & ~m["area"]).any()
    # P's ink outside the area called text, its ink inside not quite: not found in validation, found on the whole
    # sheet (its mean chance there (255 x 18 + 100 x 59) / 77 = 136)
    chance = _u8(np.where(p_ink & ~m["area"], 255, np.where(p_ink, 100, 0)))
    v = _score(sd, chance, "validation")["all"]
    assert {k: (x["found"], x["of"]) for k, x in v["words"].items()} == {
        "zoning code": (0, 1), "block number": (0, 1), "street name": (0, 0), "other": (0, 0), "all": (0, 2)}
    assert v["text_ink_px"] == ((p_ink | q_ink | r_ink) & m["area"]).sum()
    w = _score(sd, chance, "whole")["all"]["words"]
    assert {k: (x["found"], x["of"]) for k, x in w.items()} == {
        "zoning code": (1, 1), "block number": (0, 1), "street name": (0, 1), "other": (0, 1), "all": (1, 4)}
    # the other way round, P is found in validation: its ink inside called, its ink outside not
    chance = _u8(np.where(p_ink & m["area"], 255, 0))
    assert _score(sd, chance, "validation")["all"]["words"]["zoning code"] == {"found": 1, "of": 1}


def test_words_found_counts_a_box_with_half_its_ink_called_text():
    # a word is found when on average half its ink is called text: the mean chance over its ink (the ink px in its
    # rectangle) is CALLED (0.5) or more, the rule of the quarter models' scores
    a_box, b_box, c_box, d_box = [20, 20, 120, 60], [200, 20, 300, 60], [260, 20, 360, 60], [20, 100, 120, 140]
    gray = np.full((200, 400), 255, np.uint8)
    gray[30:50, 30:110] = 0        # A's ink: 80 columns
    gray[30:50, 210:250] = 0       # B's ink outside C
    gray[30:50, 265:295] = 0       # ink where B and C overlap
    gray[30:50, 310:320] = 0       # C's ink outside B
    gray[150:152, :] = 0           # a line in no box
    words = [(a_box, "zoning code", 0), (b_box, "street name", 0), (c_box, "other", 0), (d_box, "block number", 0)]
    sd = _in_memory_sheet(gray, words=words)
    ink = gray < grid.INK_BELOW
    a = _box_mask(gray.shape, a_box) & ink
    cols = np.arange(400)[None, :]

    def found(chance) -> dict[str, int]:
        return {k: x["found"] for k, x in _score(sd, chance, "whole")["all"]["words"].items()}

    def a_found(chance) -> bool:
        return found(chance)["zoning code"] == 1

    assert a_found(_u8(np.where(a, scores.CALLED, 0))) and not a_found(_u8(np.where(a, scores.CALLED - 1, 0)))
    # a float chance is read as the detector writes it (x 255, rounded): 0.5 is called text, 0.498 is not
    assert a_found(np.where(a, 0.5, 0.0)) and not a_found(np.where(a, 0.498, 0.0))
    # the mean, however the chance is spread: half at 255 and half at 1 is 128; 60% of the ink at 255 found, 40% not;
    # but 40% at 255 and the rest at 100 is found (a mean of 162), though most of its px are not called
    assert a_found(_u8(np.where(a & (cols % 2 == 0), 255, np.where(a, 1, 0))))
    assert a_found(_u8(np.where(a & (cols < 78), 255, 0))) and not a_found(_u8(np.where(a & (cols < 62), 255, 0)))
    assert a_found(_u8(np.where(a & (cols < 62), 255, np.where(a, 100, 0))))
    # paper in its box and ink in no box count for nothing
    rows = np.arange(200)[:, None]
    assert not a_found(_u8(np.where((_box_mask(gray.shape, a_box) & ~ink) | (ink & (rows >= 150)), 255, 0)))
    # overlapping words are each judged on their own ink, the ink where they overlap each one's: called there, C is
    # found (600 of its 800 ink px) and B not (600 of 1,400)
    overlap = _box_mask(gray.shape, b_box) & _box_mask(gray.shape, c_box) & ink
    f = found(_u8(np.where(overlap, 255, 0)))
    assert f["other"] == 1 and f["street name"] == 0 and f["zoning code"] == 0
    # a word with no ink (D) is not counted
    w = _score(sd, np.zeros(gray.shape, np.uint8), "whole")["all"]
    assert {k: x["of"] for k, x in w["words"].items()} == {
        "zoning code": 1, "block number": 0, "street name": 1, "other": 1, "all": 3}
    assert w["words_found"] == 0.0


def test_a_sheets_long_lines_are_found_once_per_process(monkeypatch):
    monkeypatch.setattr(scores, "_LINE_INK", {})
    calls = []
    real = scores.long_lines
    monkeypatch.setattr(scores, "long_lines", lambda ink, ft: calls.append(ft) or real(ink, ft))
    gray = _edge_page()
    sd = _in_memory_sheet(gray, words=EDGE_WORDS, points=[EDGE_POINT])
    rng = np.random.default_rng(2)
    chances = [rng.integers(0, 256, gray.shape, dtype=np.uint8) for _ in range(2)]
    first = [_score(sd, c, where) for c in chances for where in scores.WHERE]
    # two detectors' chances, on validation and on the whole sheet, and an equal copy of the sheet: the lines once
    assert _score(dataclasses.replace(sd, ink_bits=sd.ink_bits.copy()), chances[0], "validation") == first[0]
    assert len(calls) == 1
    assert np.array_equal(scores.line_ink(sd), real(data.page_ink(sd), sd.ft_per_px)) and len(calls) == 1
    # the same as found afresh
    monkeypatch.setattr(scores, "_LINE_INK", {})
    assert [_score(sd, c, where) for c in chances for where in scores.WHERE] == first and len(calls) == 2
    # another page under the same name, or the same page at another scale, is not given the kept answer
    other = gray.copy()
    other[100:102, 50:600] = 0
    scores.line_ink(_in_memory_sheet(other, words=EDGE_WORDS, points=[EDGE_POINT]))
    scores.line_ink(_in_memory_sheet(gray, ft_per_px=2.0, words=EDGE_WORDS, points=[EDGE_POINT]))
    assert calls == [4.0, 4.0, 4.0, 2.0]


def test_scores_read_the_sheets_of_the_maps_scored_and_add_them_up(maps):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    rng = np.random.default_rng(1)
    chances = {(sd.series, sd.sheet): rng.integers(0, 256, (sd.height, sd.width), dtype=np.uint8) for sd in sheets}
    # by default every sheet Terra read of the maps named, in the maps' order whatever the chances' order
    got = scores.score_sheets(dict(reversed(chances.items())), "whole")
    assert got == scores.score_sheets(chances, "whole", sheets) and list(got["series"]) == [paths.SERIES, OLD]
    # over all, the maps' counts added up
    per_map = list(got["series"].values())
    for k in ("sheets", "ink_px", "text_ink_px"):
        assert got["all"][k] == sum(s[k] for s in per_map) > 0
    assert got["all"]["words"]["all"]["of"] == sum(s["words"]["all"]["of"] for s in per_map) > 0
    assert got["all"]["long_lines"]["ink_px"] == sum(s["long_lines"]["ink_px"] for s in per_map) > 0
    # JSON as it is, and the same bytes from the same chances
    assert json.dumps(got) == json.dumps(scores.score_sheets(chances, "whole"))
    with pytest.raises(ValueError, match="1975"):
        scores.score_sheets({("1975", "1"): chances[sheets[0].series, sheets[0].sheet]}, "whole")


# --- the build: one detector, its scores on the test drawings, and the trials -----------------------------------------

def test_sheets_read_without_their_placements_train_alike_but_hold_nothing_out(maps):
    placed = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    shutil.rmtree(config.DATA_DIR / "warp")                    # no placement anywhere: none is read
    bare = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False, placement=False)
    for p, b in zip(placed, bare):
        assert b.world.shape == (0, 0, 2) and np.isnan(b.ft_per_px) and b.held == ()
        # the same crops and the same centres as with the placement: the training reads none of it
        for y, x, size, scale, turns in ((115, 140, 64, 1.0, 0), (320, 330, 96, 1.6, 3), (470, 200, 64, 0.6, 1)):
            got, want = data.crop_targets(b, y, x, size, scale, turns), data.crop_targets(p, y, x, size, scale, turns)
            assert all(np.array_equal(got[k], want[k]) for k in want)
        assert all(np.array_equal(g, w) for g, w in zip(data.centres(b, 4), data.centres(p, 4)))
        # but which px are in the validation area is not known, so it can't be held out or scored
        with pytest.raises(ValueError, match="placement"):
            data.in_validation(b, 0, 0, 8, 8)
    with pytest.raises(ValueError, match="placement"):
        data.load_sheets([paths.SERIES], hold_out_validation=True, placement=False)


def test_chance_on_some_px_is_the_whole_pages_there(monkeypatch):
    monkeypatch.setattr(predict, "TILE", 32)
    monkeypatch.setattr(predict, "PAD", 8)
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(width=8)).eval()
    rng = np.random.default_rng(3)
    ink = rng.random((100, 90)) < 0.2
    only = np.zeros(ink.shape, bool)
    only[40:50, 35:70] = True                    # in two tiles' columns, one tile's rows
    whole = predict.chance(net, ink, "cpu")
    some = predict.chance(net, ink, "cpu", only=only)
    assert np.array_equal(some[only], whole[only]) and whole[only].any()
    # the tiles holding px asked for are scored whole; the others are left 0
    tiles = np.zeros(ink.shape, bool)
    tiles[32:64, 32:96] = True
    assert np.array_equal(some[tiles], whole[tiles]) and not some[~tiles].any() and whole[~tiles].any()
    with pytest.raises(ValueError, match="page"):
        predict.chance(net, ink, "cpu", only=only[:-1])


# The weights a short fit of the stage's recipe gives on the two synthetic maps, held so that a change to how the
# detector trains is seen.  torch's CPU kernels add up in another order on another kind of machine, or with another
# number of threads for the maths libraries (conftest gives each pytest-xdist worker one), so the bits are held where
# they were recorded (the owner's Mac, in a worker and not), by machine and those threads; the losses everywhere: the
# first step's (before any update: the crops, the network and the loss) within 1e-5, the later steps' within 1e-3.
# AdamW turns the machines' tiny differences in gradients near zero into whole steps, and the loss on the ink alone
# leaves many of those: on GitHub's Linux the second and third steps' losses differ from the Mac's by 1.3e-4 and
# 2.2e-4, while a learning rate 10% lower moves the second step's text loss by 1.7%.  The recipe's name is
# held too, so a new DEFAULT_RECIPE is seen even where its first steps would train alike (re-record all three when it
# changes).
GUARD_RECIPE = "half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1"
GUARD_STEPS = 3
GUARD_WEIGHTS = {("Darwin", "arm64", "1", "1"): "4cae67dd7f6d797557d33caa1fe269fc65b4b30b6448181d83d44831f3ed4266",
                 ("Darwin", "arm64", None, None): "226bca5d04defad7871f2496f7bbe06d3ccccd616a5aaf6ecd0e8719c2eb165b"}
GUARD_LOSSES = [[0.9406127333641052, 2.443599224090576, 0.9502788782119751],       # text, kind, angle, each step
                [0.7165262699127197, 1.576537847518921, 0.8786576390266418],
                [0.6443136930465698, 1.25177800655365, 0.9018327593803406]]


def test_the_stages_recipe_trains_the_weights_recorded(maps, monkeypatch):
    assert stage.DEFAULT_RECIPE.name() == GUARD_RECIPE
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False, placement=False)
    real, parts = train.loss_fn, []

    def loss_fn(*args):
        loss, p = real(*args)
        parts.append(p.detach().tolist())
        return loss, p

    monkeypatch.setattr(train, "loss_fn", loss_fn)
    net = train.fit(sheets, dataclasses.replace(stage.DEFAULT_RECIPE, iterations=GUARD_STEPS), "cpu", seed=0)
    digest = hashlib.sha256()
    for k, v in net.state_dict().items():
        digest.update(k.encode() + v.contiguous().numpy().tobytes())
    assert np.allclose(parts[:1], GUARD_LOSSES[:1], rtol=1e-5, atol=0)
    assert np.allclose(parts[1:], GUARD_LOSSES[1:], rtol=1e-3, atol=0)
    want = GUARD_WEIGHTS.get((platform.system(), platform.machine(), os.environ.get("OMP_NUM_THREADS"),
                              os.environ.get("VECLIB_MAXIMUM_THREADS")))
    if want is not None:
        assert digest.hexdigest() == want


@pytest.fixture
def solver_dir(tmp_path, monkeypatch):
    """The solver's outputs in the test's folder."""
    solver = tmp_path / "solver"
    monkeypatch.setattr(paths, "SOLVER_DIR", solver)
    monkeypatch.setattr(paths, "TEXT_DIR", solver / "text")
    monkeypatch.setattr(paths, "TEXT_MODEL", solver / "text" / "model.pt")
    monkeypatch.setattr(paths, "TEXT_CHANCE_DIR", solver / "text" / "chance")
    monkeypatch.setattr(paths, "TEXT_SCORES_DIR", solver / "text-scores")
    monkeypatch.setattr(runs, "EXPERIMENTS_DIR", solver / "experiments")
    return solver


@pytest.fixture
def built(maps, solver_dir, monkeypatch):
    """The two synthetic maps as the build's: both trained on, the 1960 one also the test drawing; the 2000 sheet the
    one the walls read."""
    monkeypatch.setattr(paths, "TEXT_TRAIN_SERIES", (paths.SERIES, OLD))
    monkeypatch.setattr(paths, "TEXT_TEST_SERIES", (OLD,))
    monkeypatch.setattr(paths, "SHEETS", (SHEET,))
    return solver_dir


def _turns(monkeypatch, module):
    """Record the GPU's turns `module` takes (each (exclusive, device) at its start, None at its end) and, for the
    functions named later (`_inside`), whether each call is inside a turn."""
    turns = []

    @contextlib.contextmanager
    def turn(exclusive, device):
        turns.append((exclusive, device))
        yield
        turns.append(None)

    monkeypatch.setattr(module.gpu, "gpu_turn", turn)
    return turns


def _inside(monkeypatch, target, name, turns, calls):
    real = getattr(target, name)

    def wrapped(*args, **kwargs):
        calls.append((name, bool(turns) and turns[-1] is not None))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapped)


def _tree(root) -> dict[str, bytes]:
    """Every file under `root` by its path there, with its bytes."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_detector_trains_on_every_map_whole_and_scores_the_2000_sheets_in_one_exclusive_turn(built, monkeypatch):
    turns, calls, trained = _turns(monkeypatch, stage), [], []
    for target, name in ((data, "load_sheets"), (train, "loss_fn"), (predict, "chance")):
        _inside(monkeypatch, target, name, turns, calls)
    real_fit = train.fit
    monkeypatch.setattr(train, "fit", lambda sheets, *a: trained.extend(sheets) or real_fit(sheets, *a))
    monkeypatch.setattr(stage, "DEFAULT_RECIPE", TINY)
    shutil.rmtree(config.DATA_DIR / "warp")                    # it reads no placement
    paths.TEXT_CHANCE_DIR.mkdir(parents=True)
    (paths.TEXT_CHANCE_DIR / "an-old-sheet.png").write_bytes(b"from the last run")
    (paths.TEXT_DIR / "evaluation_nw.json").write_text("{}")   # a file no step of the stage writes
    s = stage.build_text(device="cpu")
    # the sheets are read before the turn; every step and every sheet scored is inside it, the one turn
    assert turns == [(True, "cpu"), None]
    assert calls == [("load_sheets", False), ("loss_fn", True), ("loss_fn", True), ("chance", True)]
    # every sheet of every training map, none with the area held out, none with its placement
    assert [(sd.series, sd.sheet, sd.held, sd.point_radius, sd.world.size) for sd in trained] == [
        (paths.SERIES, SHEET, (), data.READ_POINT_RADIUS, 0), (OLD, OLD_SHEET, (), data.READ_POINT_RADIUS, 0)]
    # the model and the 2000 sheets' chances, nothing else, nothing of the last run, no part folder left
    assert sorted(_tree(built)) == ["text/chance/2000-digital-zoning-p01.png", "text/model.pt"]
    saved = torch.load(paths.TEXT_MODEL)
    assert train.Recipe(**saved["recipe"]) == TINY and (saved["iterations"], saved["device"], saved["seed"]) == (
        2, "cpu", 0)
    # the chance is the saved model's on the page, as the scores stage reads it
    ink = data.load_page(SHEET) < grid.INK_BELOW
    want = predict.chance(predict.load_model(paths.TEXT_MODEL, "cpu"), ink, "cpu")
    assert np.array_equal(cv2.imread(str(paths.TEXT_CHANCE_DIR / f"{SHEET}.png"), cv2.IMREAD_GRAYSCALE), want)
    assert (s["iterations"], s["device"], s["trained_on"], s["scored"]) == (2, "cpu", 2, 1)
    assert list(s["seconds"]) == ["load", "train", "predict"]


def test_a_detector_run_that_stops_leaves_the_last_one_whole(built, monkeypatch):
    paths.TEXT_CHANCE_DIR.mkdir(parents=True)
    (paths.TEXT_CHANCE_DIR / f"{SHEET}.png").write_bytes(b"the last run's")
    paths.TEXT_MODEL.write_bytes(b"the last model")
    before = _tree(built)

    real = predict.chance

    def broken(*args, **kwargs):
        raise RuntimeError("stopped while scoring")

    monkeypatch.setattr(predict, "chance", broken)
    with pytest.raises(RuntimeError, match="stopped"):
        stage.build_text(TINY, device="cpu")
    assert {k: v for k, v in _tree(built).items() if k.startswith("text/")} == before
    # the next run clears what the stopped one left beside it, and replaces the last one whole
    monkeypatch.setattr(predict, "chance", real)
    stage.build_text(TINY, device="cpu")
    assert sorted(_tree(built)) == ["text/chance/2000-digital-zoning-p01.png", "text/model.pt"]


def test_the_scores_stage_scores_the_test_drawings_whole_with_the_stages_model(built, monkeypatch):
    with pytest.raises(FileNotFoundError, match="solver-text"):
        score_stage.build_scores("cpu")
    net = train.fit(data.load_sheets([paths.SERIES], False, placement=False), TINY, "cpu", 0)
    train.save_model(paths.TEXT_MODEL, net, TINY, "cpu", 0)
    paths.TEXT_SCORES_DIR.mkdir(parents=True)
    (paths.TEXT_SCORES_DIR / "evaluation_nw.json").write_text("{}")      # the quarter models' scores
    turns, calls = _turns(monkeypatch, score_stage), []
    for target, name in ((data, "load_sheets"), (predict, "chance"), (scores, "score_sheets")):
        _inside(monkeypatch, target, name, turns, calls)
    got = score_stage.build_scores("cpu")
    assert turns == [(True, "cpu"), None]
    assert calls == [("load_sheets", False), ("chance", True), ("score_sheets", False)]
    # every px of every sheet of the test drawings, by the stage's model; nothing else written, no time in it
    sheets = data.load_sheets([OLD], hold_out_validation=False)
    chance = predict.chance(predict.load_model(paths.TEXT_MODEL, "cpu"), data.page_ink(sheets[0]), "cpu")
    want = scores.score_sheets({(OLD, OLD_SHEET): chance}, "whole", sheets)
    assert got["scores"] == want and want["where"] == "whole" and list(want["series"]) == [OLD]
    assert sorted(_tree(paths.TEXT_SCORES_DIR)) == ["scores.json"]
    written = (paths.TEXT_SCORES_DIR / "scores.json").read_bytes()
    assert json.loads(written) == want
    score_stage.build_scores("cpu")
    assert (paths.TEXT_SCORES_DIR / "scores.json").read_bytes() == written


def test_the_text_commands_run_their_stages(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli
    got = []
    sd = _in_memory_sheet(_edge_page(), words=EDGE_WORDS, points=[EDGE_POINT])
    some = _score(sd, np.random.default_rng(0).integers(0, 256, (SIZE, SIZE), dtype=np.uint8), "whole")
    monkeypatch.setattr(stage, "build_text", lambda device=None: got.append(("text", device)) or {
        "iterations": 8000, "device": device, "trained_on": 87, "scored": 13,
        "seconds": {"load": 1.0, "train": 2.0, "predict": 3.0}})
    monkeypatch.setattr(score_stage, "build_scores", lambda device=None: got.append(("scores", device)) or {
        "scores": some, "seconds": {"load": 1.0, "predict": 2.0, "score": 3.0}})
    runner = CliRunner()
    res = runner.invoke(cli.app, ["solver", "text", "--device", "cpu"])
    assert res.exit_code == 0 and "87 sheets; 13 2000 sheets scored" in res.output
    res = runner.invoke(cli.app, ["solver", "text-scores", "--device", "cpu"])
    assert res.exit_code == 0 and f"{paths.SERIES}: AUC" in res.output and "all: AUC" in res.output
    assert got == [("text", "cpu"), ("scores", "cpu")]
    # one detector: no quarter to name; no device but the two
    assert runner.invoke(cli.app, ["solver", "text", "nw"]).exit_code != 0
    assert runner.invoke(cli.app, ["solver", "text", "--device", "cuda"]).exit_code != 0
    assert len(got) == 2


def test_a_recipe_is_written_as_settings_over_todays():
    assert trial.parse_recipe([]) == trial.parse_recipe("default") == trial.parse_recipe("") == train.Recipe()
    r = trial.parse_recipe("half_size=True ink_only=ink+1px iterations=2000 max_lr=6e-3 heads=text")
    assert r == train.Recipe(half_size=True, ink_only="ink+1px", iterations=2000, max_lr=6e-3, heads="text")
    assert trial.parse_recipe(["prefetch=yes", "half_size=off", "weight_average=0.999"]) == train.Recipe(
        prefetch=True, weight_average=0.999)
    for bad, said in (("iterations=2e3", "whole number"), ("half_size=maybe", "True or False"),
                      ("max_lr=fast", "a number"), ("speed=2", "no setting"), ("iterations", "name=value"),
                      ("crop=64 crop=96", "twice"), ("ink_only=all", "one of"), ("iterations=0", "at least 1")):
        with pytest.raises(ValueError, match=said):
            trial.parse_recipe(bad)
    # each recipe and seed its own folder; prefetch, which never changes the weights, is no part of its name
    assert trial.folder(r) == runs.EXPERIMENTS_DIR / f"text-{r.name()}"
    assert trial.folder(train.Recipe(), 2) == runs.EXPERIMENTS_DIR / "text-default-seed=2"
    assert trial.folder(dataclasses.replace(r, prefetch=True)) == trial.folder(r)


def test_trials_train_with_the_area_held_out_score_it_and_write_only_their_folders(built, tmp_path, monkeypatch):
    monkeypatch.setattr(scores, "_LINE_INK", {})
    first, second = TINY, dataclasses.replace(TINY, ink_only="ink")
    before = {k: v for k, v in _tree(tmp_path).items()}
    turns, calls, trained, lines = _turns(monkeypatch, trial), [], [], []
    real_fit = train.fit
    monkeypatch.setattr(train, "fit", lambda sheets, *a: trained.append([sd.held == (data.VALIDATION,)
                                                                          for sd in sheets])
                        or real_fit(sheets, *a))
    for target, name in ((data, "load_sheets"), (train, "fit"), (predict, "chance"), (scores, "score_sheets")):
        _inside(monkeypatch, target, name, turns, calls)
    real_lines = scores.long_lines
    monkeypatch.setattr(scores, "long_lines", lambda ink, ft: lines.append(ft) or real_lines(ink, ft))
    got = trial.run_trials([first, second], device="cpu")
    # the sheets read once, with the area held out; each recipe trained and predicted in its own turn, then scored
    assert turns == [(True, "cpu"), None, (True, "cpu"), None]
    assert calls == [("load_sheets", False),
                     ("fit", True), ("chance", True), ("chance", True), ("score_sheets", False),
                     ("fit", True), ("chance", True), ("chance", True), ("score_sheets", False)]
    assert trained == [[True, True], [True, True]]           # every sheet of both maps, the area held out
    assert len(lines) == 2                                    # each sheet's long lines found once, for both
    # only the recipes' folders are new, each with its four files, and each scored sheet's long lines kept for the
    # next trials; nothing else changed
    after = _tree(tmp_path)
    new = sorted(k for k in after if k not in before)
    folders = [f"solver/experiments/text-{r.name()}" for r in (first, second)]
    kept = [k for k in new if k.startswith("solver/experiments/line-masks/")]
    assert sorted(k.split("/")[3] for k in kept) == [OLD, paths.SERIES]
    assert sorted(k for k in new if k not in kept) == sorted(f"{f}/{name}" for f in folders for name in trial.FILES)
    assert {k: v for k, v in after.items() if k in before} == before
    for r, f, res in zip((first, second), folders, got):
        folder = tmp_path / f
        assert res["folder"] == folder and res["name"] == r.name()
        assert train.Recipe(**json.loads((folder / "recipe.json").read_text())) == r
        minutes = json.loads((folder / "minutes.json").read_text())
        assert list(minutes) == ["plain"] and list(minutes["plain"]) == [p for p in trial.PARTS if p != "test"]
        assert res["minutes"] == minutes["plain"] and res["test_scores"] is None
        # the scores are the validation area's, of the model's chance there, as a whole page's prediction gives it
        sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
        net = predict.load_model(folder / "model.pt", "cpu")
        chances = {(sd.series, sd.sheet): predict.chance(net, data.page_ink(sd), "cpu") for sd in sheets}
        want = scores.score_sheets(chances, "validation", sheets)
        assert res["scores"] == want == json.loads((folder / "scores.json").read_text())
        assert want["where"] == "validation" and want["all"]["words"]["all"]["of"] > 0
        assert torch.load(folder / "model.pt")["recipe"] == dataclasses.asdict(r)
    # the same recipe twice would write one folder
    with pytest.raises(ValueError, match="one folder"):
        trial.run_trials([first, dataclasses.replace(first, prefetch=True)], device="cpu")


def test_a_reused_trial_scores_the_model_in_its_folder_and_keeps_its_training_minutes(built, monkeypatch):
    folder = trial.folder(TINY)
    # nothing to reuse: refused before anything is read
    read = []
    real_load = data.load_sheets
    monkeypatch.setattr(data, "load_sheets", lambda *a, **k: read.append(a) or real_load(*a, **k))
    with pytest.raises(FileNotFoundError, match="no model"):
        trial.run_trials([TINY], device="cpu", reuse=True)
    assert not read
    # a model trained beforehand, by the recipe, with its training's minutes (as the baseline was)
    sheets = real_load([paths.SERIES, OLD], hold_out_validation=True)
    folder.mkdir(parents=True)
    train.save_model(folder / "model.pt", train.fit(sheets, TINY, "cpu", 0), TINY, "cpu", 0)
    (folder / "minutes.json").write_text(json.dumps({"load": 1.5, "train": 42.0}))
    model_bytes = (folder / "model.pt").read_bytes()
    monkeypatch.setattr(train, "fit", lambda *a, **k: pytest.fail("a reused trial trains nothing"))
    res, = trial.run_trials([TINY], device="cpu", reuse=True)
    assert (folder / "model.pt").read_bytes() == model_bytes
    minutes = json.loads((folder / "minutes.json").read_text())["plain"]     # a flat file is the plain run's
    assert list(minutes) == [p for p in trial.PARTS if p != "test"] and minutes["train"] == 42.0
    assert minutes["load"] != 1.5
    net = predict.load_model(folder / "model.pt", "cpu")
    want = scores.score_sheets({(sd.series, sd.sheet): predict.chance(net, data.page_ink(sd), "cpu")
                                for sd in sheets}, "validation", sheets)
    assert res["scores"] == want == json.loads((folder / "scores.json").read_text())
    # a model of another recipe or seed is not this recipe's
    for recipe, seed in ((dataclasses.replace(TINY, width=4), 0), (TINY, 1)):
        other = trial.folder(recipe, seed)
        other.mkdir(parents=True)
        shutil.copy(folder / "model.pt", other / "model.pt")
        with pytest.raises(ValueError, match="another recipe or seed"):
            trial.run_trials([recipe], device="cpu", seed=seed, reuse=True)


def test_the_long_lines_kept_on_disk_are_those_found_afresh_and_only_for_their_page_and_code(built, monkeypatch):
    monkeypatch.setattr(scores, "_LINE_INK", {})
    calls = []
    real = scores.long_lines
    monkeypatch.setattr(scores, "long_lines", lambda ink, ft: calls.append(ft) or real(ink, ft))
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    fresh = [scores.line_ink(sd) for sd in sheets]
    assert len(calls) == 2 and trial.keep_line_masks(sheets) == 2 and trial.keep_line_masks(sheets) == 0
    # another process (no answers kept in it) reads them back: the same px, and nothing found again
    monkeypatch.setattr(scores, "_LINE_INK", {})
    assert trial.read_line_masks(sheets) == 2
    assert all(np.array_equal(scores.line_ink(sd), f) for sd, f in zip(sheets, fresh)) and len(calls) == 2
    # another page under the same name, or another code finding the lines, has its own file: none is read
    monkeypatch.setattr(scores, "_LINE_INK", {})
    other = dataclasses.replace(sheets[0], ink_bits=sheets[0].ink_bits.copy())
    other.ink_bits[5, :] = ~other.ink_bits[5, :]       # another page: a row of it inverted
    assert trial.read_line_masks([other]) == 0
    code = trial.line_code
    monkeypatch.setattr(trial, "line_code", lambda: "another")
    assert trial.read_line_masks(sheets) == 0
    # a file whose label is not the sheet's (a clash of names) is not read
    monkeypatch.setattr(trial, "line_code", code)
    f, label = trial._line_file(sheets[1])
    with open(f, "wb") as out:
        np.savez(out, bits=np.packbits(fresh[1], axis=1), label=np.array(label + "x"))
    assert trial.read_line_masks([sheets[1]]) == 0


def test_a_prefetched_trial_keeps_the_plain_runs_minutes_beside_its_own(built, monkeypatch):
    monkeypatch.setattr(scores, "_LINE_INK", {})
    plain = TINY
    fast = dataclasses.replace(TINY, prefetch=True)
    folder = trial.folder(plain)
    assert trial.folder(fast) == folder
    trial.run_trials([plain], device="cpu")
    weights = torch.load(folder / "model.pt")["state_dict"]
    validation = (folder / "scores.json").read_bytes()
    first = json.loads((folder / "minutes.json").read_text())["plain"]
    res, = trial.run_trials([fast], device="cpu")
    minutes = json.loads((folder / "minutes.json").read_text())
    assert list(minutes) == ["plain", "prefetch"] and minutes["plain"] == first
    assert res["minutes"] == minutes["prefetch"]
    # the same weights, so the same scores
    again = torch.load(folder / "model.pt")["state_dict"]
    assert all(torch.equal(weights[k], again[k]) for k in weights)
    assert (folder / "scores.json").read_bytes() == validation
    # a reused prefetch run keeps its own training minutes, and the plain run's whole
    res, = trial.run_trials([fast], device="cpu", reuse=True)
    again = json.loads((folder / "minutes.json").read_text())
    assert again["plain"] == first and again["prefetch"]["train"] == minutes["prefetch"]["train"]


def test_a_trial_on_some_maps_has_its_own_folder_and_is_scored_on_whole_test_drawings(built, monkeypatch):
    monkeypatch.setattr(scores, "_LINE_INK", {})
    trained = []
    real_fit = train.fit
    monkeypatch.setattr(train, "fit", lambda sheets, *a: trained.append([sd.series for sd in sheets])
                        or real_fit(sheets, *a))
    # the maps in the training maps' order, whatever order they are asked in; never a test drawing or nothing
    assert trial.training_maps() == trial.training_maps([OLD, paths.SERIES]) == (paths.SERIES, OLD)
    assert trial.folder(TINY, 0, [paths.SERIES, OLD]) == trial.folder(TINY)
    assert trial.folder(TINY, 3, [OLD]) == runs.EXPERIMENTS_DIR / f"text-{TINY.name()}-train={OLD}-seed=3"
    for bad in (["1980"], []):
        with pytest.raises(ValueError, match="training map"):
            trial.training_maps(bad)
    with pytest.raises(ValueError, match="not a test map"):
        trial.test_maps([paths.SERIES])
    res, = trial.run_trials([TINY], device="cpu", train_series=[OLD], test_series=[OLD])
    folder = trial.folder(TINY, 0, [OLD])
    assert res["folder"] == folder and trained == [[OLD]] and not trial.folder(TINY).exists()
    assert set(trial.FILES) | {trial.TEST_FILE} == {f.name for f in folder.iterdir()}
    # the test drawings' scores: every px of their sheets, as the scores stage takes them
    sheets = data.load_sheets([OLD], hold_out_validation=False)
    net = predict.load_model(folder / "model.pt", "cpu")
    want = scores.score_sheets({(sd.series, sd.sheet): predict.chance(net, data.page_ink(sd), "cpu")
                                for sd in sheets}, "whole", sheets)
    assert res["test_scores"] == want == json.loads((folder / trial.TEST_FILE).read_text())
    assert "test" in json.loads((folder / "minutes.json").read_text())["plain"]


def test_batches_go_to_the_gpu_without_waiting_and_are_held_until_their_copy_is_done():
    events = []

    class Event:
        def __init__(self):
            self.n, self.synced = len(events), False
            events.append(self)

        def record(self):
            pass

        def synchronize(self):
            self.synced = True

    copies = train.Copies("cpu", event=Event)
    batches = [(torch.full((2,), float(i)),) for i in range(5)]
    for i, b in enumerate(batches):
        out = copies(b)
        assert torch.equal(out[0], b[0])
        # each batch held, with its event, until HELD steps later: then its event is waited for and it is let go
        assert [e.synced for e in events] == [j < i + 1 - train.HELD for j in range(i + 1)]
        assert [h[0] for h in copies.held] == batches[max(0, i + 1 - train.HELD):i + 1]
    copies.close()
    assert all(e.synced for e in events) and not copies.held
    # on the CPU no event is made and nothing held: the batch is its own copy
    plain = train.Copies("cpu")
    assert plain.event is None and plain(batches[0])[0] is batches[0][0] and not plain.held


def test_the_trial_command_reads_its_recipes_and_runs_them_together(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli
    asked = []
    sd = _in_memory_sheet(_edge_page(), words=EDGE_WORDS, points=[EDGE_POINT])
    some = _score(sd, np.zeros((SIZE, SIZE), np.uint8), "validation")

    def run_trials(recipes, device=None, seed=0, reuse=False, train_series=None, test_series=()):
        asked.append((recipes, device, seed, reuse, train_series, test_series))
        return [{"name": r.name(), "folder": trial.folder(r, seed, train_series), "scores": some,
                 "test_scores": some if test_series else None,
                 "minutes": {"load": 1.0, "train": 2.0, "predict": 0.5, "score": 0.25}} for r in recipes]

    monkeypatch.setattr(trial, "run_trials", run_trials)
    runner = CliRunner()
    res = runner.invoke(cli.app, ["solver", "text-trial", "--set", "half_size=True", "--set", "ink_only=ink",
                                  "--recipe", "iterations=2000", "--recipe", "default", "--seed", "1",
                                  "--device", "cpu"])
    assert res.exit_code == 0, res.output
    every = tuple(paths.TEXT_TRAIN_SERIES)
    assert asked == [([train.Recipe(half_size=True, ink_only="ink"), train.Recipe(iterations=2000), train.Recipe()],
                      "cpu", 1, False, every, ())]
    assert "half_size-ink_only=ink: AUC" in res.output and "text-iterations=2000-seed=1" in res.output
    assert runner.invoke(cli.app, ["solver", "text-trial", "--reuse", "--device", "cpu"]).exit_code == 0
    assert asked[-1] == ([train.Recipe()], "cpu", 0, True, every, ())
    # some of the maps (in the training maps' order) and test drawings to score on too
    res = runner.invoke(cli.app, ["solver", "text-trial", "--train-series", "1990-1996-zoning, 2000-digital-zoning",
                                  "--test-series", "1980", "--device", "cpu"])
    assert res.exit_code == 0, res.output
    assert asked[-1] == ([train.Recipe()], "cpu", 0, False, ("2000-digital-zoning", "1990-1996-zoning"), ("1980",))
    assert "-train=2000-digital-zoning+1990-1996-zoning" in res.output and "test 2000-digital-zoning: AUC" in res.output
    # a usage error, nothing run: a bad setting or map, or two recipes that are one (they would write one folder)
    for bad in (["--set", "speed=2"], ["--recipe", "iterations=0"], ["--device", "cuda"],
                ["--set", "half_size=True", "--recipe", "half_size=True prefetch=True"],
                ["--recipe", "default", "--recipe", "iterations=8000"],
                ["--train-series", "1980"], ["--train-series", ","], ["--test-series", "1960"]):
        res = runner.invoke(cli.app, ["solver", "text-trial", *bad])
        assert res.exit_code == 2, (bad, res.output)
    assert "one folder" in runner.invoke(cli.app, ["solver", "text-trial", "--set", "half_size=True", "--recipe",
                                                   "half_size=True"]).output
    assert len(asked) == 3


def _text_files(root: str, series_list, names) -> list[str]:
    return [f"{root}/{s}/{sheet}/{name}" for s in series_list for sheet in paths.TEXT_SHEETS[s] for name in names]


TEXTNET_CODE = ("textnet/data.py", "textnet/boxes.py", "textnet/model.py", "textnet/train.py", "textnet/predict.py")
# code a change to which must not retrain the detector or rescore it: the lots, the table, the merged page, the codes,
# the prints, the model client, the command line, the walls, the solve, their folders (runs), the GPU's turns, the
# trials
NOT_TEXT_CODE = ("solver/lots.py", "solver/table.py", "solver/codes.py", "solver/prints.py", "/llm.py", "/cli.py",
                 "solver/walls.py", "solver/wallevidence.py", "solver/textgrid.py", "solver/solve.py",
                 "solver/runs.py", "/gpu.py", "textnet/trial.py")


def test_one_detector_stage_reads_its_maps_sheets_file_by_file_and_no_placement():
    from sfzmaps import build
    table = build.stages()
    # one detector of the walls, its scores, and the code reader's own (`test_reader_stage_deps` in test_build); then
    # the text reader's three (`test_text_reader_stages_deps` in test_build)
    assert [n for n in table if n.startswith("solver-text")] == ["solver-text", "solver-text-scores",
                                                                  "solver-text-reader", "solver-text-crops",
                                                                  "solver-text-train", "solver-text-read"]
    st = table["solver-text"]
    assert build.stage_commands("solver-text") == [["solver", "text"]] and build.stage_outs(st) == ["data/solver/text"]
    assert st.code == ("sfzmaps.solver.textnet.stage",) and "sfzmaps.solver.runs" in st.cut
    deps = build.stage_deps(st)
    code = [d for d in deps if d.startswith("src/")]
    assert {f"src/sfzmaps/solver/{f}" for f in (*TEXTNET_CODE, "textnet/stage.py", "paths.py", "grid.py")} <= set(code)
    # nor the placements' code (it reads no placement; that code changes often and must not retrain the detector)
    assert not any(d.endswith((*NOT_TEXT_CODE, "textnet/scores.py", "textnet/score_stage.py", "solver/strokes.py",
                               "solver/page.py", "sfzmaps/transform.py")) for d in code)
    # of each training map's sheets only the page and Sol's points, of the saved answers Terra's boxes and the 2000
    # sheets' other passes: no placement (a warp retrains nothing), nothing of the prints or of another stage
    assert [d for d in deps if not d.startswith("src/")] == [
        "pyproject.toml", "uv.lock",
        *_text_files("data/work", paths.TEXT_TRAIN_SERIES, ("page.png", "district_labels.json")),
        "saved-reads/answers/text-on-every-sheet.jsonl.gz", "saved-reads/answers/experiments-2000/text-positions"]
    assert sum(len(paths.TEXT_SHEETS[s]) for s in paths.TEXT_TRAIN_SERIES) == 87
    assert set(paths.SHEETS) <= set(paths.TEXT_SHEETS[paths.SERIES])      # the pages it scores are among those read
    # after the stages that made those pages (frozen: they never run in a build)
    assert all(table[u].frozen for u in st.upstream) and len(st.upstream) == len(paths.TEXT_TRAIN_SERIES)


def test_the_scores_stage_reads_the_model_and_the_test_drawings_with_their_placements():
    from sfzmaps import build
    table = build.stages()
    st = table["solver-text-scores"]
    assert build.stage_commands("solver-text-scores") == [["solver", "text-scores"]]
    assert build.stage_outs(st) == ["data/solver/text-scores"]
    assert st.code == ("sfzmaps.solver.textnet.score_stage",) and "sfzmaps.solver.runs" in st.cut
    assert st.upstream == ("solver-text", *(f"warp-{s}" for s in paths.TEXT_TEST_SERIES))
    deps = build.stage_deps(st)
    code = [d for d in deps if d.startswith("src/")]
    assert {f"src/sfzmaps/solver/{f}" for f in (*TEXTNET_CODE, "textnet/scores.py", "textnet/score_stage.py",
                                                "strokes.py")} | {"src/sfzmaps/transform.py"} <= set(code)
    assert not any(d.endswith((*NOT_TEXT_CODE, "textnet/stage.py")) for d in code)
    # the stage's model (not its chances), and of each test drawing's sheets the page, Sol's points and the placement
    assert [d for d in deps if not d.startswith("src/")] == [
        "pyproject.toml", "uv.lock", "data/solver/text/model.pt",
        *_text_files("data/work", paths.TEXT_TEST_SERIES, ("page.png", "district_labels.json")),
        *_text_files("data/warp", paths.TEXT_TEST_SERIES, ("transform.json", "correction.npz")),
        "saved-reads/answers/text-on-every-sheet.jsonl.gz"]
    names = list(table)
    assert names.index("solver-text") < names.index("solver-text-scores") < names.index("solver-walls")


# --- half precision ---------------------------------------------------------------------------------------------


def test_precision_is_a_recipe_choice():
    assert "precision=float16" in train.Recipe(precision="float16").name()
    assert "precision" not in train.Recipe().name()
    with pytest.raises(ValueError):
        train.Recipe(precision="half")


def _fit(maps, seed=0):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    recipe = dataclasses.replace(SMALL, iterations=3, precision="float16")
    return recipe, train.fit(sheets, recipe, device="cpu", seed=seed)


def test_float16_fit_is_repeatable(maps):
    _, a = _fit(maps)
    _, b = _fit(maps)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert all(v.dtype == torch.float32 for v in sa.values() if v.is_floating_point())


def test_saved_float16_model_loads(maps, tmp_path):
    recipe, net = _fit(maps)
    f = train.save_model(tmp_path / "m.pt", net, recipe, "cpu")
    loaded = predict.load_model(f, "cpu")
    for k, v in net.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_the_stage_recipe_word_starts_a_trial_from_the_stages_recipe():
    assert trial.parse_recipe("stage") == stage.DEFAULT_RECIPE
    assert trial.parse_recipe(["stage"]) == stage.DEFAULT_RECIPE
    assert trial.parse_recipe("stage iterations=2000") == dataclasses.replace(stage.DEFAULT_RECIPE, iterations=2000)
    # without it, the old recipe: so existing trial folders keep their names
    assert trial.parse_recipe("default") == train.Recipe() == trial.parse_recipe("")
    assert trial.parse_recipe("iterations=2000") == train.Recipe(iterations=2000)


# --- the code reader's detectors: a quarter and the answered blocks held out ------------------------------------------

READER_SPLIT = (6000700.0, 2108700.0)  # the quarters' split moved through both synthetic pages: through page px
                                       # (384.0, 257.75) of the 2000 page and (328.5, 326.0) of the 1960 page
LOTS_ORIGIN = (4900, 3550)             # grid px (col, row) of the synthetic lots' top-left: both pages lie on them
LOTS_SHAPE = (750, 800)                # their rows and columns
LOT_PX = 10                            # grid px a square lot's side (40 ft)


def _synthetic_lots():
    """Square lots of LOT_PX grid px over a window of the grid, four to a block (2 x 2), every third block answered:
    each grid px's nearest lot (as lots.npz `grown`), each lot's block, and the answered blocks."""
    rows, cols = LOTS_SHAPE
    per_row = cols // LOT_PX
    r, c = np.mgrid[0:rows, 0:cols]
    grown = ((r // LOT_PX) * per_row + c // LOT_PX).astype(np.int32)
    lot_r, lot_c = np.divmod(np.arange(int(grown.max()) + 1), per_row)
    block = (lot_r // 2) * (per_row // 2) + lot_c // 2
    return grown, np.array([f"{b:04d}" for b in block]), {f"{b:04d}" for b in range(0, int(block.max()) + 1, 3)}


def _reference_held(series, sheet, q, grown, lot_block, blocks):
    """A page's px in quarter q and in an answered block, the plain way: each px's map point (`data.page_world`), its
    quarter by READER_SPLIT, and the lot of the grid px holding it.  Returns (either, the quarter, a block), bool."""
    gray = cv2.imread(str(config.WORK_DIR / series / sheet / "page.png"), cv2.IMREAD_GRAYSCALE)
    folder = config.DATA_DIR / "warp" / series / sheet
    x, y = data.page_world(SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder),
                           gray.shape)
    quarter = (x >= READER_SPLIT[0]).astype(int) + np.where(y >= READER_SPLIT[1], 0, 2) == q
    col = np.floor((x - grid.X0) / grid.FT_PER_PX).astype(int) - LOTS_ORIGIN[0]
    row = np.floor((grid.Y1 - y) / grid.FT_PER_PX).astype(int) - LOTS_ORIGIN[1]
    assert (col >= 0).all() and (col < LOTS_SHAPE[1]).all() and (row >= 0).all() and (row < LOTS_SHAPE[0]).all()
    block = np.isin(lot_block[grown[row, col]], sorted(blocks))
    return quarter | block, quarter, block


def _crop_of(mask, y, x, size, scale, turns):
    """A page raster's px as `data.crop_targets` crops them: moved inside the page, resized (nearest px), turned."""
    h, w = mask.shape
    src = int(round(size / scale))
    y0, x0 = int(np.clip(y - src // 2, 0, h - src)), int(np.clip(x - src // 2, 0, w - src))
    v = mask[y0:y0 + src, x0:x0 + src]
    if src != size:
        v = cv2.resize(v.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return np.rot90(v, turns)


def test_reader_detector_never_trains_on_its_quarter_or_held_blocks(maps, monkeypatch):
    monkeypatch.setattr(grid, "QUARTER_SPLIT", READER_SPLIT)
    grown, lot_block, blocks = _synthetic_lots()
    area = reader_stage.block_area(blocks, lot_block, grown, *LOTS_ORIGIN)
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    real_crop, real_loss, crops, weights = data.crop_targets, train.loss_fn, [], []
    monkeypatch.setattr(data, "crop_targets", lambda sd, *a, **k: crops.append((sd, *a)) or real_crop(sd, *a, **k))
    monkeypatch.setattr(train, "loss_fn", lambda out, t, k, a, w: weights.append(w.numpy().copy())
                        or real_loss(out, t, k, a, w))
    recipe = dataclasses.replace(stage.DEFAULT_RECIPE, iterations=4, batch=6, crop=64, width=8, prefetch=False)
    bitten = {"quarter": 0, "block": 0}
    for q in range(4):
        refs = {sd.sheet: _reference_held(sd.series, sd.sheet, q, grown, lot_block, blocks) for sd in sheets}
        held = data.hold_out(sheets, (data.Quarter(q), area))
        # no crop may centre on a px of the quarter or of an answered block, on text or on ink; held nothing, some do
        some = False
        for sd, plain in zip(held, sheets):
            either = refs[sd.sheet][0]
            for pool, every in zip(data.centres(sd, train.CENTRE_STRIDE), data.centres(plain, train.CENTRE_STRIDE)):
                assert len(pool) and not either[pool[:, 0], pool[:, 1]].any()
                some |= bool(either[every[:, 0], every[:, 1]].any())
        assert some
        # a fit on both maps: every crop it draws centres off them, and the loss weighs none of their px
        crops.clear()
        weights.clear()
        train.fit(held, recipe, "cpu", seed=q)
        w = np.concatenate(weights)
        assert len(crops) == len(w) == recipe.iterations * recipe.batch
        for (sd, y, x, size, scale, turns), wi in zip(crops, w):
            assert sd.held[0] == data.Quarter(q) and sd.held[1] is area
            assert not refs[sd.sheet][0][y, x]
            either, quarter, block = (_crop_of(m, y, x, size, scale, turns) for m in refs[sd.sheet])
            assert not real_crop(sd, y, x, size, scale, turns)["known"][either].any() and not wi[either].any()
            # px that would have carried loss, held nothing
            plain = next(p for p in sheets if p.sheet == sd.sheet)
            would = train.loss_weight([real_crop(plain, y, x, size, scale, turns)], recipe.ink_only)[0]
            bitten["quarter"] += int((would & quarter).sum())
            bitten["block"] += int((would & block & ~quarter).sum())
    assert bitten["quarter"] and bitten["block"]


def test_unknown_around_sol_points_covers_the_word(maps):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    # the radius: the longest Terra word of kind "zoning code", rounded up: R-1-D on the 1960 page, 100 px (RH-1 on the
    # 2000 page is 80; the street names and the block number are no codes)
    got = reader_stage.measure_radius(sheets)
    assert (got["px"], got["longest"]["text"], got["longest"]["sheet"], got["longest"]["length_px"], got["words"],
            got["left_out"]) == (100, "R-1-D", OLD_SHEET, 100.0, 2, 0)
    # a word whose centre is in an area held out is not measured: R-1-D's centre, page px (250, 170) of the 1960 page,
    # is on grid px (5310, 3772)
    around = data.GridArea(np.ones((16, 21), bool), col0=5300, row0=3765)
    got = reader_stage.measure_radius(sheets, (around,))
    assert (got["px"], got["longest"]["text"], got["words"], got["left_out"]) == (80, "RH-1", 1, 1)
    with pytest.raises(ValueError, match="no Terra word"):
        reader_stage.measure_radius(sheets[:1], (data.GridArea(np.ones((2000, 2000), bool), 4000, 3000),))
    # px within the radius of a point are unknown, unless in a word (text); px beyond every point's are known, unless
    # in another pass's box.  So a word of up to the radius, the point anywhere on it, running any way, is unknown
    radius = 100
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    for wide, narrow in zip(data.load_sheets([paths.SERIES, OLD], hold_out_validation=False, point_radius=radius),
                            sheets):
        assert wide.point_radius == radius and narrow.point_radius == data.READ_POINT_RADIUS == 30
        t = data.crop_targets(wide, SIZE // 2, SIZE // 2, SIZE, 1.0, 0)        # the whole page
        known, word = t["known"], t["text"] > 0
        other = np.zeros((SIZE, SIZE), bool)
        for b in (data.other_boxes(wide.sheet) if wide.series == paths.SERIES else []):
            other |= _box_mask(other.shape, b["box"])
        dist = np.stack([np.hypot(cols - x, rows - y) for x, y in wide.points])
        for d in dist:
            within = (d <= radius - 1) & ~word
            assert within.any() and not known[within].any()
        beyond = (dist.min(axis=0) >= radius + 1) & ~other
        assert beyond.any() and known[beyond].all()
        # the stage's radius of 30 px left most of them known
        short = data.crop_targets(narrow, SIZE // 2, SIZE // 2, SIZE, 1.0, 0)["known"]
        assert short[(dist.min(axis=0) >= 31) & (dist.min(axis=0) <= radius - 1) & ~word & ~other].all()


def test_outputs_returns_chance_kind_and_direction(monkeypatch):
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(width=8, half_size=True)).eval()
    ink = np.random.default_rng(3).random((100, 90)) < 0.2
    o = predict.outputs(net, ink, "cpu")
    assert (o.chance.shape, o.chance.dtype, o.kind.shape, o.kind.dtype) == ((100, 90), np.uint8, (100, 90), np.uint8)
    assert (o.kind_p.shape, o.kind_p.dtype, o.direction.shape, o.direction.dtype) == (
        (4, 100, 90), np.float16, (2, 100, 90), np.float16)
    # the chance is `chance`'s; the kinds' probabilities sum to 1, the most likely is the kind; the direction is a
    # unit vector
    assert np.array_equal(o.chance, predict.chance(net, ink, "cpu"))
    p = o.kind_p.astype(np.float32)
    assert np.allclose(p.sum(axis=0), 1, atol=2e-3)
    top = np.sort(p, axis=0)
    clear = top[-1] - top[-2] > 1e-2
    assert clear.any() and np.array_equal(o.kind[clear], p.argmax(axis=0)[clear])
    length = np.hypot(*o.direction.astype(np.float32))
    assert (length > 0).all() and np.allclose(length, 1, atol=2e-3)
    # the page in one tile: the network's own outputs, the kinds by softmax and the direction scaled to unit length
    with torch.no_grad():
        raw = net(torch.from_numpy(np.pad(ink, ((0, (-100) % net.multiple), (0, (-90) % net.multiple)))
                                   .astype(np.float32))[None, None])[0, :, :100, :90]
    assert np.allclose(p, torch.softmax(raw[1:5], 0).numpy(), atol=1e-3)
    assert np.array_equal(o.kind, torch.argmax(raw[1:5], 0).numpy().astype(np.uint8))
    cs = raw[5:7].numpy()
    assert np.allclose(o.direction.astype(np.float32), cs / np.hypot(*cs), atol=1e-3)
    # tiles holding none of the px asked for are not scored: chance 0, kind NO_KIND, no probability, no direction
    monkeypatch.setattr(predict, "TILE", 32)
    monkeypatch.setattr(predict, "PAD", 8)
    only = np.zeros(ink.shape, bool)
    only[40:50, 35:70] = True
    whole, some = predict.outputs(net, ink, "cpu"), predict.outputs(net, ink, "cpu", only=only)
    tiles = np.zeros(ink.shape, bool)
    tiles[32:64, 32:96] = True
    for a, b in zip(whole, some):
        assert np.array_equal(a[..., tiles], b[..., tiles])
    assert not some.chance[~tiles].any() and (some.kind[~tiles] == predict.NO_KIND).all()
    assert not some.kind_p[:, ~tiles].any() and not some.direction[:, ~tiles].any()
    # a network trained for text alone has no kind or direction to give
    with pytest.raises(ValueError, match="text alone"):
        predict.outputs(train.build_net(train.Recipe(width=8, heads="text")).eval(), ink, "cpu")


@pytest.fixture
def reader_built(built, monkeypatch):
    """The two synthetic maps as the reader's stage reads them: its folder in the test's, the quarters split through
    both pages, and every third block of the synthetic lots answered."""
    monkeypatch.setattr(paths, "READER_TEXT_DIR", built / "text-reader")
    monkeypatch.setattr(grid, "QUARTER_SPLIT", READER_SPLIT)
    grown, lot_block, blocks = _synthetic_lots()
    monkeypatch.setattr(reader_stage, "held_blocks", lambda: sorted(blocks))
    monkeypatch.setattr(reader_stage, "load_block_area",
                        lambda b: reader_stage.block_area(b, lot_block, grown, *LOTS_ORIGIN))
    return built, grown, lot_block, blocks


def test_the_reader_stage_trains_four_detectors_in_one_turn_and_gives_each_px_its_quarters(reader_built, monkeypatch):
    solver, grown, lot_block, blocks = reader_built
    turns, calls, trained = _turns(monkeypatch, reader_stage), [], []
    for target, name in ((data, "load_sheets"), (train, "loss_fn"), (predict, "outputs")):
        _inside(monkeypatch, target, name, turns, calls)
    real_fit = train.fit
    monkeypatch.setattr(train, "fit", lambda sheets, *a: trained.append(sheets) or real_fit(sheets, *a))
    s = reader_stage.build_reader_text("cpu", TINY)
    # the sheets are read before the turn; the four trainings and every quarter of the sheet scored in it, the one turn
    assert turns == [(True, "cpu"), None]
    assert calls == [("load_sheets", False)] + [("loss_fn", True)] * 8 + [("outputs", True)] * 4
    # each on every sheet of both maps, with its quarter and the answered blocks held out and the measured radius
    area = reader_stage.block_area(blocks, lot_block, grown, *LOTS_ORIGIN)
    radius = reader_stage.measure_radius(data.load_sheets([paths.SERIES, OLD], False), (area,))
    assert len(trained) == 4
    for q, sheets in enumerate(trained):
        assert [(sd.series, sd.sheet, sd.point_radius) for sd in sheets] == [
            (paths.SERIES, SHEET, radius["px"]), (OLD, OLD_SHEET, radius["px"])]
        for sd in sheets:
            assert sd.held[0] == data.Quarter(q) and np.array_equal(sd.held[1].mask, area.mask)
            assert (sd.held[1].col0, sd.held[1].row0) == (area.col0, area.row0) and len(sd.held) == 2
    # its folder, nothing else, no part folder left
    root = paths.READER_TEXT_DIR
    assert sorted(_tree(solver)) == sorted(
        [f"text-reader/{SHEET}/{f}" for f in ("chance.png", "direction.npz", "kind.npz")]
        + ["text-reader/held.json", "text-reader/scores.json"] + [f"text-reader/models/{q}.pt" for q in paths.QUARTERS])
    # each px from the detector whose held-out quarter holds it
    sd, = data.load_sheets([paths.SERIES], hold_out_validation=False)
    ink = data.page_ink(sd)
    chance = cv2.imread(str(root / SHEET / "chance.png"), cv2.IMREAD_GRAYSCALE)
    with np.load(root / SHEET / "kind.npz") as z:
        kind, p = z["kind"], z["p"]
    with np.load(root / SHEET / "direction.npz") as z:
        direction = z["direction"]
    assert (kind.dtype, p.dtype, p.shape, direction.dtype, direction.shape) == (
        np.uint8, np.float16, (4, SIZE, SIZE), np.float16, (2, SIZE, SIZE))
    covered = np.zeros((SIZE, SIZE), int)
    for q, name in enumerate(paths.QUARTERS):
        mine = _reference_held(paths.SERIES, SHEET, q, grown, lot_block, blocks)[1]
        covered += mine
        saved = torch.load(root / "models" / f"{name}.pt")
        assert train.Recipe(**saved["recipe"]) == TINY and (saved["device"], saved["seed"]) == ("cpu", 0)
        o = predict.outputs(predict.load_model(root / "models" / f"{name}.pt", "cpu"), ink, "cpu")
        assert mine.any() and np.array_equal(chance[mine], o.chance[mine]) and np.array_equal(kind[mine], o.kind[mine])
        assert np.array_equal(p[:, mine], o.kind_p[:, mine])
        assert np.array_equal(direction[:, mine], o.direction[:, mine])
    assert (covered == 1).all()
    # what was held out, and how the radius was measured
    held = json.loads((root / "held.json").read_text())
    assert held == {"quarters": list(paths.QUARTERS), "split": list(READER_SPLIT), "blocks": sorted(blocks),
                    "block_grid_px": int(area.mask.sum()), "unknown_radius": radius,
                    "recipe": dataclasses.asdict(TINY), "seed": 0}
    # each detector's scores on its own quarter: every word of the 2000 page counted once, in the quarter of its centre
    got = json.loads((root / "scores.json").read_text())
    assert got == s["scores"] and list(got["quarters"]) == list(paths.QUARTERS)
    assert sum(v["words"]["all"]["of"] for v in got["quarters"].values()) == 5
    assert (s["iterations"], s["device"], s["trained_on"], s["scored"], s["blocks"], s["radius"]) == (
        2, "cpu", 2, 1, len(blocks), radius)
    assert list(s["seconds"]) == ["load", "train", "predict", "score"]
    # the same inputs give the same bytes
    before = _tree(root)
    reader_stage.build_reader_text("cpu", TINY)
    assert _tree(root) == before


def test_the_text_reader_command_runs_its_stage(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli
    got = []
    radius = {"px": 328, "longest": {"series": "1968", "sheet": "p18", "text": "SH", "length_px": 328.0,
                                     "centre": [1, 2]}, "words": 6197, "left_out": 3821, "quantiles": {}}
    quarter = {"words": {"zoning code": {"found": 9, "of": 10}, "all": {"found": 9, "of": 10}}, "words_found": 0.9}
    monkeypatch.setattr(reader_stage, "build_reader_text", lambda device=None: got.append(device) or {
        "iterations": 4000, "device": device, "trained_on": 87, "scored": 13, "blocks": 715, "radius": radius,
        "scores": {"quarters": {q: quarter for q in paths.QUARTERS}},
        "seconds": {"load": 1.0, "train": 2.0, "predict": 3.0, "score": 4.0}})
    runner = CliRunner()
    res = runner.invoke(cli.app, ["solver", "text-reader", "--device", "cpu"])
    assert res.exit_code == 0, res.output
    for said in ("715 answered blocks held out", "unknown radius 328 px: SH",
                 "se on its own quarter: words found 90.0%"):
        assert said in res.output
    assert runner.invoke(cli.app, ["solver", "text-reader", "--device", "cuda"]).exit_code != 0
    assert got == ["cpu"]
