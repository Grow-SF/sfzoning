import json

import cv2
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

import synthetic_labelmap as sl
from sfzmaps import config, llm, reading
from sfzmaps.districts import labels
from sfzmaps.districts.labels import DistrictLabel


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(reading.config, "VISION_CONCURRENCY", 1)
    llm.CallCounter.reset()


def lab(code, x, y, kind="use", conf=0.9):
    return DistrictLabel(code=code, kind=kind, x=x, y=y, confidence=conf)


CODES = {"TL": "R-1", "BL": "R-2", "MID": "C-2", "RIGHT": "R-3", "ISLAND": "P"}


def _label_list(codes=CODES):
    return [lab(c, *sl.CENTRES[n]) for n, c in codes.items()]


def _region_at(regs, x, y):
    hits = [r for r in regs if r.contains(Point(x, y))]
    assert len(hits) == 1, f"{len(hits)} regions at {(x, y)}"
    return hits[0]


# --- boundaries -------------------------------------------------------------

def test_heavy_vs_thin():
    lm = sl.build()
    m = labels.heavy_line_mask(lm.img)
    assert m[300:500, 409:412].mean() > 0.95            # heavy district line
    assert m[409:412, 100:300].mean() > 0.95
    x0, y0, x1, y1 = sl.block_px(1, 1)
    assert m[y0 + 10:y1 - 10, x0:x0 + sl.THIN].mean() < 0.05   # thin block outline
    assert m[y0:y0 + sl.THIN, x0 + 10:x1 - 10].mean() < 0.05


def test_regions_closed():
    lm = sl.build(codes=CODES)
    regs = labels.regions(lm.img, lm.exclude)
    assert len(regs) == 5
    assert all(r.is_valid for r in regs)
    for name, (x, y) in {"TL": (100, 100), "BL": (100, 700), "MID": (600, 120),
                         "RIGHT": (1100, 700), "ISLAND": (960, 230)}.items():
        _region_at(regs, x, y)
    tl = _region_at(regs, 100, 100)
    assert 380 * 380 < tl.area < 410 * 410
    # the printed code sits inside its region (letters are holes only if large)
    assert _region_at(regs, *sl.CENTRES["MID"]) is _region_at(regs, 600, 120)


def test_regions_bridge_gap_in_heavy_line():
    """A heavy line that stops short (a drafting gap, or a stretch drawn over a thin block
    outline) still separates the districts."""
    lm = sl.build(codes=CODES)
    lm.img[200:260, 404:418] = 255          # 60 px gap in the x = 410 line ...
    lm.img[200:260, 400:402] = 0            # ... where only the thin block outline remains
    regs = labels.regions(lm.img, lm.exclude)
    assert len(regs) == 5
    assert _region_at(regs, 100, 100) is not _region_at(regs, 600, 120)


def test_filled_box_is_a_region():
    """A tiny district box whose bold code fills it (all heavy ink) is still a region, so its
    label doesn't land in (and split) the surrounding district."""
    lm = sl.build(codes=CODES)
    lm.img[585:625, 580:630] = 0
    regs = labels.regions(lm.img, lm.exclude)
    assert len(regs) == 6
    box_reg = _region_at(regs, 605, 605)
    assert box_reg.area < 50 * 40 * 1.2
    out = labels.assign_regions(regs, _label_list() + [lab("RH-2", 605, 605)])
    assert [c for p, c, _ in out if p.equals(box_reg)] == ["RH-2"]
    mid = _region_at(regs, 600, 120)
    assert [c for p, c, _ in out if p.equals(mid)] == ["C-2"]


# --- assignment -------------------------------------------------------------

def test_one_label_per_region():
    lm = sl.build(codes=CODES)
    regs = labels.regions(lm.img, lm.exclude)
    out = labels.assign_regions(regs, _label_list())
    for name, code in CODES.items():
        got = [c for p, c, _ in out if p.buffer(1).contains(Point(sl.CENTRES[name]))]
        assert got == [code]


def test_conflicting_labels_split():
    region = box(0, 0, 400, 100)
    out = labels.assign_regions([region], [lab("R-1", 50, 50), lab("R-3", 350, 50, conf=0.8)])
    assert sorted(c for _, c, _ in out) == ["R-1", "R-3"]
    by = {c: (p, conf) for p, c, conf in out}
    assert by["R-1"][0].contains(Point(150, 50)) and by["R-3"][0].contains(Point(250, 50))
    assert abs(by["R-1"][0].area - 20000) < 1 and abs(by["R-3"][0].area - 20000) < 1
    assert by["R-1"][1] < 0.9 and by["R-3"][1] < 0.8    # a split costs confidence
    # same code twice is not a conflict
    one = labels.assign_regions([region], [lab("R-1", 50, 50), lab("R-1", 350, 50)])
    assert len(one) == 1 and one[0][1] == "R-1"


def test_label_on_heavy_ink_snaps_to_nearest_region():
    a, b = box(0, 0, 100, 100), box(110, 0, 210, 100)
    out = labels.assign_regions([a, b], [lab("R-1", 103, 50), lab("R-2", 160, 50)])
    assert {c for p, c, _ in out if p.equals(a)} == {"R-1"}


def test_unlabelled_region_inherits_only_if_agree():
    left, mid, right = box(0, 0, 100, 100), box(106, 0, 206, 100), box(212, 0, 312, 100)
    agree = labels.assign_regions([left, mid, right], [lab("R-1", 50, 50), lab("R-1", 260, 50)])
    m = [(c, conf) for p, c, conf in agree if p.equals(mid)][0]
    assert m[0] == "R-1" and m[1] < 0.9
    differ = labels.assign_regions([left, mid, right], [lab("R-1", 50, 50), lab("C-2", 260, 50)])
    assert [c for p, c, _ in differ if p.equals(mid)] == [None]
    alone = labels.assign_regions([mid], [])
    assert alone[0][1] is None


# --- parcels ----------------------------------------------------------------

def test_parcel_majority_region(tmp_path):
    lm = sl.build(codes=CODES)
    d = sl.write_sheet("1960", "s1", lm)
    labels.save_labels(d / "district_labels.json", _label_list(), page_sha256=labels.page_sha(d))
    # a parcel straddling the x=410 district line: 70% left of it (R-1 side), 30% right
    straddle = sl.px_box_world(340, 200, 440, 240)
    p = lm.parcels.copy()
    p.loc[len(p)] = {"blklot": "9999001", "block_num": "9999", "geometry": straddle}
    out = labels.assign_parcels(p, d, lm.transform).set_index("blklot")
    assert out.loc["9999001", "code"] == "R-1"
    assert 0.5 < out.loc["9999001", "confidence"] < out.loc["0001001", "confidence"]
    x0, y0, x1, y1 = sl.block_px(*sl.ISLAND_BLOCK)
    blk = f"{sl.ISLAND_BLOCK[1] * 20 + sl.ISLAND_BLOCK[0] + 1:04d}"
    assert out.loc[blk + "001", "code"] == "P"
    assert out.loc["0001001", "code"] == "R-1"                  # block (0, 0): TL
    assert out.loc[f"{9 * 20 + 1:04d}001", "code"] == "R-2"    # block (0, 9): BL
    assert out.loc[f"{5 * 20 + 6:04d}002", "code"] == "C-2"    # block (5, 5): MID
    assert out.index.name == "blklot" and set(out.columns) >= {"code", "height_code", "confidence"}
    assert out["height_code"].isna().all()


def test_height_labels_fill_height_code():
    lm = sl.build()
    d = sl.write_sheet("2000-digital-height", "s1", lm)
    labs = [lab(c, *sl.CENTRES[n], kind="height") for n, c in
            {"TL": "40-X", "BL": "40-X", "MID": "65-A", "RIGHT": "40-X", "ISLAND": "OS"}.items()]
    labels.save_labels(d / "district_labels.json", labs, page_sha256=labels.page_sha(d))
    out = labels.assign_parcels(lm.parcels, d, lm.transform).set_index("blklot")
    assert out.loc["0001001", "height_code"] == "40-X"
    assert out["code"].isna().all()


def test_fallback_nearest_label():
    lm = sl.build(districts=False)
    d = sl.write_sheet("1960", "s1", lm)
    x0, y0, x1, y1 = sl.block_px(0, 0)
    x2, y2, x3, y3 = sl.block_px(3, 0)
    labs = [lab("R-1", x0 + 10, y0 + 30), lab("C-2", x1 - 10, y0 + 30), lab("M-1", x2 + 40, y2 + 30)]
    labels.save_labels(d / "district_labels.json", labs, page_sha256=labels.page_sha(d))
    out = labels.assign_parcels(lm.parcels, d, lm.transform).set_index("blklot")
    assert out.loc["0001001", "code"] == "R-1" and out.loc["0001002", "code"] == "C-2"
    assert out.loc["0004001", "code"] == "M-1" and out.loc["0004002", "code"] == "M-1"
    assert out.loc["0001001", "confidence"] == pytest.approx(0.3)
    assert pd.isna(out.loc["0002001", "code"])                # no label in its block


# --- reading ----------------------------------------------------------------

class Fake:
    def __init__(self, fn):
        self.fn, self.calls, self.prompts = fn, 0, []

    def call(self, png, prompt, schema):
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        self.calls += 1
        self.prompts.append(prompt)
        return self.fn(img, schema, self.calls)


def test_read_district_labels_offsets_dedupes_and_caches():
    lm = sl.build(codes=CODES)
    d = sl.write_sheet("1960", "s1", lm, page_json={"kind": "map", "mask_boxes": [
        {"x0": 1000, "y0": 700, "x1": 1220, "y1": 860}]})

    def fn(img, schema, n):
        return schema(labels=[
            {"code": "r-2", "kind": "use", "x": 10, "y": 20, "confidence": 0.8},
            {"code": "40-X", "kind": "height", "x": 5, "y": 5, "confidence": 0.7},
            {"code": "Z", "kind": "use", "x": 5000, "y": 5, "confidence": 0.9},   # off-tile
        ])

    fake = Fake(fn)
    out = labels.read_district_labels("1960", "s1", client=fake, scale=0.5, tile=400, overlap=40)
    assert fake.calls >= 1 and "district" in fake.prompts[0]
    first = [l for l in out if l.code == "R-2"]
    assert first and (first[0].x, first[0].y) == (20.0, 40.0)   # tile (0, 0) at scale 0.5
    assert all(l.code != "Z" for l in out)
    assert not any(l.x > 1000 and l.y > 700 for l in out)        # inside the mask box
    assert {l.kind for l in out} == {"use", "height"}
    # no two identical codes closer than the dedupe radius
    for a in out:
        assert sum(b.code == a.code and abs(b.x - a.x) <= labels.DEDUPE_PX
                   and abs(b.y - a.y) <= labels.DEDUPE_PX for b in out) == 1
    art = json.loads((d / "district_labels.json").read_text())
    assert art["page_sha256"] == labels.page_sha(d) and len(art["labels"]) == len(out)
    again = labels.read_district_labels("1960", "s1", client=Fake(lambda *a: 1 / 0))
    assert again == out
