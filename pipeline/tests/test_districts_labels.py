import json

import cv2
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

import synthetic_labelmap as sl
from sfzmaps import answers
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
    """A heavy line that stops short where it runs along a thin block outline still separates
    the districts: the outline under the gap carries the bridge."""
    lm = sl.build(codes=CODES)
    lm.img[200:260, 404:418] = 255          # 60 px gap in the x = 410 line ...
    lm.img[200:260, 409:413] = 0            # ... where only a block outline (under it) remains
    regs = labels.regions(lm.img, lm.exclude)
    assert len(regs) == 5
    assert _region_at(regs, 100, 100) is not _region_at(regs, 600, 120)


def test_no_bridge_across_blank_paper():
    """A gap over blank paper is not bridged: on two scans of one drawing such bridges cut the
    districts in different places (#34); the split around the gap decides instead."""
    lm = sl.build(codes=CODES)
    lm.img[200:260, 404:418] = 255          # the same gap, nothing printed in it
    regs = labels.regions(lm.img, lm.exclude)
    assert _region_at(regs, 100, 100) is _region_at(regs, 600, 120)
    off = labels.INK_SHARE
    try:
        labels.INK_SHARE = None             # without the ink test the line is bridged
        assert len(labels.regions(lm.img, lm.exclude)) == 5
    finally:
        labels.INK_SHARE = off


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


def test_a_lone_label_in_a_region_of_another_code_keeps_only_its_surroundings():
    """Mt Vernon 1960: one R-3 printed beside a small box in an unclosed region full of R-1."""
    region = box(0, 0, 2000, 400)
    r1 = [lab("R-1", x, 200) for x in (100, 900, 1700)]
    out = labels.assign_regions([region], r1 + [lab("R-3", 1300, 200)])
    by = {c: p for p, c, _ in out}
    assert by["R-3"].area == pytest.approx(np.pi * labels.SPLIT_MINOR_RADIUS_PX ** 2, rel=0.02)
    assert by["R-1"].contains(Point(1150, 50))          # beyond the disk: the main code
    # printed twice it is a district of its own (Forest Hill 1980), even if outnumbered
    many = [lab("R-1", x, 200) for x in (100, 300, 500, 700, 900)]
    two = {c: p for p, c, _ in labels.assign_regions(
        [region], many + [lab("R-3", 1500, 100), lab("R-3", 1700, 300)])}
    assert two["R-3"].contains(Point(1900, 50))
    # two codes printed alike: neither dominates, an even split
    even = {c: p for p, c, _ in labels.assign_regions(
        [region], [lab("R-1", 500, 200), lab("R-3", 1500, 200)])}
    assert even["R-3"].area == pytest.approx(even["R-1"].area)


def test_a_park_keeps_its_outline_closed_by_every_bridge():
    """McLaren Park 1980: its outline's gaps cross streets, so the ink-checked regions merge it
    with the homes around it; the fully bridged outline, holding only P, stays P."""
    merged = [(box(0, 0, 1000, 400), "RH-1", 0.8)]
    closed = [box(0, 0, 300, 400), box(300, 0, 1000, 400)]
    labs = [lab("P", 100, 200), lab("RH-1", 600, 200)]
    out = {c: p for p, c, _ in labels.closed_parks(merged, closed, labs)}
    assert out["P"].area == pytest.approx(300 * 400)
    assert out["RH-1"].area == pytest.approx(700 * 400)
    # a closed region that also prints another code is not a park outline
    shared = labels.closed_parks(merged, closed, labs + [lab("RH-1", 200, 200)])
    assert [c for _, c, _ in shared] == ["RH-1"]


def test_a_split_measures_distance_around_district_lines_not_across_them(monkeypatch):
    """A district line that stops short of closing (a gap at its foot) still keeps each side
    with the label printed on it; a straight-line split hands the far side of the line to the
    label just across it (Forest Hill 1980, Mt Vernon 1964)."""
    monkeypatch.setattr(labels, "GEO_SPLIT", True)
    region = box(0, 0, 1000, 400)
    wall = np.zeros((400, 1000), bool)
    wall[0:300, 495:505] = True                         # the line, open at the bottom
    labs = [lab("R-1", 450, 50), lab("R-2", 900, 350)]
    straight = {c: p for p, c, _ in labels.assign_regions([region], labs)}
    around = {c: p for p, c, _ in labels.assign_regions([region], labs, wall=wall)}
    assert straight["R-1"].contains(Point(600, 50))     # nearer in a straight line ...
    assert around["R-2"].contains(Point(600, 50))       # ... but the line is in the way
    assert around["R-1"].contains(Point(100, 350)) and around["R-2"].contains(Point(950, 50))
    assert around["R-1"].area + around["R-2"].area == pytest.approx(region.area, rel=0.01)


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
    answers.write(d / "district_labels.json", [])   # made from no answers (fixture)
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


def test_a_parcel_knows_how_far_its_printed_label_is(tmp_path):
    """use_label_ft: distance from a parcel to the label its code comes from; inf when its
    region inherited the code (no label printed in it)."""
    lm = sl.build(codes=CODES)
    d = sl.write_sheet("1960", "s1", lm)
    labs = [l for l in _label_list() if l.code != "R-2"] + [lab("R1", *sl.CENTRES["BL"])]
    labels.save_labels(d / "district_labels.json", labs, page_sha256=labels.page_sha(d))
    answers.write(d / "district_labels.json", [])   # made from no answers (fixture)
    out = labels.assign_parcels(lm.parcels, d, lm.transform).set_index("blklot")
    tl = lm.transform.to_world(np.array([sl.CENTRES["TL"]]))[0]
    near, far = out.loc["0001001"], out.loc[f"{9 * 20 + 1:04d}001"]
    c = lm.parcels.set_index("blklot").geometry["0001001"].centroid
    assert near["use_label_ft"] == pytest.approx(np.hypot(c.x - tl[0], c.y - tl[1]))
    assert far["code"] == "R-1"                  # printed "R1": one district, one code
    assert near["use_label_ft"] < 1e9 and np.isinf(out["height_label_ft"]).all()


def test_a_far_label_in_an_open_region_leaves_the_lot_unread(tmp_path, monkeypatch):
    """#38: a lot coded only from a piece that reaches the page edge, from a label farther than
    OPEN_REACH_FT, gets no code from this sheet; enclosed districts keep theirs."""
    lm = sl.build(codes=CODES)
    d = sl.write_sheet("1960", "s1", lm)
    labels.save_labels(d / "district_labels.json", _label_list(), page_sha256=labels.page_sha(d))
    answers.write(d / "district_labels.json", [])   # made from no answers (fixture)
    island = f"{sl.ISLAND_BLOCK[1] * 20 + sl.ISLAND_BLOCK[0] + 1:04d}001"
    monkeypatch.setattr(labels, "OPEN_REACH_FT", None)
    every = labels.assign_parcels(lm.parcels, d, lm.transform).set_index("blklot")
    monkeypatch.setattr(labels, "OPEN_REACH_FT", 1.0)
    near = labels.assign_parcels(lm.parcels, d, lm.transform).set_index("blklot")
    assert every["code"].notna().all()
    assert near.loc[island, "code"] == "P"                      # enclosed: kept
    assert near.loc["0001001", "code"] is None                  # TL reaches the page edge
    assert near["code"].isna().sum() < len(near)


def test_label_fixes_recode_move_drop_and_add_read_labels():
    """inputs/<series>/label_fixes.csv: a person's corrections, matched to the nearest read label
    of the code within FIX_PX; the cached read itself is left alone."""
    labs = [lab("M-1", 100, 100), lab("R-3", 500, 500), lab("C-2", 900, 900), lab("M-1", 300, 300)]
    fixes = [{"x": "110", "y": "95", "code": "M-1", "fix_code": "RH-1"},           # misread
             {"x": "505", "y": "500", "code": "R-3", "fix_code": "", "fix_x": "520", "fix_y": "560"},
             {"x": "900", "y": "900", "code": "C-2", "fix_code": "-"},             # not a code
             {"x": "50", "y": "60", "code": "", "fix_code": "C-2"},                 # missed
             {"x": "700", "y": "700", "code": "R-1", "fix_code": "R-2"}]            # no match
    out = {(l.code, l.x, l.y) for l in labels.apply_label_fixes(labs, fixes)}
    assert out == {("RH-1", 100, 100), ("R-3", 520, 560), ("M-1", 300, 300), ("C-2", 50, 60)}


def test_read_district_labels_applies_the_series_label_fixes(tmp_path, monkeypatch):
    from sfzmaps import inputs

    lm = sl.build(codes=CODES)
    d = sl.write_sheet("1960", "s1", lm)
    labels.save_labels(d / "district_labels.json", _label_list(), page_sha256=labels.page_sha(d),
                       key=labels.read_key("1960"))
    answers.write(d / "district_labels.json", [])   # made from no answers (fixture)
    fixes = tmp_path / "label_fixes.csv"
    x, y = sl.CENTRES["MID"]
    fixes.write_text(f"sheet,x,y,code,fix_code,fix_x,fix_y,note\ns1,{x},{y},C-2,C-3,,,\ns2,{x},{y},C-2,-,,,\n")
    monkeypatch.setattr(inputs, "label_fixes_path", lambda s: fixes)
    codes = {l.code for l in labels.read_district_labels("1960", "s1", client=Fake(lambda *a: 1 / 0))}
    assert "C-3" in codes and "C-2" not in codes
    assert {l.code for l in labels.load_labels(d / "district_labels.json")} >= {"C-2"}   # cache as read


def test_twin_sheets_pair_the_two_scans_of_one_drawing():
    assert labels.twin_of("1996-2000-zoning", "1996-2000-zoning-p35") == ("2000-digital-zoning", "2000-digital-zoning-p11")
    assert labels.twin_of("2000-digital-zoning", "2000-digital-zoning-p01") == ("1996-2000-zoning", "1996-2000-zoning-p25")
    assert labels.twin_of("1996-2000-zoning", "1996-2000-zoning-p20") is None   # not a zoning sheet's twin
    assert labels.twin_of("1980", "20180102115319417-p04") is None


def test_register_finds_the_twin_scans_shift_and_scale_from_shared_labels():
    lm = sl.build(codes=CODES)
    m_true = np.float32([[1.005, 0.0, -30.0], [0.0, 1.005, 45.0]])          # other -> img
    inv = cv2.invertAffineTransform(m_true)
    other = cv2.warpAffine(lm.img, inv, (lm.img.shape[1], lm.img.shape[0]), borderValue=255)
    pts = [(x, y) for x in range(150, 1100, 190) for y in range(150, 1100, 210)]
    labs = [lab("R-1", x, y) for x, y in pts]
    other_labs = [lab("R-1", *(inv @ np.array([x, y, 1.0]))) for x, y in pts]
    m = labels.register(lm.img, other, labs, other_labs)
    assert m is not None and np.allclose(m, m_true, atol=0.5)
    assert labels.register(lm.img, other, labs[:3], other_labs[:3]) is None     # too few to fit


def test_hyphenate():
    assert [labels.hyphenate(c) for c in ("R1", "R4", "R1D", "RH1(D)", "R-1", "P", "NC-2", None)] == \
        ["R-1", "R-4", "R-1-D", "RH-1(D)", "R-1", "P", "NC-2", None]
    assert labels.hyphenate("UPPER MARKET NCD") == "UPPER MARKET NCD"


def test_height_labels_fill_height_code():
    lm = sl.build()
    d = sl.write_sheet("2000-digital-height", "s1", lm)
    labs = [lab(c, *sl.CENTRES[n], kind="height") for n, c in
            {"TL": "40-X", "BL": "40-X", "MID": "65-A", "RIGHT": "40-X", "ISLAND": "OS"}.items()]
    labels.save_labels(d / "district_labels.json", labs, page_sha256=labels.page_sha(d))
    answers.write(d / "district_labels.json", [])   # made from no answers (fixture)
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
    answers.write(d / "district_labels.json", [])   # made from no answers (fixture)
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
    again = labels.read_district_labels("1960", "s1", client=Fake(lambda *a: 1 / 0),
                                        scale=0.5, tile=400, overlap=40)
    assert again == out
    assert art["tile"] == 400 and art["overlap"] == 40 and art["scale"] == 0.5
    assert art["inputs_sha256"] == labels.inputs_sha256()


def _legacy_artifact(tmp_path):
    lm = sl.build(codes=CODES)
    d = sl.write_sheet("1960", "s1", lm)
    lab = {"code": "R-1", "kind": "use", "x": 10.0, "y": 10.0, "confidence": 0.9}
    (d / "district_labels.json").write_text(json.dumps({"page_sha256": labels.page_sha(d),
                                                        "labels": [lab]}))
    answers.write(d / "district_labels.json", [])
    return d


def test_labels_cache_rereads_when_tiling_or_scale_changes(tmp_path):
    d = _legacy_artifact(tmp_path)
    fake = Fake(lambda img, schema, n: schema(labels=[]))
    labels.read_district_labels("1960", "s1", client=fake, tile=400)
    assert fake.calls >= 1
    art = json.loads((d / "district_labels.json").read_text())
    assert art["tile"] == 400 and art["labels"] == []
    n = fake.calls
    labels.read_district_labels("1960", "s1", client=fake, scale=0.3, tile=400)
    assert fake.calls > n


def test_legacy_labels_artifact_fresh_under_defaults(tmp_path):
    """Artifacts written before the key was stored (page sha only) match the current
    defaults: no reread, no call."""
    d = _legacy_artifact(tmp_path)
    assert labels.labels_fresh(d)
    out = labels.read_district_labels("1960", "s1", client=Fake(lambda *a: 1 / 0))
    assert [l.code for l in out] == ["R-1"]


def test_labels_cache_rereads_when_prompt_changes(tmp_path, monkeypatch):
    d = _legacy_artifact(tmp_path)
    monkeypatch.setattr(labels, "DISTRICT_PROMPT", labels.DISTRICT_PROMPT + " Be careful.")
    assert not labels.labels_fresh(d)
    fake = Fake(lambda img, schema, n: schema(labels=[]))
    labels.read_district_labels("1960", "s1", client=fake)
    assert fake.calls >= 1


def test_existing_data_artifacts_still_fresh(monkeypatch):
    """The district_labels.json / district_legend.json artifacts already in pipeline/data
    (read before the key was stored) stay fresh: the new key must not force rereads."""
    from sfzmaps.districts import pattern

    work = config.PIPELINE_DIR / "data" / "work"
    arts = sorted(work.glob("*/*/district_labels.json"))
    legends = sorted(work.glob("*/*/district_legend.json"))
    if not arts and not legends:
        pytest.skip("no pipeline/data artifacts")
    monkeypatch.setattr(config, "WORK_DIR", work)
    monkeypatch.setattr(config, "INPUTS_DIR", config.PIPELINE_DIR / "inputs")  # label_scale.json
    stale = [str(a.parent.relative_to(work)) for a in arts
             if (a.parent / "page.png").exists() and not labels.labels_fresh(a.parent)]
    stale += [str(p.relative_to(work)) for p in legends
              if not pattern.legend_fresh(p, p.parent.parent.name)]
    assert len(arts) > 0 and stale == []


def _labs(n_use, n_height):
    from sfzmaps.districts.labels import DistrictLabel
    return ([DistrictLabel(code="R-1", kind="use", x=0, y=0, confidence=1)] * n_use
            + [DistrictLabel(code="40-X", kind="height", x=0, y=0, confidence=1)] * n_height)


def test_sheet_role():
    from sfzmaps.districts.labels import sheet_role
    assert sheet_role(_labs(40, 3)) == "use"
    assert sheet_role(_labs(2, 30)) == "height"
    assert sheet_role(_labs(20, 20)) == "use"         # a combined use + height sheet reads both
    assert sheet_role(_labs(3, 2)) == "skip"          # special-use / sign sheets: few codes
    assert sheet_role([]) == "skip"


def test_overlay_labels_are_not_use_codes():
    from sfzmaps.districts.labels import DistrictLabel, use_code_labels
    labs = [DistrictLabel(code=c, kind="use", x=0, y=0, confidence=1) for c in
            ("NORTH OF MARKET RES. S.U.D.", "RINCON HILL RES. SUD", "VAN NESS SPECIAL USE DISTRICT",
             "NC-2 & FINANCIAL RUSD", "RC-4", "RINCON SUD-CU")]
    assert [lb.code for lb in use_code_labels(labs)] == ["NC-2", "RC-4"]


def test_inheritance_propagates_through_unlabelled_chain():
    a, b, c = box(0, 0, 100, 100), box(106, 0, 206, 100), box(212, 0, 312, 100)
    out = labels.assign_regions([a, b, c], [lab("R-1", 50, 50)])
    got = {round(p.bounds[0]): (code, conf) for p, code, conf in out}
    assert got[106][0] == "R-1" and got[212][0] == "R-1"
    assert got[212][1] < got[106][1] < 0.9              # each hop costs confidence


def test_disagreeing_neighbours_longest_contact_wins_at_low_confidence():
    # mid touches R-1 along its 100-long side and C-2 only along a 20-long stub
    left, mid, stub = box(0, 0, 100, 100), box(106, 0, 206, 100), box(212, 80, 312, 100)
    out = labels.assign_regions([left, mid, stub], [lab("R-1", 50, 50), lab("C-2", 260, 90)])
    m = [(c, conf) for p, c, conf in out if p.equals(mid)][0]
    assert m[0] == "R-1" and m[1] <= labels.PLURALITY_CONF


def test_hairline_lot_lines_do_not_lower_the_heavy_threshold():
    """Digital-era sheets draw lot lines as 1 px hairlines, more ink than anything else; the
    thin-stroke width must come from the block outlines, or every block outline reads heavy."""
    lm = sl.build()
    img = lm.img.copy()
    for y in range(20, img.shape[0] - 20, 6):          # dense 1 px hatching: a sea of hairlines
        img[y, 20:img.shape[1] - 20] = np.minimum(img[y, 20:img.shape[1] - 20], 0)
    m = labels.heavy_line_mask(img)
    x0, y0, x1, y1 = sl.block_px(1, 1)
    assert m[y0 + 10:y1 - 10, x0:x0 + sl.THIN].mean() < 0.3    # block outline still thin
    assert m[300:500, 409:412].mean() > 0.95


def test_public_is_never_inherited_by_plurality():
    # mid touches a park (P) along its long side and R-1 along a stub: P must not spread
    left, mid, stub = box(0, 0, 100, 100), box(106, 0, 206, 100), box(212, 80, 312, 100)
    out = labels.assign_regions([left, mid, stub], [lab("P", 50, 50), lab("R-1", 260, 90)])
    m = [c for p, c, _ in out if p.equals(mid)][0]
    assert m == "R-1"


def test_split_caps_public_cells():
    """A park's P sharing an open region with a district label takes only a disk around its
    label; the rest of the region goes to the district (2000-digital: P flooded blocks)."""
    region = box(0, 0, 2000, 400)
    out = labels.assign_regions([region], [lab("P", 100, 200), lab("RH-2", 1900, 200)])
    by = {c: p for p, c, _ in out}
    assert by["P"].area <= 3.15 * labels.SPLIT_P_RADIUS_PX ** 2 + 1
    assert by["RH-2"].contains(Point(500, 200)) and by["RH-2"].contains(Point(1000, 200))
    assert abs(by["P"].area + by["RH-2"].area - region.area) < 1


def test_small_commercial_clusters_do_not_spread():
    assert "OS" in labels.NO_PLURALITY          # open-space height district, like P
    left, mid, stub = box(0, 0, 100, 100), box(106, 0, 206, 100), box(212, 80, 312, 100)
    out = labels.assign_regions([left, mid, stub], [lab("NC-1", 50, 50), lab("RH-1(D)", 260, 90)])
    assert [c for p, c, _ in out if p.equals(mid)] == ["RH-1(D)"]


def test_an_old_artifact_whose_answers_arent_cached_is_kept_without_paying(tmp_path):
    """Made before answers were recorded, and its answers aren't cached under today's prompt:
    kept as it is, no call, no sidecar (the baseline lists it)."""
    d = _legacy_artifact(tmp_path)
    answers.sidecar(d / "district_labels.json").unlink()
    out = labels.read_district_labels("1960", "s1", client=Fake(lambda *a: 1 / 0))
    assert [l.code for l in out] == ["R-1"]
    assert not answers.sidecar(d / "district_labels.json").exists()


def test_an_old_artifact_is_replayed_from_the_cache_and_kept_byte_for_byte(tmp_path):
    d = _legacy_artifact(tmp_path)
    art = d / "district_labels.json"
    fake = Fake(lambda img, schema, n: schema(labels=[]))
    labels.read_district_labels("1960", "s1", client=fake, tile=400)     # reads, caches answers
    recorded = answers.read(art)
    assert recorded and all(k.startswith("llm/") for k in recorded)
    answers.sidecar(art).unlink()                                        # as if made before sidecars
    before, calls = (art.read_bytes(), art.stat().st_mtime_ns), fake.calls
    with llm.recording() as outer:
        labels.read_district_labels("1960", "s1", client=fake, tile=400)
    assert fake.calls == calls                                           # replayed from the cache
    assert answers.read(art) == recorded and outer == set(recorded)      # and consumed
    assert (art.read_bytes(), art.stat().st_mtime_ns) == before
