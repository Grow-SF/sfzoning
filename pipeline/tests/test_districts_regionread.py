"""Per-region read (ruling R26): unlabelled regions named by a vision call over a numbered
mosaic, inheritance only as the fallback."""
import re

import cv2
import numpy as np
import pytest
from shapely.geometry import Point

import synthetic_labelmap as sl
from sfzmaps import config, llm, reading
from sfzmaps.districts import labels, regionread
from sfzmaps.districts.labels import DistrictLabel


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(reading.config, "VISION_CONCURRENCY", 1)
    monkeypatch.setattr(regionread, "ENABLED", True)
    llm.CallCounter.reset()


def lab(code, x, y, kind="use", conf=0.9):
    return DistrictLabel(code=code, kind=kind, x=x, y=y, confidence=conf)


# TL and MID carry printed codes; BL and RIGHT (and the ISLAND box) are boxed but unlabelled.
PRINTED = {"TL": "R-1", "MID": "C-2"}


class FakeClient:
    """Answers each tile from `answer(n, candidates) -> (code, conf)`; records the mosaics."""

    def __init__(self, answer):
        self.answer, self.calls, self.images, self.prompts = answer, 0, [], []

    def call(self, png, prompt, schema):
        self.calls += 1
        self.images.append(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR))
        self.prompts.append(prompt)
        cands = {int(n): [c.strip() for c in rest.split(",")]
                 for n, rest in re.findall(r"^(\d+): (.*)$", prompt, re.M)}
        items = []
        for n, cs in sorted(cands.items()):
            code, conf = self.answer(n, cs)
            items.append({"index": n, "code": code, "confidence": conf})
        return schema(items=items)


def _setup():
    lm = sl.build(codes=PRINTED)
    regs = labels.regions(lm.img, lm.exclude)
    labs = [lab(c, *sl.CENTRES[n]) for n, c in PRINTED.items()]
    idx = {n: regs.index(next(r for r in regs if r.contains(Point(*xy))))
           for n, xy in {"TL": (100, 100), "BL": (100, 700), "MID": (600, 120),
                         "RIGHT": (1100, 700), "ISLAND": (960, 230)}.items()}
    return lm, regs, labs, idx


def test_mosaic_numbers_regions_and_outlines_them():
    lm, regs, labs, idx = _setup()
    todo = sorted([idx["BL"], idx["RIGHT"], idx["ISLAND"]])
    fake = FakeClient(lambda n, cs: ("R-1", 0.9))
    out = regionread.read_regions(lm.img, regs, todo, labs, "use", client=fake)
    assert fake.calls == 1
    img = fake.images[0]                                   # BGR
    assert max(img.shape[:2]) <= config.VISION_MAX_EDGE
    magenta = (img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] > 200)
    boxes = regionread.tile_boxes(len(todo))
    for x0, y0, x1, y1 in boxes:                           # every tile outlines its region
        assert magenta[y0:y1, x0:x1].sum() > 50
    # candidates: the codes on the sheet, nearest first; "none" is allowed
    assert "1: " in fake.prompts[0] and "none" in fake.prompts[0]
    assert set(out) == set(todo) and all(c == "R-1" for c, _ in out.values())


def test_answers_map_back_by_tile_number():
    lm, regs, labs, idx = _setup()
    todo = sorted([idx["BL"], idx["RIGHT"]])
    answers = {1: ("C-2", 0.9), 2: ("R-1", 0.8)}
    out = regionread.read_regions(lm.img, regs, todo, labs, "use",
                                  client=FakeClient(lambda n, cs: answers[n]))
    assert out == {todo[0]: ("C-2", 0.9), todo[1]: ("R-1", 0.8)}


def test_none_low_confidence_and_unknown_codes_are_dropped():
    lm, regs, labs, idx = _setup()
    todo = sorted([idx["BL"], idx["RIGHT"], idx["ISLAND"]])
    answers = {1: ("none", 0.9), 2: ("R-1", 0.2), 3: ("Q-9", 0.95)}
    out = regionread.read_regions(lm.img, regs, todo, labs, "use",
                                  client=FakeClient(lambda n, cs: answers[n]))
    assert out == {}


def test_batches_and_cache():
    lm, regs, labs, idx = _setup()
    todo = [idx["BL"], idx["RIGHT"], idx["ISLAND"]]
    fake = FakeClient(lambda n, cs: ("C-2", 0.7))
    import sfzmaps.districts.regionread as rr
    old = rr.MAX_TILES
    rr.MAX_TILES = 2
    try:
        a = regionread.read_regions(lm.img, regs, todo, labs, "use", client=fake)
        assert fake.calls == 2
        boom = FakeClient(lambda n, cs: 1 / 0)
        b = regionread.read_regions(lm.img, regs, todo, labs, "use", client=boom)
        assert boom.calls == 0 and a == b                  # cached by llm.ask_json
    finally:
        rr.MAX_TILES = old


def test_tiny_regions_are_not_read():
    lm, regs, labs, idx = _setup()
    fake = FakeClient(lambda n, cs: ("C-2", 0.9))
    from shapely.geometry import box
    regs2 = regs + [box(30, 30, 40, 40)]
    out = regionread.read_regions(lm.img, regs2, [len(regs2) - 1], labs, "use", client=fake)
    assert out == {} and fake.calls == 0


def test_read_overrides_inheritance_and_inheritance_is_the_fallback():
    lm, regs, labs, idx = _setup()
    # without reads: BL touches TL (R-1) and MID (C-2), so plurality / inheritance decides
    reads = {idx["BL"]: ("C-2", 0.9)}
    out = labels.assign_regions(regs, labs, reads=reads)
    got = {n: [(c, cf) for p, c, cf in out if p.equals(regs[i])][0] for n, i in idx.items()}
    assert got["BL"][0] == "C-2"
    assert got["BL"][1] == pytest.approx(0.9 * labels.READ_CONF)
    assert got["TL"][0] == "R-1" and got["MID"][0] == "C-2"
    assert got["RIGHT"][0] == "C-2"              # not read: inherits from its only neighbour


def test_assign_parcels_reads_unlabelled_regions():
    lm = sl.build(codes=PRINTED)
    d = sl.write_sheet("1960", "s1", lm)
    labs = [lab(c, *sl.CENTRES[n]) for n, c in PRINTED.items()]
    labels.save_labels(d / "district_labels.json", labs, page_sha256=labels.page_sha(d))
    fake = FakeClient(lambda n, cs: ("R-1", 0.9))
    out = labels.assign_parcels(lm.parcels, d, lm.transform, client=fake).set_index("blklot")
    assert fake.calls >= 1
    assert out.loc[f"{9 * 20 + 1:04d}001", "code"] == "R-1"          # block (0, 9): BL, read
    assert out.loc[f"{9 * 20 + 11:04d}001", "code"] == "R-1"         # block (10, 9): RIGHT, read
    assert out.loc[f"{5 * 20 + 6:04d}002", "code"] == "C-2"          # MID keeps its label
    regionread.ENABLED = False
    try:
        off = labels.assign_parcels(lm.parcels, d, lm.transform,
                                    client=FakeClient(lambda n, cs: 1 / 0)).set_index("blklot")
    finally:
        regionread.ENABLED = True
    assert off.loc[f"{9 * 20 + 11:04d}001", "code"] == "C-2"         # inherited instead
