import numpy as np
import cv2
import pytest

from sfzmaps import llm, reading
from sfzmaps.reading import Box, Label, PageInfo, _Labels


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    llm.CallCounter.reset()


class Fake:
    def __init__(self, fn):
        self.fn, self.shapes, self.calls = fn, [], 0

    def call(self, png, prompt, schema):
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        self.shapes.append(img.shape[:2])
        self.calls += 1
        return self.fn(img, prompt, schema, self.calls)


def page(tmp_path, w, h, ink=lambda a: None):
    a = np.full((h, w), 255, np.uint8)
    ink(a)
    p = tmp_path / "p.png"
    cv2.imwrite(str(p), a)
    return p


def lab(x, y, text="26", conf=0.9, kind="block_number"):
    return Label(kind=kind, text=text, x=x, y=y, confidence=conf)


def full_ink(a):
    a[::20, :] = 0
    a[:, ::20] = 0


def test_read_labels_offsets_tile_coords(tmp_path):
    p = page(tmp_path, 3600, 400, full_ink)  # scale .5 -> two tile columns
    seen = []

    def fn(img, prompt, schema, n):
        seen.append(n)
        return _Labels(labels=[lab(10, 10, f"t{n}")])

    out = reading.read_labels(p, scale=0.5, client=Fake(fn))
    col1 = [l for l in out if l.text == "t2"][0]
    assert (col1.x, col1.y) == (round((10 + 1568 - 160) / 0.5), 20)
    assert [l.text for l in out if l.text == "t1"][0] and \
        [l for l in out if l.text == "t1"][0].x == 20


def test_read_labels_dedupes_overlap(tmp_path):
    p = page(tmp_path, 3600, 400, full_ink)  # tiles at x0=0 and 1408 (scaled)
    # same physical label seen from both tiles: scaled x 1450 in tile0, 42 in tile1
    def fn(img, prompt, schema, n):
        return _Labels(labels=[lab(1450, 50) if n == 1 else lab(42, 50)])

    out = reading.read_labels(p, scale=0.5, client=Fake(fn))
    assert len(out) == 1


def test_blank_tiles_not_sent(tmp_path):
    def corner(a):
        a[:300, :300] = 0
    p = page(tmp_path, 4000, 4000, corner)
    c = Fake(lambda *a: _Labels(labels=[]))
    reading.read_labels(p, scale=1.0, client=c)
    assert c.calls == 1  # only the tile touching the top-left corner


def sent(p, skip):
    llm.CallCounter.reset()
    reading.read_labels(p, scale=0.5, skip=skip, client=Fake(lambda *a: _Labels(labels=[])))
    return llm.CallCounter.live + llm.CallCounter.hits


def test_masked_tiles_not_sent(tmp_path):
    p = page(tmp_path, 3600, 400, full_ink)  # scaled tiles cover full-res x 0..3136 and 2816..3600
    skip = np.zeros((400, 3600), bool)
    assert sent(p, skip) == 2
    skip[:, :2000] = True
    assert sent(p, skip) == 2  # left tile still has unmasked pixels
    skip[:, :3200] = True
    assert sent(p, skip) == 1  # left tile fully masked
    skip[:, :] = True
    assert sent(p, skip) == 0


def test_classify_thumbnail_max_1024(tmp_path):
    p = page(tmp_path, 4000, 2000, full_ink)
    c = Fake(lambda *a: PageInfo(kind="map", sheet_number="7",
                                 mask_boxes=[Box(x0=0, y0=0, x1=512, y1=256)]))
    info = reading.classify_page(p, client=c)
    assert max(c.shapes[0]) <= 1024
    assert info.mask_boxes == [Box(x0=0, y0=0, x1=2000, y1=1000)]


def test_exclusion_mask_includes_boxes(tmp_path):
    a = np.full((200, 300), 255, np.uint8)
    info = PageInfo(kind="map", mask_boxes=[Box(x0=10, y0=20, x1=50, y1=60)])
    m = reading.exclusion_mask(a, info)
    assert m[30, 20] and not m[100, 200]


def test_calibrate_picks_smallest_adequate_scale(tmp_path):
    p = page(tmp_path, 1000, 1000, full_ink)
    truth = [(50 + 90 * i, 100 + 70 * i) for i in range(10)]

    def fn(img, prompt, schema, n):
        s = img.shape[0] / 1000
        keep = 8 if abs(s - 0.4) < 0.01 or abs(s - 0.33) < 0.01 else 10
        return _Labels(labels=[lab(round(x * s), round(y * s), str(i))
                               for i, (x, y) in enumerate(truth[:keep])])

    rep = {}
    assert reading.calibrate_scale(p, client=Fake(fn), series="t", report=rep) == 0.5
    assert rep[0.4] == 0.8
    from sfzmaps import config
    assert config.label_scale("t") == 0.5
    config.LABEL_SCALE.pop("t")
