from pathlib import Path

import numpy as np
import pytest

from sfzmaps import pages


def touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


@pytest.fixture(autouse=True)
def fake_pdfinfo(monkeypatch):
    monkeypatch.setattr(pages, "pdf_page_count", lambda p: 3 if "multi" in p.name or p.stem == "1929" else 1)


def make_tree(root: Path) -> Path:
    o = root / "1948" / "originals"
    for n in ["1", "14", "2-and-3", "index", "setback", "planning-code", "12-and-13"]:
        touch(o / f"{n}.jpg")
    touch(o / "1.xcf")
    for n in ["2", "3", "1", "12", "13", "2-and-3"]:
        touch(root / "1948" / "aligned" / f"{n}.jpg")
    touch(root / "1948" / "1948.png")
    touch(root / "1921" / "2018.pdf")
    touch(root / "1929" / "1929.pdf")
    for n in [1, 2]:
        touch(root / "1960" / "aligned" / f"{n}.pdf")
    touch(root / "1960" / "originals" / "1960.pdf")
    touch(root / "1960" / "20180102104912410.pdf")
    touch(root / "1960" / "cropped" / "1.pdf")
    touch(root / "1980 Plus" / "1990 to 1996 Height Maps.PDF")
    touch(root / "1980 Plus" / "1980 - 1990  Height & Bulk Maps.PDF")
    touch(root / "1980 Plus" / "1980 - 1990  Special Use District Maps.PDF")
    touch(root / "1980 Plus" / "2000 to digital ZoningMaps.PDF")
    touch(root / "1980 Plus" / "1996 to 2000 SU-SS-Prev-Coastal.PDF")
    touch(root / "1980 Plus" / "2000 to digital Zoning Maps Introduction.PDF")
    return root


def by_series(ps):
    d = {}
    for p in ps:
        d.setdefault(p.series, []).append(p)
    return d


def test_discover_1948_prefers_single_sheets(tmp_path):
    d = by_series(pages.discover(make_tree(tmp_path)))["1948"]
    ids = {p.sheet_id for p in d}
    assert ids == {"1", "2", "3", "12", "13", "14"}
    src = {p.sheet_id: p.source for p in d}
    assert src["2"].parent.name == "aligned"
    assert src["1"].parent.name == "aligned"
    assert src["14"].parent.name == "originals"


def test_discover_skips_index_and_setback(tmp_path):
    ids = {p.sheet_id for p in by_series(pages.discover(make_tree(tmp_path)))["1948"]}
    assert not ids & {"index", "setback", "planning-code"}


def test_discover_1960_uses_aligned(tmp_path):
    d = by_series(pages.discover(make_tree(tmp_path)))["1960"]
    assert {p.sheet_id for p in d} == {"1", "2"}
    assert all(p.source.parent.name == "aligned" for p in d)


def test_discover_1980plus_series_names(tmp_path):
    s = by_series(pages.discover(make_tree(tmp_path)))
    assert "1990-1996-height" in s
    assert "1980-1990-height" in s
    assert "1980-1990-su" in s
    assert "2000-digital-zoning" in s
    assert "1996-2000-su" in s
    assert not any("introduction" in k for k in s)


def test_multipage_ids(tmp_path):
    d = by_series(pages.discover(make_tree(tmp_path)))["1929"]
    assert [p.sheet_id for p in d] == ["1929-p01", "1929-p02", "1929-p03"]
    assert [p.page_index for p in d] == [0, 1, 2]


def test_sheet_ids_unique(tmp_path):
    ps = pages.discover(make_tree(tmp_path))
    keys = [(p.series, p.sheet_id) for p in ps]
    assert len(keys) == len(set(keys))


def test_neatline_mask_synthetic():
    img = np.full((1000, 800), 255, np.uint8)
    img[100:900, 100:700] = 0
    img[120:880, 120:680] = 255
    m = pages.neatline_mask(img)
    assert m.shape == img.shape and m.dtype == bool
    assert m[130:870, 130:670].all()
    assert not m[:90].any() and not m[910:].any()
    assert not m[:, :90].any() and not m[:, 710:].any()


def test_neatline_mask_none_found():
    assert pages.neatline_mask(np.full((200, 200), 255, np.uint8)).all()


def test_render_jpg_skips_when_fresh(tmp_path):
    import cv2
    src = tmp_path / "a.jpg"
    cv2.imwrite(str(src), np.full((20, 30, 3), 200, np.uint8))
    out = tmp_path / "o" / "a.png"
    page = pages.Page("1948", "a", src, None)
    pages.render(page, out)
    assert cv2.imread(str(out)).shape == (20, 30, 3)
    t = out.stat().st_mtime_ns
    pages.render(page, out)
    assert out.stat().st_mtime_ns == t


def test_render_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    import subprocess

    src = touch(tmp_path / "a.pdf")

    def fake_run(cmd, check):
        (tmp_path / "out").mkdir(exist_ok=True)
        Path(cmd[-1] + ".png").write_bytes(b"partial")  # pdftoppm wrote, then died
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(pages.subprocess, "run", fake_run)
    out = tmp_path / "out" / "page.png"
    with pytest.raises(subprocess.CalledProcessError):
        pages.render(pages.Page("1960", "a", src, None), out)
    assert list(out.parent.iterdir()) == []


def test_discover_two_page_photo_yields_the_unscanned_page(tmp_path):
    a = tmp_path / "1948" / "aligned"
    for n in ["2-and-3", "3", "4-and-5", "4", "5", "6-and-7"]:
        touch(a / f"{n}.jpg")
    d = {p.sheet_id: p for p in pages.discover(tmp_path)}
    assert set(d) == {"2", "3", "4", "5", "6-and-7"}
    assert d["2"].source == a / "2-and-3.jpg" and d["2"].other == a / "3.jpg"
    assert d["3"].other is None and d["4"].other is None
    assert d["6-and-7"].other is None   # neither page alone: kept whole, as before
    # the page with its own scan may be the first one of the name, too
    touch(a / "8-and-9.jpg")
    touch(a / "8.jpg")
    d = {p.sheet_id: p for p in pages.discover(tmp_path)}
    assert d["9"].source == a / "8-and-9.jpg" and d["9"].other == a / "8.jpg"


def _page_art(seed: int, h: int = 900, w: int = 700) -> np.ndarray:
    """A framed 'map': a thick border around random blocks and lines (distinct per seed)."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (w - 11, h - 11), 0, 8)
    for _ in range(160):
        x, y = int(rng.integers(30, w - 80)), int(rng.integers(30, h - 80))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(8, 50))),
                      int(rng.integers(0, 120)), int(rng.choice([-1, 2])))
    for _ in range(40):
        p = rng.integers(30, min(h, w) - 30, size=4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 0, 2)
    return img


def _spread(tmp_path):
    """2-and-3.jpg: page 2 (seed 2) left, page 3 (seed 3) right; 3.jpg: page 3 alone, at
    another scale and offset (a separate photo)."""
    import cv2
    p2, p3 = _page_art(2), _page_art(3)
    spread = np.full((1000, 1700), 235, np.uint8)
    spread[40:940, 60:760] = p2
    spread[50:950, 900:1600] = p3
    a = tmp_path / "1948" / "aligned"
    a.mkdir(parents=True)
    cv2.imwrite(str(a / "2-and-3.jpg"), cv2.cvtColor(spread, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    alone = np.full((1100, 900), 235, np.uint8)
    alone[60:60 + 990, 70:70 + 770] = cv2.resize(p3, (770, 990))
    cv2.imwrite(str(a / "3.jpg"), cv2.cvtColor(alone, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return a


def test_neatline_frames_two_pages(tmp_path):
    import cv2
    a = _spread(tmp_path)
    fr = pages.neatline_frames(cv2.imread(str(a / "2-and-3.jpg"), cv2.IMREAD_GRAYSCALE))
    assert len(fr) == 2
    (l0, t0, r0, b0), (l1, t1, r1, b1) = fr
    assert abs(l0 - 66) <= 6 and abs(r0 - 754) <= 6 and abs(t0 - 46) <= 6 and abs(b0 - 934) <= 6
    assert abs(l1 - 906) <= 6 and abs(r1 - 1594) <= 6


def test_crop_box_keeps_the_page_without_its_own_scan(tmp_path):
    import cv2
    a = _spread(tmp_path)
    page = {p.sheet_id: p for p in pages.discover(tmp_path)}["2"]
    x0, y0, x1, y1 = pages.crop_box(page)
    # the left frame plus a margin, never past the middle of the gutter (x 830)
    assert x0 <= 66 and 754 <= x1 <= 830 and y0 <= 46 and y1 >= 934
    out = pages.render(page, tmp_path / "w" / "page.png")
    assert cv2.imread(str(out)).shape[:2] == (y1 - y0, x1 - x0)
    # decided by content, not position: with page 2 scanned alone, page 3 (right) is kept
    (a / "3.jpg").rename(a / "x.jpg")
    alone2 = np.full((1000, 800), 235, np.uint8)
    alone2[50:950, 50:750] = _page_art(2)
    cv2.imwrite(str(a / "2.jpg"), alone2)
    p3 = {p.sheet_id: p for p in pages.discover(tmp_path)}["3"]
    assert p3.other == a / "2.jpg"
    x0, _, x1, _ = pages.crop_box(p3)
    assert 830 <= x0 <= 906 and x1 >= 1594


def test_crop_box_refuses_undecidable_photo(tmp_path):
    import cv2
    a = _spread(tmp_path)
    cv2.imwrite(str(a / "3.jpg"), _page_art(99))   # matches neither frame
    with pytest.raises(RuntimeError, match="cannot tell"):
        pages.crop_box({p.sheet_id: p for p in pages.discover(tmp_path)}["2"])
