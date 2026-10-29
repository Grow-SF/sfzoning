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
