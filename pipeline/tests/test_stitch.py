import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sfzmaps import config, stitch


def _tif(path, x0, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((4, 20, 20), np.uint8)
    data[:3] = value
    data[3] = 255
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=4, dtype="uint8",
                       crs="EPSG:2227", transform=from_origin(x0, 2_100_000, 2, 2)) as ds:
        ds.write(data)
        ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                          rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]


def _score(sheet_dir, passed, p50, reason=None):
    (sheet_dir / "score.json").write_text(json.dumps({
        "n_scored": 50, "insample_frac_within": 0.5, "insample_p50_ft": p50, "insample_p95_ft": 9.0,
        "insample_max_ft": 12.0, "n_real_change": 0, "n_no_equivalent": 0, "passed": passed,
        "reason": reason, "holdout_p50_ft": p50, "holdout_p95_ft": 9.0, "n_holdout": 50,
        "score_version": config.SCORE_VERSION}))


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    return tmp_path


def _sid(path):
    parts = path.split("/")
    return parts[-1][:-4] if parts[-2] == "cut" else parts[-2]


def _sources(vrt):
    import xml.etree.ElementTree as ET
    out = []
    for e in ET.parse(vrt).iter("SourceFilename"):  # one per band: keep first occurrence
        if e.text not in out:
            out.append(e.text)
    return out


def test_vrt_order_passed_on_top(work):
    w = config.WORK_DIR / "1960"
    for i, (sid, p, p50, reason) in enumerate([("a", False, 6.0, "holdout_median"), ("b", True, 4.9, None),
                                               ("c", True, 3.0, None), ("d", False, 5.5, "holdout_median")]):
        _tif(w / sid / "warped.tif", 6_000_000 + 45 * i, 100)   # disjoint: no cutting
        _score(w / sid, p, p50, reason)
    vrt = stitch.build_vrt("1960")
    assert vrt == config.OUT_DIR / "1960" / "1960.vrt"
    names = [_sid(s) for s in _sources(vrt)]
    # last listed wins: needs_review at the bottom (worse first), then passing (worse first)
    assert names == ["a", "d", "b", "c"]


def test_vrt_skips_sheets_without_warp(work):
    w = config.WORK_DIR / "1960"
    _tif(w / "a" / "warped.tif", 6_000_000, 100)
    _score(w / "a", True, 3.0)
    (w / "b").mkdir()
    _score(w / "b", False, 9.0, "x")
    names = [_sid(s) for s in _sources(stitch.build_vrt("1960"))]
    assert names == ["a"]


def test_vrt_no_sheets_raises(work):
    with pytest.raises(RuntimeError):
        stitch.build_vrt("1960")


def _px(vrt, x, y=2_100_000 - 10):
    with rasterio.open(vrt) as ds:
        r, c = ds.index(x, y)
        return ds.read(window=((r, r + 1), (c, c + 1)))[:, 0, 0]


@pytest.mark.parametrize("top", ["a", "b"])
def test_overlap_shows_only_top_sheet(work, top):
    """a spans x 6000000-6000040 (value 50), b 6000020-6000060 (value 200). Only the
    higher-priority sheet has pixels in the overlap 6000020-6000040; the other keeps the rest."""
    w = config.WORK_DIR / "1960"
    _tif(w / "a" / "warped.tif", 6_000_000, 50)
    _tif(w / "b" / "warped.tif", 6_000_020, 200)
    bad = "b" if top == "a" else "a"
    _score(w / top, True, 3.0)
    _score(w / bad, False, 9.0, "holdout_median")
    vrt = stitch.build_vrt("1960")
    val = {"a": 50, "b": 200}
    for x in (6_000_024, 6_000_036):                      # overlap: top sheet only
        assert list(_px(vrt, x)) == [val[top]] * 3 + [255]
    # the lower sheet's cut raster is transparent in the overlap and untouched elsewhere
    with rasterio.open(config.OUT_DIR / "1960" / "cut" / f"{bad}.tif") as ds:
        a = ds.read(4)
        for x, want in ((6_000_030, 0), (6_000_010 if bad == "a" else 6_000_050, 255)):
            r, c = ds.index(x, 2_100_000 - 10)
            assert a[r, c] == want


def test_fully_covered_sheet_dropped(work):
    """Sanity: a fully covered lower sheet is dropped from the VRT."""
    w = config.WORK_DIR / "1960"
    _tif(w / "a" / "warped.tif", 6_000_000, 50)
    _tif(w / "b" / "warped.tif", 6_000_000, 200)
    _score(w / "a", False, 9.0, "holdout_median")
    _score(w / "b", True, 3.0)
    assert [_sid(s) for s in _sources(stitch.build_vrt("1960"))] == ["b"]


def test_polygonal_drops_lines_from_collections():
    from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, box

    from sfzmaps.stitch import polygonal
    gc = GeometryCollection([box(0, 0, 1, 1), LineString([(2, 2), (3, 3)]), Point(5, 5),
                             MultiPolygon([box(10, 10, 11, 11)])])
    g = polygonal(gc)
    assert g.geom_type == "MultiPolygon" and len(g.geoms) == 2 and g.area == 2.0
    assert polygonal(GeometryCollection([LineString([(0, 0), (1, 1)])])).is_empty
    assert polygonal(box(0, 0, 1, 1)).geom_type == "Polygon"


def test_black_ink_is_not_nodata(work):
    """A source carrying nodata=0 must not make pure-black ink transparent in the stitch."""
    w = config.WORK_DIR / "1929"
    _tif(w / "a" / "warped.tif", 6_000_000, 0)          # solid black, opaque
    with rasterio.open(w / "a" / "warped.tif", "r+") as ds:
        ds.nodata = 0
    _score(w / "a", True, 3.0)
    vrt = stitch.build_vrt("1929")
    text = vrt.read_text()
    assert "NoDataValue" not in text and "<NODATA>" not in text
    with rasterio.open(vrt) as ds:
        assert ds.read(4).min() == 255 and ds.read(1).max() == 0


def _page(sheet_dir, number):
    (sheet_dir / "page.json").write_text(json.dumps({"kind": "map", "sheet_number": number}))


def test_zoning_layer_leaves_out_height_su_and_sign_sheets(work):
    # mixed books (1964-1980) bind 1..13 with 1H..13H, SH, SU and SSD sheets; only zoning
    # sheets belong in a zoning edition's map layers
    w = config.WORK_DIR / "1972"
    nums = {"z1": "1", "z5a": "5A", "unread": None, "h": "10H", "sh": "13SH", "su": "1SUa",
            "su2": "2SU", "ssd": "SSD"}
    for i, (sid, num) in enumerate(nums.items()):
        _tif(w / sid / "warped.tif", 6_000_000 + 45 * i, 100)
        _score(w / sid, True, 3.0 + i)
        _page(w / sid, num)
    names = sorted(_sid(s) for s in _sources(stitch.build_vrt("1972")))
    assert names == ["unread", "z1", "z5a"]
    assert stitch.draw_order("1972", stitch.load_scores("1972"))  # districts still see every sheet
    assert len(stitch.draw_order("1972", stitch.load_scores("1972"))) == len(nums)


def test_height_series_keeps_its_height_sheets(work):
    w = config.WORK_DIR / "1980-1990-height"
    for i, (sid, num) in enumerate([("a", "1H"), ("b", "2H")]):
        _tif(w / sid / "warped.tif", 6_000_000 + 45 * i, 100)
        _score(w / sid, True, 3.0)
        _page(w / sid, num)
    assert sorted(_sid(s) for s in _sources(stitch.build_vrt("1980-1990-height"))) == ["a", "b"]
