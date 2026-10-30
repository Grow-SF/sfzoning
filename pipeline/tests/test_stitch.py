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


def _sources(vrt):
    import xml.etree.ElementTree as ET
    out = []
    for e in ET.parse(vrt).iter("SourceFilename"):  # one per band: keep first occurrence
        if e.text not in out:
            out.append(e.text)
    return out


def test_vrt_order_passed_on_top(work):
    w = config.WORK_DIR / "1960"
    for sid, p, p50, reason in [("a", False, 6.0, "holdout_median"), ("b", True, 4.9, None),
                                ("c", True, 3.0, None), ("d", False, 5.5, "holdout_median")]:
        _tif(w / sid / "warped.tif", 6_000_000 + (10 if sid in "ab" else 20), 100)
        _score(w / sid, p, p50, reason)
    vrt = stitch.build_vrt("1960")
    assert vrt == config.OUT_DIR / "1960" / "1960.vrt"
    names = [s.split("/")[-2] for s in _sources(vrt)]
    # last listed wins: needs_review at the bottom (worse first), then passing (worse first)
    assert names == ["a", "d", "b", "c"]


def test_vrt_skips_sheets_without_warp(work):
    w = config.WORK_DIR / "1960"
    _tif(w / "a" / "warped.tif", 6_000_000, 100)
    _score(w / "a", True, 3.0)
    (w / "b").mkdir()
    _score(w / "b", False, 9.0, "x")
    names = [s.split("/")[-2] for s in _sources(stitch.build_vrt("1960"))]
    assert names == ["a"]


def test_vrt_no_sheets_raises(work):
    with pytest.raises(RuntimeError):
        stitch.build_vrt("1960")
