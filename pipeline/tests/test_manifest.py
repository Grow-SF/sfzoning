import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sfzmaps import config, manifest
from sfzmaps.score import SheetScore
from sfzmaps.tiles import TileReport


def _score(passed, p50=3.0, p95=9.0, reason=None):
    return SheetScore(50, 0.5, p50, p95, 12.0, 0, 0, passed, reason, p50, p95, 50)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    return tmp_path


def _warp(sheet_dir):
    sheet_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(sheet_dir / "warped.tif", "w", driver="GTiff", width=10, height=10, count=4,
                       dtype="uint8", crs="EPSG:2227",
                       transform=from_origin(6_005_000, 2_115_000, 5, 5)) as ds:
        ds.write(np.zeros((4, 10, 10), np.uint8))


def _read(p):
    return json.loads(p.read_text())


def test_manifest_marks_needs_review(work):
    w = config.WORK_DIR / "1960"
    _warp(w / "a")
    _warp(w / "b")
    (w / "c").mkdir()
    (w / "b" / "score.json").write_text(json.dumps({"answer_key_fallback": True}))
    scores = {"a": _score(True), "b": _score(False, float("nan"), float("nan"), "holdout_median"),
              "c": SheetScore.empty("skipped: index")}
    m = _read(manifest.write("1960", scores))
    by = {s["sheet_id"]: s for s in m["sheets"]}
    assert by["a"]["status"] == "passed" and by["a"]["reason"] is None
    assert by["a"]["holdout_p50_ft"] == 3.0 and by["a"]["answer_key_fallback"] is False
    assert by["b"]["status"] == "needs_review" and by["b"]["reason"] == "holdout_median"
    assert by["b"]["holdout_p50_ft"] is None and by["b"]["answer_key_fallback"] is True
    lon, lat = by["a"]["bounds_wgs84"][0], by["a"]["bounds_wgs84"][1]
    assert -123 < lon < -122 and 37 < lat < 38 and len(by["a"]["bounds_wgs84"]) == 4
    assert by["c"]["status"] == "needs_review" and by["c"]["bounds_wgs84"] is None
    assert by["c"]["reason"] == "skipped: index"
    assert m["pmtiles"] == "1960.pmtiles" and m["tile_report"] is None and m["credit"] is None
    assert m["series"] == "1960" and "generated_at" in m


def test_manifest_review_status_wins(work):
    w = config.WORK_DIR / "1960"
    for sid, st in [("a", "edited"), ("b", "accepted"), ("c", "needs_work")]:
        _warp(w / sid)
        (w / sid / "review.json").write_text(json.dumps({"status": st}))
    scores = {k: _score(False, reason="holdout_median") for k in "abc"}
    by = {s["sheet_id"]: s for s in _read(manifest.write("1960", scores))["sheets"]}
    assert (by["a"]["status"], by["b"]["status"], by["c"]["status"]) == ("edited", "accepted", "needs_review")
    assert by["a"]["reason"] == "holdout_median"


def test_manifest_1948_credit_and_tile_report(work):
    rep = TileReport(bytes_total=30, bytes_by_zoom={11: 10, 12: 20})
    m = _read(manifest.write("1948", {}, rep))
    assert m["credit"] == config.RUMSEY_CREDIT
    assert m["tile_report"] == {"bytes_total": 30, "bytes_by_zoom": {"11": 10, "12": 20}}
    assert m["pmtiles"] == "1948.pmtiles"


def test_manifest_keeps_previous_tile_report(work):
    rep = TileReport(bytes_total=30, bytes_by_zoom={11: 10, 12: 20})
    manifest.write("1960", {}, rep)
    m = _read(manifest.write("1960", {}))
    assert m["tile_report"] == {"bytes_total": 30, "bytes_by_zoom": {"11": 10, "12": 20}}
    new = TileReport(bytes_total=5, bytes_by_zoom={11: 5})
    assert _read(manifest.write("1960", {}, new))["tile_report"]["bytes_total"] == 5
