import json
import os
import re
import threading
import time
import urllib.request

import pytest

from sfzmaps import config, reviewpage


def _sheet(sid, status="passed", p50=3.0, p95=9.0, reason=None, fb=False):
    return {"sheet_id": sid, "status": status, "holdout_p50_ft": p50, "holdout_p95_ft": p95,
            "reason": reason, "answer_key_fallback": fb, "bounds_wgs84": [-122.5, 37.7, -122.4, 37.8]}


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "REF_DIR", tmp_path / "ref")

    def make(series, sheets, credit=None):
        d = config.OUT_DIR / series
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({
            "series": series, "credit": credit, "pmtiles": f"{series}.pmtiles",
            "tile_report": None, "sheets": sheets}))
        return reviewpage.write(series)
    return make


def _data(path):
    html = path.read_text()
    m = re.search(r'<script type="application/json" id="sheets">(.*?)</script>', html, re.S)
    return html, json.loads(m.group(1))


def test_review_html_lists_failing_first(out):
    p = out("1960", [_sheet("1"), _sheet("2", "needs_review", 9, 30, "misplaced"),
                     _sheet("3", "edited"), _sheet("4", "needs_review", 7, 20)])
    _, d = _data(p)
    assert [s["sheet_id"] for s in d["sheets"]] == ["2", "4", "1", "3"]


def test_review_html_has_credit_for_1948(out):
    html, d = _data(out("1948", [_sheet("1")], credit=config.RUMSEY_CREDIT))
    assert config.RUMSEY_CREDIT in html and d["credit"] == config.RUMSEY_CREDIT
    html, d = _data(out("1960", [_sheet("1")]))
    assert "Rumsey" not in html and d["credit"] is None


def test_review_html_editor_link_only_for_needs_review(out):
    _, d = _data(out("1960", [_sheet("1"), _sheet("2", "needs_review", reason="x")]))
    links = {s["sheet_id"]: s["editor_url"] for s in d["sheets"]}
    assert links == {"1": None, "2": "http://127.0.0.1:8765/?series=1960&sheet=2"}


def test_changes_reprojected_to_wgs84(out):
    w = config.WORK_DIR / "1960" / "1"
    w.mkdir(parents=True)
    (w / "changes.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2227"}},
        "features": [{"type": "Feature", "properties": {"kind": "real_change"},
                      "geometry": {"type": "Point", "coordinates": [6006000, 2115000]}}]}))
    p = out("1960", [_sheet("1")])
    g = json.loads((p.parent / "changes.wgs84.geojson").read_text())
    x, y = g["features"][0]["geometry"]["coordinates"]
    assert -123 < x < -122 and 37 < y < 38
    assert g["features"][0]["properties"]["sheet_id"] == "1"


def test_stale_detection(out):
    p = out("1960", [_sheet("1")])
    assert not reviewpage.is_stale("1960")
    (p.parent / "manifest.json").touch()
    os.utime(p, (time.time() - 100, time.time() - 100))
    assert reviewpage.is_stale("1960")


def test_server_supports_range(tmp_path):
    (tmp_path / "f.bin").write_bytes(bytes(range(100)))
    srv = reviewpage.make_server(tmp_path, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/f.bin"
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=10-19"}))
        assert r.status == 206 and r.read() == bytes(range(10, 20))
        assert r.headers["Content-Range"] == "bytes 10-19/100"
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=-5"}))
        assert r.read() == bytes(range(95, 100))
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=90-"}))
        assert r.read() == bytes(range(90, 100))
        assert urllib.request.urlopen(url).status == 200
    finally:
        srv.shutdown()
