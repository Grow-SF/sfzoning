import json
import os
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from sfzmaps import config, runner
from sfzmaps.pages import Page
from sfzmaps.reading import Label, PageInfo
from sfzmaps.refine import Match, RefineResult
from sfzmaps.sources.datasf import Reference
from sfzmaps.transform import GCP, SheetTransform

GCPS = [GCP(0, 0, 6_000_000, 2_100_000, "manual"), GCP(1000, 0, 6_000_400, 2_100_000, "manual"),
        GCP(0, 1000, 6_000_000, 2_099_600, "manual"), GCP(1000, 1000, 6_000_400, 2_099_600, "manual")]
AUTO = [GCP(0, 0, 6_000_010, 2_100_010, "anchor"), GCP(1000, 0, 6_000_410, 2_100_010, "anchor"),
        GCP(0, 1000, 6_000_010, 2_099_610, "anchor"), GCP(1000, 1000, 6_000_410, 2_099_610, "anchor")]


class Fakes:
    def __init__(self, monkeypatch, kind="map", rough_ok=True):
        self.calls: dict[str, int] = {}
        self.kind, self.rough_ok = kind, rough_ok
        m = monkeypatch

        def counted(name, fn):
            def w(*a, **k):
                self.calls[name] = self.calls.get(name, 0) + 1
                return fn(*a, **k)
            return w

        def render(page, out):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.full((40, 60), 255, np.uint8))
            return out

        def refine(corners, ref_corners, start, **k):
            ms = [Match(float(i), float(i), 6e6, 2e6, 1.0, "inlier", i) for i in range(5)]
            return RefineResult(SheetTransform(AUTO, 1.0), ms, 3, True, (0.0, 0.0), "ok", 15.0)

        def warp(png, t, boxes, out):
            Path(out).write_bytes(b"tif")
            return out

        m.setattr(runner.pages, "render", counted("render", render))
        m.setattr(runner.reading, "classify_page", counted("classify", lambda p: PageInfo(kind=self.kind)))
        m.setattr(runner.reading, "exclusion_mask", lambda img, info=None: np.zeros(img.shape[:2], bool))
        m.setattr(runner.reading, "read_labels", counted("labels", lambda p, **k: [Label(kind="block_number", text="1", x=1, y=2, confidence=1.0)]))
        m.setattr(runner.linework, "extract_blocks", counted("blocks", lambda img, exclude=None: [box(1, 1, 10, 10)]))
        m.setattr(runner.linework, "block_corners_px", lambda polys: np.array([[1.0, 1.0], [10.0, 10.0]]))
        m.setattr(runner.anchors, "match_labels", lambda labels, ref: [])
        m.setattr(runner.anchors, "rough_fit",
                  counted("rough", lambda c, **k: SimpleRough(SheetTransform(AUTO, 1.0)) if self.rough_ok else None))
        m.setattr(runner.anchors, "street_snap", lambda fit, labels, ref: fit)
        m.setattr(runner.refine_mod, "refine", counted("refine", refine))
        m.setattr(runner.refine_mod, "classify_changes", counted("changes", lambda r: r))
        m.setattr(runner.refine_mod, "changes_geojson", lambda r: {"type": "FeatureCollection", "features": []})
        m.setattr(runner.warp, "warp_sheet", counted("warp", warp))


class SimpleRough:
    def __init__(self, t):
        self.transform = t


@pytest.fixture
def env(tmp_path, monkeypatch):
    for n in ("WORK_DIR", "OUT_DIR", "REF_DIR", "RAW_DIR"):
        monkeypatch.setattr(config, n, tmp_path / n.lower())
    config.REF_DIR.mkdir()
    (config.REF_DIR / "parcels.parquet").write_bytes(b"x")
    src = tmp_path / "src.jpg"
    src.write_bytes(b"x")
    os.utime(src, ns=(1, 1))
    page = Page("1948", "7", src, None)
    ref = Reference(*(gpd.GeoDataFrame({"block_num": []}, geometry=[], crs="EPSG:2227") for _ in range(4)))
    return page, ref, config.WORK_DIR / "1948" / "7"


def write_review(d, status="edited"):
    (d / "review.json").write_text(json.dumps({
        "status": status, "reviewer": "me", "at": "2026-01-01T00:00:00Z", "note": "",
        "gcps": [g.__dict__ for g in GCPS], "change_areas": {"type": "FeatureCollection", "features": []}}))


def test_rerun_keeps_human_points(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    d.mkdir(parents=True)
    write_review(d)
    runner.run_sheet(page, ref)
    assert "refine" not in f.calls and "rough" not in f.calls
    t = json.loads((d / "transform.json").read_text())
    assert [(g["px"], g["py"], g["wx"], g["wy"]) for g in t["gcps"]] == [(g.px, g.py, g.wx, g.wy) for g in GCPS]
    assert (d / "score.json").exists() and f.calls["warp"] == 1


def test_force_auto_ignores_review(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    d.mkdir(parents=True)
    write_review(d)
    runner.run_sheet(page, ref)
    runner.run_sheet(page, ref, force_auto=True)
    assert f.calls["refine"] == 1
    t = json.loads((d / "transform.json").read_text())
    assert t["gcps"][0]["wx"] == AUTO[0].wx
    assert (d / "review.json").exists()  # never deleted
    # and a later normal run goes back to the review
    runner.run_sheet(page, ref)
    t = json.loads((d / "transform.json").read_text())
    assert t["gcps"][0]["wx"] == GCPS[0].wx


def test_non_map_page_skipped(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch, kind="legend")
    s = runner.run_sheet(page, ref)
    assert s.reason == "skipped: legend" and not s.passed
    assert "labels" not in f.calls and "warp" not in f.calls


def test_stage_skip_when_fresh(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    s1 = runner.run_sheet(page, ref)
    assert set(f.calls) == {"render", "classify", "labels", "blocks", "rough", "refine", "changes", "warp"}
    f.calls.clear()
    s2 = runner.run_sheet(page, ref)
    assert f.calls == {}
    assert s2.reason == s1.reason and s2.n_scored == s1.n_scored
    runner.run_sheet(page, ref, force=True)
    assert f.calls["refine"] == 1


def test_stale_input_reruns_downstream(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    runner.run_sheet(page, ref)
    f.calls.clear()
    p = d / "labels.json"
    os.utime(p, ns=(p.stat().st_mtime_ns + 10**9,) * 2)
    runner.run_sheet(page, ref)
    assert set(f.calls) == {"rough", "refine", "changes", "warp"}
    assert "labels" not in f.calls and "render" not in f.calls


def test_rough_failure_uses_answer_key(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch, rough_ok=False)
    pts = page.source.parent / "7.jpg.points"
    pts.write_text("mapX,mapY,pixelX,pixelY,enable\n" + "\n".join(
        f"{g.wx},{g.wy},{g.px},{-g.py},1" for g in GCPS))
    monkeypatch.setattr(runner, "answer_key_path", lambda p: pts)
    s = runner.run_sheet(page, ref)
    assert "refine" not in f.calls and "changes" not in f.calls
    assert s.reason == "insufficient_anchors"
    sj = json.loads((d / "score.json").read_text())
    assert sj["answer_key_fallback"] is True
    t = json.loads((d / "transform.json").read_text())
    assert t["gcps"][0]["source"] == "answer_key"
    assert f.calls["warp"] == 1


def test_run_series_logs_failure_and_continues(env, monkeypatch, capsys):
    page, ref, d = env
    Fakes(monkeypatch)
    bad = Page("1948", "8", page.source, None)
    orig = runner.reading.classify_page

    def boom(p):
        if "/8/" in str(p):
            raise RuntimeError("kaput")
        return orig(p)

    monkeypatch.setattr(runner.reading, "classify_page", boom)
    res = runner.run_series("1948", pages_list=[bad, page], ref=ref)
    assert res["8"].reason.startswith("error: RuntimeError: kaput")
    assert res["7"].reason is None or res["7"].reason != ""
    logs = list(config.OUT_DIR.glob("run-*.log"))
    assert len(logs) == 1 and "Traceback" in logs[0].read_text() and "kaput" in logs[0].read_text()
    out = capsys.readouterr().out
    assert "holdout_p50" in out and "insample_p95" in out


def test_rough_failure_after_pass_clears_score_and_warp(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    runner.run_sheet(page, ref)
    assert (d / "warped.tif").exists()
    f.rough_ok = False
    p = d / "labels.json"
    os.utime(p, ns=(p.stat().st_mtime_ns + 10**9,) * 2)
    s = runner.run_sheet(page, ref)
    assert s.reason == "insufficient_anchors" and not s.passed
    assert json.loads((d / "score.json").read_text())["reason"] == "insufficient_anchors"
    assert not (d / "warped.tif").exists() and not (d / "transform.json").exists()


def test_review_edit_triggers_rewarp(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    d.mkdir(parents=True)
    write_review(d)
    runner.run_sheet(page, ref)
    assert f.calls["warp"] == 1
    runner.run_sheet(page, ref)
    assert f.calls["warp"] == 1
    p = d / "review.json"
    r = json.loads(p.read_text())
    r["gcps"][0]["wx"] += 5
    p.write_text(json.dumps(r))
    os.utime(p, ns=(p.stat().st_mtime_ns + 10**9,) * 2)
    runner.run_sheet(page, ref)
    assert f.calls["warp"] == 2
    assert json.loads((d / "transform.json").read_text())["gcps"][0]["wx"] == GCPS[0].wx + 5


def test_review_without_change_areas_removes_stale_changes(env, monkeypatch):
    page, ref, d = env
    Fakes(monkeypatch)
    runner.run_sheet(page, ref)
    assert (d / "changes.geojson").exists()
    write_review(d)
    r = json.loads((d / "review.json").read_text())
    del r["change_areas"]
    (d / "review.json").write_text(json.dumps(r))
    runner.run_sheet(page, ref)
    assert not (d / "changes.geojson").exists()


def test_run_series_aborts_on_auth_error(env, monkeypatch, capsys):
    page, ref, d = env
    Fakes(monkeypatch)
    from sfzmaps import llm

    def boom(p):
        raise llm.LLMAuthError("OPENAI_API_KEY missing or invalid")

    monkeypatch.setattr(runner.reading, "classify_page", boom)
    pl = [Page("1948", str(i), page.source, None) for i in range(3)]
    res = runner.run_series("1948", pages_list=pl, ref=ref)
    assert list(res) == ["0"]
    assert capsys.readouterr().out.count("aborting series") == 1


def _grid_polys(street):
    return [box(j * (1500 + street), i * (1000 + street), j * (1500 + street) + 1500,
                i * (1000 + street) + 1000) for i in range(3) for j in range(3)]


def _capture_hint(monkeypatch):
    got = {}

    def rough(c, **k):
        got.update(k)
        return SimpleRough(SheetTransform(AUTO, 1.0))

    monkeypatch.setattr(runner.anchors, "rough_fit", rough)
    return got


def test_wide_street_sheet_passes_measured_hint(env, monkeypatch):
    page, ref, d = env
    Fakes(monkeypatch)
    got = _capture_hint(monkeypatch)
    monkeypatch.setattr(runner.linework, "extract_blocks", lambda img, exclude=None: _grid_polys(200))
    runner.run_sheet(page, ref)
    assert got["scale_hint"] == pytest.approx(runner.linework.STREET_FT / 200, rel=0.05)


def test_unmeasured_street_width_passes_no_hint(env, monkeypatch):
    page, ref, d = env
    Fakes(monkeypatch)
    got = _capture_hint(monkeypatch)  # one block: default width, not a measurement
    runner.run_sheet(page, ref)
    assert got["scale_hint"] is None
    assert json.loads((d / "rough.json").read_text())["scale_hint_ft_per_px"] is None


def test_old_format_score_is_stale_and_rescored(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    runner.run_sheet(page, ref)
    sj = d / "score.json"
    old = json.loads(sj.read_text())
    for k in ("holdout_p50_ft", "holdout_p95_ft", "n_holdout"):
        old.pop(k)
    old["passed"], old["reason"] = True, None
    sj.write_text(json.dumps(old))
    os.utime(sj, ns=(sj.stat().st_mtime_ns + 10**9,) * 2)  # newer than every input
    assert not runner._score_current(old)
    s = runner.run_sheet(page, ref)
    assert "holdout_p50_ft" in json.loads(sj.read_text())
    assert not (s.passed and s.n_holdout == 0)
    # and an until-stage early return never reports the old file as passing
    sj.write_text(json.dumps(old))
    assert not runner.run_sheet(page, ref, until="rough").passed


def test_misplaced_with_key_warps_from_key(env, monkeypatch):
    page, ref, d = env
    f = Fakes(monkeypatch)
    pts = page.source.parent / "7.jpg.points"
    pts.write_text("mapX,mapY,pixelX,pixelY,enable\n" + "\n".join(
        f"{g.wx},{g.wy},{g.px},{-g.py},1" for g in GCPS))
    monkeypatch.setattr(runner, "answer_key_path", lambda p: pts)
    from sfzmaps.score import SheetScore
    real = runner.score_mod.score
    monkeypatch.setattr(runner.score_mod, "score",
                        lambda res, **k: SheetScore(50, 0.9, 3.0, 9.0, 20.0, 0, 0, False, "misplaced")
                        if res is not None else real(res, **k))
    s = runner.run_sheet(page, ref)
    assert s.reason == "misplaced" and not s.passed
    sj = json.loads((d / "score.json").read_text())
    assert sj["answer_key_fallback"] is True and sj["reason"] == "misplaced" and sj["passed"] is False
    t = json.loads((d / "transform.json").read_text())
    assert t["origin"] == "answer_key" and t["gcps"][0]["source"] == "answer_key"
    assert (d / "warped.tif").exists()
    # a rerun stays consistent
    s2 = runner.run_sheet(page, ref)
    assert s2.reason == "misplaced" and json.loads((d / "transform.json").read_text())["origin"] == "answer_key"
