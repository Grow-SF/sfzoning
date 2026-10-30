import json
import shutil
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from sfzmaps import config, runner
from sfzmaps.editor import demo as demo_mod
from sfzmaps.editor import server


@pytest.fixture
def env(tmp_path, monkeypatch):
    for n in ("WORK_DIR", "RAW_DIR", "REF_DIR", "OUT_DIR"):
        monkeypatch.setattr(config, n, getattr(config, n))  # restored after the test
    demo = demo_mod.build_demo(tmp_path)
    monkeypatch.setattr(server, "_get_ref", lambda: demo.ref)
    monkeypatch.setattr(server, "_find_page", lambda series, sheet: demo.page)
    monkeypatch.setattr(server, "_git_email", lambda: "tester@example.org")
    rebuilt = []
    monkeypatch.setattr(server, "_rebuild_series", lambda series: rebuilt.append(series))
    server.configure(year=None, sheet=None)
    return demo, TestClient(server.app), rebuilt


def _gcps(gcps):
    return [asdict(g) for g in gcps]


def _bad(demo):
    t = json.loads((demo.work_dir / demo.series / demo.sheet_id / "transform.json").read_text())
    return t["gcps"]


def url(demo, tail=""):
    return f"/api/sheet/{demo.series}/{demo.sheet_id}{tail}"


def test_queue_needs_review_first(env):
    demo, c, _ = env
    src = demo.work_dir / demo.series / demo.sheet_id
    ok = demo.work_dir / demo.series / "0"
    shutil.copytree(src, ok)
    s = json.loads((ok / "score.json").read_text())
    s.update(passed=True, reason=None)
    (ok / "score.json").write_text(json.dumps(s))
    q = c.get("/api/queue").json()
    assert [(r["sheet_id"], r["status"]) for r in q] == [("1", "needs_review"), ("0", "passed")]
    assert q[0]["passed"] is False and q[0]["reason"] and "holdout_p95_ft" in q[0]
    assert {"series", "holdout_p50_ft", "answer_key_fallback"} <= set(q[0])


def test_queue_year_and_sheet_filter(env):
    demo, c, _ = env
    server.configure(year="1948", sheet=None)
    assert c.get("/api/queue").json() == []
    server.configure(year=demo.series, sheet="nope")
    assert c.get("/api/queue").json() == []
    server.configure(year=demo.series, sheet=demo.sheet_id)
    assert len(c.get("/api/queue").json()) == 1


def test_sheet_payload(env):
    demo, c, _ = env
    r = c.get(url(demo)).json()
    assert len(r["gcps"]) == 25 and all("residual_ft" in g for g in r["gcps"])
    assert max(g["residual_ft"] for g in r["gcps"]) > 20  # the corrupted points
    assert r["matches"] and r["score"]["passed"] is False
    assert r["image_size"]["width"] > 100 and len(r["world_bounds"]) == 4
    assert r["origin"] == "auto" and "{z}" in r["image_url"]
    assert c.get("/api/sheet/demo/zzz").status_code == 404


def test_sheet_starts_from_review_json(env):
    demo, c, _ = env
    d = demo.work_dir / demo.series / demo.sheet_id
    (d / "review.json").write_text(json.dumps({
        "status": "edited", "reviewer": "x", "at": "t", "note": "", "gcps": _gcps(demo.true_gcps[:6]),
        "change_areas": None}))
    r = c.get(url(demo)).json()
    assert len(r["gcps"]) == 6 and r["origin"] == "review" and r["status"] == "edited"


def test_preview_moving_bad_point_improves_score(env):
    demo, c, _ = env
    base = c.post(url(demo, "/preview"), json={"gcps": _bad(demo), "change_areas": None}).json()
    fixed = c.post(url(demo, "/preview"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None}).json()
    assert fixed["score"]["holdout_p95_ft"] < base["score"]["holdout_p95_ft"] / 3
    assert fixed["score"]["holdout_p50_ft"] < base["score"]["holdout_p50_ft"]
    assert fixed["score"]["passed"] is True and base["score"]["passed"] is False
    assert len(fixed["residuals"]) == 25
    # low-res warped PNG with a world file
    png = c.get(fixed["preview_png_url"])
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    pgw = (demo.work_dir / demo.series / demo.sheet_id / "preview.pgw").read_text().split()
    assert len(pgw) == 6
    from PIL import Image
    import io
    assert max(Image.open(io.BytesIO(png.content)).size) <= 2048


def test_preview_excludes_change_areas(env):
    demo, c, _ = env
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[5e6, 1e6], [7e6, 1e6], [7e6, 3e6], [5e6, 3e6], [5e6, 1e6]]]}}]}
    r = c.post(url(demo, "/preview"), json={"gcps": _bad(demo), "change_areas": gj}).json()
    assert r["score"]["n_scored"] == 0 and r["score"]["n_real_change"] > 0


def test_preview_bad_input(env):
    demo, c, _ = env
    r = c.post(url(demo, "/preview"), json={"gcps": _gcps(demo.true_gcps[:2]), "change_areas": None})
    assert r.status_code == 400


def test_save_writes_review_json_and_reruns(env, monkeypatch):
    demo, c, rebuilt = env
    calls = []
    real = runner.run_sheet
    monkeypatch.setattr(runner, "run_sheet", lambda *a, **k: calls.append(a) or real(*a, **k))
    r = c.post(url(demo, "/save"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None,
                                         "status": "edited", "note": "fixed 3 points"})
    assert r.status_code == 200, r.text
    d = demo.work_dir / demo.series / demo.sheet_id
    rv = json.loads((d / "review.json").read_text())
    assert rv["status"] == "edited" and rv["reviewer"] == "tester@example.org"
    assert rv["note"] == "fixed 3 points" and len(rv["gcps"]) == 25 and rv["at"]
    assert not list(d.glob("*.tmp"))
    assert len(calls) == 1 and calls[0][0] is demo.page
    assert json.loads((d / "transform.json").read_text())["origin"] == "review"
    body = r.json()
    assert body["score"]["passed"] is True, body
    assert body["status"] == "edited"
    assert (d / "warped.tif").exists()
    assert rebuilt == [demo.series]  # background task ran
    # and the queue now shows it as edited
    assert c.get("/api/queue").json()[0]["status"] == "edited"


def test_accept_below_threshold_requires_note(env):
    demo, c, rebuilt = env
    body = {"gcps": _bad(demo), "change_areas": None, "status": "accepted", "note": "  "}
    r = c.post(url(demo, "/save"), json=body)
    assert r.status_code == 400 and "note" in r.json()["detail"]
    assert not (demo.work_dir / demo.series / demo.sheet_id / "review.json").exists()
    assert rebuilt == []
    ok = c.post(url(demo, "/save"), json={**body, "note": "scan damaged; best effort"})
    assert ok.status_code == 200 and ok.json()["status"] == "accepted"
    # a passing edit may be accepted without a note
    good = c.post(url(demo, "/save"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None,
                                            "status": "accepted", "note": ""})
    assert good.status_code == 200


def test_save_rejects_bad_status(env):
    demo, c, _ = env
    r = c.post(url(demo, "/save"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None,
                                         "status": "bogus", "note": ""})
    assert r.status_code in (400, 422)


def test_deep_zoom_tiles(env):
    demo, c, _ = env
    info = c.get(url(demo)).json()
    z = info["image_max_zoom"]
    t0 = c.get(url(demo, "/image/0/0/0.png"))
    assert t0.status_code == 200 and t0.content[:4] == b"\x89PNG"
    from PIL import Image
    import io
    assert max(Image.open(io.BytesIO(t0.content)).size) <= info["image_tile_size"]
    top = c.get(url(demo, f"/image/{z}/0/0.png"))
    assert Image.open(io.BytesIO(top.content)).size == (min(info["image_tile_size"], info["image_size"]["width"]),
                                                        min(info["image_tile_size"], info["image_size"]["height"]))
    assert (demo.work_dir / demo.series / demo.sheet_id / "dz").is_dir()
    assert c.get(url(demo, "/image/0/5/5.png")).status_code == 404
    assert c.get(url(demo, f"/image/{z + 1}/0/0.png")).status_code == 404


def test_rebuild_series_skips_missing_modules(monkeypatch, caplog):
    import sys
    for m in ("stitch", "tiles", "manifest", "reviewpage"):
        monkeypatch.setitem(sys.modules, f"sfzmaps.{m}", None)  # import raises ImportError
    with caplog.at_level("INFO"):
        server._rebuild_series("1948")  # must not raise
    assert "skip" in caplog.text.lower()


def test_rebuild_series_calls_modules_in_order(monkeypatch, tmp_path):
    import sys
    import types
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    calls = []
    stitch = types.SimpleNamespace(build_vrt=lambda s: calls.append(("vrt", s)) or tmp_path / "x.vrt")
    tiles = types.SimpleNamespace(make_pmtiles=lambda vrt, out, **k: calls.append(("tiles", out.name)))
    manifest = types.SimpleNamespace(write=lambda s, scores: calls.append(("manifest", s)))
    page = types.SimpleNamespace(write=lambda s: calls.append(("page", s)))
    for n, m in (("stitch", stitch), ("tiles", tiles), ("manifest", manifest), ("reviewpage", page)):
        monkeypatch.setitem(sys.modules, f"sfzmaps.{n}", m)
    import sfzmaps
    for n, m in (("stitch", stitch), ("tiles", tiles), ("manifest", manifest), ("reviewpage", page)):
        monkeypatch.setattr(sfzmaps, n, m, raising=False)
    server._rebuild_series("1948")
    assert [c[0] for c in calls] == ["vrt", "tiles", "manifest", "page"]
    assert calls[1][1] == "1948.pmtiles"


def test_cli_edit_demo_binds_localhost(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli

    seen = {}
    monkeypatch.setattr(server, "serve", lambda port=8765: seen.update(port=port, host=server.HOST))
    for n in ("WORK_DIR", "RAW_DIR", "REF_DIR", "OUT_DIR"):
        monkeypatch.setattr(config, n, getattr(config, n))
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix="": str(tmp_path))
    r = CliRunner().invoke(cli.app, ["edit", "--demo"])
    assert r.exit_code == 0, r.output
    assert seen == {"port": 8765, "host": "127.0.0.1"}
    q = TestClient(server.app).get("/api/queue").json()
    assert [(x["series"], x["status"]) for x in q] == [("demo", "needs_review")]
    server.configure(year=None, sheet=None)
    server._pages.clear()
    server.set_reference(None)
