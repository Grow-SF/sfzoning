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


def test_rebuild_series_failing_steps_do_not_raise(monkeypatch, tmp_path, caplog):
    from sfzmaps import manifest, reviewpage, stitch
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    seen = []

    def boom(series, scores=None):
        raise RuntimeError("no warped sheets")

    monkeypatch.setattr(stitch, "build_vrt", boom)
    monkeypatch.setattr(manifest, "write", lambda s, sc, tr=None: seen.append(tr))
    monkeypatch.setattr(reviewpage, "write", lambda s: (_ for _ in ()).throw(RuntimeError("no blocks")))
    with caplog.at_level("WARNING"):
        server._rebuild_series("1948")  # must not raise
    assert "skipping" in caplog.text
    assert seen == [None]  # manifest still rewritten, no tile report (previous one is kept)


def test_rebuild_series_calls_task14_functions_in_order(monkeypatch, tmp_path):
    from sfzmaps import manifest, reviewpage, stitch, tiles
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    calls = []
    sentinel = {"s1": object()}
    monkeypatch.setattr(stitch, "load_scores", lambda s: sentinel)
    monkeypatch.setattr(stitch, "build_vrt", lambda s, sc=None: calls.append(("vrt", sc)) or tmp_path / "x.vrt")
    monkeypatch.setattr(tiles, "make_pmtiles",
                        lambda vrt, out, **k: calls.append(("tiles", out.name)) or "REPORT")
    monkeypatch.setattr(manifest, "write", lambda s, sc, tr=None: calls.append(("manifest", tr)))
    monkeypatch.setattr(reviewpage, "write", lambda s: calls.append(("review", s)))
    server._rebuild_series("1948")
    assert calls == [("vrt", sentinel), ("tiles", "1948.pmtiles"), ("manifest", "REPORT"), ("review", "1948")]


def test_rebuilds_are_serialized_and_coalesced(monkeypatch):
    import threading
    import time
    spans = []

    def slow(series):
        t0 = time.monotonic()
        time.sleep(0.3)
        spans.append((t0, time.monotonic()))

    monkeypatch.setattr(server, "_rebuild_series", slow)
    server._rb_state.clear()
    ths = []
    for i in range(3):  # three rapid saves
        th = threading.Thread(target=server._schedule_rebuild, args=("1948",))
        th.start()
        ths.append(th)
        time.sleep(0.05)
    for th in ths:
        th.join()
    assert len(spans) == 2                      # running one + exactly one trailing
    assert spans[0][1] <= spans[1][0]           # no overlap
    server._schedule_rebuild("1948")            # idle again: runs normally
    assert len(spans) == 3


def test_rebuild_failure_does_not_wedge_scheduler(monkeypatch):
    n = []

    def boom(series):
        n.append(1)
        raise RuntimeError("x")

    monkeypatch.setattr(server, "_rebuild_series", boom)
    server._rb_state.clear()
    server._schedule_rebuild("s")
    server._schedule_rebuild("s")
    assert len(n) == 2


@pytest.mark.parametrize("series,sheet", [("..", "1"), ("demo", ".."), ("demo", ".hidden"),
                                           ("demo", "a b"), ("%2e%2e", "1"), ("demo", "..%2f1")])
def test_sheet_ids_are_strict(env, series, sheet):
    demo, c, _ = env
    assert c.get(f"/api/sheet/{series}/{sheet}").status_code == 404
    r = c.post(f"/api/sheet/{series}/{sheet}/save", json={
        "gcps": _gcps(demo.true_gcps), "change_areas": None, "status": "edited", "note": ""})
    assert r.status_code == 404


def test_unusable_transform_is_400_not_500(env, monkeypatch):
    demo, c, _ = env
    monkeypatch.setattr(server, "_bounds", lambda t, size: None)
    r = c.post(url(demo, "/preview"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None})
    assert r.status_code == 400


def test_preview_tmp_files_cleaned_up(env):
    demo, c, _ = env
    c.post(url(demo, "/preview"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None})
    d = demo.work_dir / demo.series / demo.sheet_id
    assert not list(d.glob("preview.*.tmp.*"))


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


def test_blocks_geojson_near_sheet(env):
    demo, c, _ = env
    r = c.get(url(demo, "/blocks.geojson"))
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 81
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert -123 < lon < -122 and 37 < lat < 38           # WGS84, San Francisco
    assert "block_num" in fc["features"][0]["properties"]


def test_blocks_geojson_streets_empty_and_404(env):
    demo, c, _ = env
    assert c.get(url(demo, "/blocks.geojson?layer=streets")).json()["features"] == []
    assert c.get(url(demo, "/blocks.geojson?layer=nope")).status_code == 422
    assert c.get("/api/sheet/demo/nope/blocks.geojson").status_code == 404


def test_blocks_geojson_without_reference(env, monkeypatch):
    demo, c, _ = env
    monkeypatch.setattr(server, "_get_ref", lambda: None)
    assert c.get(url(demo, "/blocks.geojson")).json()["features"] == []


def test_static_front_end_served(env):
    _, c, _ = env
    r = c.get("/")
    assert r.status_code == 200 and "editor.js" in r.text
    assert c.get("/editor.js").status_code == 200 and c.get("/editor.css").status_code == 200
    assert c.post("/nothing").status_code == 404
