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
    rebuilt = []
    monkeypatch.setattr(server, "_rebuild_series", lambda series: rebuilt.append(series))
    server.configure(year=None, sheet=None)
    return demo, TestClient(server.app, base_url="http://127.0.0.1"), rebuilt


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
    assert rv["status"] == "edited" and rv["reviewer"] == "owner"
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
    q = TestClient(server.app, base_url="http://127.0.0.1").get("/api/queue").json()
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


# -- C1: a review seeded from a dense automatic transform -----------------------------

def _dense_auto(demo):
    """Turn the demo into a dense automatic sheet: one "auto" GCP per (noisy) modern
    corner, smoothing TPS_SMOOTHING, transform.json GCPs == refine.json inliers, scored
    by the automatic path. Returns (sheet dir, its score.json)."""
    import numpy as np

    from sfzmaps import score as score_mod
    from sfzmaps.refine import Match, RefineResult
    from sfzmaps.transform import GCP, SheetTransform

    d = demo.work_dir / demo.series / demo.sheet_id
    rf = json.loads((d / "refine.json").read_text())
    rng = np.random.default_rng(1)
    ms = []
    for m in rf["matches"]:
        dx, dy = rng.normal(0, 2.0, 2)
        ms.append(Match(m["px"], m["py"], m["wx"] + dx, m["wy"] + dy, 0.0, "inlier", m["ref_idx"]))
    t = SheetTransform([GCP(m.px, m.py, m.wx, m.wy, "auto") for m in ms], config.TPS_SMOOTHING)
    W = t.to_world([[m.px, m.py] for m in ms])
    for m, w in zip(ms, W):
        m.dist_ft = float(np.hypot(w[0] - m.wx, w[1] - m.wy))
    res = RefineResult(t, ms, 1, True, (0.0, 0.0), "ok", 15.0)
    runner._write_json(d / "refine.json", runner._result_json(res))
    runner._write_json(d / "transform.json", runner._transform_json(t, "auto"))
    sc = score_mod.score(runner._result_load(runner._result_json(res), demo.ref, t),
                         shape=runner.page_shape(d / "page.png"))
    runner._write_json(d / "score.json", runner._score_json(sc))
    return d, json.loads((d / "score.json").read_text())


def test_preview_of_auto_gcps_follows_edits(env):
    demo, c, _ = env
    d, auto = _dense_auto(demo)
    seed = c.get(url(demo)).json()["gcps"]
    assert len(seed) > 100 and {g["source"] for g in seed} == {"auto"}
    gcps = [{k: g[k] for k in ("px", "py", "wx", "wy", "source", "enabled")} for g in seed]

    same = c.post(url(demo, "/preview"), json={"gcps": gcps, "change_areas": None}).json()["score"]
    assert abs(same["holdout_p50_ft"] - auto["holdout_p50_ft"]) < 0.1
    assert abs(same["holdout_p95_ft"] - auto["holdout_p95_ft"]) < 0.1

    moved = [{**g, "wx": g["wx"] + 300.0} for g in gcps]
    sm = c.post(url(demo, "/preview"), json={"gcps": moved, "change_areas": None}).json()["score"]
    assert sm["holdout_p50_ft"] > 250 and sm["passed"] is False

    half = [{**g, "wx": g["wx"] + (300.0 if i % 2 else 0.0)} for i, g in enumerate(gcps)]
    sh = c.post(url(demo, "/preview"), json={"gcps": half, "change_areas": None}).json()["score"]
    assert sh["holdout_p50_ft"] > same["holdout_p50_ft"] + 20

    off = [{**g, "enabled": i % 10 == 0} for i, g in enumerate(gcps)]
    so = c.post(url(demo, "/preview"), json={"gcps": off, "change_areas": None}).json()["score"]
    assert so["holdout_p50_ft"] != same["holdout_p50_ft"]


def test_resave_unchanged_auto_sheet_keeps_score(env):
    demo, c, _ = env
    d, auto = _dense_auto(demo)
    assert auto["passed"] is True
    gcps = [{k: g[k] for k in ("px", "py", "wx", "wy", "source", "enabled")}
            for g in c.get(url(demo)).json()["gcps"]]
    r = c.post(url(demo, "/save"), json={"gcps": gcps, "change_areas": None, "status": "edited"})
    assert r.status_code == 200, r.text
    rv = json.loads((d / "review.json").read_text())
    assert rv["smoothing"] == config.TPS_SMOOTHING
    assert json.loads((d / "transform.json").read_text())["smoothing"] == config.TPS_SMOOTHING
    s = r.json()["score"]
    assert s["passed"] is True
    assert abs(s["holdout_p50_ft"] - auto["holdout_p50_ft"]) < 0.1
    assert abs(s["holdout_p95_ft"] - auto["holdout_p95_ft"]) < 0.1
    assert abs(r.json()["preview_score"]["holdout_p50_ft"] - s["holdout_p50_ft"]) < 1e-9
    # the editor reopens it with the stored smoothing
    assert c.get(url(demo)).json()["smoothing"] == config.TPS_SMOOTHING


def test_sparse_review_uses_manual_smoothing(env):
    demo, c, _ = env
    r = c.post(url(demo, "/save"), json={"gcps": _gcps(demo.true_gcps), "change_areas": None,
                                         "status": "edited"})
    d = demo.work_dir / demo.series / demo.sheet_id
    assert r.status_code == 200 and json.loads((d / "review.json").read_text())["smoothing"] == \
        config.MANUAL_TPS_SMOOTHING


def test_stored_review_smoothing_is_used_by_runner(env):
    demo, c, _ = env
    d = demo.work_dir / demo.series / demo.sheet_id
    (d / "review.json").write_text(json.dumps({
        "status": "edited", "reviewer": "x", "at": "t", "note": "", "gcps": _gcps(demo.true_gcps),
        "change_areas": None, "smoothing": 7.5}))
    runner.run_sheet(demo.page, demo.ref)
    assert json.loads((d / "transform.json").read_text())["smoothing"] == 7.5
    assert c.get(url(demo)).json()["smoothing"] == 7.5


# -- M6 / M7 ---------------------------------------------------------------------------

def test_bad_host_header_rejected(env):
    demo, c, _ = env
    assert c.get("/api/queue").status_code == 200
    assert c.get("/api/queue", headers={"host": "localhost:8765"}).status_code == 200
    assert c.get("/api/queue", headers={"host": "evil.example.com"}).status_code == 400


def test_reviewer_default_and_env(env, monkeypatch):
    demo, c, _ = env
    d = demo.work_dir / demo.series / demo.sheet_id
    body = {"gcps": _gcps(demo.true_gcps), "change_areas": None, "status": "edited"}
    monkeypatch.delenv("SFZMAPS_REVIEWER", raising=False)
    assert c.post(url(demo, "/save"), json=body).json()["reviewer"] == "owner"
    monkeypatch.setenv("SFZMAPS_REVIEWER", "jane")
    c.post(url(demo, "/save"), json=body)
    rv = json.loads((d / "review.json").read_text())
    assert rv["reviewer"] == "jane" and "@" not in json.dumps(rv)


def test_editor_js_reads_series_and_sheet_from_url():
    """I4: the review page links to EDITOR_URL?series=..&sheet=..; editor.js must read them."""
    from pathlib import Path

    from sfzmaps import reviewpage

    js = (Path(server.__file__).parent / "static" / "editor.js").read_text()
    assert "URLSearchParams" in js and 'get("series")' in js and 'get("sheet")' in js
    assert reviewpage.EDITOR_URL.rstrip("/").endswith(f"{server.HOST}:{server.PORT}")
    import inspect
    src = inspect.getsource(reviewpage)
    assert "?series=" in src and "&sheet=" in src


# -- auto-fill ---------------------------------------------------------------------------

def _seed(demo, n=6):
    pts = _gcps(demo.true_gcps[:: max(1, len(demo.true_gcps) // n)][:n])
    for p in pts:
        p["source"] = "manual"
    return pts


def _sheet_dir_of(demo):
    return demo.work_dir / demo.series / demo.sheet_id


def test_autofill_returns_dense_set_keeping_manual_points(env):
    demo, c, _ = env
    d = _sheet_dir_of(demo)
    seed = _seed(demo)
    names = ("refine.json", "transform.json", "rough.json")
    before = {n: (d / n).stat().st_mtime_ns for n in names if (d / n).exists()}
    r = c.post(url(demo, "/autofill"), json={"gcps": seed})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["gcps"]) > 25
    manual = [g for g in body["gcps"] if g["source"] == "manual"]
    assert len(manual) == 6
    assert {(g["px"], g["py"]) for g in manual} == {(g["px"], g["py"]) for g in seed}
    assert body["n_matches"] >= 40
    assert body["score"]["passed"] is True, body["score"]
    assert {"residuals", "matches", "preview_png_url", "preview_bounds", "preview_size"} <= set(body)
    af = json.loads((d / "autofill.json").read_text())
    assert {"seed", "raw", "result", "n_matches"} <= set(af)
    for n, m in before.items():
        assert (d / n).stat().st_mtime_ns == m, n


def test_autofill_needs_four_enabled_points(env):
    demo, c, _ = env
    r = c.post(url(demo, "/autofill"), json={"gcps": _seed(demo, 3)})
    assert r.status_code == 400 and "4" in r.json()["detail"]


def test_autofill_counts_enabled_points_only(env):
    demo, c, _ = env
    seed = _seed(demo, 5)
    seed[0]["enabled"] = seed[1]["enabled"] = False
    assert c.post(url(demo, "/autofill"), json={"gcps": seed}).status_code == 400
    seed = _seed(demo, 6)
    seed[1]["enabled"] = False   # not seed[0]: the other five would then be collinear
    r = c.post(url(demo, "/autofill"), json={"gcps": seed})
    assert r.status_code == 200, r.text
    off = [g for g in r.json()["gcps"] if g["enabled"] is False]
    assert len(off) == 1 and off[0]["source"] == "manual"


def test_autofill_without_blocks_is_409(env):
    demo, c, _ = env
    (_sheet_dir_of(demo) / "blocks_px.geojson").unlink()
    r = c.post(url(demo, "/autofill"), json={"gcps": _seed(demo)})
    assert r.status_code == 409 and "run the pipeline" in r.json()["detail"]


def test_autofill_degenerate_is_422_and_writes_nothing(env, monkeypatch):
    from sfzmaps.refine import RefineResult
    from sfzmaps.transform import GCP, SheetTransform
    demo, c, _ = env
    d = _sheet_dir_of(demo)
    w, h = runner.page_shape(d / "page.png")[::-1]
    gs = [GCP(px, py, wx, wy, "auto") for (px, py), (wx, wy) in
          zip([(0, 0), (w, 0), (0, h), (w, h)], [(0, 0), (1, 0), (0, 1), (1, 1)])]
    t = SheetTransform(gs, config.MANUAL_TPS_SMOOTHING).fit()
    monkeypatch.setattr(server.refine_mod, "refine", lambda *a, **k: RefineResult(t, [], 1, True))
    r = c.post(url(demo, "/autofill"), json={"gcps": _seed(demo)})
    assert r.status_code == 422, r.text
    assert not (d / "autofill.json").exists()


def test_preview_autofill_scores_against_autofill_json(env):
    demo, c, _ = env
    d = _sheet_dir_of(demo)
    a = c.post(url(demo, "/autofill"), json={"gcps": _seed(demo)})
    assert a.status_code == 200, a.text
    gcps = [{k: g[k] for k in ("px", "py", "wx", "wy", "source", "enabled")} for g in a.json()["gcps"]]
    p = c.post(url(demo, "/preview"), json={"gcps": gcps, "autofill": True})
    assert p.status_code == 200, p.text
    assert p.json()["score"]["n_scored"] == a.json()["score"]["n_scored"]
    (d / "refine.json").unlink()
    p = c.post(url(demo, "/preview"), json={"gcps": gcps, "autofill": True})
    assert p.status_code == 200, p.text
    assert p.json()["score"]["n_scored"] == a.json()["score"]["n_scored"]


def test_preview_autofill_without_file_is_409(env):
    demo, c, _ = env
    r = c.post(url(demo, "/preview"), json={"gcps": _seed(demo), "autofill": True})
    assert r.status_code == 409


def test_autofill_does_not_duplicate_manual_points_refine_kept(env, monkeypatch):
    """refine._fit appends the seed's non-auto GCPs on sparse sheets; they must not come back twice."""
    from sfzmaps.refine import RefineResult
    from sfzmaps.transform import GCP, SheetTransform
    demo, c, _ = env
    seed = _seed(demo)

    def fake_refine(corners, ref, s):
        gs = [GCP(g.px, g.py, g.wx, g.wy, "auto") for g in demo.true_gcps] + \
             [g for g in s.gcps if g.source != "auto"]
        ms = [runner._match_load(m) for m in
              json.loads((_sheet_dir_of(demo) / "refine.json").read_text())["matches"]]
        return RefineResult(SheetTransform(gs, config.TPS_SMOOTHING).fit(), ms, 1, True)

    monkeypatch.setattr(server.refine_mod, "refine", fake_refine)
    monkeypatch.setattr(server.refine_mod, "classify_changes", lambda r: r)
    r = c.post(url(demo, "/autofill"), json={"gcps": seed})
    assert r.status_code == 200, r.text
    gcps = r.json()["gcps"]
    assert len([g for g in gcps if g["source"] == "manual"]) == 6
    assert len(gcps) == 25 + 6


# -- save promotes the auto-fill ------------------------------------------------------------

def _autofill_and_save(demo, c, d, status="edited"):
    a = c.post(url(demo, "/autofill"), json={"gcps": _seed(demo)})
    assert a.status_code == 200, a.text
    af = json.loads((d / "autofill.json").read_text())
    gcps = [{k: g[k] for k in ("px", "py", "wx", "wy", "source", "enabled")} for g in a.json()["gcps"]]
    r = c.post(url(demo, "/save"), json={"gcps": gcps, "status": status, "autofill": True})
    return a, af, r


def test_save_autofill_promotes_and_matches_preview(env):
    demo, c, _ = env
    d = _sheet_dir_of(demo)
    a, af, r = _autofill_and_save(demo, c, d)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["score"]["passed"] is True
    assert body["score"]["holdout_p50_ft"] == pytest.approx(body["preview_score"]["holdout_p50_ft"])
    assert json.loads((d / "refine.json").read_text()) == af["result"]
    assert json.loads((d / "refine_raw.json").read_text()) == af["raw"]
    assert json.loads((d / "rough.json").read_text())["seeded_from"] == "editor autofill"
    m = [(d / n).stat().st_mtime_ns for n in ("rough.json", "refine_raw.json", "refine.json")]
    assert m == sorted(m)
    assert not (d / "autofill.json").exists()
    review = json.loads((d / "review.json").read_text())
    assert review["autofill"] == {"seed_points": 6, "matches": af["n_matches"]}
    assert review["smoothing"] == config.TPS_SMOOTHING


def test_save_without_autofill_flag_does_not_promote(env):
    demo, c, _ = env
    d = _sheet_dir_of(demo)
    before = (d / "refine.json").read_bytes()
    a = c.post(url(demo, "/autofill"), json={"gcps": _seed(demo)})
    assert a.status_code == 200, a.text
    r = c.post(url(demo, "/save"), json={"gcps": _gcps(demo.true_gcps), "status": "edited",
                                         "autofill": False})
    assert r.status_code == 200, r.text
    assert (d / "refine.json").read_bytes() == before
    assert "autofill" not in json.loads((d / "review.json").read_text())
    assert (d / "autofill.json").exists()


def test_force_auto_after_autofill_save_reproduces_refine(env):
    demo, c, _ = env
    d = _sheet_dir_of(demo)
    a, af, r = _autofill_and_save(demo, c, d)
    assert r.status_code == 200, r.text
    runner.run_sheet(demo.page, demo.ref, force_auto=True)
    assert json.loads((d / "transform.json").read_text())["origin"] == "auto"
    assert json.loads((d / "refine.json").read_text())["matches"] == af["result"]["matches"]


def test_front_end_has_autofill_and_current_pass_bar():
    from pathlib import Path

    static = Path(server.__file__).parent / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "editor.js").read_text(encoding="utf-8")
    assert 'id="tool-autofill"' in html
    assert "Auto-fill from my points" in html
    assert "≤ 7 ft" in html and "≤ 15 ft" in html
    assert "≤ 5 ft" not in html and "≤ 12 ft" not in html
    assert "&lt; 7 ft" in html and "&lt; 15 ft" in html
    assert "/autofill" in js and "autofill: S.autofill" in js
    assert "PASS_P50 = 7" in js and "PASS_P95 = 15" in js
    assert "above 5 ft" not in js
