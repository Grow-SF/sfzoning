import contextlib
import csv
import dataclasses
import hashlib
import json
import types

import numpy as np
import pytest
from PIL import Image

from sfzmaps.frame import calibrate, data, decide, layout, run, scores, shape
from sfzmaps.frame.geometry import Frame

H, W = 40, 48
RULE = 4            # the toy frame rule's distance in from the page edge
B = RULE + 2        # the label's edge: map is [B, W - B] x [B, H - B], outlines included


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Few resamples and head steps, two head seeds, no process pool: the real counts are for real pages."""
    monkeypatch.setattr(calibrate, "N_CAL", 2_000)
    monkeypatch.setattr(calibrate, "CAL_BOOT", 5)
    monkeypatch.setattr(scores, "BOOT_N", 200)
    monkeypatch.setattr(layout, "COMPONENTS", 5)
    monkeypatch.setattr(layout, "HEAD_MAX_STEPS", 5)
    monkeypatch.setattr(layout, "HEAD_SEEDS", (0, 1))
    monkeypatch.setattr(run, "PROCESSES", 1)


def _labelled_lum(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lum = np.ones((H, W), np.float32)
    lum[RULE:H - RULE, RULE:W - RULE] = 0.1                   # the rule: 2 px wide, outside the label
    lum[B:H - B, B:W - B] = 1.0
    for _ in range(3):
        r = int(rng.integers(RULE + 4, H - RULE - 6))
        lum[r:r + 2, RULE + 4:W - RULE - 4] = 0.3
    lum[30:34, 30:36] = 0.2                                  # a printed block
    return lum


def _unlabelled_lum(i: int) -> np.ndarray:
    lum = np.ones((H, W), np.float32)
    lum[:, 10 + 3 * i] = 0.1
    lum[5:9, 3:3 + i + 1] = 0.6
    return lum


def _save(lum: np.ndarray, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(lum * 255).astype(np.uint8)).save(path)


UNLABELLED = [("a", "u0"), ("b", "u0"), ("b", "u1")]


@pytest.fixture
def fake_world(tmp_path, monkeypatch, tiny_backbone):
    """Two series, each with three labelled map pages (s0..s2; each s1 has a removed block), unlabelled map pages
    (a/u0, b/u0, b/u1) and a cover (c0), under tmp_path; the stand-in backbone; plenty of free disk.  The data paths
    and BASELINE point there.  Returns (labelled sheets, out dir)."""
    work, labels, kinds = tmp_path / "work", tmp_path / "labels", tmp_path / "page_kinds.csv"
    monkeypatch.setattr(data, "WORK", work)
    monkeypatch.setattr(data, "LABELS", labels)
    monkeypatch.setattr(data, "PAGE_KINDS", kinds)
    monkeypatch.setattr(data, "OUT", tmp_path / "out")
    monkeypatch.setattr(run, "BASELINE", tmp_path / "baseline")
    monkeypatch.setattr(layout, "load_backbone", lambda device: tiny_backbone)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1e12, used=0, free=1e12))
    rows, labelled = [], []
    pages = ([(s, f"s{j}", "map") for s in "ab" for j in range(3)] + [(s, h, "map") for s, h in UNLABELLED]
             + [("a", "c0", "cover")])
    for i, (s, sheet, kind) in enumerate(pages):
        if sheet.startswith("s"):
            _save(_labelled_lum(i), data.page_path(s, sheet, work))
            label = {"polygons": [[[B, B], [W - B, B], [W - B, H - B], [B, H - B]]]}
            if sheet == "s1":
                label["remove"] = [[[29, 29], [37, 29], [37, 35], [29, 35]]]
            (labels / s).mkdir(parents=True, exist_ok=True)
            (labels / s / f"{sheet}.json").write_text(json.dumps(label))
            labelled.append((s, sheet))
        elif kind == "map":
            _save(_unlabelled_lum(i), data.page_path(s, sheet, work))
        else:
            _save(np.ones((H, W), np.float32), data.page_path(s, sheet, work))
        rows.append({"series": s, "sheet": sheet, "kind": kind, "checked": "test", "maps": ""})
    with open(kinds, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["series", "sheet", "kind", "checked", "maps"])
        wr.writeheader()
        wr.writerows(rows)
    return sorted(labelled), tmp_path / "out"


def _pixel_shape(p, lum, params, bounds, maps=1, timing=None, cache=None):
    """A fast stand-in for the shape decision (for the tests of the run's orchestration): the per-pixel cut."""
    return decide.decide(p, lum, maps), Frame(), [], []


@pytest.fixture
def pixel_shape(monkeypatch):
    monkeypatch.setattr(shape, "decide", _pixel_shape)


def _lum(sid):
    return data.load_luminance(data.page_path(*sid, work=data.WORK))


def _cls(sid):
    return data.class_mask(data.load_label(*sid, root=data.LABELS), data.load_remove(*sid, root=data.LABELS),
                           (H, W))


def _feat(out, sid):
    return np.load(layout.features_dir(out) / sid[0] / f"{sid[1]}.npy")


def _files(root):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def test_layout_chances_are_held_out(fake_world, pixel_shape, monkeypatch):
    """Orchestration test (the shape decision is stubbed with the per-pixel cut)."""
    labelled, out = fake_world
    calls, real, inside = [], layout.train_head, []

    def spy(feats, targets, seed, device="cpu"):
        head = real(feats, targets, seed, device)
        calls.append({"feats": [np.asarray(f) for f in feats], "seed": seed, "head": head, "turn": bool(inside)})
        return head
    monkeypatch.setattr(layout, "train_head", spy)

    @contextlib.contextmanager
    def turn():
        inside.append(1)
        yield
        inside.pop()
    result = run.held_out(labelled, "cpu", out, turn=turn)
    assert all(c["turn"] for c in calls)                      # every head inside the GPU turn

    feats = {sid: _feat(out, sid) for sid in labelled}

    def trained_on(call):
        return {sid for sid, f in feats.items() if any(np.array_equal(f, g) for g in call["feats"])}
    assert len(calls) == len(labelled) * len(layout.HEAD_SEEDS)
    heads = {}
    for c in calls:
        left_out = set(labelled) - trained_on(c)
        assert len(left_out) == 1 and len(c["feats"]) == len(labelled) - 1   # every other sheet, never its own
        heads.setdefault(left_out.pop(), {})[c["seed"]] = c["head"]
    assert sorted(heads) == labelled and all(sorted(h) == list(layout.HEAD_SEEDS) for h in heads.values())

    written = json.loads((out / run.FOLDS_FILE).read_text())
    temperature = written["temperature"]
    assert temperature == result["temperature"] and temperature > 0
    assert sorted((s["series"], s["sheet"]) for s in written["sheets"]) == labelled
    for sid in labelled:                                       # each sheet's chances: its own held-out heads at T
        zs = [layout.page_logits(layout.head_logits(heads[sid][s], feats[sid]), (H, W)) for s in layout.HEAD_SEEDS]
        p = layout.combine(zs, temperature)
        with np.load(run.sheet_dir(out, sid) / run.PROB_FILE) as z:
            np.testing.assert_array_equal(z["q"], np.round(255 * (p[1] + p[2])).astype(np.uint8))
        row = next(s for s in written["sheets"] if (s["series"], s["sheet"]) == sid)
        want = decide.decide(p, _lum(sid), 1)
        assert row["pixel"] == pytest.approx(scores.sheet_scores(p, want.deleted, _cls(sid), _lum(sid)),
                                             nan_ok=True)
        assert set(row["side_ink_left"]) == {"top", "bottom", "left", "right"}
    assert set(written) >= {"params", "temperature", "sheets", "pooled", "baseline", "search"}
    assert written["reducer"] == {"file": "reducer.npz", "pages": [f"{s}/{h}" for s, h in run.feature_pages(labelled)],
                                  "sha256": hashlib.sha256((out / "reducer.npz").read_bytes()).hexdigest()}
    assert set(written["pooled"]) == {"shape", "pixel"}


RIGGED = shape.Params(lam_notch=0, beta=10, lam_block=1e4)


def _rigged_decide(p, lum, params, bounds, maps=1, timing=None, cache=None):
    """Deletes exactly what the label calls outside, except that each of the numbers off RIGGED leaves one side's
    rule in: the pooled cost falls one coordinate at a time towards RIGGED."""
    off = sum(getattr(params, k) != getattr(RIGGED, k) for k in ("lam_notch", "beta", "lam_block"))
    deleted = np.ones(lum.shape, bool)
    deleted[B:H - B + 1, B:W - B + 1] = False
    for k, sl in enumerate([np.s_[RULE:B, :], np.s_[H - B:H - RULE, :], np.s_[:, RULE:B]][:off]):
        deleted[sl] = False
    _rigged_decide.calls.append((cache, hashlib.sha1(np.ascontiguousarray(p).tobytes()).hexdigest(), params))
    return decide.Decision(deleted, np.zeros_like(deleted), 0.0, 0.0, 0.0), Frame(), [], []


def test_params_minimise_the_pooled_cost(fake_world, monkeypatch):
    """Orchestration test (the shape decision is stubbed with `_rigged_decide`)."""
    labelled, out = fake_world
    _rigged_decide.calls = []
    monkeypatch.setattr(shape, "decide", _rigged_decide)
    result = run.held_out(labelled, "cpu", out)
    assert result["params"] == dataclasses.asdict(RIGGED)
    search = result["search"]
    points = [(r["lam_notch"], r["beta"], r["lam_block"]) for r in search]
    assert len(points) == len(set(points)) == 15          # 4 + 3 + 2 new points in the first round, 3 + 3 + 0 in the second
    assert result["rounds"] == 2                           # the second round changed nothing
    assert points[0] == (1e4, 3, 1e3)                    # the start first
    costs = {pt: r["cost"] for pt, r in zip(points, search)}
    assert min(costs, key=costs.get) == (0, 10, 1e4)
    assert costs[(0, 10, 1e4)] < sorted(costs.values())[1]
    assert json.loads((out / run.FOLDS_FILE).read_text())["search"] == search
    # every search cache saw a single page (the calls hold the caches, so no two share an id)
    pages_of = {}
    for cache, page, _ in _rigged_decide.calls:
        if cache is not None:
            pages_of.setdefault(id(cache), set()).add(page)
    assert pages_of and all(len(v) == 1 for v in pages_of.values())


def test_search_pool_matches_one_process(fake_world, monkeypatch):
    """The process pool (spawned workers, the real shape decision) gives what one process gives."""
    labelled, out = fake_world
    rng = np.random.default_rng(0)
    jobs = [run.PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK), labels=data.LABELS,
                        zs=tuple(rng.normal(size=(3, 6, 7)).astype(np.float32) for _ in range(2)),
                        temperature=1.0, maps=1, bounds=run.blk.BlockBounds((20.0, 10.0), (10.0, 8.0)))
            for sid in labelled[:3]]
    points = (shape.Params(1e4, 3, 1e3), shape.Params(0, 0, 0))
    one = run.pool_map(run.search_page, jobs, points)
    monkeypatch.setattr(run, "PROCESSES", 2)
    assert run.pool_map(run.search_page, jobs, points) == one
    assert len(one) == 3 and all(len(r) == 2 and {"cost", "frame_kept", "map_cut"} <= set(r[0]) for r in one)


def test_sample_draws_the_same_px_as_calibrate():
    rng = np.random.default_rng(1)
    zs = [rng.normal(size=(3, 5, 6)).astype(np.float32) for _ in range(2)]
    lum = rng.random((40, 48)).astype(np.float32)
    cls = rng.integers(0, 3, (40, 48)).astype(np.uint8)
    got = run.sample_patch_logits(zs, cls, lum, np.random.default_rng(7), 500)
    want = calibrate.sample([layout.page_logits(z, lum.shape) for z in zs], cls, lum, np.random.default_rng(7), 500)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_a_held_out_run_drops_the_old_summary_first(fake_world, monkeypatch):
    labelled, out = fake_world
    out.mkdir()
    (out / run.FOLDS_FILE).write_text("{}")

    def boom(*args, **kwargs):
        raise RuntimeError("crash mid-run")
    monkeypatch.setattr(layout, "train_head", boom)
    with pytest.raises(RuntimeError):
        run.held_out(labelled, "cpu", out)
    assert not (out / run.FOLDS_FILE).exists()


def test_baseline_only_on_shared_sheets(fake_world, pixel_shape):
    labelled, out = fake_world
    shared = ("a", "s1")
    for sid in (shared, ("a", "u0")):                           # an unlabelled sheet's file is not compared
        d = run.sheet_dir(run.BASELINE, sid)
        d.mkdir(parents=True)
        np.savez_compressed(d / "map_prob.npz", map=np.ones((H, W), np.float16))   # all map: nothing cut
    result = run.held_out(labelled, "cpu", out)

    base = result["baseline"]
    assert [(s["series"], s["sheet"]) for s in base["sheets"]] == [shared]
    lum, cls = _lum(shared), _cls(shared)
    frame_ink = float((1.0 - lum.astype(np.float64))[cls != 0].sum())
    old = base["sheets"][0]
    assert old["frame_kept"] == pytest.approx(frame_ink) and old["map_cut"] == 0.0
    assert old["cost"] == pytest.approx(decide.COST_FRAME * frame_ink)
    new = next(s["scores"] for s in result["sheets"] if (s["series"], s["sheet"]) == shared)
    assert set(base["paired"]) == {"frame_kept", "map_cut", "cost"}
    for key in base["paired"]:
        diff = new[key] - old[key]
        assert base["paired"][key] == pytest.approx({"mean": diff, "lo": diff, "hi": diff})
    text = run.format_scores(result)
    first = next(line for line in text.splitlines() if "left in" in line)
    assert "%" in first.split("left in")[1][:12]           # percentages first


@pytest.mark.parametrize("mode", ["RGB", "1"])
def test_write_outputs_keeps_colour(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(data, "WORK", tmp_path / "work")
    sid = ("s", "p")
    rng = np.random.default_rng(0)
    paper, frame, ink = (250, 240, 220), (120, 20, 20), (30, 30, 30)
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[...] = frame
    rgb[5:H - 5, 5:W - 5] = paper
    rgb[12:14, 8:40] = ink
    path = data.page_path(*sid, work=data.WORK)
    path.parent.mkdir(parents=True)
    if mode == "RGB":
        Image.fromarray(rgb).save(path)
    else:
        Image.fromarray(rgb.mean(axis=2) > 128).save(path)       # 1-bit: paper True (white), the rest black
    with Image.open(path) as im:
        assert im.mode == mode
        page = np.array(im.convert("L") if mode == "1" else im)

    outside = np.ones((H, W), bool)
    outside[5:H - 5, 5:W - 5] = False
    removed = np.zeros((H, W), bool)
    removed[25:30, 20:28] = True
    decision = decide.Decision(outside, removed, 0.0, 0.0, 0.0)
    p = rng.dirichlet((1.0, 1.0, 1.0), (H, W)).transpose(2, 0, 1).astype(np.float32)
    run.write_outputs(tmp_path / "out", sid, p, decision, 2)

    d = run.sheet_dir(tmp_path / "out", sid)
    with Image.open(d / run.ERASED_PNG) as im:
        assert im.mode == ("RGB" if mode == "RGB" else "L")
        erased = np.asarray(im)
    deleted = decision.deleted
    fill = np.round(np.median(page[~deleted], axis=0)).astype(np.uint8)
    np.testing.assert_array_equal(fill, paper if mode == "RGB" else 255)
    np.testing.assert_array_equal(erased[deleted], np.broadcast_to(fill, erased[deleted].shape))
    np.testing.assert_array_equal(erased[~deleted], page[~deleted])
    with Image.open(d / run.OUTSIDE_PNG) as im:
        assert im.mode == "1"
        np.testing.assert_array_equal(np.asarray(im), deleted)
    with np.load(d / run.PROB_FILE) as z:
        np.testing.assert_array_equal(z["q"], np.round(255 * (p[1] + p[2])).astype(np.uint8))
    outline = json.loads((d / run.OUTLINE_FILE).read_text())
    assert outline["polygons"] == json.loads(json.dumps(decide.outline(deleted))) and outline["maps"] == 2
    assert len(outline["polygons"]) == 1 and len(outline["polygons"][0]["holes"]) == 1
    assert outline["shape"] is None


def test_train_final_then_predict_every_other_page(fake_world):
    """The real shape decision on the toy pages; the labelled sheets keep their held-out outputs."""
    labelled, out = fake_world
    held = {"params": dataclasses.asdict(shape.Params(1e4, 3, 1e3)), "temperature": 1.25,
            "reducer": run.extract_features(labelled, "cpu", out)}
    run.train_final(labelled, "cpu", out, held)
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    assert meta["params"] == held["params"] and meta["temperature"] == 1.25
    assert meta["head_seeds"] == list(layout.HEAD_SEEDS) and meta["sheets"] == [list(s) for s in labelled]
    assert set(meta["bounds"]) == {"rect_max", "ellipse_max"} and meta["bounds"]["rect_max"][0] > 0
    assert meta["reducer"] == held["reducer"]
    assert meta["reducer"]["sha256"] == hashlib.sha256((out / "reducer.npz").read_bytes()).hexdigest()
    heads, got = run.load_model(out)
    assert len(heads) == len(layout.HEAD_SEEDS) and got["temperature"] == 1.25

    for i, sid in enumerate(labelled):                          # held-out outputs a held-out run left
        d = run.sheet_dir(out, sid)
        d.mkdir(parents=True)
        (d / run.OUTSIDE_PNG).write_bytes(b"held-out")
    (out / run.FOLDS_FILE).write_text(json.dumps({"sheets": [
        {"series": s, "sheet": h, "filled_ink": 0.0, "dropped_ink": 2.5, "removed_ink": 0.0,
         "side_ink_left_total": float(i)} for i, (s, h) in enumerate(labelled)]}))
    run.predict_all("cpu", out)

    for sid in labelled:
        assert (run.sheet_dir(out, sid) / run.OUTSIDE_PNG).read_bytes() == b"held-out"
    assert not run.sheet_dir(out, ("a", "c0")).exists()
    for sid in UNLABELLED:
        d = run.sheet_dir(out, sid)
        assert {f.name for f in d.iterdir()} == {run.OUTSIDE_PNG, run.PROB_FILE, run.ERASED_PNG, run.OUTLINE_FILE}
        assert json.loads((d / run.OUTLINE_FILE).read_text())["shape"] is not None
    pages = json.loads((out / run.PAGES_FILE).read_text())
    assert {(p["series"], p["sheet"]) for p in pages} == set(UNLABELLED) | set(labelled)
    side = [p["side_ink_left_total"] for p in pages]
    assert side == sorted(side, reverse=True)
    assert sorted(p["side_rank"] for p in pages) == list(range(1, len(pages) + 1))
    by_drop = sorted(pages, key=lambda p: p["dropped_rank"])
    assert [p["dropped_ink"] for p in by_drop] == sorted((p["dropped_ink"] for p in pages), reverse=True)
    assert all(p["held_out"] == ((p["series"], p["sheet"]) in labelled) for p in pages)


def test_predict_refuses_a_full_disk(monkeypatch, fake_world):
    labelled, out = fake_world
    monkeypatch.setattr(run.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1e12, used=0, free=19e9))
    with pytest.raises(run.DiskFull, match="20 GB"):
        run.predict_all("cpu", out)
    assert not out.exists() or not _files(out)


def test_predict_refuses_a_changed_reducer(fake_world):
    labelled, out = fake_world
    run.train_final(labelled, "cpu", out, {"params": dataclasses.asdict(shape.Params(1e4, 3, 1e3)),
                                           "temperature": 1.0, "reducer": run.extract_features(labelled, "cpu", out)})
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    meta["reducer"]["sha256"] = "0" * 64
    (out / run.MODEL_DIR / "model.json").write_text(json.dumps(meta))
    with pytest.raises(run.StaleFeatures, match="reducer"):
        run.predict_all("cpu", out)


def test_a_new_page_after_the_held_out_run_is_refused_not_refitted(fake_world, tiny_backbone):
    labelled, out = fake_world
    record = run.extract_features(labelled, "cpu", out)
    held = {"params": dataclasses.asdict(shape.Params(1e4, 3, 1e3)), "temperature": 1.0, "reducer": record}
    run.train_final(labelled, "cpu", out, held)
    stamp = (out / "reducer.npz").stat().st_mtime_ns
    _save(_unlabelled_lum(9), data.page_path("b", "u9", data.WORK))       # a page added since
    with open(data.PAGE_KINDS, "a", newline="") as f:
        f.write("b,u9,map,test,\n")
    with pytest.raises(run.StaleFeatures, match="--held-out"):
        run.train_final(labelled, "cpu", out, held)
    with pytest.raises(run.StaleFeatures, match="--held-out"):
        run.predict_all("cpu", out)
    with pytest.raises(run.StaleFeatures, match="--held-out"):
        run.train_final(labelled, "cpu", out, {k: v for k, v in held.items() if k != "reducer"})   # an older run
    assert (out / "reducer.npz").stat().st_mtime_ns == stamp                 # never refitted
    assert not (layout.features_dir(out) / "b" / "u9.npy").exists()


def test_an_old_model_is_refused(tmp_path):
    (tmp_path / run.MODEL_DIR).mkdir()
    (tmp_path / run.MODEL_DIR / "model.json").write_text(json.dumps({"step": 3000, "temperature": 1.0,
                                                                     "seeds": [0, 1]}))
    with pytest.raises(FileNotFoundError, match="old"):
        run.load_model(tmp_path)


def test_pages_rank_by_side_ink_then_dropped():
    rows = [{"series": "s", "sheet": str(i), "side_ink_left_total": a, "dropped_ink": b}
            for i, (a, b) in enumerate([(1.0, 5.0), (3.0, 0.0), (1.0, 9.0), (0.0, 7.0)])]
    got = run.rank_pages(rows)
    assert [r["sheet"] for r in got] == ["1", "2", "0", "3"]
    assert {r["sheet"]: r["dropped_rank"] for r in got} == {"2": 1, "3": 2, "0": 3, "1": 4}
    assert [r["side_rank"] for r in got] == [1, 2, 3, 4]
