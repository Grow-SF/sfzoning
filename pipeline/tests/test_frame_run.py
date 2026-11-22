import contextlib
import csv
import hashlib
import json
import types

import numpy as np
import pytest
from PIL import Image

from sfzmaps.frame import calibrate, data, decide, layout, learned, rules, run, scores, shape
from sfzmaps.frame.geometry import Frame

H, W = 40, 48
REAL_RUN = rules.RUN
RULE = 4            # the toy frame rule's distance in from the page edge
B = RULE + 2        # the label's edge: map is [B, W - B] x [B, H - B], outlines included


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Few resamples and head steps, two head seeds, no process pool: the real counts are for real pages."""
    monkeypatch.setattr(calibrate, "N_CAL", 2_000)
    monkeypatch.setattr(calibrate, "CAL_BOOT", 5)
    monkeypatch.setattr(layout, "COMPONENTS", 5)
    monkeypatch.setattr(layout, "HEAD_MAX_STEPS", 5)
    monkeypatch.setattr(layout, "HEAD_SEEDS", (0, 1))
    monkeypatch.setattr(run, "PROCESSES", 1)
    monkeypatch.setattr(rules, "RUN", 15)          # the toy pages' rules are 32-40 px long


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
TEST = [("a", "u0"), ("b", "u1")]          # unlabelled for training; labelled in the test folder
TEST_BOX = [[2, 4], [11, 4], [11, 10], [2, 10]]   # over a/u0's printed block


@pytest.fixture
def fake_world(tmp_path, monkeypatch, tiny_backbone):
    """Two series, each with three labelled map pages (s0..s2; each s1 has a removed block), unlabelled map pages
    (a/u0, b/u0, b/u1; a/u0 and b/u1 are the test sheets, labelled in their own folder, a/u0 with a box) and a cover
    (c0), under tmp_path; the stand-in backbone; plenty of free disk.  The data paths point there.  Returns (training
    sheets, out dir)."""
    work, labels, kinds = tmp_path / "work", tmp_path / "labels", tmp_path / "page_kinds.csv"
    monkeypatch.setattr(data, "WORK", work)
    monkeypatch.setattr(data, "LABELS", labels)
    monkeypatch.setattr(data, "TEST_LABELS", tmp_path / "labels-test")
    monkeypatch.setattr(data, "PAGE_KINDS", kinds)
    monkeypatch.setattr(data, "OUT", tmp_path / "out")
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
    for s, sheet in TEST:
        label = {"polygons": [[[B, B], [W - B, B], [W - B, H - B], [B, H - B]]]}
        if s == "a":
            label["remove"] = [TEST_BOX]
        (tmp_path / "labels-test" / s).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels-test" / s / f"{sheet}.json").write_text(json.dumps(label))
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


def _feat(out, sid):
    return np.load(layout.features_dir(out) / sid[0] / f"{sid[1]}.npy")


def _files(root):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def spy_train_head(monkeypatch):
    """Every head trained (the real training): the features of the pages it trained on, its seed and the head."""
    calls, real = [], layout.train_head

    def spy(feats, targets, seed, device="cpu"):
        head = real(feats, targets, seed, device)
        calls.append({"feats": [np.asarray(f) for f in feats], "seed": seed, "head": head})
        return head
    monkeypatch.setattr(layout, "train_head", spy)
    return calls


def _trained_on(call, feats):
    return {sid for sid, f in feats.items() if any(np.array_equal(f, g) for g in call["feats"])}


def _q(path):
    with np.load(path) as z:
        return z["q"]


def _written(p):
    return np.round(np.clip(p[1] + p[2], 0.0, 1.0) * 255.0).astype(np.uint8)


def test_fit_refuses_a_test_sheet_that_is_also_trained_on(fake_world):
    labelled, out = fake_world
    (data.TEST_LABELS / "a" / "s0.json").write_text((data.LABELS / "a" / "s0.json").read_text())
    with pytest.raises(ValueError, match="a/s0"):
        run.fit("cpu", out)
    assert not out.exists()                                     # before any features or heads


def test_no_test_sheet_reaches_training(fake_world, spy_train_head, monkeypatch):
    labelled, out = fake_world
    sampled, real_fit = [], calibrate.fit_temperature

    def spy_fit(samples):
        sampled.append(len(samples))
        return real_fit(samples)
    monkeypatch.setattr(calibrate, "fit_temperature", spy_fit)
    run.fit("cpu", out)
    feats = {sid: _feat(out, sid) for sid in labelled + TEST}
    assert len(spy_train_head) == (run.TEMPERATURE_FOLDS + 1) * len(layout.HEAD_SEEDS)
    for call in spy_train_head:
        seen = _trained_on(call, feats)
        assert not seen & set(TEST) and len(seen) == len(call["feats"])
    assert sampled and all(n == len(labelled) for n in sampled)      # the temperature: the training sheets only
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    assert meta["sheets"] == [list(s) for s in labelled]


def test_dev_sheets_are_cut_by_heads_that_did_not_train_on_them(fake_world, spy_train_head, pixel_shape, caplog):
    """Orchestration test (the shape decision is stubbed with the per-pixel cut)."""
    labelled, out = fake_world
    caplog.set_level("INFO", logger=run.__name__)
    marks = []

    @contextlib.contextmanager
    def turn():
        marks.append(len(spy_train_head))
        yield
        marks.append(len(spy_train_head))
    fitted = run.fit("cpu", out, turn=turn)
    total = (run.TEMPERATURE_FOLDS + 1) * len(layout.HEAD_SEEDS)
    assert marks == [0, total]                                   # every head inside the one GPU turn

    feats = {sid: _feat(out, sid) for sid in labelled}
    folds = [labelled[k::run.TEMPERATURE_FOLDS] for k in range(run.TEMPERATURE_FOLDS)]
    heads, final = {}, []
    for c in spy_train_head:
        left_out = set(labelled) - _trained_on(c, feats)
        if not left_out:
            final.append(c["seed"])
            continue
        k = next(k for k, f in enumerate(folds) if set(f) == left_out)    # a fold, held out whole
        heads.setdefault(k, {})[c["seed"]] = c["head"]
    assert sorted(heads) == list(range(run.TEMPERATURE_FOLDS))
    assert all(sorted(h) == list(layout.HEAD_SEEDS) for h in heads.values())
    assert sorted(final) == list(layout.HEAD_SEEDS)

    temperature = fitted["temperature"]
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    assert meta["temperature"] == temperature > 0 and len(meta["temperature_interval"]) == 2
    assert meta["reducer"] == {"file": "reducer.npz",
                               "pages": [f"{s}/{h}" for s, h in run.feature_pages(labelled + TEST)],
                               "sha256": hashlib.sha256((out / "reducer.npz").read_bytes()).hexdigest()}
    saved = {}
    with np.load(out / run.MODEL_DIR / run.DEV_LOGITS) as z:
        assert sorted(z.files) == sorted(f"{s}/{h}" for s, h in labelled)
        for k, fold in enumerate(folds):
            for sid in fold:
                saved[sid] = z[f"{sid[0]}/{sid[1]}"]
                want = np.stack([layout.head_logits(heads[k][s], feats[sid]) for s in layout.HEAD_SEEDS])
                np.testing.assert_array_equal(saved[sid], want)           # exact: the follow-up fits on them

    result = run.score_dev(out)
    chances = learned.Chances.from_json(meta["chances"])
    for sid in labelled:                                         # each sheet cut from its fold's logits at T
        p = layout.combine([layout.page_logits(z, (H, W)) for z in saved[sid]], temperature)
        np.testing.assert_array_equal(_q(run.sheet_dir(out / run.DEV_DIR, sid) / run.PROB_FILE),
                                      _written(learned.apply(p, _lum(sid), chances)))
    written = json.loads((out / run.DEV_DIR / run.SCORES_FILE).read_text())
    assert [(r["series"], r["sheet"]) for r in written["sheets"]] == labelled
    assert written["temperature"] == temperature and written["sheets"] == result["sheets"]
    assert written["pooled"] == pytest.approx(scores.pooled_split([r["split"] for r in written["sheets"]]),
                                              nan_ok=True)
    boxes = {(r["series"], r["sheet"]): len(r["split"]["boxes"]) for r in written["sheets"]}
    assert boxes == {sid: int(sid[1] == "s1") for sid in labelled}
    fits = json.loads((out / run.DEV_DIR / run.FITS_FILE).read_text())
    assert sorted(fits) == sorted(f"{s}/{h}" for s, h in labelled)
    text = run.format_split(result)
    for sid in labelled:                                         # one progress line per finished sheet
        assert sum(f"{sid[0]}/{sid[1]}: frame left" in r.getMessage() for r in caplog.records) == 1
    assert len(text.splitlines()) >= len(labelled) + 1
    assert "a/s1: frame left " in text and ("(0 of 1 deleted)" in text or "(1 of 1 deleted)" in text)
    assert text.splitlines()[-1].startswith("pooled")


def test_test_sheets_are_cut_by_the_final_heads(fake_world):
    """The real shape decision on the toy pages."""
    labelled, out = fake_world
    run.fit("cpu", out)
    heads, meta = run.load_model(out)
    result = run.score_test(out)
    assert [(r["series"], r["sheet"]) for r in result["sheets"]] == TEST
    for sid in TEST:
        f = _feat(out, sid)
        p = layout.combine([layout.page_logits(layout.head_logits(h, f), (H, W)) for h in heads],
                           meta["temperature"])
        d = run.sheet_dir(out / run.TEST_DIR, sid)
        np.testing.assert_array_equal(_q(d / run.PROB_FILE),
                                      _written(learned.apply(p, _lum(sid), learned.Chances.from_json(meta["chances"]))))
        assert json.loads((d / run.OUTLINE_FILE).read_text())["shape"] is not None
    row = result["sheets"][0]
    assert len(row["split"]["boxes"]) == 1 and row["split"]["box_ink"] > 0     # scored on the test folder's label
    fits = json.loads((out / run.TEST_DIR / run.FITS_FILE).read_text())
    assert sorted(fits) == ["a/u0", "b/u1"]
    fit = fits["a/u0"]
    assert set(fit) == {"frame", "notches", "boxes"} and set(fit["frame"]) == {"angle", "sides", "arcs"}
    assert json.loads((out / run.TEST_DIR / run.SCORES_FILE).read_text())["sheets"] == result["sheets"]


def _fit(angle=0.5, box_x=100.0, notch_d=3.0):
    side = {"offset": 40.0, "tilt": 0.0, "bow": 1.5}
    return {"frame": {"angle": angle, "sides": {"top": side, "bottom": side, "left": "page edge", "right": side},
                      "arcs": {"top-left": 12.0, "top-right": 0.0, "bottom-left": 0.0, "bottom-right": 0.0}},
            "notches": [{"side": "top", "psi": 0.5, "a0": 10.0, "a1": 90.0, "d": notch_d}],
            "boxes": [{"kind": "rect", "cx": box_x, "cy": 50.0, "theta": 0.5, "p": [-5.0, 5.0, -3.0, 3.0]}]}


def test_fit_changes_names_a_sheet_whose_frame_moved(tmp_path):
    before = {"s/1": _fit(), "s/2": _fit(), "s/3": _fit(), "s/4": _fit(), "s/6": _fit()}
    after = {"s/1": _fit(angle=0.6), "s/2": _fit(box_x=120.0), "s/3": _fit(notch_d=4.0), "s/4": _fit(),
             "s/5": _fit()}
    (tmp_path / "before.json").write_text(json.dumps(before))
    (tmp_path / "after.json").write_text(json.dumps(after))
    # the boxes may change (label-box work); a sheet in one file only is not a change of its fit
    assert run.fit_changes(tmp_path / "before.json", tmp_path / "after.json") == ["s/1", "s/3"]


def test_fits_are_plain_rounded_numbers():
    from sfzmaps.frame import blocks, notches

    fr = Frame(phi=0.123456)
    fr.offset["top"], fr.tilt["top"], fr.bow["top"] = 40.00049, 0.1111, 2.0
    fr.arc[("top", "left")] = 12.3456
    fit = run.fit_json(fr, [notches.Notch("top", 0.5, 10.0, 90.123456, 3.0, gain=9.9)],
                       [blocks.Block("rect", 100.0004, 50.0, 0.5, (-5.0, 5.0, -3.0, 3.0), gain=7.0)])
    assert fit["frame"]["angle"] == 0.123
    assert fit["frame"]["sides"]["top"] == {"offset": 40.0, "tilt": 0.111, "bow": 2.0}
    assert fit["frame"]["sides"]["bottom"] == "page edge"
    assert fit["frame"]["arcs"]["top-left"] == 12.346
    assert fit["notches"] == [{"side": "top", "psi": 0.5, "a0": 10.0, "a1": 90.123, "d": 3.0}]
    assert fit["boxes"] == [{"kind": "rect", "cx": 100.0, "cy": 50.0, "theta": 0.5, "p": [-5.0, 5.0, -3.0, 3.0]}]
    assert json.loads(json.dumps(fit)) == fit


def test_the_model_is_stale_until_fitted_and_when_its_inputs_change(fake_world):
    labelled, out = fake_world
    run.extract_features(labelled + TEST, "cpu", out)
    stamp = (out / "reducer.npz").stat().st_mtime_ns
    assert "missing" in run.model_stale(out)
    run.fit("cpu", out)
    assert (out / "reducer.npz").stat().st_mtime_ns == stamp                 # the features were reused
    assert run.model_stale(out) is None
    new = data.LABELS / "b" / "u0.json"                                      # a training label added
    new.write_text((data.LABELS / "b" / "s0.json").read_text())
    assert "training sheets" in run.model_stale(out)
    new.unlink()
    assert run.model_stale(out) is None
    meta_path = out / run.MODEL_DIR / "model.json"
    meta = json.loads(meta_path.read_text())
    meta["reducer"]["sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    assert "reducer" in run.model_stale(out)


def test_sample_draws_the_same_px_as_calibrate():
    rng = np.random.default_rng(1)
    zs = [rng.normal(size=(3, 5, 6)).astype(np.float32) for _ in range(2)]
    lum = rng.random((40, 48)).astype(np.float32)
    cls = rng.integers(0, 3, (40, 48)).astype(np.uint8)
    got = run.sample_patch_logits(zs, cls, lum, np.random.default_rng(7), 500)
    want = calibrate.sample([layout.page_logits(z, lum.shape) for z in zs], cls, lum, np.random.default_rng(7), 500)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_a_fit_that_stops_part_way_leaves_no_model(fake_world, monkeypatch):
    labelled, out = fake_world
    run.fit("cpu", out)

    def boom(*args, **kwargs):
        raise RuntimeError("crash mid-run")
    monkeypatch.setattr(layout, "train_head", boom)
    with pytest.raises(RuntimeError):
        run.fit("cpu", out)
    assert not (out / run.MODEL_DIR / "model.json").exists() and run.model_stale(out)


def test_score_pool_matches_one_process(fake_world, tmp_path, monkeypatch):
    """The process pool (spawned workers, the real shape decision) gives what one process gives."""
    labelled, out = fake_world
    monkeypatch.setattr(rules, "RUN", REAL_RUN)                  # what the spawned workers import
    rng = np.random.default_rng(0)
    jobs = [run.PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK), labels=data.LABELS,
                        zs=tuple(rng.normal(size=(3, 6, 7)).astype(np.float32) for _ in range(2)),
                        temperature=1.0, maps=1, bounds=run.blk.BlockBounds((20.0, 10.0), (10.0, 8.0)),
                        chances=learned.Chances(rule=(-6.0, 2.0, 1.5), ink=(0.2, 1.0)))
            for sid in labelled[:3]]
    one = run.pool_map(run.score_page, jobs, run.PARAMS, tmp_path / "one")
    monkeypatch.setattr(run, "PROCESSES", 2)
    assert run.pool_map(run.score_page, jobs, run.PARAMS, tmp_path / "two") == one
    assert [(r["series"], r["sheet"]) for r in one] == labelled[:3]


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


def test_predict_every_other_page_with_the_fitted_model(fake_world):
    """The real shape decision on the toy pages; the training sheets take their dev outputs."""
    labelled, out = fake_world
    run.fit("cpu", out)
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    assert meta["head_seeds"] == list(layout.HEAD_SEEDS) and meta["sheets"] == [list(s) for s in labelled]
    assert set(meta["bounds"]) == {"rect_max", "ellipse_max"} and meta["bounds"]["rect_max"][0] > 0
    heads, got = run.load_model(out)
    assert len(heads) == len(layout.HEAD_SEEDS) and got["temperature"] == meta["temperature"]

    for i, sid in enumerate(labelled):                          # the outputs a dev run left
        d = run.sheet_dir(out / run.DEV_DIR, sid)
        d.mkdir(parents=True)
        for name in (run.OUTSIDE_PNG, run.PROB_FILE, run.ERASED_PNG, run.OUTLINE_FILE):
            (d / name).write_bytes(b"dev")
    (out / run.DEV_DIR / run.SCORES_FILE).write_text(json.dumps({"sheets": [
        {"series": s, "sheet": h, "filled_ink": 0.0, "dropped_ink": 2.5, "removed_ink": 0.0, "side_ink_left": {},
         "side_ink_left_total": float(i)} for i, (s, h) in enumerate(labelled)]}))
    run.predict_all("cpu", out)

    for sid in labelled:
        d = run.sheet_dir(out, sid)
        assert all((d / n).read_bytes() == b"dev"
                   for n in (run.OUTSIDE_PNG, run.PROB_FILE, run.ERASED_PNG, run.OUTLINE_FILE))
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
    run.fit("cpu", out)
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    meta["reducer"]["sha256"] = "0" * 64
    (out / run.MODEL_DIR / "model.json").write_text(json.dumps(meta))
    with pytest.raises(run.StaleFeatures, match="reducer"):
        run.predict_all("cpu", out)


def test_a_new_page_after_the_fit_is_refused_not_refitted(fake_world, tiny_backbone):
    labelled, out = fake_world
    run.fit("cpu", out)
    stamp = (out / "reducer.npz").stat().st_mtime_ns
    _save(_unlabelled_lum(9), data.page_path("b", "u9", data.WORK))       # a page added since
    with open(data.PAGE_KINDS, "a", newline="") as f:
        f.write("b,u9,map,test,\n")
    assert "reducer" in run.model_stale(out) or "pages" in run.model_stale(out)
    with pytest.raises(run.StaleFeatures, match="refit"):
        run.predict_all("cpu", out)
    with pytest.raises(run.StaleFeatures, match="refit"):
        run.score_test(out)
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


def test_model_without_chances_is_stale(fake_world):
    labelled, out = fake_world
    run.fit("cpu", out)
    assert run.model_stale(out) is None
    path = out / run.MODEL_DIR / "model.json"
    meta = json.loads(path.read_text())
    got = meta["chances"]
    assert len(got["rule"]) == 3 and len(got["ink"]) == 2
    assert got["fitted_on"]["rules"] > 0 and got["fitted_on"]["rule_ink"] > 0 and got["fitted_on"]["other_ink"] > 0
    assert learned.Chances.from_json(got) == learned.Chances(tuple(got["rule"]), tuple(got["ink"]))
    del meta["chances"]
    path.write_text(json.dumps(meta))
    assert "chances" in run.model_stale(out)


def test_fit_adds_the_chances_to_a_current_model_without_retraining(fake_world, spy_train_head):
    labelled, out = fake_world
    run.fit("cpu", out)
    d = out / run.MODEL_DIR
    first = json.loads((d / "model.json").read_text())
    kept = {f: (d / f).read_bytes() for f in [run.DEV_LOGITS, *(f"head-{s}.pt" for s in layout.HEAD_SEEDS)]}
    del first["chances"]
    (d / "model.json").write_text(json.dumps(first))
    spy_train_head.clear()
    again = run.fit("cpu", out)
    assert spy_train_head == []                                  # no head trained
    assert {f: (d / f).read_bytes() for f in kept} == kept      # heads and dev logits as they were
    assert again["temperature"] == first["temperature"] and run.model_stale(out) is None
    assert again["level"] == decide.cut_level() == 0.5


@pytest.fixture
def spy_apply(monkeypatch):
    """Every `learned.apply` call: the chances it was given, the fit, and what it returned."""
    calls, real = [], learned.apply

    def spy(p, lum, chances, labels=None):
        q = real(p, lum, chances, labels)
        calls.append({"p": np.array(p), "chances": chances, "q": q})
        return q
    monkeypatch.setattr(learned, "apply", spy)
    return calls


def test_dev_and_test_pages_use_the_learned_chances(fake_world, spy_apply, monkeypatch):
    """Orchestration test (the shape decision is stubbed with the per-pixel cut, and records its chances)."""
    labelled, out = fake_world
    decided = []

    def stub(p, lum, params, bounds, maps=1, timing=None, cache=None):
        decided.append(np.array(p))
        return _pixel_shape(p, lum, params, bounds, maps)
    monkeypatch.setattr(shape, "decide", stub)
    run.fit("cpu", out)
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    want = learned.Chances.from_json(meta["chances"])
    logits = run.load_dev_logits(out)
    heads, _ = run.load_model(out)

    def check(where, pages, model_p):
        assert len(spy_apply) == len(decided) == len(pages)
        for call, p_decided, sid in zip(spy_apply, decided, pages):
            assert call["chances"] == want
            np.testing.assert_array_equal(call["p"], model_p(sid))          # the model's chances, corrected
            np.testing.assert_array_equal(p_decided, call["q"])              # the corrected ones are decided
            np.testing.assert_array_equal(_q(run.sheet_dir(where, sid) / run.PROB_FILE), _written(call["q"]))
        spy_apply.clear()
        decided.clear()

    def final(sid):
        return run.chances([layout.head_logits(h, _feat(out, sid)) for h in heads], (H, W), meta["temperature"])

    run.score_dev(out)
    check(out / run.DEV_DIR, labelled, lambda sid: run.chances(list(logits[sid]), (H, W), meta["temperature"]))
    run.score_test(out)
    check(out / run.TEST_DIR, TEST, final)
    run.predict_all("cpu", out)
    check(out, sorted(set(UNLABELLED)), final)
