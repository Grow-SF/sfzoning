import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from sfzmaps.frame import calibrate, data, decide, folds, learn, model, run, scores
from sfzmaps.frame.learn import Recipe

H, W = 40, 48
RULE = 4            # the toy frame rule's distance in from the page edge
SPLIT = 10          # an unlabelled page's dark line: the column cutting off its smaller piece
PATCH = {("a", "u0"): 2, ("b", "u0"): 4, ("b", "u1"): 3}   # side of the grey patch in that smaller piece
GREY = 153          # the patch's grey (0.6): map, not frame, under `lum_logits`
RECIPE = Recipe(steps=2, check_every=1, prefetch=False)
CURVE = [[1, 0.2, 1.0], [2, 0.3, 1.0]]                     # every fake held curve is lowest at step 1


@pytest.fixture(autouse=True)
def quick(small_model, monkeypatch):
    """Tiny nets and few resamples: the real counts are for real pages."""
    monkeypatch.setattr(calibrate, "N_CAL", 2_000)
    monkeypatch.setattr(calibrate, "CAL_BOOT", 5)
    monkeypatch.setattr(scores, "BOOT_N", 200)


def _labelled_lum(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lum = np.ones((H, W), np.float32)
    lum[RULE:H - RULE, RULE:W - RULE] = 0.1
    lum[RULE + 2:H - RULE - 2, RULE + 2:W - RULE - 2] = 1.0
    for _ in range(3):
        r = int(rng.integers(RULE + 4, H - RULE - 6))
        lum[r:r + 2, RULE + 4:W - RULE - 4] = 0.3
    lum[30:34, 30:36] = 0.2                                  # a printed block
    return lum


def _unlabelled_lum(sid) -> np.ndarray:
    """A dark line from the top edge to the bottom one at column SPLIT, and a grey patch left of it."""
    lum = np.ones((H, W), np.float32)
    lum[:, SPLIT] = 0.1
    n = PATCH[sid]
    lum[5:5 + n, 3:3 + n] = GREY / 255
    return lum


def _save(lum: np.ndarray, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(lum * 255).astype(np.uint8)).save(path)


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    """Two series, each with three labelled map pages (s0..s2; each s1 has a removed block), unlabelled map pages
    (a/u0, b/u0, b/u1) and a cover (c0), under tmp_path.  The data paths and BASELINE point there.  Returns
    (labelled sheets, out dir)."""
    work, labels, kinds = tmp_path / "work", tmp_path / "labels", tmp_path / "page_kinds.csv"
    monkeypatch.setattr(data, "WORK", work)
    monkeypatch.setattr(data, "LABELS", labels)
    monkeypatch.setattr(data, "PAGE_KINDS", kinds)
    monkeypatch.setattr(data, "OUT", tmp_path / "out")
    monkeypatch.setattr(run, "BASELINE", tmp_path / "baseline")
    rows, labelled = [], []
    for i, (s, sheet, kind) in enumerate([(s, f"s{j}", "map") for s in "ab" for j in range(3)]
                                         + [("a", "u0", "map"), ("b", "u0", "map"), ("b", "u1", "map"),
                                            ("a", "c0", "cover")]):
        if sheet.startswith("s"):
            _save(_labelled_lum(i), data.page_path(s, sheet, work))
            b = RULE + 2
            label = {"polygons": [[[b, b], [W - b, b], [W - b, H - b], [b, H - b]]]}
            if sheet == "s1":
                label["remove"] = [[[29, 29], [37, 29], [37, 35], [29, 35]]]
            (labels / s).mkdir(parents=True, exist_ok=True)
            (labels / s / f"{sheet}.json").write_text(json.dumps(label))
            labelled.append((s, sheet))
        elif kind == "map":
            _save(_unlabelled_lum((s, sheet)), data.page_path(s, sheet, work))
        else:
            _save(np.ones((H, W), np.float32), data.page_path(s, sheet, work))
        rows.append({"series": s, "sheet": sheet, "kind": kind, "checked": "test", "maps": ""})
    with open(kinds, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["series", "sheet", "kind", "checked", "maps"])
        wr.writeheader()
        wr.writerows(rows)
    return sorted(labelled), tmp_path / "out"


def _varied_net() -> torch.nn.Module:
    """A tiny random net whose chances vary across a page: a fresh one calls nearly every px "not map", so every page
    would be deleted whole and two seeds would always agree.  Its head is scaled up and leans to map."""
    net = model.new_net().eval()
    with torch.no_grad():
        net.net.head.weight.mul_(5.0)
        net.net.head.bias[:model.FOLD ** 2] += 2.0          # the head's first FOLD² channels are class 0, map
    return net


class FakeTrain:
    """A stand-in for `learn.train`: records each call and saves a tiny real net (a different random one per call and
    step, `_varied_net`) as the snapshot at each check, with the held curve CURVE.  Returns the last step's net."""

    def __init__(self):
        self.calls = []

    def __call__(self, pages, recipe, device, held=(), snapshots=None, label=""):
        self.calls.append({"label": label, "seed": recipe.seed, "steps": recipe.steps,
                           "train": [pg.id for pg in pages], "held": [pg.id for pg in held],
                           "snapshots": snapshots, "pages": {pg.id: pg for pg in (*pages, *held)}})
        net = None
        for step in range(recipe.check_every, recipe.steps + 1, recipe.check_every):
            torch.manual_seed(1000 * len(self.calls) + step)
            net = _varied_net()
            if snapshots is not None:
                snapshots.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), snapshots / f"step-{step:05d}.pt")
        return net, {"train_curve": [[s, 0.5] for s, _, _ in CURVE[:recipe.steps]],
                     "held_curve": [list(c) for c in CURVE[:recipe.steps]] if held else []}


@pytest.fixture
def fake_train(monkeypatch):
    fake = FakeTrain()
    monkeypatch.setattr(learn, "train", fake)
    return fake


def _lum(sid):
    return data.load_luminance(data.page_path(*sid, work=data.WORK))


def _cls(sid):
    return data.class_mask(data.load_label(*sid, root=data.LABELS), data.load_remove(*sid, root=data.LABELS),
                           (H, W))


def _files(root):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def test_choose_step_pools_ink_and_breaks_ties_early():
    a = [(500, 0.30, 10.0), (1000, 0.20, 10.0)]
    b = [(500, 0.10, 30.0), (1000, 0.20, 30.0)]
    assert run.choose_step([a, b]) == 500          # pooled: 0.15 vs 0.20
    assert run.choose_step([[(500, 0.2, 1.0), (1000, 0.2, 1.0)]]) == 500


def test_every_labelled_sheet_held_out_once(fake_world, fake_train, monkeypatch):
    labelled, out = fake_world
    # Random nets fit a large T, which flattens every chance to about 1/3 and deletes whole pages; the real fit
    # still runs on the run's samples, but T is 0.8 so the cuts below are not trivial.
    fitted, fit = [], calibrate.fit_temperature

    def fixed_fit(samples):
        fitted.append(samples)
        fit(samples)
        return 0.8
    monkeypatch.setattr(calibrate, "fit_temperature", fixed_fit)
    result = run.run_folds(labelled, RECIPE, "cpu", out=out)
    outer = folds.split(labelled, folds.OUTER_K)
    assert len(fitted[0]) == len(labelled)                         # one sample per held sheet, both seeds' logits
    assert all(z.shape == (len(run.SEEDS), 3, calibrate.N_CAL) for z, _ in fitted[0])

    # every fold x seed trains once, on its own sheets, holding out its own, snapshots where snapshot_path says
    calls = {(c["label"], c["seed"]): c for c in fake_train.calls}
    assert len(fake_train.calls) == len(outer) * len(run.SEEDS)
    for k, fold in enumerate(outer):
        for s in run.SEEDS:
            c = next(c for c in fake_train.calls if c["seed"] == s and c["held"] == list(fold.held))
            assert c["train"] == list(fold.train)
            assert c["snapshots"] == run.snapshot_path(out, k, s, 1).parent
            for sid, pg in c["pages"].items():                     # each page is its own sheet's
                np.testing.assert_array_equal(pg.lum, _lum(sid))
                np.testing.assert_array_equal(pg.cls, _cls(sid))
    assert len(calls) == len(fake_train.calls)
    assert any((pg.cls == 2).any() for c in fake_train.calls for pg in c["pages"].values())

    written = json.loads((out / run.FOLDS_FILE).read_text())
    assert written["step"] == result["step"] == 1                   # the pooled curve's lowest, not the last
    by_sheet = {(s["series"], s["sheet"]): s for s in written["sheets"]}
    assert sorted(by_sheet) == labelled
    temperature = written["temperature"]
    assert temperature > 0 and len(written["temperature_interval"]) == 2
    assert len(written["reliability"]) == 10 and len(written["curves"]) == len(outer) * len(run.SEEDS)

    # each sheet's outputs are its fold's two seeds at the chosen step, averaged at the fitted temperature
    for k, fold in enumerate(outer):
        nets = [run.load_snapshot(run.snapshot_path(out, k, s, 1)) for s in run.SEEDS]
        for sid in fold.held:
            assert by_sheet[sid]["fold"] == k
            lum = _lum(sid)
            zs = [model.logits(n, lum, "cpu") for n in nets]
            p = model.combine(zs, temperature)
            d = run.sheet_dir(out, sid)
            with np.load(d / run.PROB_FILE) as z:
                assert z["q"].dtype == np.uint8
                np.testing.assert_array_equal(z["q"], np.round(255 * (p[1] + p[2])).astype(np.uint8))
            want = decide.decide(p, lum, 1)
            with Image.open(d / run.OUTSIDE_PNG) as im:
                assert im.mode == "1"
                np.testing.assert_array_equal(np.asarray(im), want.deleted)
            assert (d / run.ERASED_PNG).exists() and (d / run.OUTLINE_FILE).exists()
            assert by_sheet[sid]["scores"] == pytest.approx(scores.sheet_scores(p, want.deleted, _cls(sid), lum),
                                                            nan_ok=True)
            # seed disagreement: each seed alone, decided at T, compared on ink
            own = [decide.decide(model.combine([z], temperature), lum, 1).deleted for z in zs]
            ink = 1.0 - lum.astype(np.float64)
            assert by_sheet[sid]["seed_disagreement"] == pytest.approx(ink[own[0] != own[1]].sum() / ink.sum())
            assert set(by_sheet[sid]["ratios"]) == {f"{r:g}" for r in run.RATIOS_SHOWN}
            assert 0 < want.deleted.mean() < 1                      # the fake nets' cuts are not trivial
    assert all(s["seed_disagreement"] > 0 for s in written["sheets"])

    assert set(written["pooled"]) >= {"frame_kept", "map_cut", "cost", "log_loss", "hd95_px"}
    assert set(written["ratios"]) == {f"{r:g}" for r in run.RATIOS_SHOWN}
    assert written["seed_disagreement"]["mean"] == pytest.approx(
        np.mean([s["seed_disagreement"] for s in written["sheets"]]))
    assert written["baseline"]["sheets"] == []
    text = run.format_scores(result)
    assert "step 1" in text and f"{temperature:.3f}" in text


def test_one_fold_run_stays_in_its_folder(fake_world, fake_train):
    labelled, out = fake_world
    result = run.run_folds(labelled, RECIPE, "cpu", out=out, only=1)
    held = folds.split(labelled, folds.OUTER_K)[1].held
    assert [(c["seed"], c["held"]) for c in fake_train.calls] == [(0, list(held))]
    assert result["step"] == RECIPE.steps                          # the recipe's last, not the curve's lowest
    assert {p.parts[0] for p in _files(out)} == {"fold-1"}
    sub = out / "fold-1"
    assert run.snapshot_path(sub, 1, 0, RECIPE.steps).exists()
    assert sorted((s["series"], s["sheet"]) for s in result["sheets"]) == sorted(held)
    for sid in held:
        assert (run.sheet_dir(sub, sid) / run.OUTSIDE_PNG).exists()
    assert result["seed_disagreement"] is None
    assert json.loads((sub / run.FOLDS_FILE).read_text())["step"] == RECIPE.steps


def test_a_full_run_drops_the_old_summary_first(fake_world, monkeypatch):
    labelled, out = fake_world
    out.mkdir()
    (out / run.FOLDS_FILE).write_text("{}")

    def boom(*args, **kwargs):
        raise RuntimeError("crash mid-run")
    monkeypatch.setattr(learn, "train", boom)
    with pytest.raises(RuntimeError):
        run.run_folds(labelled, RECIPE, "cpu", out=out)
    assert not (out / run.FOLDS_FILE).exists()


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
    assert outline == json.loads(json.dumps({"polygons": decide.outline(deleted), "maps": 2}))
    assert len(outline["polygons"]) == 1 and len(outline["polygons"][0]["holes"]) == 1


def lum_logits(net, lum, device):
    """Stand-in logits read off the page: dark px outside the frame, the rest map, never a removed block."""
    a = np.float32(20.0)
    return np.stack([a * (lum - 0.5), a * (0.5 - lum), np.full_like(lum, -a)]).astype(np.float32)


def test_predict_all_skips_labelled_and_ranks_dropped(fake_world, fake_train, monkeypatch):
    labelled, out = fake_world
    run.train_final(labelled, RECIPE, "cpu", out, step=1, temperature=1.5)
    assert [(c["seed"], c["steps"], c["held"], c["snapshots"]) for c in fake_train.calls] == \
        [(s, 1, [], None) for s in run.SEEDS]
    assert all(c["train"] == labelled for c in fake_train.calls)
    meta = json.loads((out / run.MODEL_DIR / "model.json").read_text())
    assert meta["step"] == 1 and meta["temperature"] == 1.5 and meta["seeds"] == list(run.SEEDS)
    assert meta["sheets"] == [list(s) for s in labelled] and meta["level"] == decide.cut_level()
    nets, temperature = run.load_model(out)
    assert temperature == 1.5 and len(nets) == len(run.SEEDS)
    saved = torch.load(out / run.MODEL_DIR / "seed-1.pt", weights_only=True)
    assert all(torch.equal(saved[k], v) for k, v in nets[1].state_dict().items())

    for sid in labelled:                                        # held-out outputs a run of the folds left
        d = run.sheet_dir(out, sid)
        d.mkdir(parents=True)
        (d / run.OUTSIDE_PNG).write_bytes(b"held-out")
    (out / run.FOLDS_FILE).write_text(json.dumps({"sheets": [
        {"series": s, "sheet": h, "filled_ink": 0.5, "dropped_ink": 2.5} for s, h in labelled]}))

    monkeypatch.setattr(model, "logits", lum_logits)
    run.predict_all("cpu", out)

    for sid in labelled:
        assert (run.sheet_dir(out, sid) / run.OUTSIDE_PNG).read_bytes() == b"held-out"
    assert not run.sheet_dir(out, ("a", "c0")).exists()
    ink = 1.0 - np.float32(GREY / 255)
    pages = json.loads((out / run.PAGES_FILE).read_text())
    got = {(p["series"], p["sheet"]): p for p in pages}
    assert set(got) == set(PATCH) | set(labelled)
    for sid, n in PATCH.items():
        assert (run.sheet_dir(out, sid) / run.OUTSIDE_PNG).exists()
        assert got[sid]["dropped_ink"] == pytest.approx(n * n * ink, rel=1e-5)
        assert got[sid]["filled_ink"] == 0.0 and not got[sid]["held_out"]
    assert all(got[sid]["held_out"] and got[sid]["dropped_ink"] == 2.5 for sid in labelled)
    drops = [p["dropped_ink"] for p in pages]
    assert drops == sorted(drops, reverse=True)
    assert [(p["series"], p["sheet"]) for p in pages][:2] == [("b", "u0"), ("b", "u1")]


def test_baseline_only_on_shared_sheets(fake_world, fake_train):
    labelled, out = fake_world
    shared = ("a", "s1")
    for sid in (shared, ("a", "u0")):                           # an unlabelled sheet's file is not compared
        d = run.sheet_dir(run.BASELINE, sid)
        d.mkdir(parents=True)
        np.savez_compressed(d / "map_prob.npz", map=np.ones((H, W), np.float16))   # all map: nothing cut
    result = run.run_folds(labelled, RECIPE, "cpu", out=out)

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
