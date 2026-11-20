import collections
import csv
import json
import logging
import math

import numpy as np
import pytest
import torch
from PIL import Image

from sfzmaps.frame import data, folds, look, predict, run, train
from sfzmaps.frame.train import Recipe

SERIES = ("a", "b", "c")
H, W = 48, 64
RULE = 6          # the toy frame rule's distance in from the page edge
RECIPE = Recipe(steps=2, check_every=1, batch=1)


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    """Toy-sized nets: the real sizes would take minutes on cpu."""
    monkeypatch.setattr(predict, "TILE", 32)
    monkeypatch.setattr(predict, "COARSE_SIDE", 64)
    monkeypatch.setattr(predict, "COARSE_FOLD", 4)
    monkeypatch.setattr(predict, "COARSE_LEVELS", 3)
    monkeypatch.setattr(predict, "COARSE_WIDTH", 4)
    monkeypatch.setattr(predict, "FINE_FOLD", 4)
    monkeypatch.setattr(predict, "FINE_LEVELS", 3)
    monkeypatch.setattr(predict, "FINE_WIDTH", 8)


def _toy_lum(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lum = np.ones((H, W), np.float32)
    lum[RULE:H - RULE, RULE:W - RULE] = 0.0
    lum[RULE + 2:H - RULE - 2, RULE + 2:W - RULE - 2] = 1.0
    for _ in range(3):
        r = int(rng.integers(RULE + 4, H - RULE - 6))
        lum[r:r + 2, RULE + 4:W - RULE - 4] = 0.2
    return lum


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """Three toy series of four labelled map pages each, plus one unlabelled map page and one cover per series, under
    tmp_path (the data paths point there).  Returns (labelled sheets, out dir)."""
    work, labels, kinds = tmp_path / "work", tmp_path / "labels", tmp_path / "page_kinds.csv"
    monkeypatch.setattr(data, "WORK", work)
    monkeypatch.setattr(data, "LABELS", labels)
    monkeypatch.setattr(data, "PAGE_KINDS", kinds)
    rows, labelled = [], []
    seed = 0
    for s in SERIES:
        for sheet, kind in [(f"s{i}", "map") for i in range(4)] + [("u0", "map"), ("c0", "cover")]:
            p = data.page_path(s, sheet, work)
            p.parent.mkdir(parents=True)
            Image.fromarray(np.round(_toy_lum(seed) * 255).astype(np.uint8)).save(p)
            seed += 1
            rows.append({"series": s, "sheet": sheet, "kind": kind, "checked": "test"})
            if sheet.startswith("s"):
                b = RULE + 2
                (labels / s).mkdir(parents=True, exist_ok=True)
                (labels / s / f"{sheet}.json").write_text(json.dumps(
                    {"polygons": [[[b, b], [W - b, b], [W - b, H - b], [b, H - b]]]}))
                labelled.append((s, sheet))
    with open(kinds, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["series", "sheet", "kind", "checked"])
        wr.writeheader()
        wr.writerows(rows)
    return sorted(labelled), tmp_path / "out"


class Const(torch.nn.Module):
    """A network whose every output logit is `logit`: its probability identifies it."""

    def __init__(self, in_ch: int, logit: float):
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(float(logit)))
        self.in_ch = in_ch

    def forward(self, x):
        assert x.shape[1] == self.in_ch
        return x[:, :1] * 0 + self.b


def _sig(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


class Recorder:
    """Fake trainers: each call records the sheets it trained on ("trained"), every sheet it was given, training and
    validation ("seen"), and, for the pixel pass, each page's prior.  The net's state after each step (one check per
    step) is a Const whose probability names (call, step); the call returns the last one and, when asked, the
    snapshots of every step, as `train._fit` does.  Both curves bottom at step 1, short of the recipe's 2 steps, so a
    net at the chosen step differs from the net a call returns."""

    CURVES = {"coarse": [(1, 0.4), (2, 0.5)], "fine": [(1, 0.3), (2, 0.6)]}

    def __init__(self):
        self.coarse, self.fine = [], []   # per call: {"trained", "seen", "steps"[, "priors"]}
        self.tags = {"coarse": [], "fine": []}   # per kind: [(call, step)], a tag's index fixes its logit
        self.labels = []                         # every call's label, in call order

    @staticmethod
    def _logit(kind: str, t: int) -> float:
        return (-2.0 if kind == "coarse" else 0.1) + 0.06 * t

    def which(self, kind: str, p: np.ndarray) -> tuple[int, int]:
        """(call, step) of the net state that gave the constant probability p."""
        p = np.asarray(p, np.float64)
        assert float(p.max() - p.min()) < 1e-3, "a toy net's output is constant"
        dist = [abs(_sig(self._logit(kind, t)) - float(p.mean())) for t in range(len(self.tags[kind]))]
        t = int(np.argmin(dist))
        assert dist[t] < 2e-3
        return self.tags[kind][t]

    def _train(self, kind, in_ch, pages, recipe, val, snapshots, label, record):
        self.labels.append(label)
        k = len(getattr(self, kind))
        getattr(self, kind).append({"trained": {pg.id for pg in pages}, "seen": {pg.id for pg in pages + (val or [])},
                                    "steps": recipe.steps, **record})
        snaps = {}
        for step in range(1, recipe.steps + 1):
            self.tags[kind].append((k, step))
            snaps[step] = Const(in_ch, self._logit(kind, len(self.tags[kind]) - 1)).state_dict()
        net = Const(in_ch, 0.0)
        net.load_state_dict(snaps[recipe.steps])
        info = {"val_curve": self.CURVES[kind] if val else []}
        if snapshots:
            info["snapshots"] = snaps
        return net, info

    def train_coarse(self, pages, recipe, device, val=None, snapshots=False, label="page pass"):
        return self._train("coarse", 1, pages, recipe, val, snapshots, label, {})

    def train_fine(self, pages, recipe, device, val=None, snapshots=False, label="pixel pass"):
        priors = {pg.id: np.asarray(pg.prior, np.float32) for pg in pages + (val or [])}
        return self._train("fine", 2, pages, recipe, val, snapshots, label, {"priors": priors})


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(train, "train_coarse", rec.train_coarse)
    monkeypatch.setattr(train, "train_fine", rec.train_fine)
    return rec


def test_run_folds_never_scores_a_trained_sheet(toy, recorder):
    labelled, out = toy
    result = run.run_folds(labelled, RECIPE, "cpu", out=out)
    outer = folds.split(labelled, folds.OUTER_K)

    # every trainer call (outer or inner) sees exactly one fold's training sheets, never its held-out ones
    per_fold, home = collections.Counter(), {}
    for kind, calls in (("coarse", recorder.coarse), ("fine", recorder.fine)):
        for i, call in enumerate(calls):
            homes = [k for k, f in enumerate(outer) if call["seen"] == set(f.train)]
            assert len(homes) == 1, call["seen"]
            assert not call["seen"] & set(outer[homes[0]].held)
            per_fold[(homes[0], kind)] += 1
            home[(kind, i)] = homes[0]
    assert per_fold == {(k, kind): 4 for k in range(folds.OUTER_K) for kind in ("coarse", "fine")}
    chosen = {f["fold"]: f for f in result["folds"]}

    # every pixel-pass page's prior came from an inner page pass that never trained on that page (it validates on the
    # sheets it then predicts: validation only records its curve), taken at the step the fold chose, not the last
    for i, call in enumerate(recorder.fine):
        for sid, prior in call["priors"].items():
            c, step = recorder.which("coarse", prior)
            assert sid not in recorder.coarse[c]["trained"]
            assert recorder.coarse[c]["trained"] != recorder.coarse[c]["seen"]      # an inner page pass
            assert step == chosen[home[("fine", i)]]["coarse_steps"] < RECIPE.steps

    # each held-out sheet is predicted by nets that saw nothing of its fold's held-out sheets, nor did the page
    # passes that made the pixel pass's priors; the page pass that made its prior ran the chosen steps
    for k, f in enumerate(outer):
        for sid in f.held:
            with np.load(out / sid[0] / sid[1] / "map_prob.npz") as z:
                assert z["coarse"].dtype == np.float16 and z["map"].dtype == np.float16
                assert z["coarse"].shape == z["map"].shape == (H, W)
                (c, c_step), (m, m_step) = recorder.which("coarse", z["coarse"]), recorder.which("fine", z["map"])
            assert c_step == recorder.coarse[c]["steps"] == chosen[k]["coarse_steps"]
            assert m_step == recorder.fine[m]["steps"] == chosen[k]["fine_steps"]
            assert not recorder.coarse[c]["seen"] & set(f.held)
            assert not recorder.fine[m]["seen"] & set(f.held)
            for prior in recorder.fine[m]["priors"].values():
                assert not recorder.coarse[recorder.which("coarse", prior)[0]]["seen"] & set(f.held)

    # every sheet has its scores; folds.json has them with the fold, the means and the chosen steps
    written = json.loads((out / "folds.json").read_text())
    assert written == json.loads(json.dumps(result))
    by_sheet = {(s["series"], s["sheet"]): s for s in written["sheets"]}
    assert set(by_sheet) == set(labelled)
    for k, f in enumerate(outer):
        for sid in f.held:
            assert json.loads((out / sid[0] / sid[1] / "scores.json").read_text()) == by_sheet[sid]["scores"]
            assert by_sheet[sid]["fold"] == k
    assert set(written["series"]) == set(SERIES)
    assert written["overall"]["log_loss_all_ink"]["sheets"] == len(labelled)
    assert written["steps"] == {"coarse": [1] * folds.OUTER_K, "fine": [1] * folds.OUTER_K}
    # every validation curve is in folds.json
    for f in written["folds"]:
        assert f["coarse_curves"] == [[list(p) for p in Recorder.CURVES["coarse"]]] * folds.INNER_K
        assert f["fine_curves"] == [[list(p) for p in Recorder.CURVES["fine"]]] * folds.INNER_K
    # each training is labelled by its fold, pass and inner fold
    want = []
    for k in range(folds.OUTER_K):
        want += [f"fold {k} page inner {i}" for i in range(folds.INNER_K)] + [f"fold {k} page all"]
        want += [f"fold {k} pixel inner {i}" for i in range(folds.INNER_K)] + [f"fold {k} pixel all"]
    assert recorder.labels == want
    # the outer nets train for the chosen steps; the inner ones for the recipe's
    assert collections.Counter(c["steps"] for c in recorder.coarse) == {2: 12, 1: 4}
    assert collections.Counter(c["steps"] for c in recorder.fine) == {2: 12, 1: 4}


def test_uneven_series_each_counted(toy, recorder):
    """A series with one labelled sheet runs with the rest; every per-series mean counts its own sheets; and the
    leak checks hold with folds that hold out several sheets of one series."""
    labelled, out = toy
    labelled = [s for s in labelled if s[0] != "c" or s[1] == "s0"]          # c has one sheet: 9 in all
    result = run.run_folds(labelled, RECIPE, "cpu", out=out)
    outer = folds.split(labelled, folds.OUTER_K)
    assert [len(f.held) for f in outer] == [3, 2, 2, 2]                       # 9 dealt into 4
    for call in recorder.coarse + recorder.fine:
        homes = [k for k, f in enumerate(outer) if call["seen"] == set(f.train)]
        assert len(homes) == 1 and not call["seen"] & set(outer[homes[0]].held)
    assert {(s["series"], s["sheet"]) for s in result["sheets"]} == set(labelled)
    counts = {name: m["log_loss_all_ink"]["sheets"] for name, m in result["series"].items()}
    assert counts == {"a": 4, "b": 4, "c": 1}


def test_labelled_page_cuts_holes_and_ink_skips_the_strip(toy):
    labelled, _ = toy
    s, sheet = labelled[0]
    b = RULE + 2
    (data.LABELS / s / f"{sheet}.json").write_text(json.dumps(
        {"polygons": [[[b, b], [W - b, b], [W - b, H - b], [b, H - b]]], "holes": [[[20, 20], [30, 20], [30, 30],
                                                                                   [20, 30]]]}))
    pg = run._labelled_page((s, sheet))
    assert pg.mask[25, 25] == 0 and pg.mask[15, 40] == 1
    from sfzmaps.frame import scores
    keep = ~scores.unsure(pg.mask)
    ink = run._ink(pg.mask, pg.lum)
    assert ink["all"] == pytest.approx(float(data.ink_weight(pg.lum)[keep].sum(dtype=np.float64)))
    assert ink["all"] < float(data.ink_weight(pg.lum).sum(dtype=np.float64))


def test_a_curve_best_at_the_cap_is_logged(toy, recorder, monkeypatch, caplog):
    labelled, out = toy
    monkeypatch.setattr(Recorder, "CURVES", {"coarse": [(1, 0.4), (2, 0.5)], "fine": [(1, 0.6), (2, 0.3)]})
    with caplog.at_level(logging.INFO, logger="sfzmaps.frame.run"):
        run.run_folds(labelled, RECIPE, "cpu", out=out, only=1)
    capped = [r.getMessage() for r in caplog.records if "best at the cap" in r.getMessage()]
    assert capped == [f"fold 1 pixel inner {i}: best at the cap; more steps may help" for i in range(folds.INNER_K)]


def _files(root):
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_one_fold_run_leaves_the_main_tree_alone(toy, recorder):
    labelled, out = toy
    run.run_folds(labelled, RECIPE, "cpu", out=out)
    before = _files(out)
    run.run_folds(labelled, RECIPE, "cpu", out=out, only=0)
    after = _files(out)
    assert {p: after[p] for p in before} == before
    new = sorted(p for p in after if p not in before)
    held = folds.split(labelled, folds.OUTER_K)[0].held
    from pathlib import Path
    want = [Path("fold-0") / "folds.json"] + [Path("fold-0", s, h, name) for s, h in held
                                              for name in ("map_prob.npz", "scores.json")]
    assert new == sorted(want)
    summary = json.loads((out / "fold-0" / "folds.json").read_text())
    assert [f["fold"] for f in summary["folds"]] == [0]
    assert {(s["series"], s["sheet"]) for s in summary["sheets"]} == set(held)


def test_a_full_run_drops_the_old_summary_first(toy, recorder, monkeypatch):
    labelled, out = toy
    out.mkdir()
    (out / "folds.json").write_text("{}")

    def boom(*args, **kwargs):
        raise RuntimeError("crash mid-run")
    monkeypatch.setattr(train, "train_coarse", boom)
    with pytest.raises(RuntimeError):
        run.run_folds(labelled, RECIPE, "cpu", out=out)
    assert not (out / "folds.json").exists()


def test_final_priors_are_held_out_predictions(toy, recorder):
    labelled, out = toy
    for i, sid in enumerate(labelled):
        d = out / sid[0] / sid[1]
        d.mkdir(parents=True)
        np.savez_compressed(d / "map_prob.npz", coarse=np.full((H, W), 0.05 + 0.07 * i, np.float16),
                            map=np.zeros((H, W), np.float16))
    run.train_final(labelled, RECIPE, "cpu", {"coarse": [1, 2, 4, 5], "fine": [2, 2, 4, 6]}, out=out)

    assert [c["seen"] for c in recorder.coarse] == [set(labelled)]
    assert [c["seen"] for c in recorder.fine] == [set(labelled)]
    assert recorder.labels == ["final page", "final pixel"]
    assert recorder.coarse[0]["steps"] == 3 and recorder.fine[0]["steps"] == 3
    for i, sid in enumerate(labelled):
        np.testing.assert_array_equal(recorder.fine[0]["priors"][sid],
                                      np.full((H, W), np.float16(0.05 + 0.07 * i), np.float32))
    assert (out / "model" / "coarse.pt").exists() and (out / "model" / "fine.pt").exists()
    recipe = json.loads((out / "model" / "recipe.json").read_text())
    assert recipe["coarse_steps"] == 3 and recipe["fine_steps"] == 3
    assert recipe["recipe"]["seed"] == RECIPE.seed and recipe["recipe"]["lr"] == RECIPE.lr


def test_predict_all_keeps_held_out_and_draws_every_page(toy, monkeypatch):
    labelled, out = toy
    (out / "model").mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(predict.new_coarse_net().state_dict(), out / "model" / "coarse.pt")
    torch.save(predict.new_fine_net().state_dict(), out / "model" / "fine.pt")
    sentinel = np.full((H, W), 0.25, np.float16)
    for sid in labelled:
        d = out / sid[0] / sid[1]
        d.mkdir(parents=True)
        np.savez_compressed(d / "map_prob.npz", coarse=sentinel, map=sentinel)
        (d / "scores.json").write_text(json.dumps({"log_loss_all_ink": 0.125}))

    run.predict_all("cpu", out=out)
    for sid in labelled:
        with np.load(out / sid[0] / sid[1] / "map_prob.npz") as z:
            np.testing.assert_array_equal(z["map"], sentinel)
    for s in SERIES:
        with np.load(out / s / "u0" / "map_prob.npz") as z:
            assert z["map"].shape == (H, W) and z["map"].dtype == np.float16
        assert not (out / s / "c0").exists()

    paths = look.series_mosaics(out)
    assert sorted(p.name for p in paths) == [f"{s}.png" for s in SERIES]
    assert all(p.parent == out / "look" and p.exists() for p in paths)
    monkeypatch.setattr(look, "ODD_CASES", [("a", "s1"), ("b", "u0"), ("zz", "missing")])
    odd = look.odd_cases(out)
    assert odd == out / "look" / "odd_cases.png" and odd.exists()


def test_sheet_picture_colours_ink_by_probability():
    lum = np.ones((1000, 2000), np.float32)
    lum[:, :1000] = 0.0                     # ink on the left half
    prob = np.zeros(lum.shape, np.float32)
    prob[:500] = 1.0                        # map on the top half
    img = look.sheet_picture(lum, prob)
    assert img.size == (1400, 700)
    a = np.asarray(img).astype(int)
    assert tuple(a[100, 100]) == look.MAP_RGB            # ink, map
    assert tuple(a[600, 100]) == look.FRAME_RGB          # ink, frame
    assert tuple(a[100, 1300]) == (255, 255, 255)        # paper stays white
    assert tuple(a[350, 1300]) == look.CONTOUR_RGB       # the 1/2 contour
    assert tuple(a[600, 1300]) == (255, 255, 255)


def test_means_skip_nan_and_weigh_by_ink():
    sheets = [
        {"scores": {k: 1.0 for k in run.REGION_OF}, "ink": {"frame_band": 1.0, "map_band": 1.0, "all": 1.0}},
        {"scores": {k: 4.0 for k in run.REGION_OF}, "ink": {"frame_band": 3.0, "map_band": 3.0, "all": 3.0}},
        {"scores": {k: float("nan") for k in run.REGION_OF},
         "ink": {"frame_band": 0.0, "map_band": 0.0, "all": 0.0}},
    ]
    means = run.means(sheets)
    assert set(means) == set(run.REGION_OF)
    for k in run.REGION_OF:
        assert means[k] == {"mean": pytest.approx(13.0 / 4.0), "sheets": 2}
