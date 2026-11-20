import csv
import json
import math
from pathlib import Path

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
    tmp_path (the data paths point there).  Each sheet's label is a rectangle of its own, so a mask names its sheet.
    Returns (labelled sheets, out dir)."""
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
            if sheet.startswith("s"):
                b, r = RULE + 2, W - RULE - 2 - len(labelled)
                (labels / s).mkdir(parents=True, exist_ok=True)
                (labels / s / f"{sheet}.json").write_text(json.dumps(
                    {"polygons": [[[b, b], [r, b], [r, H - b], [b, H - b]]]}))
                labelled.append((s, sheet))
            seed += 1
            rows.append({"series": s, "sheet": sheet, "kind": kind, "checked": "test"})
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
    """A fake `train.train_joint`: each call records its label and every page it was given, training and validation
    (each page's id, luminance and mask), and returns a page pass and a pixel pass that are Consts whose
    probabilities name the call."""

    CURVE = [(1, 0.5, 0.4), (2, 0.3, 0.2)]

    def __init__(self):
        self.calls = []          # per call: {"label", "trained", "seen", "pages"}

    @staticmethod
    def _logit(kind: str, call: int) -> float:
        return (-2.0 if kind == "coarse" else 0.1) + 0.3 * call

    def which(self, kind: str, p: np.ndarray) -> int:
        """The call whose `kind` net gave the constant probability p."""
        p = np.asarray(p, np.float64)
        assert float(p.max() - p.min()) < 1e-3, "a toy net's output is constant"
        dist = [abs(_sig(self._logit(kind, c)) - float(p.mean())) for c in range(len(self.calls))]
        c = int(np.argmin(dist))
        assert dist[c] < 2e-3
        return c

    def train_joint(self, pages, recipe, device, val=None, label=""):
        given = pages + (val or [])
        self.calls.append({"label": label, "trained": {pg.id for pg in pages}, "seen": {pg.id for pg in given},
                           "pages": [(pg.id, pg.lum, pg.mask) for pg in given]})
        c = len(self.calls) - 1
        return (Const(3, self._logit("coarse", c)), Const(4, self._logit("fine", c))), \
            {"train_curve": self.CURVE, "val_curve": []}


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(train, "train_joint", rec.train_joint)
    return rec


def _check_no_leak(recorder, labelled, outer):
    """Each outer fold's one training sees exactly its training sheets: their own pages and labels, never a
    held-out sheet's (by id, and by the page and mask given)."""
    truth = {sid: run._labelled_page(sid) for sid in labelled}
    assert [c["label"] for c in recorder.calls] == [f"fold {k}" for k in range(len(outer))]
    for k, (call, fold) in enumerate(zip(recorder.calls, outer)):
        assert call["seen"] == call["trained"] == set(fold.train)
        assert not call["seen"] & set(fold.held)
        for sid, lum, mask in call["pages"]:
            assert np.array_equal(lum, truth[sid].lum) and np.array_equal(mask, truth[sid].mask)
            for held in fold.held:
                assert not np.array_equal(mask, truth[held].mask)
                assert not np.array_equal(lum, truth[held].lum)


def test_run_folds_never_scores_a_trained_sheet(toy, recorder):
    labelled, out = toy
    result = run.run_folds(labelled, RECIPE, "cpu", out=out)
    outer = folds.split(labelled, folds.OUTER_K)
    _check_no_leak(recorder, labelled, outer)

    # each held-out sheet is predicted by its own fold's two nets, the pixel pass reading the page pass's prior
    for k, f in enumerate(outer):
        for sid in f.held:
            with np.load(out / sid[0] / sid[1] / "map_prob.npz") as z:
                assert z["coarse"].dtype == np.float16 and z["map"].dtype == np.float16
                assert z["coarse"].shape == z["map"].shape == (H, W)
                assert recorder.which("coarse", z["coarse"]) == k and recorder.which("fine", z["map"]) == k

    # every sheet has its scores; folds.json has them with the fold, the means and the fold's logged curves
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
    assert [f["fold"] for f in written["folds"]] == list(range(folds.OUTER_K))
    for f, fold in zip(written["folds"], outer):
        assert f["held"] == [list(s) for s in fold.held]
        assert f["train_curve"] == [list(p) for p in Recorder.CURVE] and f["val_curve"] == []
    assert "steps" not in written


def test_uneven_series_each_counted(toy, recorder):
    """A series with one labelled sheet runs with the rest; every per-series mean counts its own sheets; and the
    leak checks hold with folds that hold out several sheets of one series."""
    labelled, out = toy
    labelled = [s for s in labelled if s[0] != "c" or s[1] == "s0"]          # c has one sheet: 9 in all
    result = run.run_folds(labelled, RECIPE, "cpu", out=out)
    outer = folds.split(labelled, folds.OUTER_K)
    assert [len(f.held) for f in outer] == [3, 2, 2, 2]                       # 9 dealt into 4
    _check_no_leak(recorder, labelled, outer)
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
    monkeypatch.setattr(train, "train_joint", boom)
    with pytest.raises(RuntimeError):
        run.run_folds(labelled, RECIPE, "cpu", out=out)
    assert not (out / "folds.json").exists()


def test_final_model_trains_on_every_labelled_sheet(toy, recorder):
    labelled, out = toy
    run.train_final(labelled, RECIPE, "cpu", out=out)
    assert [c["label"] for c in recorder.calls] == ["final"]
    assert recorder.calls[0]["seen"] == recorder.calls[0]["trained"] == set(labelled)
    for name, kind in (("coarse.pt", "coarse"), ("fine.pt", "fine")):
        state = torch.load(out / "model" / name, weights_only=True)
        assert float(state["b"]) == pytest.approx(Recorder._logit(kind, 0))
    recipe = json.loads((out / "model" / "recipe.json").read_text())
    assert recipe["recipe"]["seed"] == RECIPE.seed and recipe["recipe"]["lr"] == RECIPE.lr
    assert recipe["recipe"]["steps"] == RECIPE.steps
    assert recipe["sheets"] == [list(s) for s in labelled]
    assert recipe["train_curve"] == [list(p) for p in Recorder.CURVE]


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
