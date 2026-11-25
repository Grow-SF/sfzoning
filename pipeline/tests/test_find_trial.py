"""Trial data roots for placement (`find.trial`, `sfzmaps find trial`): a root holds copies (never links) of a sheet's
work files up to its blocks, links only raw/ ref/ cache/, refuses a root inside today's data and a sheet that does not
test the blocks seam, and is prepared once; the numbers are read from a root's own files, the same way for every root;
two roots are compared file by file with no tolerance; the share of a sheet's ground under the run's lot-labelled
training sheets and the run's hand crops on it."""
import json
import os

import numpy as np
import pytest
from shapely.geometry import box, mapping

from find_fixtures import invoke
from sfzmaps import config, inputs
from sfzmaps.find import pages, trial
from sfzmaps.find.trial import KEPT

DOWNSTREAM = ("rough.json", "refine_raw.json", "refine.json", "changes.geojson", "transform.json", "score.json",
              "warped.tif", "warped.footprint.geojson")
SIDECARS = ("page.png.inputs.json", "page.json.inputs.json", "page.json.answers", "labels.json.inputs.json",
            "labels.json.answers", "blocks_px.geojson.inputs.json", "warped_paper.tif.inputs.json")


def _transform(gcps):
    return {"smoothing": 0.1, "gcps": [{"px": x, "py": y, "wx": x * 2, "wy": y * 2, "source": "auto", "enabled": e}
                                       for x, y, e in gcps], "origin": "auto"}


def _fill(root, series="s", sheet="a", chosen="probabilistic"):
    """A placed sheet in data root `root`: every KEPT file, the sidecars, every downstream work file and a warp
    folder, with known numbers (3 blocks, 2 corners, 2 of 3 control points enabled)."""
    d = root / "work" / series / sheet
    d.mkdir(parents=True)
    for f in (*KEPT, *SIDECARS, *DOWNSTREAM):
        (d / f).write_text(f"{f} of {sheet}")
    blocks = {"type": "FeatureCollection", "corners": [[1, 2], [None, 3]],
              "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(i, 0, i + 1, 1))}
                           for i in range(3)]}
    (d / "blocks_px.geojson").write_text(json.dumps(blocks))
    (d / "transform.json").write_text(json.dumps(_transform([(1.5, 2.5, True), (3, 4, False), (5, 6, True)])))
    (d / "refine.json").write_text(json.dumps({"matches": [{"px": 1.5, "py": 2.5, "label": "inlier"}]}))
    (d / "score.json").write_text(json.dumps({"passed": True, "reason": None, "holdout_p50_ft": 5.01,
                                              "holdout_p95_ft": 12.28}))
    w = root / "warp" / series / sheet
    w.mkdir(parents=True)
    (w / "scores.json").write_text(json.dumps({"within1_first": 0.6, "within1_probabilistic": 0.627,
                                               "chosen": chosen, "device": "mps"}))
    np.savez(w / "correction.npz", Gc=np.arange(6, dtype=np.float32).reshape(2, 3), spacing=np.int64(4),
             kind=np.str_(chosen))


@pytest.fixture
def src(tmp_path):
    root = tmp_path / "src"
    for x in ("raw", "ref", "cache"):
        (root / x).mkdir(parents=True)
    _fill(root)
    return root


def test_a_trial_root_holds_copies_of_the_inputs_and_nothing_after_them(src, tmp_path):
    root = tmp_path / "t" / "traced"
    copied = trial.prepare(root, [("s", "a")], source=src)
    d = root / "work/s/a"
    assert {p.name for p in d.iterdir()} == {*KEPT, *SIDECARS}
    assert sorted(copied) == sorted(d.iterdir())
    assert not any((d / f).is_symlink() or os.stat(d / f).st_ino == os.stat(src / "work/s/a" / f).st_ino
                   for f in (*KEPT, *SIDECARS))
    (d / "page.json").write_text("x")
    assert (src / "work/s/a/page.json").read_text() != "x"
    assert all((root / x).is_symlink() and (root / x).resolve() == (src / x).resolve() for x in ("raw", "ref", "cache"))
    assert not any((d / f).exists() for f in DOWNSTREAM)
    assert not (root / "warp").exists()


@pytest.mark.parametrize("inside", ["", "work/x", "warp/x", "out/x"])
def test_a_root_in_todays_data_is_refused(src, inside):
    root = src / inside if inside else src
    before = sorted(p.relative_to(src) for p in src.rglob("*"))
    with pytest.raises(ValueError, match="today's data"):
        trial.prepare(root, [("s", "a")], source=src)
    assert sorted(p.relative_to(src) for p in src.rglob("*")) == before        # nothing made


@pytest.mark.parametrize("why", ["duplicate", "review", "seed", "missing"])
def test_a_sheet_that_does_not_test_the_seam_is_refused(src, tmp_path, why):
    if why == "duplicate":
        (config.INPUTS_DIR / "s").mkdir(parents=True)
        (config.INPUTS_DIR / "s" / "duplicate_scans.csv").write_text("sheet,kept,reason\na,b,same scan\n")
    elif why == "review":
        inputs.review_path("s", "a").parent.mkdir(parents=True)
        inputs.review_path("s", "a").write_text("{}")
    elif why == "seed":
        inputs.seed_path("s", "a").parent.mkdir(parents=True)
        inputs.seed_path("s", "a").write_text("{}")
    else:
        (src / "work/s/a/blocks_px.geojson").unlink()
    root = tmp_path / "t" / "traced"
    with pytest.raises(ValueError, match="s/a") as e:
        trial.prepare(root, [("s", "a")], source=src)
    assert {"duplicate": "duplicate scan", "review": "review", "seed": "seed",
            "missing": "blocks_px.geojson"}[why] in str(e.value)
    assert not (root / "work").exists() and not root.exists()


def test_a_root_is_prepared_once(src, tmp_path):
    root = tmp_path / "t" / "traced"
    trial.prepare(root, [("s", "a")], source=src)
    (root / "work/s/a/blocks_px.geojson").write_text("the finder's")
    with pytest.raises(FileExistsError):
        trial.prepare(root, [("s", "a")], source=src)
    assert (root / "work/s/a/blocks_px.geojson").read_text() == "the finder's"


def test_numbers_are_read_from_the_root(src):
    got = trial.numbers(src, ("s", "a"))
    assert got == {"blocks": 3, "good_corners": 2, "control_points": 2, "passed": True, "reason": None,
                   "holdout_p50_ft": 5.01, "holdout_p95_ft": 12.28, "warp": "probabilistic", "within1": 0.627}
    (src / "warp/s/a/scores.json").write_text(json.dumps({"uncorrected": "nothing to fit",
                                                          "placement_check": {"passed": True, "reason": None}}))
    got = trial.numbers(src, ("s", "a"))
    assert got["warp"] == "uncorrected: nothing to fit" and got["within1"] is None


def test_reproduced_says_which_file_differs(src, tmp_path):
    other = tmp_path / "other"
    _fill(other)
    assert trial.reproduced(src, other, ("s", "a")) == {"transform": True, "refine": True, "score": True,
                                                        "warp_scores": True, "correction": True}
    t = other / "work/s/a/transform.json"
    t.write_text(json.dumps(_transform([(1.5, 2.5000001, True), (3, 4, False), (5, 6, True)])))
    s = other / "warp/s/a/scores.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "device": "cpu"}))
    assert trial.reproduced(src, other, ("s", "a")) == {"transform": False, "refine": True, "score": True,
                                                        "warp_scores": True, "correction": True}
    np.savez(other / "warp/s/a/correction.npz", Gc=np.arange(6, dtype=np.float32).reshape(2, 3) + 1e-6,
             spacing=np.int64(4), kind=np.str_("probabilistic"))
    assert trial.reproduced(src, other, ("s", "a"))["correction"] is False


def _footprint(root, series, sheet, geom):
    d = root / "warp" / series / sheet
    d.mkdir(parents=True, exist_ok=True)
    (d / "warped.footprint.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(geom)}]}))


def test_trained_ground_is_the_share_under_lot_labelled_sheets(tmp_path):
    source, run = tmp_path / "data", tmp_path / "run"
    _footprint(source, "s", "a", box(0, 0, 10, 10))
    _footprint(source, "s", "l", box(5, 0, 15, 10))
    _footprint(source, "s", "far", box(50, 50, 60, 60))
    run.mkdir()
    (run / "log.json").write_text(json.dumps({"items": [
        {"name": "s/l"}, {"name": "s/far"}, {"name": "train/s/a__0_0_64_64"}, {"name": "train/s/b__0_0_64_64"},
        {"name": "train/s/ab__0_0_64_64"}]}))
    got = trial.trained_ground(("s", "a"), run, source)
    assert got["lot_labelled_share"] == pytest.approx(0.5)
    assert got["hand_crops"] == ["train/s/a__0_0_64_64"]


def test_the_trial_commands_prepare_a_root_and_write_the_numbers(src, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", src)
    monkeypatch.setattr(trial, "TRIAL", tmp_path / "find" / "trial")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    r = invoke(["find", "trial", "root", "--trial", "T", "--arm", "traced", "--sheet", "s/a"])
    assert r.exit_code == 0, r.output
    root = trial.arm_root("T", "traced")
    assert root == tmp_path / "find" / "trial" / "T" / "traced"
    assert str(root / "work/s/a/blocks_px.geojson") in r.output.splitlines()
    again = invoke(["find", "trial", "root", "--trial", "T", "--arm", "traced", "--sheet", "s/a"])
    assert again.exit_code == 1 and "prepared once" in again.output
    bad = invoke(["find", "trial", "root", "--trial", "T", "--arm", "other", "--sheet", "s/a"])
    assert bad.exit_code != 0
    _fill(tmp_path / "copy")                          # the traced arm's placement, as `sfzmaps run` would leave it
    for x in ("work", "warp"):
        for f in (tmp_path / "copy" / x / "s" / "a").iterdir():
            (root / x / "s" / "a").mkdir(parents=True, exist_ok=True)
            (root / x / "s" / "a" / f.name).write_bytes(f.read_bytes())
    _footprint(src, "s", "a", box(0, 0, 10, 10))
    run = tmp_path / "find" / "runs" / "all-pieces"
    run.mkdir(parents=True)
    (run / "log.json").write_text(json.dumps({"items": [{"name": "train/s/a__0_0_64_64"}]}))
    r = invoke(["find", "trial", "numbers", "--trial", "T"])
    assert r.exit_code == 0, r.output
    rows = [ln for ln in r.output.splitlines() if ln.startswith("s/a ") and " blocks, " in ln]
    assert [ln.split()[1] for ln in rows] == ["today:", "traced:"]
    assert "traced as today: transform same" in r.output and "hand crops on it: train/s/a__0_0_64_64" in r.output
    assert "3 blocks" in rows[0] and "2 control points" in rows[0] and "probabilistic" in rows[0]
    saved = json.loads((tmp_path / "find" / "trial" / "T" / "numbers.json").read_text())
    got = saved["sheets"]["s/a"]
    assert got["today"] == got["traced"] == trial.numbers(src, ("s", "a"))
    assert all(got["reproduced"].values())
    assert got["trained_ground"] == {"lot_labelled_share": 0.0, "by_sheet": {}, "hand_crops": ["train/s/a__0_0_64_64"]}
