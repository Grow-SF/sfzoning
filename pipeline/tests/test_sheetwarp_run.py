import json

import numpy as np
import pytest
from typer.testing import CliRunner

from sfzmaps import cli, config
from sfzmaps.sheetwarp import run, scores
from sfzmaps.sheetwarp.sheet import PLACED_FILES, NothingToFit
from sfzmaps.transform import Correction, SheetTransform

GCPS = [{"px": x, "py": y, "wx": 100.0 * x, "wy": -100.0 * y, "source": "manual", "enabled": True}
        for x, y in ((0, 0), (10, 0), (0, 10), (10, 10))]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """One placed sheet, "s"/"a", in a temporary data folder."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "REF_DIR", tmp_path / "ref")
    monkeypatch.setattr(run, "OUT", tmp_path / "warp")
    config.REF_DIR.mkdir()
    for f in ("parcels.parquet", "streets.parquet"):
        (config.REF_DIR / f).write_bytes(b"ref")
    folder = config.WORK_DIR / "s" / "a"
    folder.mkdir(parents=True)
    for f in PLACED_FILES:
        (folder / f).write_bytes(b"x")
    (folder / "transform.json").write_text(json.dumps({"smoothing": 1.0, "gcps": GCPS}))
    (folder / "score.json").write_text(json.dumps({"passed": False, "reason": "misplaced"}))
    for f in ("warped_paper.tif", "warped_paper.footprint.geojson"):
        (folder / f).write_bytes(b"paper")
    return folder


def _correction():
    return Correction(grid=np.ones((2, 3, 3)), spacing=50.0, origin=np.array([0.0, 0.0]))


def _fake_fit(calls, kind="first"):
    def fit(sheet):
        calls.append(sheet.name)
        return _correction(), kind, {"within1_first": 0.5}
    return fit


def _fake_warp(sheet, correction, out_dir):
    for f in ("warped.tif", "warped_paper.tif"):
        (out_dir / f).write_bytes(b"warp")


def test_choose_takes_the_probabilistic_correction_only_when_it_fits_as_well_and_never_folds():
    base = {"within1_first": 0.8, "within1_probabilistic": 0.8, "smallest_cell_probabilistic": 0.4}
    assert scores.choose(base) == "probabilistic"
    assert scores.choose({**base, "within1_probabilistic": 0.9}) == "probabilistic"
    assert scores.choose({**base, "within1_probabilistic": 0.79}) == "first"
    assert scores.choose({**base, "smallest_cell_probabilistic": 0.0}) == "first"
    assert scores.choose({**base, "smallest_cell_probabilistic": -0.2, "within1_probabilistic": 1.0}) == "first"


def test_a_sheet_with_nothing_to_fit_is_written_uncorrected(world, monkeypatch):
    def nothing(sheet):
        raise NothingToFit("no block sides on the page")
    monkeypatch.setattr(run, "fit_sheet", nothing)
    result = run.correct_sheet("s", "a")
    out = run.OUT / "s" / "a"
    assert result == "uncorrected: no block sides on the page"
    assert "correction" not in json.loads((out / "transform.json").read_text())
    assert not (out / "correction.npz").exists()
    assert json.loads((out / "scores.json").read_text())["uncorrected"] == "no block sides on the page"
    for f in ("warped.tif", "warped_paper.tif", "warped.footprint.geojson", "warped_paper.footprint.geojson"):
        assert (out / f).read_bytes() == (world / f).read_bytes()


def test_a_sheet_that_raises_anything_else_is_not_written(world, monkeypatch):
    def broken(sheet):
        raise ValueError("boom")
    monkeypatch.setattr(run, "fit_sheet", broken)
    with pytest.raises(ValueError):
        run.correct_sheet("s", "a")
    assert not (run.OUT / "s" / "a" / "transform.json").exists()


def test_a_second_run_on_unchanged_inputs_fits_nothing(world, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "fit_sheet", _fake_fit(calls))
    monkeypatch.setattr(run.apply, "warp_with_correction", _fake_warp)
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert run.correct_sheet("s", "a") == "skipped"
    assert calls == ["a"]
    assert run.correct_sheet("s", "a", force=True) == "corrected (first)"
    assert calls == ["a", "a"]
    (world / "refine.json").write_bytes(b"changed")
    assert run.correct_sheet("s", "a") == "corrected (first)"


def test_the_correction_is_named_relative_to_its_folder(world, monkeypatch):
    monkeypatch.setattr(run, "fit_sheet", _fake_fit([], "street edges and later blocks"))
    monkeypatch.setattr(run.apply, "warp_with_correction", _fake_warp)
    run.correct_sheet("s", "a")
    out = run.OUT / "s" / "a"
    d = json.loads((out / "transform.json").read_text())
    assert d["correction"] == {"file": "correction.npz"}
    t = SheetTransform.from_json(d, folder=out)
    assert np.allclose(t.correction.grid, 1.0)
    assert str(np.load(out / "correction.npz")["kind"]) == "street edges and later blocks"
    s = json.loads((out / "scores.json").read_text())
    assert s["chosen"] == "street edges and later blocks" and s["within1_first"] == 0.5
    assert s["placement_check"] == {"passed": False, "reason": "misplaced"}


def test_a_sheet_interrupted_before_its_placement_is_written_is_redone(world, monkeypatch):
    calls, warps = [], []
    monkeypatch.setattr(run, "fit_sheet", _fake_fit(calls))

    def warp(sheet, correction, out_dir):
        warps.append(1)
        if len(warps) == 1:
            raise RuntimeError("interrupted")
        _fake_warp(sheet, correction, out_dir)
    monkeypatch.setattr(run.apply, "warp_with_correction", warp)
    with pytest.raises(RuntimeError):
        run.correct_sheet("s", "a")
    assert not (run.OUT / "s" / "a" / "transform.json").exists()
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert calls == ["a", "a"]


def test_a_series_finishes_every_sheet_then_raises_naming_the_ones_that_failed(world, monkeypatch):
    other = config.WORK_DIR / "s" / "b"
    other.mkdir()
    for f in PLACED_FILES:
        (other / f).write_bytes(b"x")
    done = []

    def correct(series, name, force=False):
        done.append(name)
        if name == "a":
            raise ValueError("boom")
        return "corrected (first)"
    monkeypatch.setattr(run, "correct_sheet", correct)
    with pytest.raises(RuntimeError, match="s/a"):
        run.correct_series("s", jobs=1)
    assert done == ["a", "b"]


def test_warp_rejects_an_unknown_series():
    res = CliRunner().invoke(cli.app, ["warp", "NOPE"])
    assert res.exit_code != 0
    assert "NOPE" in res.output and "1921" in res.output
