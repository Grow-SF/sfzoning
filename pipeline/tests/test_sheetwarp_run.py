import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from sfzmaps import build, cli, config, fresh
from sfzmaps.sheetwarp import apply, run, scores
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
    monkeypatch.setattr(run, "numeric_versions", lambda: {"torch": "2.15.0", "gdal": "GDAL 3.13.3"}, raising=False)
    return folder


def _correction():
    return Correction(grid=np.ones((2, 3, 3)), spacing=50.0, origin=np.array([0.0, 0.0]))


def _fake_fit(calls, kind="first"):
    def fit(sheet):
        calls.append(sheet.name)
        return _correction(), kind, {"within1_first": 0.5}
    return fit


def _fake_warp(sheet, correction, out_dir):
    for f in ("warped.tif", "warped.footprint.geojson", "warped_paper.tif", "warped_paper.footprint.geojson"):
        (out_dir / f).write_bytes(b"warp")


def _nothing_to_fit(sheet):
    raise NothingToFit("no block sides on the page")


def test_choose_takes_the_probabilistic_correction_only_when_it_fits_as_well_and_never_folds():
    base = {"within1_first": 0.8, "within1_probabilistic": 0.8, "smallest_cell_probabilistic": 0.4}
    assert scores.choose(base) == "probabilistic"
    assert scores.choose({**base, "within1_probabilistic": 0.9}) == "probabilistic"
    assert scores.choose({**base, "within1_probabilistic": 0.79}) == "first"
    assert scores.choose({**base, "smallest_cell_probabilistic": 0.0}) == "first"
    assert scores.choose({**base, "smallest_cell_probabilistic": -0.2, "within1_probabilistic": 1.0}) == "first"


def _constant(ux, uy, interpolation, shape=(20, 30)):
    """the resampling by a correction of (ux, uy) px everywhere on a page of `shape` (the grid reaches past it)"""
    grid = np.empty((2, 4, 4), np.float32)
    grid[0], grid[1] = ux, uy
    return apply._resampled_by(Correction(grid=grid, spacing=50.0, origin=np.array([-50.0, -50.0])), interpolation)


def _page(shape=(20, 30)):
    return np.random.default_rng(0).integers(0, 256, (*shape, 4), dtype=np.uint8)


def test_resampling_by_a_correction_takes_each_pixel_from_where_the_correction_points():
    # the corrected placement puts a map point where today's puts it plus u, so output pixel p holds page pixel p + u
    page = _page()
    out = _constant(3.0, -2.0, cv2.INTER_LINEAR)(page)
    assert np.array_equal(out[2:, :-3], page[:-2, 3:])
    assert not out[:2].any() and not out[:, -3:].any()       # from off the page: transparent


@pytest.mark.parametrize("interpolation", [cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_resampling_by_a_zero_correction_copies_the_page_exactly(interpolation):
    page = _page()
    assert np.array_equal(_constant(0.0, 0.0, interpolation)(page), page)


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


def test_a_sheet_interrupted_after_a_finished_run_is_redone(world, monkeypatch):
    calls, warps = [], []
    monkeypatch.setattr(run, "fit_sheet", _fake_fit(calls))

    def warp(sheet, correction, out_dir):
        warps.append(1)
        if len(warps) == 2:
            raise RuntimeError("interrupted")
        _fake_warp(sheet, correction, out_dir)
    monkeypatch.setattr(run.apply, "warp_with_correction", warp)
    out = run.OUT / "s" / "a"
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert (out / "transform.json").exists()
    with pytest.raises(RuntimeError):
        run.correct_sheet("s", "a", force=True)
    assert not (out / "transform.json").exists()
    assert not fresh.sidecar(out / "transform.json").exists()
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert calls == ["a", "a", "a"]


def test_a_series_finishes_every_sheet_then_raises_naming_the_ones_that_failed(world, monkeypatch, caplog):
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
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="s/a"):
            run.correct_series("s", jobs=1)
    assert "Traceback" in caplog.text and "ValueError: boom" in caplog.text and "in correct" in caplog.text
    assert done == ["a", "b"]


def test_warp_rejects_an_unknown_series():
    res = CliRunner().invoke(cli.app, ["warp", "NOPE"])
    assert res.exit_code != 0
    assert "NOPE" in res.output and "1921" in res.output


def test_warp_checks_that_its_tools_come_from_pixi(monkeypatch):
    monkeypatch.delenv("SFZMAPS_SYSTEM_TOOLS", raising=False)
    checked, corrected = [], []
    monkeypatch.setattr(build, "require_toolchain", checked.append)
    monkeypatch.setattr(run, "correct_series", lambda series, jobs, force: corrected.append(series) or {})
    runner = CliRunner()
    assert runner.invoke(cli.app, ["warp", "1921"]).exit_code == 0
    assert runner.invoke(cli.app, ["warp", "1921", "--system-tools"]).exit_code == 0
    # how `sfzmaps build --system-tools` passes it on to the warp stages
    assert build.SYSTEM_TOOLS_ENV == cli.SYSTEM_TOOLS_ENV
    assert runner.invoke(cli.app, ["warp", "1921"], env={build.SYSTEM_TOOLS_ENV: "1"}).exit_code == 0
    assert runner.invoke(cli.app, ["warp", "1921"], env={build.SYSTEM_TOOLS_ENV: "0"}).exit_code == 0
    assert checked == [False, True, True, False]
    assert corrected == ["1921"] * 4

    def refuse(system_tools):
        raise SystemExit("these tools aren't from the pixi environment")
    monkeypatch.setattr(build, "require_toolchain", refuse)
    assert runner.invoke(cli.app, ["warp", "1921"]).exit_code != 0
    assert corrected == ["1921"] * 4


def test_the_code_the_warp_runs_is_pinned():
    """Every module whose code the warp runs, and no other: a change to one redoes every sheet, so adding one (an
    import in any of them) is a deliberate change here."""
    root = Path(run.__file__).resolve().parents[1]
    assert [p.relative_to(root).as_posix() for p in run.warp_code_files()] == [
        "__init__.py", "config.py", "districts/__init__.py", "districts/lots.py", "districts/read.py",
        "districts/series.py", "inputs.py", "linework.py", "pages.py", "reading.py",
        "sheetwarp/__init__.py", "sheetwarp/apply.py", "sheetwarp/device.py", "sheetwarp/first.py",
        "sheetwarp/probabilistic.py",
        "sheetwarp/run.py", "sheetwarp/scores.py", "sheetwarp/sheet.py",
        "sheetwarp/street_edges_and_later_blocks.py", "sheetwarp/targets.py", "transform.py", "warp.py"]


def test_the_versions_that_change_the_numbers_are_pinned():
    assert set(run.numeric_versions()) == {"torch", "numpy", "scipy", "opencv-python-headless", "shapely",
                                           "scikit-image", "gdal"}


def test_a_new_version_of_a_library_the_fits_compute_with_redoes_the_sheet(world, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "fit_sheet", _fake_fit(calls))
    monkeypatch.setattr(run.apply, "warp_with_correction", _fake_warp)
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert run.correct_sheet("s", "a") == "skipped"
    monkeypatch.setattr(run, "numeric_versions", lambda: {"torch": "2.16.0", "gdal": "GDAL 3.13.3"})
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert calls == ["a", "a"]


def test_the_paper_warp_and_the_placement_check_are_inputs(world, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "fit_sheet", _fake_fit(calls))
    monkeypatch.setattr(run.apply, "warp_with_correction", _fake_warp)
    assert run.correct_sheet("s", "a") == "corrected (first)"
    for f in ("warped_paper.tif", "warped_paper.footprint.geojson"):     # copied as they are when uncorrected
        (world / f).write_bytes(b"changed")
        assert run.correct_sheet("s", "a") == "corrected (first)"
    (world / "score.json").write_text(json.dumps({"passed": True, "reason": None}))
    assert run.correct_sheet("s", "a") == "corrected (first)"
    (world / "score.json").unlink()                     # only when there is one
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert run.correct_sheet("s", "a") == "skipped"
    assert len(calls) == 5


@pytest.mark.parametrize("missing", ["correction.npz", "scores.json", "warped.tif", "warped.footprint.geojson",
                                     "warped_paper.tif", "warped_paper.footprint.geojson"])
def test_a_corrected_sheet_missing_an_output_is_redone(world, monkeypatch, missing):
    calls = []
    monkeypatch.setattr(run, "fit_sheet", _fake_fit(calls))
    monkeypatch.setattr(run.apply, "warp_with_correction", _fake_warp)
    assert run.correct_sheet("s", "a") == "corrected (first)"
    (run.OUT / "s" / "a" / missing).unlink()
    assert run.correct_sheet("s", "a") == "corrected (first)"
    assert calls == ["a", "a"]


@pytest.mark.parametrize("missing", ["scores.json", "warped.tif", "warped.footprint.geojson", "warped_paper.tif",
                                     "warped_paper.footprint.geojson"])
def test_an_uncorrected_sheet_missing_an_output_is_redone(world, monkeypatch, missing):
    monkeypatch.setattr(run, "fit_sheet", _nothing_to_fit)
    assert run.correct_sheet("s", "a").startswith("uncorrected")
    assert run.correct_sheet("s", "a") == "skipped"      # complete without a correction.npz
    (run.OUT / "s" / "a" / missing).unlink()
    assert run.correct_sheet("s", "a").startswith("uncorrected")


def test_a_correction_that_is_not_finite_leaves_the_sheet_uncorrected(world, monkeypatch):
    c = _correction()
    c.grid[1, 1, 1] = np.nan
    monkeypatch.setattr(run, "fit_sheet", lambda sheet: (c, "probabilistic", {"within1_first": 0.5}))
    result = run.correct_sheet("s", "a")
    out = run.OUT / "s" / "a"
    assert result.startswith("uncorrected: ") and "probabilistic" in result
    assert "correction" not in json.loads((out / "transform.json").read_text())
    assert not (out / "correction.npz").exists()
    assert json.loads((out / "scores.json").read_text())["uncorrected"] == result.removeprefix("uncorrected: ")
    for f in ("warped.tif", "warped_paper.tif", "warped.footprint.geojson", "warped_paper.footprint.geojson"):
        assert (out / f).read_bytes() == (world / f).read_bytes()
