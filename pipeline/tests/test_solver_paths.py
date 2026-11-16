import pytest

from sfzmaps import config
from sfzmaps.solver import paths, runs


def test_solver_folders_sit_under_the_data_dir():
    assert paths.SOLVER_DIR == config.DATA_DIR / "solver"
    assert paths.PAGE_DIR == paths.SOLVER_DIR / "page"
    assert paths.TEXT_DIR == paths.SOLVER_DIR / "text" and paths.TEXT_MODEL == paths.TEXT_DIR / "model.pt"
    assert paths.TEXT_CHANCE_DIR == paths.TEXT_DIR / "chance"
    assert paths.TEXT_SCORES_DIR == paths.SOLVER_DIR / "text-scores"
    assert runs.WALLS_DIR == paths.SOLVER_DIR / "walls" and runs.SOLVE_DIR == paths.SOLVER_DIR / "solve"


def test_sheet_names_are_the_warped_sheets_with_a_transform(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    root = tmp_path / "warp" / paths.SERIES
    (root / "p10").mkdir(parents=True)
    (root / "p10" / "transform.json").write_text("{}")
    (root / "p02").mkdir()
    (root / "p02" / "transform.json").write_text("{}")
    (root / "p05").mkdir()
    (root / "stray.txt").write_text("x")
    assert paths.sheet_names() == ["p02", "p10"]


def test_no_sheets_when_the_series_was_not_warped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert paths.sheet_names() == []


def _warp(root, names) -> None:
    for name in names:
        (root / "warp" / paths.SERIES / name).mkdir(parents=True, exist_ok=True)
        (root / "warp" / paths.SERIES / name / "transform.json").write_text("{}")


def test_the_fixed_sheet_list_is_the_sheets_the_warp_places_and_a_stage_reading_it_checks_so(tmp_path, monkeypatch):
    # the 13 pages of the 2000 zoning map, as the warp places them (13 folders, each with its transform.json)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    _warp(tmp_path, [f"{paths.SERIES}-p{i:02d}" for i in range(13, 0, -1)])
    assert paths.sheet_names() == list(paths.SHEETS)
    assert paths.check_sheets() == list(paths.SHEETS)
    # a sheet placed that the list doesn't name: refused, not silently left out of the detector's training
    _warp(tmp_path, [f"{paths.SERIES}-p14"])
    with pytest.raises(ValueError, match="p14"):
        paths.check_sheets()
    # a listed sheet no longer placed: refused too
    (tmp_path / "warp" / paths.SERIES / f"{paths.SERIES}-p14" / "transform.json").unlink()
    (tmp_path / "warp" / paths.SERIES / f"{paths.SERIES}-p07" / "transform.json").unlink()
    with pytest.raises(ValueError, match="p07"):
        paths.check_sheets()
    # nothing warped at all
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "empty")
    with pytest.raises(ValueError, match=f"run the warp-{paths.SERIES} stage"):
        paths.check_sheets()


def test_the_stage_run_is_the_solve_folder_and_every_other_run_an_experiment():
    assert runs.solve_dir(runs.FINAL_RUN) == runs.SOLVE_DIR == paths.SOLVER_DIR / "solve"
    assert runs.solve_dir("j7-walls") == paths.SOLVER_DIR / "experiments" / "j7-walls"
    for name in ("", "a/b", ".solve.part", "../solve"):
        with pytest.raises(ValueError):
            runs.solve_dir(name)


def test_a_solve_never_takes_the_folder_of_walls_fitted_as_an_experiment():
    walls = runs.walls_experiment_dir(["street_crossing", "context"])
    assert walls == runs.EXPERIMENTS_DIR / "walls-context+street_crossing"           # named by the sorted list
    assert runs.walls_experiment_dir(["context", "street_crossing"]) == walls
    assert runs.walls_experiment_dir([]) == runs.EXPERIMENTS_DIR / "walls-none"
    # a solve built afresh in that folder would delete the walls it reads
    with pytest.raises(ValueError, match="walls"):
        runs.solve_dir(walls.name)


def test_a_solve_never_takes_the_folder_of_a_text_detector_trial():
    from sfzmaps.solver.textnet import train, trial
    for folder in (trial.folder(train.Recipe()), trial.folder(train.Recipe(half_size=True), seed=1)):
        with pytest.raises(ValueError, match="text detector"):
            runs.solve_dir(folder.name)


def test_a_solve_never_takes_the_text_detectors_cache_of_long_line_ink():
    from sfzmaps.solver.textnet import trial
    with pytest.raises(ValueError, match="cache"):
        runs.solve_dir(trial.line_masks_dir().name)


def test_a_folder_built_afresh_replaces_the_old_one_whole(tmp_path):
    folder = tmp_path / "walls"
    folder.mkdir()
    (folder / "text_chance.png").write_text("from an earlier fit")
    (tmp_path / ".walls.part").mkdir()
    (tmp_path / ".walls.part" / "half.npz").write_text("from a fit that died")
    part = runs.fresh_folder(folder)
    assert part == tmp_path / ".walls.part" and not any(part.iterdir())
    (part / "walls_nw.npz").write_text("new")
    runs.replace_folder(part, folder)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["walls"]
    assert sorted(p.name for p in folder.iterdir()) == ["walls_nw.npz"]
    # the first time there is nothing to replace
    part = runs.fresh_folder(tmp_path / "solve")
    runs.replace_folder(part, tmp_path / "solve")
    assert (tmp_path / "solve").is_dir()


def test_a_default_wall_fit_leaves_no_text_chance_of_an_earlier_candidates_fit(tmp_path, monkeypatch):
    import numpy as np

    from sfzmaps.solver import walls
    monkeypatch.setattr(runs, "WALLS_DIR", tmp_path / "walls")
    runs.WALLS_DIR.mkdir()
    (runs.WALLS_DIR / "text_chance.png").write_bytes(b"chance read by text_weighted")
    # the fit itself stubbed: no text read (inputs.text None), one pair
    inputs = walls.Inputs(lots=None, mapped=None, strokes={}, log_area=None, geometries=None, letter=None, text=None,
                          ink=None)
    monkeypatch.setattr(walls, "load_inputs", lambda configs: inputs)
    monkeypatch.setattr(walls, "candidate_design", lambda inputs, candidates, workers: np.zeros((1, 1)))
    monkeypatch.setattr(walls, "fit_walls", lambda X, lots_: ({q: np.zeros(1) for q in paths.QUARTERS},
                                                             {"pairs": 1, "features": 1, "quarters": {}}))
    walls.build_walls(workers=1)
    assert sorted(p.name for p in runs.WALLS_DIR.iterdir()) == [
        "evaluation.json", "walls_ne.npz", "walls_nw.npz", "walls_se.npz", "walls_sw.npz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["walls"]
