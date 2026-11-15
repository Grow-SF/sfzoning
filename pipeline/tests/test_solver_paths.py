import pytest

from sfzmaps import config
from sfzmaps.solver import paths


def test_solver_folders_sit_under_the_data_dir():
    assert paths.SOLVER_DIR == config.DATA_DIR / "solver"
    assert paths.PAGE_DIR == paths.SOLVER_DIR / "page"
    assert paths.text_dir("ne") == paths.SOLVER_DIR / "text-ne"


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


def test_the_stage_run_is_the_solve_folder_and_every_other_run_an_experiment():
    assert paths.solve_dir(paths.FINAL_RUN) == paths.SOLVE_DIR == paths.SOLVER_DIR / "solve"
    assert paths.solve_dir("j7-walls") == paths.SOLVER_DIR / "experiments" / "j7-walls"
    for name in ("", "a/b", ".solve.part", "../solve"):
        with pytest.raises(ValueError):
            paths.solve_dir(name)


def test_a_folder_built_afresh_replaces_the_old_one_whole(tmp_path):
    folder = tmp_path / "walls"
    folder.mkdir()
    (folder / "text_chance.png").write_text("from an earlier fit")
    (tmp_path / ".walls.part").mkdir()
    (tmp_path / ".walls.part" / "half.npz").write_text("from a fit that died")
    part = paths.fresh_folder(folder)
    assert part == tmp_path / ".walls.part" and not any(part.iterdir())
    (part / "walls_nw.npz").write_text("new")
    paths.replace_folder(part, folder)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["walls"]
    assert sorted(p.name for p in folder.iterdir()) == ["walls_nw.npz"]
    # the first time there is nothing to replace
    part = paths.fresh_folder(tmp_path / "solve")
    paths.replace_folder(part, tmp_path / "solve")
    assert (tmp_path / "solve").is_dir()


def test_a_default_wall_fit_leaves_no_text_chance_of_an_earlier_candidates_fit(tmp_path, monkeypatch):
    import numpy as np

    from sfzmaps.solver import walls
    monkeypatch.setattr(paths, "WALLS_DIR", tmp_path / "walls")
    paths.WALLS_DIR.mkdir()
    (paths.WALLS_DIR / "text_chance.png").write_bytes(b"chance read by text_weighted")
    # the fit itself stubbed: no text read (inputs.text None), one pair
    inputs = walls.Inputs(lots=None, mapped=None, strokes={}, log_area=None, geometries=None, letter=None, text=None,
                          ink=None)
    monkeypatch.setattr(walls, "load_inputs", lambda configs: inputs)
    monkeypatch.setattr(walls, "candidate_design", lambda inputs, candidates, workers: np.zeros((1, 1)))
    monkeypatch.setattr(walls, "fit_walls", lambda X, lots_: ({q: np.zeros(1) for q in paths.QUARTERS},
                                                             {"pairs": 1, "features": 1, "quarters": {}}))
    walls.build_walls(workers=1)
    assert sorted(p.name for p in paths.WALLS_DIR.iterdir()) == [
        "evaluation.json", "walls_ne.npz", "walls_nw.npz", "walls_se.npz", "walls_sw.npz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["walls"]
