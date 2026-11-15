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
