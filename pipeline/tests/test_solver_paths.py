from sfzmaps import config
from sfzmaps.solver import paths


def test_solver_folders_sit_under_the_data_dir():
    assert paths.SOLVER_DIR == config.DATA_DIR / "solver"
    assert paths.PAGE_DIR == paths.SOLVER_DIR / "page"
    assert paths.text_dir("ne") == paths.SOLVER_DIR / "text-ne"


def test_sheet_names_are_the_placed_sheets_of_the_series(tmp_path, monkeypatch):
    from sfzmaps.sheetwarp import sheet
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    d = tmp_path / paths.SERIES / "b"
    d.mkdir(parents=True)
    for f in sheet.PLACED_FILES:
        (d / f).write_text("x")
    (tmp_path / paths.SERIES / "a").mkdir()
    assert paths.sheet_names() == ["b"]
