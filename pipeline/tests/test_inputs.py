import json

from sfzmaps import config, inputs, reading


def test_paths_live_under_the_series_folder():
    i = config.INPUTS_DIR
    assert inputs.review_path("1948", "6a") == i / "1948" / "reviews" / "6a.json"
    assert inputs.seed_path("1921", "s8") == i / "1921" / "seeds" / "s8.json"
    assert inputs.page_patch_path("1990-1996-zoning", "p05") == i / "1990-1996-zoning" / "pages" / "p05.json"
    assert inputs.overrides_path("1975") == i / "1975" / "overrides.csv"
    assert inputs.label_scale_path() == i / "label_scale.json" == config._scale_file()


def _page(tmp_path, series="1990-1996-zoning", sheet="p05"):
    pj = tmp_path / "work" / series / sheet / "page.json"
    pj.parent.mkdir(parents=True)
    pj.write_text(json.dumps({"kind": "map", "sheet_number": "5", "neighbor_refs": []}))
    return pj


def test_page_patch_replaces_top_level_fields(tmp_path):
    pj = _page(tmp_path)
    p = inputs.page_patch_path("1990-1996-zoning", "p05")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"kind": "text", "sheet_number": None}))
    info = reading.load_page(pj)
    assert info.kind == "text" and info.sheet_number is None and info.neighbor_refs == []
    assert inputs.page_dict(pj)["kind"] == "text"
    assert json.loads(pj.read_text())["kind"] == "map"     # the machine file is never rewritten


def test_page_without_patch_is_the_machine_page(tmp_path):
    pj = _page(tmp_path)
    assert reading.load_page(pj).kind == "map" and inputs.page_dict(pj)["sheet_number"] == "5"


def test_zoning_layer_reads_the_patched_sheet_number(tmp_path, monkeypatch):
    from sfzmaps import stitch
    from sfzmaps.districts.series import SERIES
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    pj = _page(tmp_path, "1980-1990-zoning", "a")
    _page(tmp_path, "1980-1990-zoning", "b")
    p = inputs.page_patch_path("1980-1990-zoning", "a")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"sheet_number": "13SH"}))       # a height sheet, by its patch
    assert SERIES["1980-1990-zoning"].role == "use" and pj.exists()
    assert stitch.layer_sheets("1980-1990-zoning", ["a", "b"]) == ["b"]
