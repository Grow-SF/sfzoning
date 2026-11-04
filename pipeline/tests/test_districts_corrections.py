import json

import pandas as pd
import pytest

from sfzmaps import config
from sfzmaps.districts import assemble, check, corrections as corr

BODY = """Corrections suggested on sfzoning.com: what the old map prints for these lots.

| Era | Block/lot | Site shows | Map prints | Note |
|---|---|---|---|---|
| 1980 | 2842/007 | RH-1 | P | Laguna Honda |

<!-- Read by the pipeline: please leave this block as it is. -->
```sfzoning-corrections
""" + json.dumps([
    {"era": "1980", "series": "1980", "blklot": "2842007", "shown": "RH-1", "proposed": "P", "note": "Laguna Honda"},
    {"era": "2000", "series": "2000-digital-zoning", "blklot": "1313023", "shown": "RH-2", "proposed": "p", "note": ""},
    {"era": "today", "series": "today", "blklot": "2842007", "shown": "P", "proposed": "RH-1", "note": ""},
    {"era": "1980", "series": "1980", "blklot": "0000000", "shown": "RH-1", "proposed": "P", "note": ""},
    {"era": "1980", "series": "1980", "blklot": "2842007", "shown": "RH-1", "proposed": "ZZ-9", "note": ""},
], indent=1) + "\n```\n"

PARCELS = pd.DataFrame({"blklot": ["2842007", "1313023"]})


def test_the_site_issue_block_parses_and_bad_rows_are_explained():
    rows = corr.parse(BODY)
    assert len(rows) == 5
    got = corr.check(rows, PARCELS)
    assert [(r["series"], r["blklot"], r["proposed"]) for r in got.ok] == [
        ("1980", "2842007", "P"), ("2000-digital-zoning", "1313023", "P")]   # code normalised
    assert len(got.problems) == 3
    assert any("today" in p for p in got.problems) and any("0000000" in p for p in got.problems)
    assert any("ZZ-9" in p for p in got.problems)
    assert corr.parse("no block here") == []


def test_a_correction_to_one_scan_of_a_drawing_applies_to_its_twin():
    rows = corr.with_twins([{"series": "2000-digital-zoning", "blklot": "1313023", "shown": "RH-2",
                             "proposed": "P", "note": ""}])
    assert {(r["series"], r["shown"]) for r in rows} == {("2000-digital-zoning", "RH-2"), ("1996-2000-zoning", "")}


def test_apply_writes_overrides_with_a_note_the_lot_card_shows(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    (tmp_path / "inputs" / "1980").mkdir(parents=True)
    (tmp_path / "inputs" / "1980" / "overrides.csv").write_text("blklot,block,code\n2842007,,RH-1\n1234567,,P\n")
    rows = [{"series": "1980", "blklot": "2842007", "shown": "RH-1", "proposed": "P", "note": ""}]
    assert corr.write_overrides(rows, 99) == {"1980": 1}
    ov = pd.read_csv(tmp_path / "inputs" / "1980" / "overrides.csv", dtype=str, keep_default_na=False)
    assert ov.set_index("blklot").loc["2842007", "code"] == "P"            # replaced, not duplicated
    assert len(ov) == 2 and "issue #99" in ov.set_index("blklot").loc["2842007", "note"]
    df = pd.DataFrame({"blklot": ["2842007", "1234567"], "code": ["RH-1", "RH-2"], "confidence": [0.5, 0.5]})
    out = assemble.apply_overrides(df, "1980", blocks=pd.Series(["2842", "1234"]))
    out["note"] = [None, "Park since 1900"]
    out = assemble.add_override_notes(out).set_index("blklot")
    assert out.loc["2842007", "code"] == "P" and "issue #99" in out.loc["2842007", "note"]
    assert out.loc["1234567", "note"] == "Park since 1900"                  # no note on the row: kept
    assert "override_note" not in out


def test_a_drawn_area_is_checked_and_written_for_publishing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    ring = [[-122.4, 37.8], [-122.39, 37.8], [-122.39, 37.81]]
    rows = [{"era": "1980", "series": "1980", "blklot": "area-x", "shown": "", "proposed": "rh-1", "note": "line through 0028016", "area": ring},
            {"era": "1980", "series": "1980", "blklot": "area-y", "shown": "", "proposed": "RH-1", "note": "", "area": [[1, 2]]},
            {"era": "1980", "series": "1980", "blklot": "2842007~in", "shown": "C-2", "proposed": "P", "note": ""}]
    got = corr.check(rows, PARCELS)
    assert [r.get("area") is not None for r in got.ok] == [True, False]
    assert got.ok[1]["blklot"] == "2842007"                      # a split piece names its lot
    assert any("3+" in p for p in got.problems)
    assert corr.write_overrides(got.ok, 5) == {"1980": 1}         # the lot row only
    assert corr.write_areas(got.ok, 5) == {"1980": 1}
    assert corr.write_areas(got.ok, 5) == {"1980": 1}             # re-run: replaced, not doubled
    fc = json.loads((tmp_path / "inputs" / "1980" / "area_overrides.geojson").read_text())
    assert len(fc["features"]) == 1
    f = fc["features"][0]
    assert f["properties"]["code"] == "RH-1" and "issue #5" in f["properties"]["note"]
    assert f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][0][-1]   # closed


def test_apply_adds_a_review_case_per_code_change(tmp_path, monkeypatch):
    cases = tmp_path / "district_cases.csv"
    cases.write_text("id,series,x0,y0,x1,y1,wrong,right,kind,note\nold,1980,0,0,1,1,R-1,R-2,seam,\n")
    monkeypatch.setattr(check, "CASES_CSV", cases)
    cent = pd.DataFrame({"blklot": ["a", "b", "c"], "x": [100.0, 200.0, 900.0], "y": [100.0, 150.0, 900.0]})
    rows = [{"series": "1980", "blklot": "a", "shown": "RH-1", "proposed": "P", "note": ""},
            {"series": "1980", "blklot": "b", "shown": "RH-1", "proposed": "P", "note": ""},
            {"series": "1980", "blklot": "c", "shown": "", "proposed": "P", "note": ""}]     # unread: no case
    assert corr.write_cases(rows, 7, cent) == 1
    assert corr.write_cases(rows, 7, cent) == 1                              # re-run replaces, not appends
    got = pd.read_csv(cases, dtype=str, keep_default_na=False).set_index("id")
    assert len(got) == 2
    new = got.loc[[i for i in got.index if i.startswith("issue7-")][0]]
    assert (new.wrong, new.right, int(new.x0), int(new.x1)) == ("RH-1", "P", 85, 215)
