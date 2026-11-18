"""Maps published from their series' solve (districts/solved.py, the solver-publish stages), 2000's first."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box as _box

from sfzmaps import config, inputs
from sfzmaps.districts import assemble, solved
from sfzmaps.districts.categories import category_for
from sfzmaps.solver import paths, runs

X0, Y0 = 6006000, 2110000  # San Francisco, EPSG:2227
S = "2000-digital-zoning"


def box(a, b, c, d):
    return _box(a + X0, b + Y0, c + X0, d + Y0)


def codes_of(lots, table, codes):
    return solved.majority_codes([box(*b) if b else None for b in lots], [box(*b) for b in table], codes)


# --- the lots -------------------------------------------------------------------------------

def test_a_lot_takes_the_code_covering_most_of_its_area():
    code, how = codes_of([(0, 0, 100, 100)], [(0, 0, 60, 100), (60, 0, 100, 100)], ["RH-2", "RH-1"])
    assert code.tolist() == ["RH-2"] and how.tolist() == ["area"]


def test_the_overlaps_of_one_code_add_up():
    # B is the largest single table lot (40%), but A's two lots cover 60%
    code, _ = codes_of([(0, 0, 100, 100)], [(0, 0, 30, 100), (30, 0, 60, 100), (60, 0, 100, 100)],
                       ["RM-1", "RM-1", "NC-2"])
    assert code.tolist() == ["RM-1"]


def test_a_lot_no_table_lot_overlaps_takes_the_nearest_ones_code():
    code, how = codes_of([(0, 0, 10, 10)], [(30, 0, 40, 10), (12, 0, 20, 10)], ["M-1", "P"])
    assert code.tolist() == ["P"] and how.tolist() == ["nearest"]


def test_table_lots_the_solve_left_blank_take_no_part():
    code, how = codes_of([(0, 0, 100, 100), (300, 0, 310, 10)], [(0, 0, 90, 100), (90, 0, 100, 100), (250, 0, 305, 10)],
                         ["", "RH-3", ""])
    assert code.tolist() == ["RH-3", "RH-3"] and how.tolist() == ["area", "nearest"]


def test_a_lot_with_no_shape_gets_no_code():
    code, how = codes_of([None, (0, 0, 10, 10)], [(0, 0, 10, 10)], ["RH-1"])
    assert code.tolist() == ["", "RH-1"] and how.tolist() == ["", "area"]


# --- the spellings --------------------------------------------------------------------------

@pytest.mark.parametrize("name, published", [
    ("RH-1(D)", "RH-1(D)"), ("rh1", "RH-1"), ("NC2", "NC-2"), ("C-3-O(SD)", "C-3-O(SD)"),
    ("UPR MARKET", "UPPER MARKET NCD"), ("NO BEACH NCD", "NORTH BEACH NCD"), ("HAIGHT ST", "HAIGHT NCD"),
    ("24TH-NOE", "24TH-NOE VALLEY NCD"), ("WEST PORTAL", "WEST PORTAL AVENUE NCD"), ("", None)])
def test_a_solved_code_is_published_in_the_one_spelling_with_ncd_on_a_named_district(name, published):
    assert solved.published_code(name, 2000) == published


def test_every_named_district_has_a_category_and_a_code_without_one_stops_the_publish():
    assert all(category_for(solved.published_code(n, 2000), 2000) != "unknown" for n in solved.NAMED_NCDS)
    assert category_for(solved.published_code("BROADWAY", 2000), 2000) == "homes_mid"
    with pytest.raises(ValueError, match="M-2 \\(MB\\)"):
        solved.published_code("M-2 (MB)", 2000)


# --- the published map ----------------------------------------------------------------------

PARCELS = {"1000001": (0, 0, 100, 100), "1000002": (100, 0, 200, 100), "1001001": (0, 200, 50, 300),
           "1001002": (50, 200, 100, 300), "1002001": (300, 0, 400, 50)}


def _solve(names, pred, table, scored=None, wrong=1):
    """The solver's files in their places: the table lots (blklot, box) and the solve's names and pred."""
    paths.LOTS_DIR.mkdir(parents=True, exist_ok=True)
    (runs.SOLVE_DIR / "runs").mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"blklot": [b for b, _ in table], "zoning": [""] * len(table)},
                     geometry=[box(*g) for _, g in table], crs="EPSG:2227").to_parquet(solved.table_file())
    np.savez_compressed(solved.run_file(), names=np.array(names), pred=np.array(pred))
    np.savez_compressed(solved.lots_file(), scored=np.array(scored or [True] * len(table)))
    solved.report_file().write_text(json.dumps({"measures": {"every_lot": {"wrong_lots": wrong}}}))


@pytest.fixture
def city(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "OUT_DIR", "WORK_DIR", "REF_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(runs, "SOLVE_DIR", tmp_path / "solver" / "solve")
    monkeypatch.setattr(paths, "LOTS_DIR", tmp_path / "solver" / "lots")
    config.DATA_DIR.mkdir()
    parcels = gpd.GeoDataFrame({"blklot": list(PARCELS), "block_num": [b[:4] for b in PARCELS], "active": True},
                               geometry=[box(*g) for g in PARCELS.values()], crs="EPSG:2227")
    monkeypatch.setattr(assemble, "load_parcels", lambda *a, **k: parcels.copy())
    # the read: every lot, with the reader's own codes, a sheet each and no heights
    raw = pd.DataFrame({"blklot": list(PARCELS), "code": pd.Series(["RH-2"] * 5, dtype=object),
                        "height_code": pd.Series([None] * 5, dtype=object), "confidence": 0.5,
                        "source_sheet": ["p01", "p01", "p02", "p02", "p03"]})
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: raw.copy())
    # the table: 1000001 under RH-1 and 40% of 1000002; the rest of 1000002 under UPPER MARKET; block 1001 under
    # NC-2 (two rows sharing one blklot, like the table's 45); 1002001 under no table lot, nearest UPPER MARKET
    _solve(["RH-1", "UPPER MARKET", "NC-2", "P"], [0, 1, 2, 2, 3],
           [("1000001", (0, 0, 140, 100)), ("1000002", (140, 0, 200, 100)), ("1001001", (0, 200, 50, 300)),
            ("1001001", (50, 200, 100, 300)), ("9999001", (1000, 1000, 1100, 1100))], wrong=1)
    return parcels


def _published():
    return gpd.read_file(assemble.out_dir(S) / "parcels.fgb").set_index("blklot")


def test_reading_the_2000_map_leaves_its_publish_to_the_solve(city):
    assemble.run_series("2000-digital-height")
    assemble.run_series(S)
    assert assemble._raw_path(S).exists() and (assemble.out_dir(S) / "height.parquet").exists()
    assert not any((assemble.out_dir(S) / n).exists() for n in solved.PUBLISHED)


def test_the_2000_map_is_published_from_the_solve_with_its_other_steps(city, monkeypatch):
    inputs.overrides_path(S).parent.mkdir(parents=True, exist_ok=True)
    inputs.overrides_path(S).write_text("blklot,block,code\n1001002,,P\n")
    heights = pd.DataFrame({"blklot": ["1000001"], "height_code": ["40-X"], "confidence": 0.9, "source_sheet": "h1"})
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: heights.copy() if "height" in s else
                        pd.DataFrame({"blklot": list(PARCELS), "code": None, "height_code": None, "confidence": 0.5,
                                      "source_sheet": ["p01", "p01", "p02", "p02", "p03"]}))
    assemble.run_series("2000-digital-height")
    assemble.run_series(S)
    d = solved.publish()
    g = _published()
    assert g["code"].to_dict() == {"1000001": "RH-1", "1000002": "UPPER MARKET NCD", "1001001": "NC-2",
                                   "1001002": "P", "1002001": "UPPER MARKET NCD"}
    assert g.loc["1000002", "category"] == "homes_low" and g.loc["1001002", "confidence"] == 1.0
    assert g.loc["1000001", "confidence"] == 0.8                  # 1 wrong of 5 scored
    assert g.loc["1000001", "height_code"] == "40-X" and g.loc["1002001", "source_sheet"] == "p03"
    assert json.loads((d / "unknown_codes.json").read_text()) == {}
    st = json.loads((d / "stats.json").read_text())
    assert st["provenance"]["codes"]["confidence"] == 0.8 and st["provenance"]["use"]["reader"] == "B"


def test_a_new_solve_publishes_again_and_an_unchanged_one_does_not(city):
    assemble.run_series(S)
    solved.publish()
    before = (assemble.out_dir(S) / "parcels.fgb").stat().st_mtime_ns
    assert assemble.publish_current(S)
    solved.publish()
    assert (assemble.out_dir(S) / "parcels.fgb").stat().st_mtime_ns == before
    _solve(["RH-1", "RH-3"], [0, 1, 1, 1, 1], [("1000001", (0, 0, 140, 100)), ("1000002", (140, 0, 400, 100)),
                                              ("1001001", (0, 200, 50, 300)), ("1001001", (50, 200, 100, 300)),
                                              ("9999001", (1000, 1000, 1100, 1100))])
    assert not assemble.publish_current(S)
    solved.publish()
    assert _published().loc["1001001", "code"] == "RH-3"


def test_a_solve_of_other_lots_than_the_tables_stops_the_publish(city):
    assemble.run_series(S)
    np.savez_compressed(solved.run_file(), names=np.array(["RH-1"]), pred=np.array([0, 0]))
    with pytest.raises(ValueError, match="sfzmaps solver lots"):
        solved.publish(force=True)


def test_the_stage_command_publishes_and_fails_on_a_code_with_no_category(city):
    from typer.testing import CliRunner

    from sfzmaps import cli

    assemble.run_series(S)
    res = CliRunner().invoke(cli.app, ["solver", "publish"])
    assert res.exit_code == 0, res.output
    assert "the read's lots 5, 0 of them with no code; codes with no category 0" in res.output
    _solve(["RH-1", "M-2 (MB)"], [0, 1, 1, 1, 1], [("1000001", (0, 0, 140, 100)), ("1000002", (140, 0, 400, 100)),
                                                 ("1001001", (0, 200, 50, 300)), ("1001001", (50, 200, 100, 300)),
                                                 ("9999001", (1000, 1000, 1100, 1100))])
    res = CliRunner().invoke(cli.app, ["solver", "publish"])
    assert res.exit_code == 1 and "M-2 (MB)" in res.output


# --- any series: its solve under its root ---------------------------------------------------

@pytest.mark.parametrize("name, year, published", [
    ("1R", 1921, "1R"), ("C2", 1964, "C-2"), ("R-1-D", 1960, "R-1-D"), ("NOE VALLEY NCD", 1990, "NOE VALLEY NCD"),
    ("HAYES NCD", 1990, "HAYES-GOUGH NCD"), ("UPR MARKET", 1996, "UPPER MARKET NCD")])
def test_a_solved_code_is_spelled_and_categorised_in_its_series_era(name, year, published):
    assert solved.published_code(name, year) == published
    assert category_for(published, year) != "unknown"


@pytest.mark.parametrize("name, year", [("R-C-1", 1960), ("VALENCIA NCT", 1990), ("RM", 1980)])
def test_a_code_the_era_has_no_category_for_stops_the_publish(name, year):
    with pytest.raises(ValueError, match="no category"):
        solved.published_code(name, year)


E = "1960"
# 1960's lots: the city fixture's, and a Port lot (block 9900, unzoned until 1969)
ERA_PARCELS = {**PARCELS, "9900001": (600, 0, 700, 100)}


def _solve_at(root, names, pred, table, posterior=None, scored=None, wrong=None):
    """A series' solver files under `root`, in 2000's layout: unscored unless `scored` and `wrong` are given."""
    (root / "lots").mkdir(parents=True, exist_ok=True)
    (root / "solve" / "runs").mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"blklot": [b for b, _ in table], "zoning": [""] * len(table)},
                     geometry=[box(*g) for _, g in table], crs="EPSG:2227").to_parquet(root / "lots" / "table.parquet")
    run = {"names": np.array(names), "pred": np.array(pred)}
    if posterior is not None:
        run["posterior"] = np.array(posterior, float)
    np.savez_compressed(root / "solve" / "runs" / "final.npz", **run)
    np.savez_compressed(root / "lots" / "lots.npz", scored=np.array(scored or [False] * len(table)),
                        pairs=np.array([[0, 1, 1, 0], [2, 3, 1, 0], [3, 4, 1, 0]]))
    report = {"measures": {"every_lot": {"wrong_lots": wrong}}} if wrong is not None else {"measures": {}}
    (root / "solve" / "report.json").write_text(json.dumps(report))


# 1000001 under R-1 (0.9); 1000002 40% R-1, 60% C2 (0.6); 1001001 20% M-1 at 0.8 and 80% M-1 at 0.4; 1001002 under
# R-1-D (0.7); 1002001 under none, nearest C2; the Port lot under none, nearest the blank solve lot, which takes no part
ERA_TABLE = [("a", (0, 0, 140, 100)), ("b", (140, 0, 200, 100)), ("c", (0, 200, 10, 300)), ("d", (10, 200, 50, 300)),
             ("e", (50, 200, 100, 300)), ("f", (650, 0, 700, 50))]
ERA_SOLVE = (["R-1", "C2", "M-1", "R-1-D", ""], [0, 1, 2, 2, 3, 4], [0.9, 0.6, 0.8, 0.4, 0.7, np.nan])


@pytest.fixture
def era(city, tmp_path, monkeypatch):
    """1960 read by the old reader (its own codes) and published from that read, and its solve at the default root of
    a series other than 2000."""
    monkeypatch.setattr(paths, "SOLVER_DIR", tmp_path / "solver")
    parcels = gpd.GeoDataFrame({"blklot": list(ERA_PARCELS), "block_num": [b[:4] for b in ERA_PARCELS],
                                "active": True}, geometry=[box(*g) for g in ERA_PARCELS.values()], crs="EPSG:2227")
    monkeypatch.setattr(assemble, "load_parcels", lambda *a, **k: parcels.copy())
    raw = pd.DataFrame({"blklot": list(ERA_PARCELS), "code": pd.Series(["R-2"] * 6, dtype=object),
                        "height_code": pd.Series([None] * 6, dtype=object), "confidence": 0.5,
                        "source_sheet": ["s1", "s1", "s2", "s2", "s3", "s3"]})
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: raw.copy())
    _solve_at(solved.default_root(E), *ERA_SOLVE[:2], ERA_TABLE, posterior=ERA_SOLVE[2])
    assemble.run_series(E)                  # 1960 is not in SOLVED: the old reader's map is published
    return solved.default_root(E)


def test_another_series_is_published_from_its_solve_with_its_posterior(era):
    assert era == paths.SOLVER_DIR / E
    d = solved.publish(E)
    g = gpd.read_file(d / "parcels.fgb").set_index("blklot")
    assert g["code"].fillna("").to_dict() == {"1000001": "R-1", "1000002": "C-2", "1001001": "M-1",
                                              "1001002": "R-1-D", "1002001": "C-2", "9900001": ""}   # the Port: unzoned
    conf = g["confidence"].to_dict()
    assert conf["1000001"] == 0.9 and conf["1000002"] == 0.6 and conf["1001002"] == 0.7
    assert conf["1001001"] == pytest.approx(0.2 * 0.8 + 0.8 * 0.4)    # weighted by each solve lot's overlap
    assert conf["1002001"] == 0.6                                      # the nearest solve lot's
    assert g.loc["1000002", "category"] == "commercial_light" and g.loc["1000001", "source_sheet"] == "s1"
    assert json.loads((d / "unknown_codes.json").read_text()) == {}
    codes = json.loads((d / "stats.json").read_text())["provenance"]["codes"]
    assert codes["source"] == "the 1960 solver's run final" and codes["confidence_from"] == "posterior"
    assert "confidence" not in codes


def test_a_scored_solve_gives_its_share_right_even_with_a_posterior(era):
    _solve_at(era, *ERA_SOLVE[:2], ERA_TABLE, posterior=ERA_SOLVE[2], scored=[True] * 4 + [False] * 2, wrong=1)
    d = solved.publish(E)
    g = gpd.read_file(d / "parcels.fgb").set_index("blklot")
    assert g.loc["1001001", "confidence"] == 0.75 and g.loc["1000002", "confidence"] == 0.75     # 1 wrong of 4
    assert json.loads((d / "stats.json").read_text())["provenance"]["codes"]["confidence"] == 0.75


def test_a_solve_with_no_score_and_no_posterior_stops_the_publish(era):
    _solve_at(era, *ERA_SOLVE[:2], ERA_TABLE)
    with pytest.raises(ValueError, match="no score .* and no posterior"):
        solved.publish(E, force=True)


def test_a_solve_under_another_root_is_published_from_there(era, tmp_path):
    other = tmp_path / "elsewhere"
    _solve_at(other, ["R-3"], [0] * 6, ERA_TABLE, posterior=[0.5] * 6)
    d = solved.publish(E, root=other)
    assert set(gpd.read_file(d / "parcels.fgb")["code"].dropna()) == {"R-3"}
    assert not solved.is_solved(E)                 # only for the length of the publish


def test_a_series_in_solved_hands_its_publish_to_its_own_stage(era, monkeypatch):
    monkeypatch.setattr(solved, "SOLVED", solved.SOLVED | {E})
    for n in assemble.PUBLISHED:
        (assemble.out_dir(E) / n).unlink()
    assemble.run_series(E, force=True)            # reads, and leaves the publish to solver-publish-1960
    assert not any((assemble.out_dir(E) / n).exists() for n in assemble.PUBLISHED)
    assemble.republish(E)                          # `districts --republish`: from the solve
    assert gpd.read_file(assemble.out_dir(E) / "parcels.fgb").set_index("blklot").loc["1000002", "code"] == "C-2"


def test_the_publish_command_takes_any_zoning_series(era):
    from typer.testing import CliRunner

    from sfzmaps import cli

    res = CliRunner().invoke(cli.app, ["solver", "publish", "--series", E])
    assert res.exit_code == 0, res.output
    # the Port lot has no code in 1960 and doesn't count against it
    assert "the read's lots 6, 0 of them with no code; codes with no category 0; confidence from the solve's " \
           "posterior" in res.output
    assert CliRunner().invoke(cli.app, ["solver", "publish", "--series", "1960-height"]).exit_code == 2
