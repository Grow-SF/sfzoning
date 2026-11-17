import json
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from sfzmaps.solver import (borders, city, energy, grid, labels, lots, measures, paths, placement, report, runs, solve,
                            table)

SHAPE = (70, 100)                       # grid rows, columns: 280 x 400 ft
COLUMNS, ROWS, SIDE = 8, 5, 40.0        # 40 lots, 40-ft squares, touching
DISTRICT = ("RH-1",) * 3 + ("RH-2",) * 3 + ("NC-1",) * 2     # each column's district
LABELLED = (1, 4, 6)                    # the column of each district's one print, in the middle row
NAMES = np.array(["NC-1", "RH-1", "RH-2"])


def _lot(column: int, row: int) -> int:
    return row * COLUMNS + column


def _quarter(column: int, row: int) -> int:
    """nw 0, ne 1, sw 2, se 3: the left four columns are west, the top two rows north."""
    return (row >= 2) * 2 + (column >= 4)


def _write_readings(q: str, names, print_index, code_index, trust) -> None:
    np.savez(paths.LABELS_DIR / f"readings_{q}.npz", names=np.asarray(names), print_index=np.asarray(print_index),
             code_index=np.asarray(code_index), trust=np.asarray(trust, float))


def _write_streetside(q: str, lot_b, on_street, freeway, r_a: float, r_b: float) -> None:
    np.savez(paths.LABELS_DIR / f"streetside_{q}.npz", lotB=np.asarray(lot_b, np.int64),
             onst=np.asarray(on_street, bool), isfw=np.asarray(freeway, bool), rA_st=r_a, rB_st=r_b, rA_fw=0.5,
             rB_fw=0.5)


def _write_prints(lab_lot, street) -> None:
    n = len(lab_lot)
    pd.DataFrame({"sheet": ["p01"] * n, "x": np.zeros(n), "y": np.zeros(n),
                  "lab_lot": np.asarray(lab_lot, np.int32), "street": np.asarray(street, bool)}).to_parquet(
        paths.PRINTS_DIR / "prints.parquet")


UNDER = [_lot(c, 2) for c in LABELLED]     # the lot under each print's text


def _solved(run: str, q: str) -> np.ndarray:
    """The codes of the run's solve that held out quarter q."""
    R = np.load(city.run_file(run, f"-{q}"), allow_pickle=True)
    return R["names"].astype(str)[R["pred"]]


def _write_walls(folder, odds) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for q in paths.QUARTERS:
        np.savez(folder / f"walls_{q}.npz", lo_ab=np.asarray(odds, np.float32))


def _write_placement(lot_under, walls_dir=None, run=runs.FINAL_RUN) -> None:
    """The run's placement: every label stays on the lot under its text, for each held-out model, fitted with the
    walls of walls_dir."""
    n = len(lot_under)
    folder = runs.solve_dir(run)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "placement.npz", label=np.tile(np.arange(n), 4), lot=np.tile(lot_under, 4),
             chance=np.ones(4 * n), stay=np.ones(4 * n, bool), holdout=np.repeat(np.arange(4), n),
             walls=city.walls_digest(walls_dir))


@pytest.fixture
def synthetic_city(tmp_path, monkeypatch):
    """A city of 40 lots (8 columns x 5 rows) in three districts by column, one print in each district read right
    with trust 0.95, every label staying on its own lot, and walls the reader never saw (NaN: a line everywhere).
    Returns the City."""
    for module, name in ((paths, "PAGE_DIR"), (paths, "LOTS_DIR"), (paths, "PRINTS_DIR"), (paths, "LABELS_DIR"),
                         (runs, "WALLS_DIR"), (runs, "SOLVE_DIR"), (runs, "EXPERIMENTS_DIR"), (paths, "TABLE_DIR")):
        monkeypatch.setattr(module, name, tmp_path / name.lower())
        (tmp_path / name.lower()).mkdir()
    monkeypatch.setattr(grid, "HEIGHT", SHAPE[0])
    monkeypatch.setattr(grid, "WIDTH", SHAPE[1])
    cells = [(c, r) for r in range(ROWS) for c in range(COLUMNS)]
    boxes = [shapely.box(grid.X0 + 40 + c * SIDE, grid.Y1 - 40 - (r + 1) * SIDE,
                         grid.X0 + 40 + (c + 1) * SIDE, grid.Y1 - 40 - r * SIDE) for c, r in cells]
    t = gpd.GeoDataFrame({"blklot": [f"{r:02d}{c:02d}" for c, r in cells], "zoning": [DISTRICT[c] for c, _ in cells]},
                         geometry=boxes, crs="EPSG:2227")
    known, codes = table.known_and_codes(t)
    lot, grown = lots.lot_rasters(t, SHAPE)
    pairs = lots.neighbour_pairs(lot, grown)
    np.savez(paths.LOTS_DIR / "lots.npz", lot=lot, grown=grown, pairs=pairs, scored=known, known=known,
             quarter=np.array([_quarter(c, r) for c, r in cells], np.int8), table=codes)
    t.to_parquet(paths.LOTS_DIR / "table.parquet")
    (paths.TABLE_DIR / table.CORRECTIONS_FILE).write_text("group,verdict,pass,A,B,lots,blklots\n")
    under = [_lot(c, 2) for c in LABELLED]
    _write_prints(under, [False] * 3)
    for q in paths.QUARTERS:
        _write_readings(q, NAMES, [0, 1, 2], [1, 2, 0], [0.95] * 3)
        _write_streetside(q, [-1] * 3, [False] * 3, [False] * 3, 0.8, 0.8)
    _write_walls(runs.WALLS_DIR, np.full(len(pairs), np.nan))
    cv2.imwrite(str(paths.PAGE_DIR / "ink.png"), np.full(SHAPE, 255, np.uint8))      # a blank page
    city.city.cache_clear()
    borders.border_paths.cache_clear()
    yield city.city()
    city.city.cache_clear()
    borders.border_paths.cache_clear()


def test_true_border_walls_recover_the_table(synthetic_city, tmp_path):
    C = synthetic_city
    assert C.n == 40 and len(set(C.table.tolist())) == 3
    # the walls the reader never saw are a line everywhere, so the table's own borders decide
    _write_placement(UNDER, run="tb")
    solve.solve_quarter("tb", "nw", true_borders=True)
    np.testing.assert_array_equal(_solved("tb", "nw"), C.table)
    # the same borders read as odds from another walls folder, with a placement fitted on them
    both = C.known[C.a] & C.known[C.b]
    odds = np.where(both & (C.table[C.a] != C.table[C.b]), 6.0, -6.0)
    _write_walls(tmp_path / "true-walls", odds)
    for run in ("walls", "again"):
        _write_placement(UNDER, tmp_path / "true-walls", run)
    solve.solve_quarter("walls", "se", walls_dir=tmp_path / "true-walls")
    np.testing.assert_array_equal(_solved("walls", "se"), C.table)
    # discrete and seeded: a second solve gives every lot the same code
    solve.solve_quarter("again", "se", walls_dir=tmp_path / "true-walls")
    np.testing.assert_array_equal(_solved("again", "se"), _solved("walls", "se"))


def test_the_city_has_one_centroid_per_lot_even_for_a_lot_with_no_area(synthetic_city):
    # a lot of the table with an empty outline: its row is NaN, and every other lot keeps its own row
    t = gpd.read_parquet(paths.LOTS_DIR / "table.parquet")
    t.loc[5, "geometry"] = shapely.Polygon()
    t.to_parquet(paths.LOTS_DIR / "table.parquet")
    city.city.cache_clear()
    C = city.city()
    assert C.centroid.shape == (C.n, 2) and np.isnan(C.centroid[5]).all()
    others = np.delete(np.arange(C.n), 5)
    np.testing.assert_array_equal(C.centroid[others],
                                  shapely.get_coordinates(shapely.centroid(t.geometry.values[others])))


def test_city_scores_on_the_corrected_table_and_fits_on_the_raw_one(synthetic_city):
    # a by-eye check says lot 0 (RH-1 in the table) is printed NC-1
    (paths.TABLE_DIR / table.CORRECTIONS_FILE).write_text(
        "group,verdict,pass,A,B,lots,blklots\n0,A,1,NC-1,RH-1,1,0000\n")
    city.city.cache_clear()
    C = city.city()
    assert C.table[0] == "NC-1" and C.table_raw[0] == "RH-1"
    np.testing.assert_array_equal(np.delete(C.table, 0), np.delete(C.table_raw, 0))
    # lots.npz and table.parquet in different lot orders: refused, not misaligned
    z = dict(np.load(paths.LOTS_DIR / "lots.npz"))
    z["table"] = z["table"][::-1]
    np.savez(paths.LOTS_DIR / "lots.npz", **z)
    city.city.cache_clear()
    with pytest.raises(ValueError, match="different codes"):
        city.city()


def test_wall_odds_are_one_per_pair_and_a_pair_never_read_is_a_line(synthetic_city, tmp_path):
    odds = np.full(len(synthetic_city.a), -2.0)
    odds[3] = np.nan
    _write_walls(tmp_path / "w", odds)
    got = city.wall_odds("ne", tmp_path / "w")
    assert got[3] == city.UNREAD_ODDS and np.all(np.delete(got, 3) == -2.0)
    _write_walls(tmp_path / "short", odds[:-1])
    with pytest.raises(ValueError, match="pairs"):
        solve.solve_quarter("short", "ne", walls_dir=tmp_path / "short")


def test_energy_charges_both_wall_directions():
    # three lots in a row: no line between 0 and 1 (odds -2), a line between 1 and 2 (odds +3)
    a, b, odds = np.array([0, 1]), np.array([1, 2]), np.array([-2.0, 3.0])
    U = np.zeros((3, 2))
    U[0, 1] = 0.5

    def e(codes, same_weight=1.0):
        return energy.energy(U, a, b, odds, np.array(codes), same_weight)

    assert e([0, 0, 1]) == 0.0                         # codes change only across the line
    assert e([0, 1, 1]) == 2.0 + 3.0                   # a change with no line, and the same code across the line
    assert e([0, 0, 0]) == 3.0                         # the same code across the line only
    assert e([0, 0, 0], same_weight=0.5) == 1.5        # same-code costs scale by the same weight
    assert e([1, 0, 1]) == 0.5 + 2.0                   # a label cost and a change with no line


def test_readings_add_far_side_copies_at_the_ratio(synthetic_city):
    # print 0 on an ordinary street with a lot across (20); print 1 by a freeway; print 2 on a street with no lot
    # across; print 3 off the street
    _write_prints([8, 9, 10, 11], [True, True, True, False])
    names = ["NC-1", "RH1", "RH-2", "BLUE SUD", "NORTH OF"]
    _write_readings("nw", names, [0, 0, 1, 2, 3, 0, 3], [1, 2, 1, 1, 0, 3, 4], [0.9, 0.2, 0.8, 0.7, 0.6, 0.5, 0.4])
    _write_streetside("nw", [20, 21, -1, 22], [True, True, True, False], [False, True, False, False], 0.8, 0.6)
    _write_streetside("ne", [20, 21, -1, 22], [True, True, True, False], [False, True, False, False], 0.5, 0.7)

    rd = labels.readings("nw", vocabulary={"NC-1", "RH-1", "RH-2"})
    got = sorted(zip(rd.label.tolist(), rd.lot.tolist(), [rd.names[k] for k in rd.code], rd.trust.round(6).tolist(),
                     rd.far.tolist()))
    # the overlay (BLUE SUD) and the name of no code (NORTH OF) are dropped; print 0's readings are copied to the lot
    # across at trust x rB / rA = 0.75; the freeway's, the one with no lot across and the one off the street are not
    assert got == [(0, 8, "RH-1", 0.9, False), (0, 8, "RH-2", 0.2, False),
                   (0, 20, "RH-1", 0.675, True), (0, 20, "RH-2", 0.15, True),
                   (1, 9, "RH-1", 0.8, False), (2, 10, "RH-1", 0.7, False), (3, 11, "NC-1", 0.6, False)]
    assert rd.names[:3] == ["NC-1", "RH-1", "RH-2"]    # the one spelling
    # with the far side read more often than the near side, the copy keeps the reading's whole trust
    _write_readings("ne", names, [0], [1], [0.9])
    rd = labels.readings("ne")
    assert sorted(zip(rd.lot.tolist(), rd.trust.tolist())) == [(8, 0.9), (20, 0.9)]


def test_join_takes_each_lot_from_its_quarters_solve(synthetic_city):
    # each quarter's solve gives lot i the code RH-(100 k + i), k the quarter it held out
    C = synthetic_city
    for k, q in enumerate(paths.QUARTERS):
        city.save_run(city.run_file("j", f"-{q}"), np.array([f"RH-{100 * k + i}" for i in range(C.n)]))
    solve.join("j")
    assert set(C.quarter.tolist()) == {0, 1, 2, 3}
    assert city.load_run("j").tolist() == [f"RH-{100 * C.quarter[i] + i}" for i in range(C.n)]


def test_a_solve_refuses_a_placement_fitted_on_other_walls(synthetic_city, tmp_path):
    odds = np.full(len(synthetic_city.a), -1.0)
    _write_walls(tmp_path / "other", odds)
    _write_placement(UNDER, run="other")
    with pytest.raises(ValueError, match="other walls"):
        solve.solve_quarter("other", "nw", walls_dir=tmp_path / "other")
    # fitted on them, then the same odds written again: the same walls
    _write_placement(UNDER, tmp_path / "other", "other")
    _write_walls(tmp_path / "other", odds.copy())
    solve.solve_quarter("other", "nw", walls_dir=tmp_path / "other")
    # one quarter's odds changed: other walls
    changed = np.where(np.arange(len(odds)) == 0, 2.0, odds).astype(np.float32)
    np.savez(tmp_path / "other" / "walls_se.npz", lo_ab=changed)
    with pytest.raises(ValueError, match="other walls"):
        solve.solve_quarter("other", "nw", walls_dir=tmp_path / "other")
    # a placement that names no walls is refused too
    _write_placement(UNDER, run="old")
    P = dict(np.load(placement.placement_file(runs.solve_dir("old"))))
    del P["walls"]
    np.savez(placement.placement_file(runs.solve_dir("old")), **P)
    with pytest.raises(ValueError, match="other walls"):
        solve.solve_quarter("old", "nw")


def _right_but_lot_0(C) -> None:
    """The run "r": the table's codes, but lot 0 (RH-1, in the top-left corner) given NC-1."""
    codes = C.table.astype(object).copy()
    codes[0] = "NC-1"
    city.save_run(city.run_file("r"), codes.astype(str))


def test_the_report_gives_six_measures_with_and_without_the_data_errors_against_j7(synthetic_city):
    C = synthetic_city
    _right_but_lot_0(C)
    np.savez(measures.data_errors_file(runs.solve_dir("r")), lots=np.array([0]))
    got = report.build_report("r", placement_summary={"labels": 3}, hole_causes={"BIG AREA": 0})
    left_out, every = got["measures"]["data_errors_left_out"], got["measures"]["every_lot"]
    assert set(report.J7) <= set(left_out) and set(report.J7) <= set(every) and got["j7"] == report.J7
    # lot 0's two borders are drawn and are no table line: they count only when the data error is counted
    assert left_out["drawn_are_table_lines"] == 1.0 > every["drawn_are_table_lines"]
    assert left_out["lines_drawn"] == every["lines_drawn"] == 1.0
    # wrong lots and incidents count every scored lot either way, as J7's did
    assert left_out["wrong_lots"] == every["wrong_lots"] == 1 and left_out["incidents"] == every["incidents"] == 1
    assert got["wrong_lots_by_quarter"] == {"nw": 1, "ne": 0, "sw": 0, "se": 0}
    assert got["data_errors"] == {"parcels": 1, "j7_parcels": 33} and len(got["definition_changes"]) == 7
    # the placement's leak across the quarter split, inherited from J7, is named
    assert any("placement" in c and "quarter split" in c and "as J7's did" in c for c in got["definition_changes"])
    md = report.markdown(got)
    assert "96.3%" in md and "8,625" in md and all(change in md for change in got["definition_changes"])


def test_a_report_written_again_at_another_time_is_the_same_bytes(synthetic_city, monkeypatch):
    import time
    _right_but_lot_0(synthetic_city)
    report.write_report(report.build_report("r", placement_summary={"labels": 3}, hole_causes={}))
    before = {f: (runs.solve_dir("r") / f).read_bytes() for f in ("report.json", "report.md")}
    monkeypatch.setattr(time, "time", lambda: 4e9)
    monkeypatch.setattr(time, "perf_counter", lambda: 4e9)
    city.city.cache_clear()
    report.write_report(report.build_report("r", placement_summary={"labels": 3}, hole_causes={}))
    assert {f: (runs.solve_dir("r") / f).read_bytes() for f in before} == before
    assert json.loads(before["report.json"])["run"] == "r"


def test_the_trust_check_reads_each_label_by_the_fit_that_held_its_lot_out(synthetic_city):
    # the labels' lots are in sw (print 0) and se (prints 1, 2); the other fits say print 0 is NC-1 at 0.99
    for q in ("nw", "ne", "se"):
        _write_readings(q, NAMES, [0, 1, 2], [0, 2, 0], [0.99, 0.95, 0.95])
    _write_readings("sw", NAMES, [0, 1, 2], [1, 2, 0], [0.6, 0.95, 0.95])
    got = report.top_reading_calibration()
    assert got["labels"] == 3 and got["right"] == 3
    bins = {(b["from"], b["to"]): b for b in got["bins"]}
    assert bins[(0.5, 0.7)]["labels"] == 1 and bins[(0.5, 0.7)]["right"] == 1.0
    assert bins[(0.9, 0.97)]["labels"] == 2
    assert got["used"] == list(energy.TOP_READING_FIT)


def test_the_report_scores_the_walls_by_quarter_against_the_scoring_table(synthetic_city, tmp_path):
    C = synthetic_city
    both = C.known[C.a] & C.known[C.b]
    _write_walls(tmp_path / "true", np.where(both & (C.table[C.a] != C.table[C.b]), 6.0, -6.0))
    before = report.walls_evaluation(tmp_path / "true")["held_out_log_loss"]
    assert before == pytest.approx({q: np.log1p(np.exp(-6.0)) for q in paths.QUARTERS}, abs=1e-6)
    # a by-eye correction puts lot 0 (nw) in NC-1: its two borders become table lines the walls call open, in nw only
    (paths.TABLE_DIR / table.CORRECTIONS_FILE).write_text(
        "group,verdict,pass,A,B,lots,blklots\n0,A,1,NC-1,RH-1,1,0000\n")
    city.city.cache_clear()
    after = report.walls_evaluation(tmp_path / "true")["held_out_log_loss"]
    assert after["nw"] > before["nw"] and all(after[q] == before[q] for q in ("ne", "sw", "se"))


def test_a_quarter_log_is_the_same_bytes_at_another_time(synthetic_city, capsys, monkeypatch):
    # what solve_quarter prints is the quarter's log (runs/RUN-Q.log, in the stage's output)
    import itertools
    import time
    _write_placement(UNDER, run="t")
    solve.solve_quarter("t", "nw")
    first = capsys.readouterr().out
    ticks = itertools.count(1e9, 1000.0)              # the clock jumps 1000 s at every reading
    monkeypatch.setattr(time, "time", lambda: next(ticks))
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    solve.solve_quarter("t", "nw")
    assert first and capsys.readouterr().out == first


def test_an_experiment_writes_only_in_its_own_folder(synthetic_city):
    _write_placement(UNDER, run="trial")
    solve.solve_quarter("trial", "sw", keep_costs=True)
    assert sorted(p.name for p in (runs.EXPERIMENTS_DIR / "trial" / "runs").iterdir()) == [
        "trial-sw-costs.npz", "trial-sw.npz"]
    assert not any(runs.SOLVE_DIR.iterdir())
    # each run's data errors are its own: the stage's list leaves nothing out of an experiment's score
    np.savez(measures.data_errors_file(runs.SOLVE_DIR), lots=np.array([0, 1]))
    assert measures.data_error_lots(runs.solve_dir("trial")).sum() == 0
    assert measures.data_error_lots(runs.solve_dir(runs.FINAL_RUN)).sum() == 2


def test_the_stage_run_is_refused_other_walls_the_true_borders_or_no_compound_fusion(synthetic_city, tmp_path):
    _write_walls(tmp_path / "other", np.full(len(synthetic_city.a), -1.0))
    for kwargs in ({"walls_dir": tmp_path / "other"}, {"true_borders": True}, {"compound_fusion": False}):
        with pytest.raises(ValueError, match="solver-solve stage"):
            solve.solve_and_report(runs.FINAL_RUN, **kwargs)
        with pytest.raises(ValueError, match="solver-solve stage"):
            solve.solve_quarter(runs.FINAL_RUN, "nw", **kwargs)
    # nothing was written: no part-built folder, nothing in the stage's
    assert not any(runs.SOLVE_DIR.iterdir()) and not runs.SOLVE_DIR.with_name(".solve.part").exists()


def test_the_cli_keeps_hand_runs_out_of_the_stage_run(synthetic_city):
    from typer.testing import CliRunner

    from sfzmaps import cli
    for args in (["placement", "final"], ["holes", "final", "--save-data-errors"]):
        got = CliRunner().invoke(cli.app, ["solver", *args])
        assert got.exit_code != 0 and "solver-solve stage's run" in got.output, args
    assert not any(runs.SOLVE_DIR.iterdir())


def test_the_solve_and_the_placement_run_in_a_process_pinned_to_one_thread(synthetic_city, monkeypatch):
    """`sfzmaps solver solve` (the placement, the report, ...) and `placement` compute in a process started with one
    thread for the maths libraries: pinned, they compute there; not, they start themselves again pinned with the same
    arguments.  A run they refuse is refused at once, in this process."""
    from typer.testing import CliRunner

    from sfzmaps import cli
    from sfzmaps.solver import threads
    solved, placed, reruns = [], [], []
    monkeypatch.setattr(solve, "solve_and_report", lambda run, walls, true_borders, compound_fusion=False:
                        solved.append((run, walls, true_borders, compound_fusion)) or {"run": run})
    monkeypatch.setattr(report, "markdown", lambda found: "the report")
    monkeypatch.setattr(placement, "fit", lambda folder, walls: placed.append((folder, walls)))
    monkeypatch.setattr(threads, "run_pinned", lambda argv: reruns.append(argv) or 0)
    runner = CliRunner()
    for k, v in threads.ONE_THREAD.items():
        monkeypatch.setenv(k, v)
    assert runner.invoke(cli.app, ["solver", "solve", "final"]).exit_code == 0
    assert runner.invoke(cli.app, ["solver", "placement", "trial", "--walls", "w"]).exit_code == 0
    assert runner.invoke(cli.app, ["solver", "solve", "trial", "--no-compound-fusion"]).exit_code == 0
    assert solved == [("final", None, False, True), ("trial", None, False, False)]
    assert placed == [(runs.solve_dir("trial"), Path("w"))] and not reruns
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert runner.invoke(cli.app, ["solver", "solve", "trial", "--walls", "w", "--true-borders"]).exit_code == 0
    assert runner.invoke(cli.app, ["solver", "placement", "trial"]).exit_code == 0
    assert runner.invoke(cli.app, ["solver", "solve", "trial", "--no-compound-fusion"]).exit_code == 0
    assert reruns == [["solver", "solve", "trial", "--walls", "w", "--true-borders", "--compound-fusion"],
                      ["solver", "placement", "trial"], ["solver", "solve", "trial", "--no-compound-fusion"]]
    for args in (["solve", "final", "--walls", "w"], ["solve", "walls-context"], ["placement", "final"],
                 ["solve", "final", "--no-compound-fusion"]):
        assert runner.invoke(cli.app, ["solver", *args]).exit_code != 0, args
    assert len(reruns) == 3 and len(solved) == 2 and len(placed) == 1

FAKE_QUARTER = '''
import os, sys
import numpy as np
run, q, folder = sys.argv[3], sys.argv[4], sys.argv[sys.argv.index("--folder") + 1]
os.makedirs(os.path.join(folder, "runs"), exist_ok=True)
np.savez(os.path.join(folder, "runs", f"{run}-{q}.npz"), names=np.array(["RH-1"]),
         pred=np.zeros(int(os.environ["FAKE_LOTS"]), np.int64))
print(f"{q}: solved, threads {os.environ.get('VECLIB_MAXIMUM_THREADS')} {os.environ.get('OMP_NUM_THREADS')}")
print(f"{q}: a library's warning", file=sys.stderr)
'''


def test_a_quarters_log_holds_what_it_prints_and_its_warnings_go_to_the_console(synthetic_city, tmp_path,
                                                                                 monkeypatch, capfd):
    # each quarter's process (a stand-in that writes its solve) runs pinned to one thread; its log (runs/RUN-Q.log, a
    # stage output) holds what it prints, never a library's warning on stderr, which goes to the console
    import sys
    script = tmp_path / "fake_quarter.py"
    script.write_text(FAKE_QUARTER)
    monkeypatch.setattr(solve.threads, "sfzmaps_command", lambda: [sys.executable, str(script)])
    monkeypatch.setenv("FAKE_LOTS", str(synthetic_city.n))
    solve.solve_all("t", folder=tmp_path / "out")
    for q in paths.QUARTERS:
        assert (tmp_path / "out" / "runs" / f"t-{q}.log").read_text() == f"{q}: solved, threads 1 1\n"
    err = capfd.readouterr().err
    assert all(f"{q}: a library's warning" in err for q in paths.QUARTERS)
    assert city.load_run("t", tmp_path / "out").tolist() == ["RH-1"] * synthetic_city.n


def test_the_compound_fusion_option_adds_its_move_to_the_search_loop_and_eight_rounds(monkeypatch):
    """solve_joint's loop: region, compound and expansion moves, at most 4 rounds; with compound_fusion (the default)
    the compound fusion move after the compound moves, at most 8 rounds.  Off, the move is never called."""
    from sfzmaps.solver import search
    calls = []
    for name in ("region_moves", "compound_moves", "compound_fusion_moves", "expansion_moves"):
        monkeypatch.setattr(search, name, lambda *args, name=name, **kw: calls.append(name) or args[5])
    falling = iter(range(10 ** 6, 0, -1))                # the energy falls at every look: the loop runs to its cap
    monkeypatch.setattr(solve.E, "energy", lambda *args: float(next(falling)))
    rd = labels.Readings(np.array([0]), np.array([0]), np.array([0.9]), np.array([0]), np.array([False]),
                         ["RH-1", "RH-2"])
    a, b, odds = np.array([0]), np.array([1]), np.array([-1.0])
    for on, rounds in ((False, 4), (True, 8), (None, 8)):
        calls.clear()
        kw = {} if on is None else {"compound_fusion": on}
        solve.solve_joint(2, a, b, odds, rd, log=lambda s: None, **kw)
        on = on is not False
        one_round = ["region_moves", "compound_moves"] + ["compound_fusion_moves"] * on + ["expansion_moves"]
        assert calls == ["expansion_moves"] * 2 + one_round * rounds     # the two expansion stages, then the loop


FAKE_QUARTER_ARGS = '''
import os, sys
import numpy as np
run, q, folder = sys.argv[3], sys.argv[4], sys.argv[sys.argv.index("--folder") + 1]
os.makedirs(os.path.join(folder, "runs"), exist_ok=True)
np.savez(os.path.join(folder, "runs", f"{run}-{q}.npz"), names=np.array(["RH-1"]),
         pred=np.zeros(int(os.environ["FAKE_LOTS"]), np.int64))
print(f"{q}: " + " ".join(a for a in sys.argv[5:] if a.startswith("--") and a != "--folder"))
'''


def test_each_quarters_process_gets_the_compound_fusion_option(synthetic_city, tmp_path, monkeypatch):
    import sys
    script = tmp_path / "fake_quarter.py"
    script.write_text(FAKE_QUARTER_ARGS)
    monkeypatch.setattr(solve.threads, "sfzmaps_command", lambda: [sys.executable, str(script)])
    monkeypatch.setenv("FAKE_LOTS", str(synthetic_city.n))
    for on, flags in ((False, "--no-compound-fusion"), (True, "--compound-fusion")):
        solve.solve_all("t", compound_fusion=on, folder=tmp_path / "out")
        for q in paths.QUARTERS:
            assert (tmp_path / "out" / "runs" / f"t-{q}.log").read_text() == f"{q}: {flags}\n"


def test_the_cli_passes_the_compound_fusion_option_to_a_quarters_solve(synthetic_city, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli
    got = []
    monkeypatch.setattr(solve, "solve_quarter", lambda *args, **kw: got.append((args, kw)))
    for extra, on in (([], True), (["--no-compound-fusion"], False), (["--compound-fusion"], True)):
        assert CliRunner().invoke(cli.app, ["solver", "solve-quarter", "trial", "nw", *extra]).exit_code == 0
        assert got[-1][1]["compound_fusion"] is on
