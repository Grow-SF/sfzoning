import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from sfzmaps.solver import city, energy, grid, labels, lots, paths, solve, table

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


def _write_walls(folder, odds) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for q in paths.QUARTERS:
        np.savez(folder / f"walls_{q}.npz", lo_ab=np.asarray(odds, np.float32))


@pytest.fixture
def synthetic_city(tmp_path, monkeypatch):
    """A city of 40 lots (8 columns x 5 rows) in three districts by column, one print in each district read right
    with trust 0.95, every label staying on its own lot, and walls the reader never saw (NaN: a line everywhere).
    Returns the City."""
    for name in ("LOTS_DIR", "PRINTS_DIR", "LABELS_DIR", "WALLS_DIR", "SOLVE_DIR", "TABLE_DIR"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
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
    np.savez(paths.SOLVE_DIR / "placement.npz", label=np.tile([0, 1, 2], 4), lot=np.tile(under, 4),
             chance=np.ones(12), stay=np.ones(12, bool), holdout=np.repeat(np.arange(4), 3))
    _write_walls(paths.WALLS_DIR, np.full(len(pairs), np.nan))
    city.city.cache_clear()
    yield city.city()
    city.city.cache_clear()


def test_true_border_walls_recover_the_table(synthetic_city, tmp_path):
    C = synthetic_city
    assert C.n == 40 and len(set(C.table.tolist())) == 3
    # the walls the reader never saw are a line everywhere, so the table's own borders decide
    solve.solve_quarter("tb", "nw", true_borders=True)
    np.testing.assert_array_equal(city.load_run("tb-nw"), C.table)
    # the same borders read as odds from another walls folder
    both = C.known[C.a] & C.known[C.b]
    odds = np.where(both & (C.table[C.a] != C.table[C.b]), 6.0, -6.0)
    _write_walls(tmp_path / "true-walls", odds)
    solve.solve_quarter("walls", "se", walls_dir=tmp_path / "true-walls")
    np.testing.assert_array_equal(city.load_run("walls-se"), C.table)
    # discrete and seeded: a second solve gives every lot the same code
    solve.solve_quarter("walls-again", "se", walls_dir=tmp_path / "true-walls")
    np.testing.assert_array_equal(city.load_run("walls-again-se"), city.load_run("walls-se"))


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
        city.save_run(city.run_file(f"j-{q}"), np.array([f"RH-{100 * k + i}" for i in range(C.n)]))
    solve.join("j")
    assert set(C.quarter.tolist()) == {0, 1, 2, 3}
    assert city.load_run("j").tolist() == [f"RH-{100 * C.quarter[i] + i}" for i in range(C.n)]
