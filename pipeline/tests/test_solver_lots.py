import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sfzmaps.solver import grid, lots, table as table_mod


def _lot(c0, r0, c1, r1):
    """A square lot whose corners are the grid corners (c0, r0) and (c1, r1)."""
    (x0, y0), (x1, y1) = grid.to_map([c0, r0]), grid.to_map([c1, r1])
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _px(c0, r0, c1, r1):
    """A lot covering exactly the grid px c0..c1-1, r0..r1-1: its corners are 0.1 px inside their edges, because a
    corner exactly on a px centre's boundary is up to fillPoly's edge rule."""
    return _lot(c0 + 0.1, r0 + 0.1, c1 - 0.1, r1 - 0.1)


def _table(geoms, zoning=None):
    z = zoning if zoning is not None else ["RH-1"] * len(geoms)
    return gpd.GeoDataFrame({"zoning": z, "geometry": geoms}, crs="EPSG:2227")


SHAPE = (12, 20)


def test_lot_raster_fills_largest_first_and_grows_to_nearest():
    # lot 0 is the small one inside lot 1, so the order of the rows must not decide who is on top
    t = _table([_px(8, 4, 12, 8), _px(0, 0, 20, 12)])
    lot, grown = lots.lot_rasters(t, SHAPE)
    assert lot.dtype == np.int32 and grown.dtype == np.int32
    assert (lot[4:8, 8:12] == 0).all()
    assert (lot == 0).sum() == 16 and (lot == 1).sum() == 20 * 12 - 16
    far = _table([_px(2, 2, 6, 6), _px(14, 2, 18, 6)])
    lot, grown = lots.lot_rasters(far, SHAPE)
    assert (lot[:, 6:14] == -1).all() and (lot[2:6, 2:6] == 0).all()
    assert (grown[:, :9] == 0).all() and (grown[:, 11:] == 1).all()   # cols 9 and 10 are the tie, either way


def test_a_lots_raster_is_the_grid_pixels_whose_centres_fall_inside_it():
    # corners at fractions of a pixel: the pixels with centres (c + .5, r + .5) in [2.3, 6.6) x [1.2, 4.7)
    t = _table([_lot(2.3, 1.2, 6.6, 4.7)])
    lot, _ = lots.lot_rasters(t, SHAPE)
    cols = [c for c in range(SHAPE[1]) if 2.3 <= c + 0.5 < 6.6]
    rows = [r for r in range(SHAPE[0]) if 1.2 <= r + 0.5 < 4.7]
    want = np.zeros(SHAPE, bool)
    want[np.ix_(rows, cols)] = True
    assert ((lot == 0) == want).all()
    # a whole-pixel square covers exactly its pixels
    lot, _ = lots.lot_rasters(_table([_px(3, 2, 7, 5)]), SHAPE)
    assert ((lot == 0) == np.pad(np.ones((3, 4), bool), ((2, 7), (3, 13)))).all()


def test_pairs_count_touching_and_street_pixels():
    # A | B share a 4 px edge; C is across a 3 px gap from B
    t = _table([_px(2, 2, 6, 6), _px(6, 2, 10, 6), _px(13, 2, 17, 6)])
    lot, grown = lots.lot_rasters(t, SHAPE)
    pairs = lots.neighbour_pairs(lot, grown)
    assert pairs.dtype == np.int64 and pairs.shape[1] == 4
    keys = [(int(a), int(b)) for a, b, *_ in pairs]
    assert keys == sorted(keys) and all(a < b for a, b in keys)
    got = {(int(a), int(b)): (int(tw), int(st)) for a, b, tw, st in pairs}
    assert set(got) == {(0, 1), (1, 2)}
    assert got[(0, 1)] == (4, SHAPE[0] - 4)      # the edge itself; the rest of the line to the grid's edges is street
    assert got[(1, 2)] == (0, SHAPE[0])          # the whole line between them runs through the gap


def test_scored_needs_known_code_and_majority_mapped():
    lot = np.full((4, 10), -1, np.int32)
    lot[:, 0:2] = 0       # 8 px, all mapped
    lot[:, 3:5] = 1       # 8 px, half mapped: not more than half
    lot[:, 6:8] = 2       # 8 px, 5 mapped
    lot[:, 8:10] = 3      # 8 px, all mapped but not known
    mapped = np.zeros(lot.shape, bool)
    mapped[:, 0:2] = True
    mapped[:2, 3:5] = True
    mapped[:, 6] = True
    mapped[0, 7] = True
    mapped[:, 8:10] = True
    known = np.array([True, True, True, False, True])    # lot 4 has no px
    s = lots.scored_lots(lot, mapped, known)
    assert s.tolist() == [True, False, True, False, False]


def test_quarter_of_uses_the_split():
    sx, sy = lots.QUARTER_SPLIT
    pts = np.array([[sx - 1, sy + 1], [sx, sy], [sx - 1, sy - 1], [sx + 1, sy - 1], [sx, sy + 5]])
    q = lots.quarter_of(pts)
    assert q.tolist() == [0, 1, 2, 3, 1] and q.dtype == np.int8


def test_an_empty_lot_keeps_every_later_lot_in_its_own_quarter():
    from shapely.geometry import Polygon
    sx, sy = grid.QUARTER_SPLIT
    geoms = [box(sx - 20, sy + 10, sx - 10, sy + 20), Polygon(), box(sx + 10, sy - 20, sx + 20, sy - 10),
             box(sx + 10, sy + 10, sx + 20, sy + 20)]
    c = lots.lot_centroids(geoms)
    assert c.shape == (4, 2) and np.isnan(c[1]).all()
    assert lots.quarter_of(c)[[0, 2, 3]].tolist() == [0, 3, 1]       # nw, se, ne: lot order kept past the empty one


def test_lots_table_codes_follow_table_row_order(tmp_path, monkeypatch):
    t = _table([_px(2, 2, 6, 6), _px(6, 2, 10, 6), _px(13, 2, 17, 6)], ["rh1", "RM-2/RH-1", " nc-3 "])
    monkeypatch.setattr(lots.table, "load_table", lambda: t)
    monkeypatch.setattr(lots.page, "load_sheets", lambda: np.ones(SHAPE, np.uint8))
    monkeypatch.setattr(grid, "HEIGHT", SHAPE[0])
    monkeypatch.setattr(grid, "WIDTH", SHAPE[1])
    monkeypatch.setattr(lots.paths, "LOTS_DIR", tmp_path)
    summary = lots.build_lots()
    z = np.load(tmp_path / "lots.npz")
    back = gpd.read_parquet(tmp_path / "table.parquet")
    known, codes = table_mod.known_and_codes(back)
    assert z["table"].tolist() == codes.tolist() == ["RH-1", "", "NC-3"]
    assert z["known"].tolist() == known.tolist() == [True, False, True]
    assert z["scored"].tolist() == [True, False, True]
    assert back["zoning"].tolist() == t["zoning"].tolist()
    assert summary["scored"] == 2
    loaded = lots.load_lots()
    assert loaded.n == 3 and len(loaded.a) == len(loaded.border_ft)
