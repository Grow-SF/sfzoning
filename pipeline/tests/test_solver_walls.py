import json

import numpy as np
from scipy import ndimage

from sfzmaps import config
from sfzmaps.solver import city, grid, lots, paths, strokes, wallevidence, walls
from sfzmaps.transform import GCP, SheetTransform

LOT_PX = 15         # the synthetic lots are squares of 15 px (60 ft)


def _lots(lot: np.ndarray, codes, quarter=None) -> lots.Lots:
    """A Lots from a lot raster (every px may be a lot, or -1 for street) and each lot's code ("" unknown)."""
    idx = ndimage.distance_transform_edt(lot < 0, return_distances=False, return_indices=True)
    grown = lot[idx[0], idx[1]]
    pairs = lots.neighbour_pairs(lot, grown)
    codes = np.asarray(codes, dtype=object).astype(str)
    known = codes != ""
    n = len(codes)
    quarter = np.zeros(n, np.int8) if quarter is None else np.asarray(quarter, np.int8)
    return lots.Lots(lot=lot, grown=grown, a=pairs[:, 0], b=pairs[:, 1], touching=pairs[:, 2], street=pairs[:, 3],
                     scored=known, known=known, quarter=quarter, codes=codes)


def _lot_grid(rows: int, cols: int, size: int = LOT_PX) -> np.ndarray:
    """A raster of rows x cols square lots of `size` px, numbered row by row."""
    r, c = np.mgrid[0:rows * size, 0:cols * size]
    return ((r // size) * cols + c // size).astype(np.int32)


def _pair_rows(L: lots.Lots, first, second) -> np.ndarray:
    """The pair rows of (first[i], second[i]) (either order)."""
    a, b = np.minimum(first, second), np.maximum(first, second)
    key = L.a.astype(np.int64) * L.n + L.b
    rows = np.searchsorted(key, a.astype(np.int64) * L.n + b)
    assert (key[rows] == a.astype(np.int64) * L.n + b).all()
    return rows


def _level(width_ft: float, closing_ft: float) -> int:
    """The column of `enclosure`'s output for one level."""
    return (wallevidence.ENCLOSURE_WIDTHS.index(width_ft) * len(wallevidence.ENCLOSURE_CLOSINGS)
            + wallevidence.ENCLOSURE_CLOSINGS.index(closing_ft))


def _separation(L: lots.Lots, ink: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    s = strokes.strokes(ink)
    sid = wallevidence.stroke_raster(s, ink.shape)
    found = wallevidence.closings(s, L.lot, mapped)
    return wallevidence.enclosure(L, mapped, sid, s["wmed"], found)


def _district_city():
    """24 x 24 lots of 10 px in 4 x 4 districts of 6 x 6 lots; every two neighbouring districts have different codes,
    and a 6 px wide line is printed along every district border.  Quarters: the city's four quarters."""
    size, n_side = 10, 24
    lot = _lot_grid(n_side, n_side, size)
    r, c = np.divmod(np.arange(n_side * n_side), n_side)
    codes = np.array([f"C{(i + 2 * j) % 4}" for i, j in zip(r // 6, c // 6)])
    quarter = np.where(r < 12, 0, 2) + (c >= 12)
    ink = np.zeros(lot.shape, bool)
    for k in (60, 120, 180):
        ink[:, k - 3:k + 3] = True
        ink[k - 3:k + 3, :] = True
    return _lots(lot, codes, quarter), ink, r, c


def test_inked_border_scores_higher_than_bare_border(tmp_path):
    L, ink, r, c = _district_city()
    s = strokes.strokes(ink)
    F = wallevidence.border_features(L, np.ones(ink.shape, bool), s, np.zeros(len(s["wmed"])))
    assert F.shape == (len(L.a), wallevidence.N_FEATURES)
    X = walls.design(F, L, np.full(L.n, np.log(1600.0)))
    odds, evaluation = walls.fit_walls(X, L)
    district = L.codes[L.a] != L.codes[L.b]
    # a bare border at least two lots from every printed line: both lots' rows and columns are 2-3 inside a district
    inner = np.isin(r % 6, (2, 3)) & np.isin(c % 6, (2, 3))
    bare = inner[L.a] & inner[L.b]
    assert district.sum() > 50 and bare.sum() > 20
    for q, name in enumerate(("nw", "ne", "sw", "se")):
        lo = odds[name]
        assert lo.shape == (len(L.a),)
        held = (L.quarter[L.a] == q) & (L.quarter[L.b] == q)
        assert lo[held & district].min() > lo[held & bare].max()
        assert lo[held & district].mean() > 0 > lo[held & ~district].mean()
        assert evaluation["quarters"][name]["held_out_pairs"] == int((held & L.known[L.a] & L.known[L.b]).sum())
    # written one value per pair row, as the solver reads them
    walls.write_walls(tmp_path, odds, evaluation)
    for name in ("nw", "ne", "sw", "se"):
        np.testing.assert_allclose(city.wall_odds(name, tmp_path), odds[name].astype(np.float32))


def test_fit_ignores_pairs_in_its_quarter():
    rng = np.random.default_rng(0)
    n = 60
    lot = np.arange(n, dtype=np.int32).reshape(6, 10)
    quarter = rng.integers(0, 4, n)
    codes = rng.choice(["RH-1", "RH-2", "NC-1"], n).astype(object)
    codes[[3, 17]] = ""
    L = _lots(lot, codes, quarter)
    X = rng.normal(size=(len(L.a), 5))
    q = 1
    train = walls.training_pairs(L, q)
    touches_q = (L.quarter[L.a] == q) | (L.quarter[L.b] == q)
    assert not train[touches_q].any()                       # a pair with either lot in q is never trained on
    assert train[~touches_q & L.known[L.a] & L.known[L.b]].all()
    y = walls.table_lines(L)
    w = walls.fit_quarter(X, y, train)
    # every row touching q changed, features and target: the fit is the same to the bit
    X2, L2 = X.copy(), _lots(lot, codes.copy(), quarter)
    X2[touches_q] = rng.normal(size=(int(touches_q.sum()), 5))
    in_q = np.flatnonzero(quarter == q)
    L2.codes[in_q] = np.where(L2.codes[in_q] == "RH-1", "M-1", "RH-1")
    assert not np.array_equal(walls.table_lines(L2), y)
    np.testing.assert_array_equal(walls.fit_quarter(X2, walls.table_lines(L2), walls.training_pairs(L2, q)), w)
    # ... while a change outside q does change it
    X3 = X.copy()
    X3[~touches_q] += 1.0
    assert not np.array_equal(walls.fit_quarter(X3, y, train), w)


def test_enclosure_separates_two_walled_regions():
    lot = _lot_grid(8, 8)
    L = _lots(lot, ["RH-1"] * 64)
    ink = np.zeros(lot.shape, bool)
    # a ring around lots 2-5 of each row and column, drawn 3 px off the lot lines into the outer lots on the
    # right and bottom and into the inner lots on the left and top: each lot sits where most of its px are
    lo_, hi_ = 2 * LOT_PX + 3, 6 * LOT_PX + 3
    for k in (lo_, hi_):
        ink[lo_ - 3:hi_ + 4, k - 3:k + 4] = True
        ink[k - 3:k + 4, lo_ - 3:hi_ + 4] = True
    sep = _separation(L, ink, np.ones(lot.shape, bool))
    r, c = np.divmod(np.arange(64), 8)
    inner = (r >= 2) & (r <= 5) & (c >= 2) & (c <= 5)
    across = inner[L.a] != inner[L.b]
    assert across.sum() == 16
    assert sep[across].all()                     # every level: the ring is wide and closed
    assert not sep[~across].any()


def test_city_limit_is_a_wall():
    """A printed line that stops 8 px short of the mapped area's edge closes on it, as it closes on a printed line
    standing where the edge is (along a lot line, so its own ends close on the raster's edge)."""
    lot = _lot_grid(8, 8)
    L = _lots(lot, ["RH-1"] * 64)
    edge = 7 * LOT_PX                                        # the lot line between lot columns 6 and 7
    bar = np.zeros(lot.shape, bool)
    bar[4 * LOT_PX - 3:4 * LOT_PX + 4, :edge - 8] = True    # along the lot line between lot rows 3 and 4
    limit = np.zeros(lot.shape, bool)
    limit[:, :edge] = True                                   # the mapped area ends there
    upright = bar.copy()
    upright[:, edge - 3:edge + 4] = True                     # ... or a printed line stands there instead
    sep_limit = _separation(L, bar, limit)
    sep_ink = _separation(L, upright, np.ones(lot.shape, bool))
    rows = _pair_rows(L, 3 * 8 + np.arange(6), 4 * 8 + np.arange(6))   # the lots on either side of the bar
    for wl in wallevidence.ENCLOSURE_WIDTHS:
        assert not sep_limit[rows, _level(wl, 0.0)].any()   # no closings: the gap is open
        assert sep_limit[rows, _level(wl, 60.0)].all()      # closed on the city limit
    np.testing.assert_array_equal(sep_limit[rows], sep_ink[rows])
    s = strokes.strokes(bar)
    found = wallevidence.closings(s, lot, limit)
    assert (found["tgt"] == len(s["wmed"])).all()            # both ends close on the city limit


def test_letter_share_counts_stroke_points_in_text():
    # two strokes, one along row 5 and one along row 20; text covers columns 0-9 of row 5's stroke only
    ink = np.zeros((30, 40), bool)
    ink[4:7, 2:38] = True
    ink[19:22, 2:38] = True
    s = strokes.strokes(ink)
    text = np.zeros(ink.shape, bool)
    text[:10, :20] = True
    share = wallevidence.letter_share(s, text)
    rows = np.array([grid.to_grid(s["xy"][s["off"][k]:s["off"][k + 1]])[:, 1].mean() for k in range(len(share))])
    top = rows < 10
    assert top.sum() == 1 and len(share) == 2
    assert 0.4 < share[top][0] < 0.6                        # half its points are in the text
    assert share[~top][0] == 0


def test_text_rings_drop_boxes_outside_the_frame_and_in_cuts():
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    boxes = [{"box": [10, 10, 30, 20], "angle_deg": 0},       # kept
             {"box": [150, 10, 170, 20], "angle_deg": 0},     # outside the frame
             {"box": [60, 60, 80, 70], "angle_deg": 0},       # inside a cut box
             {"box": [10, 40, 20, 80], "angle_deg": 90}]      # kept, turned
    rings = wallevidence.text_rings(boxes, quad, [[50, 50, 99, 99]])
    assert len(rings) == 2
    np.testing.assert_allclose(np.sort(rings[0][:, 0]), [10, 10, 30, 30])
    np.testing.assert_allclose(np.sort(rings[1][:, 1]), [40, 40, 80, 80])


def test_terra_text_puts_page_boxes_on_the_grid_through_the_sheet_placement(tmp_path, monkeypatch):
    """Page px (x right, y down) -> map ft (2 ft per page px, x east, y north) -> grid (col east, row south): a box
    keeps its place and its long side; a box outside the sheet's frame is left out."""
    sheet = f"{paths.SERIES}-p01"
    corners = [(0, 0), (400, 0), (0, 400), (400, 400)]
    placement = SheetTransform([GCP(x, y, grid.X0 + 2 * x, grid.Y1 - 2 * y, "manual") for x, y in corners], 1.0)
    warp = tmp_path / "data" / "warp" / paths.SERIES / sheet
    warp.mkdir(parents=True)
    (warp / "transform.json").write_text(json.dumps(placement.to_json()))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    for name, folder, content in (("PRINTS_DIR", "prints", {"terra_boxes.json": {sheet: [
            {"box": [100, 40, 140, 60], "angle_deg": 0},          # wide, near the top: grid cols 50-70, rows 20-30
            {"box": [20, 200, 30, 260], "angle_deg": 90},          # tall, lower left: cols 10-15, rows 100-130
            {"box": [100, 380, 140, 399], "angle_deg": 0}]}}),     # its centre below the frame: left out
            ("PAGE_DIR", "page", {"frames.json": {sheet: [[0, 0], [400, 0], [400, 300], [0, 300]]}})):
        (tmp_path / folder).mkdir()
        for f, data in content.items():
            (tmp_path / folder / f).write_text(json.dumps(data))
        monkeypatch.setattr(paths, name, tmp_path / folder)
    (tmp_path / "cuts.json").write_text("{}")
    monkeypatch.setattr(paths, "CUTS_FILE", tmp_path / "cuts.json")
    text = walls.terra_text((200, 100))
    rows, cols = np.nonzero(text)
    wide = rows < 50
    assert (rows[wide].min(), rows[wide].max(), cols[wide].min(), cols[wide].max()) == (20, 30, 50, 70)
    assert (rows[~wide].min(), rows[~wide].max(), cols[~wide].min(), cols[~wide].max()) == (100, 130, 10, 15)
    assert text.sum() == 11 * 21 + 31 * 6                   # two full rectangles, nothing from the third box


def test_hats_are_a_partition_of_unity():
    kn = np.array([0.0, 6.0, 14.0, 26.0])
    h = wallevidence.hats(np.array([-5.0, 0.0, 3.0, 14.0, 20.0, 99.0]), kn)
    np.testing.assert_allclose(h.sum(1), 1.0)
    np.testing.assert_allclose(h[2], [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(h[4], [0, 0, 0.5, 0.5])
    np.testing.assert_allclose(h[[0, 5]], [[1, 0, 0, 0], [0, 0, 0, 1]])


def test_closings_do_not_depend_on_the_number_of_workers():
    L, ink, _, _ = _district_city()
    ink[100:103, 30:50] = True                              # loose strokes with ends to close
    ink[150:170, 140:143] = True
    s = strokes.strokes(ink)
    mapped = np.ones(ink.shape, bool)
    one = wallevidence.closings(s, L.lot, mapped, workers=1)
    many = wallevidence.closings(s, L.lot, mapped, workers=2, band_rows=40)
    assert len(one["src"]) > 4
    for k in one:
        np.testing.assert_array_equal(one[k], many[k])
