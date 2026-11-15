import json
import shutil

import cv2
import geopandas as gpd
import numpy as np
import pytest
import shapely
from scipy import ndimage

from sfzmaps import config
from sfzmaps.solver import city, grid, lots, page, paths, strokes, textgrid, wallevidence, walls
from sfzmaps.transform import GCP, SheetTransform

LOT_PX = 15         # the synthetic lots are squares of 15 px (60 ft)
SPLIT_COL, SPLIT_ROW = 5220, 5593   # the grid px holding grid.QUARTER_SPLIT (col 5220.2, row 5593.1)


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




# --- the detector's text chance on the grid, and the candidate features -----------------------------------------


def _text_folders(tmp_path, monkeypatch):
    """Point the data, work, page, solver and walls folders into tmp_path, with no cut boxes."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(paths, "SOLVER_DIR", tmp_path / "solver")
    monkeypatch.setattr(paths, "PAGE_DIR", tmp_path / "page")
    monkeypatch.setattr(paths, "WALLS_DIR", tmp_path / "walls")
    (tmp_path / "cuts.json").write_text("{}")
    monkeypatch.setattr(paths, "CUTS_FILE", tmp_path / "cuts.json")
    (tmp_path / "page").mkdir()
    (tmp_path / "page" / "frames.json").write_text("{}")


SHEET_PX = 200      # the synthetic sheets' pages: 200 x 200 px of 4 ft
OFFSET_FT = 0.4     # ... set off so that a grid px reads a page px whose own map point is 1.6 ft west and north of it


def _page_point(col0, row0, x, y) -> np.ndarray:
    """The map point of page px (x, y) of the sheet placed at grid (col0, row0) (see `_sheet`)."""
    return np.c_[np.ravel(grid.X0 + grid.FT_PER_PX * col0 + OFFSET_FT + 4 * x),
                 np.ravel(grid.Y1 - grid.FT_PER_PX * row0 - OFFSET_FT - 4 * y)]


def _sheet(tmp_path, name, col0, row0, page_image, chances):
    """A sheet at 4 ft per page px whose page px (x, y) is the map point `_page_point(col0, row0, x, y)`: grid px
    (col0 + x, row0 + y) has its centre at page (x + 0.4, y + 0.4), so it reads page px (x, y), whose own map point is
    1.6 ft west and 1.6 ft north of the grid px's centre.  Framed by its whole page, with each quarter's chance of
    text (`chances`, page px)."""
    x0, y1 = _page_point(col0, row0, 0, 0)[0]
    corners = [(0, 0), (SHEET_PX, 0), (0, SHEET_PX), (SHEET_PX, SHEET_PX)]
    placement = SheetTransform([GCP(x, y, x0 + 4 * x, y1 - 4 * y, "manual") for x, y in corners], 1.0)
    warp = tmp_path / "data" / "warp" / paths.SERIES / name
    warp.mkdir(parents=True)
    (warp / "transform.json").write_text(json.dumps(placement.to_json()))
    work = tmp_path / "work" / paths.SERIES / name
    work.mkdir(parents=True)
    cv2.imwrite(str(work / "page.png"), page_image)
    frames = json.loads((tmp_path / "page" / "frames.json").read_text())
    last = SHEET_PX - 1
    frames[name] = [[0, 0], [last, 0], [last, last], [0, last]]
    (tmp_path / "page" / "frames.json").write_text(json.dumps(frames))
    for q, chance in chances.items():
        folder = tmp_path / "solver" / f"text-{q}" / "chance"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{name}.png"), chance)


def test_out_of_fold_text_takes_each_pixels_own_quarter_model(tmp_path, monkeypatch):
    """Review Focus 4: every grid px takes the chance of the model that held out the page px it reads: the quarter of
    that page px's own map point, as the detector held it out (`textnet.data.page_quarters`), even on the row and the
    column by the split where the grid px's centre lies in another quarter.  Where two sheets inked a px the larger
    chance wins; a sheet that covers a px without inking it gives nothing; a px no sheet inked is 0.  Nothing is
    written."""
    _text_folders(tmp_path, monkeypatch)
    model = {"nw": 11, "ne": 22, "sw": 33, "se": 44}
    col_a, col_b, row0 = SPLIT_COL - 100, SPLIT_COL, SPLIT_ROW - 100      # sheet a straddles the split
    rise = np.broadcast_to((np.arange(SHEET_PX) // 20).astype(np.uint8), (SHEET_PX, SHEET_PX))   # +1 every 20 px
    page_a = np.zeros((SHEET_PX, SHEET_PX), np.uint8)                      # all ink ...
    page_a[:, 50:55] = 255                                                 # ... but a paper stripe
    _sheet(tmp_path, f"{paths.SERIES}-p01", col_a, row0, page_a, {q: v + rise for q, v in model.items()})
    page_b = np.full((SHEET_PX, SHEET_PX), 255, np.uint8)
    page_b[:, :50] = 0                                                     # sheet b inks its west quarter only
    _sheet(tmp_path, f"{paths.SERIES}-p02", col_b, row0, page_b,
           {q: np.full((SHEET_PX, SHEET_PX), 200, np.uint8) for q in model})
    # what each grid px of the window should get, from the placement alone
    r, c = np.mgrid[row0:row0 + SHEET_PX, col_a:col_b + SHEET_PX]
    xa, xb, y = c - col_a, c - col_b, r - row0                             # the page px each sheet reads
    qa = grid.quarter_of(_page_point(col_a, row0, xa, y)).reshape(c.shape)
    qb = grid.quarter_of(_page_point(col_b, row0, xb, y)).reshape(c.shape)
    centre_quarter = grid.quarter_of(grid.to_map(np.c_[c.ravel() + 0.5, r.ravel() + 0.5])).reshape(c.shape)
    inked_a = (xa < SHEET_PX) & ~((xa >= 50) & (xa < 55))
    inked_b = (xb >= 0) & (xb < 50)
    by_index = np.array([model[q] for q in paths.QUARTERS])

    def expected(quarters):
        wanted = [paths.QUARTERS.index(q) for q in quarters]
        a = np.where(inked_a & np.isin(qa, wanted), by_index[qa] + xa // 20, 0)
        b = np.where(inked_b & np.isin(qb, wanted), 200, 0)
        return np.maximum(a, b).astype(np.uint8)

    window = np.s_[row0:row0 + SHEET_PX, col_a:col_b + SHEET_PX]
    every = textgrid.text_chance()
    assert every.shape == (grid.HEIGHT, grid.WIDTH) and every.dtype == np.uint8
    np.testing.assert_array_equal(every[window], expected(paths.QUARTERS))
    assert every.sum(dtype=np.int64) == every[window].sum(dtype=np.int64)     # nothing outside the sheets
    for qi, q in enumerate(paths.QUARTERS):                 # sheet a's own px: model q's chance in quarter q
        mine = inked_a & ~inked_b & (qa == qi)
        assert mine.sum() > 1000
        np.testing.assert_array_equal(every[window][mine] - xa[mine] // 20, model[q])
    # the row by the split: its grid px centres are south of it, the page px they read north: the north models
    moved = inked_a & ~inked_b & (qa != centre_quarter)
    assert moved.sum() > 100 and (r[moved] == SPLIT_ROW).all()
    np.testing.assert_array_equal(every[window][moved] - xa[moved] // 20, by_index[qa[moved]])
    assert (by_index[qa[moved]] != by_index[centre_quarter[moved]]).all()
    assert (every[window][inked_a & inked_b] == 200).all() and (every[window][inked_a & ~inked_b] < 200).all()
    # the px some sheet inked are the merged page's ink, laid by the merge's own sampler
    ink, covered = np.zeros(every.shape, bool), np.zeros(every.shape, np.uint8)
    frames = json.loads((tmp_path / "page" / "frames.json").read_text())
    for name in paths.sheet_names():
        folder = config.DATA_DIR / "warp" / paths.SERIES / name
        tr = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
        pg = cv2.imread(str(config.WORK_DIR / paths.SERIES / name / "page.png"), cv2.IMREAD_GRAYSCALE)
        page.lay_sheet(tr, pg, page.crop_mask(pg.shape, frames[name], []), ink, covered,
                       page.grid_box(tr, np.array(frames[name]), ink.shape))
    np.testing.assert_array_equal(every > 0, ink)
    assert not paths.WALLS_DIR.exists()                     # computed, not written
    # a look at some quarters: the others' page px give 0
    some = textgrid.text_chance(["nw", "se"])
    np.testing.assert_array_equal(some[window], expected(("nw", "se")))
    assert some.sum(dtype=np.int64) == some[window].sum(dtype=np.int64)
    # a quarter without its chances: the whole set can't be made, a look without that quarter can
    shutil.rmtree(paths.chance_dir("sw"))
    with pytest.raises(FileNotFoundError, match="text-sw"):
        textgrid.text_chance()
    np.testing.assert_array_equal(textgrid.text_chance(["nw"])[window], expected(("nw",)))


def test_text_chance_is_written_whole(tmp_path):
    """The file appears under its name only once it is whole (written beside it, then renamed), and replaces an older
    one; nothing else is left in the folder."""
    chance = np.random.default_rng(4).integers(0, 256, (50, 70)).astype(np.uint8)
    folder = tmp_path / "walls"
    f = textgrid.write_text_chance(folder, np.zeros_like(chance))
    assert textgrid.write_text_chance(folder, chance) == f == folder / "text_chance.png"
    np.testing.assert_array_equal(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), chance)
    assert [p.name for p in folder.iterdir()] == ["text_chance.png"]


def _columns(name, options=()) -> np.ndarray:
    """The column numbers of one feature group of `border_features` under `options`."""
    s = wallevidence.columns(options)[name]
    return np.arange(s.start, s.stop)


STROKE_INK = np.array([i * 10 + j for i in range(6) for j in range(6)])   # stroke evidence: near x width hats
STROKE_LETTER = np.array([i * 10 + 6 for i in range(6)])                   # stroke evidence: near x letter share
PROFILE_LETTER = 160 + np.array([level * 8 + 6 for level in range(6)])     # profile: near x letter share


def test_text_ink_counts_less_than_line_ink():
    """Two borders crossed by the same ink, which the detector calls text on one and a line on the other.  The
    default features can't tell them apart (no Terra box on either); with the ink weighted by (1 - text chance) the
    text's border keeps almost no stroke evidence and the line's keeps all of it, and the text share says which is
    text."""
    lot = _lot_grid(2, 4)
    L = _lots(lot, ["RH-1"] * 8)
    ink = np.zeros(lot.shape, bool)
    ink[2:28, 13:18] = True                    # across the borders 0 | 1 and 4 | 5 (between cols 14 and 15)
    ink[2:28, 43:48] = True                    # the same across 2 | 3 and 6 | 7 (between cols 44 and 45)
    text = np.where(ink & (np.arange(lot.shape[1]) < 30), 255, 0).astype(np.uint8)
    s = strokes.strokes(ink)
    mapped = np.ones(lot.shape, bool)
    plain = wallevidence.border_features(L, mapped, s, np.zeros(len(s["wmed"])))
    weighted = wallevidence.border_features(L, mapped, s, None, options=["text_weighted"], text=text)
    rows = _pair_rows(L, np.array([0, 2]), np.array([1, 3]))        # the text's border, the line's border
    before, after = plain[rows][:, STROKE_INK].sum(1), weighted[rows][:, STROKE_INK].sum(1)
    assert before[0] == pytest.approx(before[1]) and before[1] > 0.5   # the same ink, the same evidence
    assert after[1] == pytest.approx(before[1])                         # the line keeps it
    assert after[0] < 0.05 * after[1]                                   # text counts (almost) nothing
    share = weighted[rows][:, STROKE_LETTER].sum(1)
    assert share[0] == pytest.approx(before[0]) and share[1] == 0      # the text share: all of it, none of it


def test_text_weighted_changes_only_the_stroke_evidence_and_profile():
    """text_weighted: the stroke evidence and the profile weigh the nearest stroke px by (1 - text chance) and read
    its text chance where Terra's letter share was; every other column is the default's to the bit, and with no text
    anywhere the whole matrix is the default's (with no Terra letters)."""
    L, ink, _, _ = _district_city()
    s = strokes.strokes(ink)
    mapped = np.ones(ink.shape, bool)
    plain = wallevidence.border_features(L, mapped, s, np.zeros(len(s["wmed"])))
    text = np.where(ink, np.random.default_rng(1).integers(0, 256, ink.shape), 0).astype(np.uint8)
    weighted = wallevidence.border_features(L, mapped, s, None, options=["text_weighted"], text=text)
    assert weighted.shape == plain.shape
    changed = np.r_[_columns("stroke evidence"), _columns("profile")]
    same = np.setdiff1d(np.arange(wallevidence.N_FEATURES), changed)
    np.testing.assert_array_equal(weighted[:, same], plain[:, same])
    assert not np.array_equal(weighted[:, _columns("stroke evidence")], plain[:, _columns("stroke evidence")])
    assert not np.array_equal(weighted[:, _columns("profile")], plain[:, _columns("profile")])
    no_text = wallevidence.border_features(L, mapped, s, None, options=["text_weighted"], text=np.zeros_like(text))
    np.testing.assert_array_equal(no_text, plain)


def test_text_weighted_beside_letters_keeps_terra_letter_share_too():
    """text_weighted_beside_letters: text_weighted's columns, then the default's Terra letter-share columns (stroke
    evidence and profile) to the bit."""
    L, ink, _, _ = _district_city()
    s = strokes.strokes(ink)
    mapped = np.ones(ink.shape, bool)
    rng = np.random.default_rng(2)
    letter = rng.random(len(s["wmed"]))
    text = np.where(ink, rng.integers(0, 256, ink.shape), 0).astype(np.uint8)
    plain = wallevidence.border_features(L, mapped, s, letter)
    weighted = wallevidence.border_features(L, mapped, s, None, options=["text_weighted"], text=text)
    beside = wallevidence.border_features(L, mapped, s, letter, options=["text_weighted_beside_letters"], text=text)
    n = wallevidence.N_FEATURES
    assert beside.shape[1] == n + 12 == sum(k for _, k in wallevidence.groups(["text_weighted_beside_letters"]))
    np.testing.assert_array_equal(beside[:, :n], weighted)
    np.testing.assert_array_equal(beside[:, _columns("Terra letter share", ["text_weighted_beside_letters"])],
                                  plain[:, np.r_[STROKE_LETTER, PROFILE_LETTER]])
    with pytest.raises(ValueError):
        wallevidence.border_features(L, mapped, s, letter, options=["text_weighted", "text_weighted_beside_letters"],
                                     text=text)
    with pytest.raises(ValueError):
        wallevidence.border_features(L, mapped, s, letter, options=["no_such_feature"])


def _street_city():
    """Two rows of four 15 px lots with a 15 px (60 ft) street between them.  A line is printed along the street's
    north edge in front of lots 0-1, and text of the same shape in front of lots 2-3."""
    lot = np.full((45, 60), -1, np.int32)
    lot[:15] = np.arange(60) // 15
    lot[30:] = 4 + np.arange(60) // 15
    L = _lots(lot, ["RH-1"] * 8)
    ink = np.zeros(lot.shape, bool)
    ink[15:18, 1:29] = True
    ink[15:18, 31:59] = True
    text = np.where(ink & (np.arange(60) >= 30), 255, 0).astype(np.uint8)
    return L, ink, text


def test_street_crossing_reads_ink_across_the_whole_street():
    """street_crossing: the ink met crossing the street from one lot's edge to the other's, weighted by (1 - text
    chance).  A line along the street's edge is far from the border px in the street's middle, so the default near
    hats miss it; the crossing finds it, and finds nothing where the same ink is text.  The default columns come
    first, to the bit."""
    L, ink, text = _street_city()
    s = strokes.strokes(ink)
    mapped = np.ones(ink.shape, bool)
    letter = np.zeros(len(s["wmed"]))
    plain = wallevidence.border_features(L, mapped, s, letter)
    with_crossing = wallevidence.border_features(L, mapped, s, letter, options=["street_crossing"], text=text,
                                                 ink=ink)
    n = wallevidence.N_FEATURES
    assert with_crossing.shape[1] == n + len(wallevidence.WIDTH_KNOTS)
    np.testing.assert_array_equal(with_crossing[:, :n], plain)
    line, text_row = _pair_rows(L, np.array([1, 2]), np.array([5, 6]))
    assert L.touching[line] == 0 and L.touching[text_row] == 0
    assert plain[line, 0:6].sum() < 0.05                    # no stroke within 6 ft of the street's middle
    crossing = with_crossing[:, _columns("street crossing", ["street_crossing"])]
    assert crossing[line, 0] < 0.2                          # nearly every crossing meets line ink ...
    assert crossing[line, 1:3].sum() > 0.8                  # ... about 12 ft of it (hats at 8 and 14 ft)
    assert crossing[text_row, 0] == 1.0                     # text only: no line ink across
    with pytest.raises(ValueError):
        wallevidence.border_features(L, mapped, s, letter, options=["street_crossing"], text=text)   # no ink given


def test_context_tells_the_touching_kind_and_the_street_width():
    """context: touching pairs get hats of how their centre line runs along the lots' long axes (0 side by side, 0.5
    between, 1 back to back); pairs across a street get hats of the street's width.  The default design comes first,
    to the bit."""
    def box(x0, y0, x1, y1):            # ft from the grid's corner, y down
        return shapely.box(grid.X0 + x0, grid.Y1 - y1, grid.X0 + x1, grid.Y1 - y0)
    geometries = np.array([box(0, 0, 25, 100),            # 0
                           box(25, 0, 50, 100),           # 1: side by side with 0
                           box(0, 100, 25, 200),          # 2: back to back with 0
                           box(0, 268.75, 25, 368.75)],   # 3: across a 68.75 ft street from 2
                          dtype=object)
    lot, _ = lots.lot_rasters(gpd.GeoDataFrame(geometry=geometries, crs="EPSG:2227"), (100, 20))
    L = _lots(lot, ["RH-1"] * 4)
    context = walls.context(geometries, L)
    side, back, street = _pair_rows(L, np.array([0, 0, 2]), np.array([1, 2, 3]))
    kinds, widths = slice(0, 3), slice(3, 3 + len(walls.STREET_WIDTH_KNOTS))
    np.testing.assert_allclose(context[side, kinds], [1, 0, 0])
    np.testing.assert_allclose(context[back, kinds], [0, 0, 1])
    assert context[[side, back], widths].sum() == 0 and context[street, kinds].sum() == 0
    np.testing.assert_allclose(context[street, widths], walls.STREET_WIDTH_KNOTS == 68.75)
    F = np.random.default_rng(3).random((len(L.a), wallevidence.N_FEATURES)).astype(np.float32)
    log_area = np.log(np.full(L.n, 2500.0))
    plain = walls.design(F, L, log_area)
    with_context = walls.design(F, L, log_area, context)
    np.testing.assert_array_equal(with_context[:, :plain.shape[1]], plain)
    np.testing.assert_array_equal(with_context[:, plain.shape[1]:], context)


def test_no_city_limit_leaves_the_limit_open():
    """no_city_limit (an ablation): the city limit is neither a closing's end nor an enclosure wall, so a line
    stopping short of the mapped edge stays open; the columns that read neither are the default's to the bit."""
    lot = _lot_grid(8, 8)
    L = _lots(lot, ["RH-1"] * 64)
    edge = 7 * LOT_PX
    bar = np.zeros(lot.shape, bool)
    bar[4 * LOT_PX - 3:4 * LOT_PX + 4, :edge - 8] = True
    mapped = np.zeros(lot.shape, bool)
    mapped[:, :edge] = True
    s = strokes.strokes(bar)
    found = wallevidence.closings(s, lot, mapped, city_limit_wall=False)
    assert not (found["tgt"] == len(s["wmed"])).any()
    letter = np.zeros(len(s["wmed"]))
    plain = wallevidence.border_features(L, mapped, s, letter)
    open_limit = wallevidence.border_features(L, mapped, s, letter, options=["no_city_limit"])
    rows = _pair_rows(L, 3 * 8 + np.arange(6), 4 * 8 + np.arange(6))
    enclosure = _columns("enclosure")
    assert plain[rows][:, enclosure[_level(6.0, 60.0)]].all()       # the default: closed on the city limit
    assert not open_limit[rows][:, enclosure].any()                 # without it: open at every level
    same = np.r_[_columns("stroke evidence"), _columns("whole line"), _columns("straight closing"),
                 _columns("profile")]
    np.testing.assert_array_equal(open_limit[:, same], plain[:, same])


def _city_inputs(text=True) -> walls.Inputs:
    """The district city as the walls read it (`walls.Inputs`); the text chance 0 everywhere, or none."""
    L, ink, _, _ = _district_city()
    s = strokes.strokes(ink)
    return walls.Inputs(lots=L, mapped=np.ones(ink.shape, bool), strokes=s, log_area=np.full(L.n, np.log(1600.0)),
                        geometries=None, letter=np.zeros(len(s["wmed"])),
                        text=np.zeros(ink.shape, np.uint8) if text else None, ink=ink)


def test_evaluate_candidates_scores_each_configuration_from_inputs_read_once(tmp_path, monkeypatch):
    """Several configurations in one call: the inputs are read once, each configuration gets the held-out log loss
    of each quarter's fit and its own features, in the order given, and nothing is written."""
    inputs, reads = _city_inputs(), []
    monkeypatch.setattr(walls, "load_inputs", lambda configs: reads.append(configs) or inputs)
    monkeypatch.setattr(paths, "WALLS_DIR", tmp_path / "walls")
    base, crossing = walls.evaluate_candidates([[], ["street_crossing"]])
    assert len(reads) == 1 and [list(c) for c in reads[0]] == [[], ["street_crossing"]]
    for result in (base, crossing):
        assert set(result["quarters"]) == set(paths.QUARTERS)
        for q in paths.QUARTERS:
            assert 0 < result["quarters"][q]["held_out_log_loss"] < 1
    assert base["candidates"] == [] and crossing["candidates"] == ["street_crossing"]
    assert crossing["features"] == base["features"] + 2 * len(wallevidence.WIDTH_KNOTS)
    assert not (tmp_path / "walls").exists()
    with pytest.raises(ValueError):
        walls.evaluate_candidates([["text_weighted", "text_weighted_beside_letters"]])
    with pytest.raises(TypeError):
        walls.evaluate_candidates(["street_crossing"])      # a configuration is a list of names


def test_build_walls_writes_the_text_chance_it_read(tmp_path, monkeypatch):
    """build_walls writes the walls, the candidates in evaluation.json and, when a candidate reads it, the text chance
    it fitted with; the default set reads none and writes none."""
    monkeypatch.setattr(paths, "WALLS_DIR", tmp_path / "walls")
    inputs = _city_inputs()
    inputs.text[inputs.ink] = 7
    monkeypatch.setattr(walls, "load_inputs", lambda configs: inputs)
    walls.build_walls(1, ["street_crossing"])
    assert json.loads((tmp_path / "walls" / "evaluation.json").read_text())["candidates"] == ["street_crossing"]
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "walls" / "text_chance.png"), cv2.IMREAD_GRAYSCALE),
                                  inputs.text)
    shutil.rmtree(tmp_path / "walls")
    monkeypatch.setattr(walls, "load_inputs", lambda configs: _city_inputs(text=False))
    walls.build_walls(1)
    assert sorted(p.name for p in (tmp_path / "walls").iterdir()) == ["evaluation.json"] + [
        f"walls_{q}.npz" for q in sorted(paths.QUARTERS)]
