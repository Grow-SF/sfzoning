"""The learned reader's training data (`sfzmaps.solver.hatchnet.data`): the scan as printed (a sheet scanned on its side
turned back exactly), today's blocks placed on it and labelled once where scans overlap, nothing erased from the ink,
and the lot table's one read per lot and the owner's answers."""
import dataclasses
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import shapely
from scipy import ndimage

from sfzmaps.solver import grid, hatching
from sfzmaps.solver.hatchnet import data
from sfzmaps.transform import GCP, SheetTransform

SERIES = "1921"
PAGE = (150, 100)          # an upright sheet's page px (rows, cols): portrait, as every 1921 section
FT_PER_PX = 1.1            # not a divisor of the grid's 4 ft: no page px centre lands on a grid px edge
Y_FT = 30.21               # the sheets' top edge, ft below the grid's top
INSET = 5                  # the neatline: px inside the page edge
LOTS = [                   # (rows, cols) of the solver grid (4 ft px), blklot, quarter
    ((15, 25), (8, 12), "0001001", 0),      # on A only
    ((15, 25), (24, 28), "0002001", 1),     # in the overlap of A and C
    ((15, 25), (38, 42), "0003001", 2),     # on C only
    ((28, 36), (14, 18), "0004001", 3),     # on A only
    ((28, 36), (43, 46), "0005001", 1),     # on C only
    ((28, 36), (24, 28), "0006001", 3),     # in the overlap of A and C: C owns its centre
]
TODAY = {                  # today's parcels (DataSF) that the solver's April 2000 table lacks: world box (ft from X0, Y1)
    "0003001B": (154.0, 62.0, 166.0, 98.0),     # inside lot 2
    "0003001C": (156.0, 64.0, 164.0, 96.0),     # inside lot 2
    "0004001Z": (58.0, 116.0, 70.0, 140.0),     # inside lot 3
    "0009001": (73.0, 62.0, 79.0, 98.0),        # on a street (no solver lot)
}


def placement(shape, x_ft: float, y_ft: float = Y_FT, ft_per_px: float = FT_PER_PX) -> SheetTransform:
    """North up: page px (x, y) -> (X0 + x_ft + x ft_per_px, Y1 - y_ft - y ft_per_px), fitted from the 4 corners."""
    h, w = shape
    gcps = [GCP(px, py, grid.X0 + x_ft + px * ft_per_px, grid.Y1 - y_ft - py * ft_per_px, "manual")
            for px, py in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))]
    return SheetTransform(gcps, 1.0)


def placement_turned(shape, x_ft: float, y_ft: float = Y_FT, ft_per_px: float = FT_PER_PX) -> SheetTransform:
    """The placement of `np.rot90(upright, -1)` (scanned a quarter turn clockwise: north to the right), `shape` the
    upright page's: turned px (x, y) is upright px (y, h - 1 - x)."""
    h, w = shape
    gcps = [GCP(px, py, grid.X0 + x_ft + py * ft_per_px, grid.Y1 - y_ft - (h - 1 - px) * ft_per_px, "manual")
            for px, py in ((0, 0), (h - 1, 0), (h - 1, w - 1), (0, w - 1))]
    return SheetTransform(gcps, 1.0)


def scan(seed: int = 0) -> np.ndarray:
    """A grey page as printed: diagonal hatch, a solid block, block outlines, a white text knockout with letters in
    it, and grey px of every value (a scanned edge's anti-aliasing)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:PAGE[0], 0:PAGE[1]]
    pg = np.full(PAGE, 255, np.uint8)
    pg[np.mod(x + y, 9) < 2] = 0                       # 1R-like hatch
    pg[100:130, 10:40] = 0                             # a solid (C) block
    cv2.rectangle(pg, (8, 8), (90, 60), 0, 2)          # a block outline
    pg[30:42, 30:70] = 255                             # a text knockout
    cv2.putText(pg, "AB", (32, 41), cv2.FONT_HERSHEY_PLAIN, 0.8, 0, 1)
    noise = rng.integers(0, 256, PAGE).astype(np.uint8)
    edge = rng.random(PAGE) < 0.05
    pg[edge] = noise[edge]
    return pg


def write_sheet(root, sheet: str, page: np.ndarray, tr: SheetTransform, crop: np.ndarray, reads=()):
    d = root / "work" / SERIES / sheet
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "page.png"), page)
    (d / "transform.json").write_text(json.dumps(tr.to_json()))
    parcels = [{"blklot": b, "code": c, "confidence": p, "via": v} for b, c, p, v in reads]
    (d / "district_pattern.json").write_text(json.dumps({"series": SERIES, "sheet": sheet, "parcels": parcels}))
    cut = root / "frame" / SERIES / sheet                 # the frame finder's cut: nonzero where it deleted
    cut.mkdir(parents=True)
    cv2.imwrite(str(cut / hatching.FRAME_CUT), np.where(crop, 0, 255).astype(np.uint8))


def write_lots(root):
    """The solver's lot raster (a 80 x 80 corner of its grid), quarters and table (geometry in EPSG:2227 ft)."""
    lot = np.full((80, 80), -1, np.int32)
    geoms, rows = [], []
    for i, ((r0, r1), (c0, c1), blklot, _) in enumerate(LOTS):
        lot[r0:r1, c0:c1] = i
        geoms.append(shapely.box(grid.X0 + 4 * c0, grid.Y1 - 4 * r1, grid.X0 + 4 * c1, grid.Y1 - 4 * r0))
        rows.append({"blklot": blklot, "block_num": blklot[:4]})
    d = root / "solver" / SERIES / "lots"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "lots.npz", lot=lot, quarter=np.array([q for *_, q in LOTS], np.int8))
    pd.DataFrame(rows).assign(geometry=shapely.to_wkb(np.array(geoms))).to_parquet(d / "table.parquet")
    return lot


def write_today(root):
    """Today's parcels (blklot, WKB geometry in EPSG:2227 ft), one with no geometry."""
    geoms = [shapely.box(grid.X0 + x0, grid.Y1 - y1, grid.X0 + x1, grid.Y1 - y0) for x0, y0, x1, y1 in TODAY.values()]
    wkb = list(shapely.to_wkb(np.array(geoms))) + [None]
    pd.DataFrame({"blklot": list(TODAY) + ["0009003"], "geometry": wkb}).to_parquet(root / "parcels.parquet")


STREETS = [                # today's centre lines: world (x, y) ft from (X0, Y1), x east and y south
    ((78.0, 30.0), (78.0, 190.0)),          # north-south down grid column 19, on sheet A only
    ((10.0, 162.0), (190.0, 162.0)),        # east-west along grid row 40, across both sheets
]


def write_streets(root):
    """Today's street centre lines (WKB LineStrings in EPSG:2227 ft, as data/ref/streets.parquet), one not active."""
    lines = [shapely.LineString([(grid.X0 + x, grid.Y1 - y) for x, y in pts]) for pts in STREETS]
    pd.DataFrame({"cnn": np.arange(len(lines)), "active": [True, False],
                  "geometry": list(shapely.to_wkb(np.array(lines)))}).to_parquet(root / "streets.parquet")


def sources(root, order=None) -> data.Sources:
    parcels, streets = root / "parcels.parquet", root / "streets.parquet"
    return data.Sources(series=SERIES, work=root / "work" / SERIES, solver=root / "solver" / SERIES,
                        legends=root / "legends.json", answers=root / "answers", cases=root / "cases",
                        parcels=parcels if parcels.is_file() else None, order=tuple(order) if order else None,
                        streets=streets if streets.is_file() else None, frame=root / "frame" / SERIES)


def upright_crop(shape=PAGE):
    c = np.zeros(shape, bool)
    c[INSET:shape[0] - INSET, INSET:shape[1] - INSET] = True
    return c


A_FT, C_FT = 20.37, 80.37  # the two sheets' west edges, ft east of the grid's corner


def two_sheets(root, reads_a=(), reads_c=(), page_a=None, page_c=None):
    """Sheets A (world x 20-130 ft) and C (80-190 ft) side by side, overlapping 50 ft; C is drawn above A.  Their
    pages are `scan(0)` and `scan(1)` unless given."""
    write_lots(root)
    write_sheet(root, "sheetA", scan(0) if page_a is None else page_a, placement(PAGE, A_FT),
                upright_crop(), reads_a)
    write_sheet(root, "sheetC", scan(1) if page_c is None else page_c, placement(PAGE, C_FT),
                upright_crop(), reads_c)
    return sources(root, order=("sheetA", "sheetC"))


def lot_box(i: int, x_ft: float, pad: int = 3) -> tuple[slice, slice]:
    """The page px (rows, cols) over lot i on the sheet whose west edge is `x_ft`, `pad` px beyond it each way."""
    (r0, r1), (c0, c1), *_ = LOTS[i]
    rows = slice(int(np.floor((4 * r0 - Y_FT) / FT_PER_PX)) - pad, int(np.ceil((4 * r1 - Y_FT) / FT_PER_PX)) + pad)
    cols = slice(int(np.floor((4 * c0 - x_ft) / FT_PER_PX)) - pad, int(np.ceil((4 * c1 - x_ft) / FT_PER_PX)) + pad)
    return rows, cols


def lot_px_on(src, sheet: str, i: int) -> np.ndarray:
    """Where lot i lies on the sheet's page inside its neatline (the solver's lot raster read at every page px through
    the placement): its px, whichever sheet owns them."""
    sh = data.load_sheet(src, sheet)
    lot, _ = data.solver_grid(src)
    (v,) = hatching.page_values(hatching.PageWorld(sh.transform, sh.shape), sh.shape, [(lot, -1)])
    return (v == i) & sh.crop


def printed(page: np.ndarray, rows: slice, cols: slice, solid: bool = False) -> np.ndarray:
    """The page with the box printed over: solid ink, or hatch inside an outline."""
    pg = page.copy()
    y, x = np.mgrid[rows, cols]
    pg[rows, cols] = 0 if solid else np.where(np.mod(x + y, 9) < 2, 0, 255)
    if not solid:
        cv2.rectangle(pg, (cols.start, rows.start), (cols.stop - 1, rows.stop - 1), 0, 1)
    return pg


def blank(page: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    pg = page.copy()
    pg[rows, cols] = 255
    return pg


# --- the tests ------------------------------------------------------------------------------------------------------

def test_turned_sheet_matches_upright(tmp_path):
    # the same sheet scanned upright and on its side (a quarter turn clockwise: north to the right, as
    # 20180102100525362) gives the same scan and the same rasters once turned back
    page = scan(0)
    up, side = tmp_path / "up", tmp_path / "side"
    for root in (up, side):
        write_lots(root)
        write_streets(root)
    write_sheet(up, "sheet", page, placement(PAGE, 20.37), upright_crop())
    h, w = PAGE
    write_sheet(side, "sheet", np.ascontiguousarray(np.rot90(page, -1)), placement_turned(PAGE, 20.37),
                np.rot90(upright_crop(), -1))
    a = data.load_scan(SERIES, "sheet", sources(up))
    b = data.load_scan(SERIES, "sheet", sources(side))
    assert a.dtype == np.float32 and a.shape == PAGE
    assert np.array_equal(a, b)
    ra, rb = data.sheet_rasters(SERIES, "sheet", sources(up)), data.sheet_rasters(SERIES, "sheet", sources(side))
    assert (ra.turn, rb.turn) == (0, 1)
    for k in ("ink", "block", "lot", "crop", "street", "street_w", "centre"):
        assert np.array_equal(getattr(ra, k), getattr(rb, k)), k
    assert (ra.block == 1).sum() > 0 and (ra.block == 0).sum() > 0 and (ra.lot >= 0).sum() > 0
    assert (ra.street == 1).sum() > 0 and ra.centre.sum() > 0
    assert np.isclose(ra.ft_per_px, rb.ft_per_px) and np.isclose(ra.ft_per_px, FT_PER_PX)
    # a box on the scanned page lands on the same px of the turned-back page
    assert data.turn_box((10, 20, 30, 50), (w, h), 1) == (20, h - 30, 50, h - 10)
    box = data.turn_box((10, 20, 30, 50), (w, h), 1)
    side_page = np.rot90(page, -1)
    assert np.array_equal(np.rot90(side_page, 1)[box[1]:box[3], box[0]:box[2]],
                          np.rot90(side_page[20:50, 10:30], 1))


def test_overlap_owned_once(tmp_path):
    # C is drawn above A, and prints the overlap's two blocks (lots 1 and 5) solid, so it owns them by its ink (R9) as
    # well as the streets between them by the draw order
    src = two_sheets(tmp_path, page_c=printed(printed(scan(1), *lot_box(1, C_FT), solid=True), *lot_box(5, C_FT),
                                              solid=True))
    ra = data.sheet_rasters(SERIES, "sheetA", src)
    rc = data.sheet_rasters(SERIES, "sheetC", src)
    ta, tc = placement(PAGE, A_FT), placement(PAGE, C_FT)

    def at(r, tr, world):
        x, y = np.round(tr.to_pixel(np.array([world]))[0]).astype(int)
        return int(r.block[y, x]), int(r.lot[y, x])

    overlap = (grid.X0 + 105.3, grid.Y1 - 80.3)                  # inside both neatlines, on lot 1: C's
    assert at(ra, ta, overlap)[0] == 255 and at(ra, ta, overlap)[1] == -1
    assert at(rc, tc, overlap)[0] != 255 and at(rc, tc, overlap)[1] == 1
    only_a = (grid.X0 + 40.3, grid.Y1 - 80.3)
    assert at(ra, ta, only_a)[0] != 255 and at(ra, ta, only_a)[1] == 0
    # outside the neatline: unknown; inside it on no other sheet: labelled
    assert (ra.block[~ra.crop] == 255).all() and (ra.lot[~ra.crop] == -1).all()
    assert (rc.block[rc.crop] != 255).all()                      # C keeps its whole neatline
    # each place is labelled on one sheet: A's labelled px reach no further east than C's neatline begins
    ys, xs = np.nonzero(ra.block != 255)
    east = ta.to_world(np.c_[xs, ys].astype(float))[:, 0].max()
    c_west = tc.to_world(np.array([[INSET, 50.0]]))[0, 0]
    assert east < c_west and east > c_west - 2 * FT_PER_PX


def at_world(r, x_ft: float, world) -> tuple[int, int]:
    """(block, lot) of the sheet's rasters at a world point (ft from X0, Y1)."""
    x, y = np.round(placement(PAGE, x_ft).to_pixel(np.array([[grid.X0 + world[0], grid.Y1 - world[1]]]))[0])
    return int(r.block[int(y), int(x)]), int(r.lot[int(y), int(x)])


def test_blank_corner_yields_to_printing_sheet(tmp_path):
    # R9: C is drawn above A, but its page is blank paper over lot 1's block (a join corner the other sheet prints),
    # which A prints with outline and hatch: every px of the block on A is A's and none on C is C's.  The street beside
    # it, in no block, keeps the draw order (C's).  The lot table's centre sheet follows the same rule
    src = two_sheets(tmp_path, page_a=printed(scan(0), *lot_box(1, A_FT)), page_c=blank(scan(1), *lot_box(1, C_FT)))
    ra, rc = data.sheet_rasters(SERIES, "sheetA", src), data.sheet_rasters(SERIES, "sheetC", src)
    on_a, on_c = lot_px_on(src, "sheetA", 1), lot_px_on(src, "sheetC", 1)
    assert on_a.sum() > 400 and on_c.sum() > 400
    assert np.array_equal(ra.lot == 1, on_a) and (ra.block[on_a] == 1).all()
    assert not (rc.lot == 1).any() and (rc.block[on_c] == data.UNKNOWN).all()
    street = (105.3, 106.3)                                      # between lots 1 and 5: in no block
    assert at_world(ra, A_FT, street)[0] == data.UNKNOWN and at_world(rc, C_FT, street) == (0, -1)
    assert data.lot_table(SERIES, src).loc[1, "centre_sheet"] == "sheetA"


def test_both_printed_goes_to_more_ink(tmp_path):
    # both sheets print lots 1's and 5's blocks.  Lot 1's: A (drawn below) solid, C hatched: A has more ink, so the
    # block is A's.  Lot 5's: C solid, A hatched: C's
    src = two_sheets(tmp_path,
                     page_a=printed(printed(scan(0), *lot_box(1, A_FT), solid=True), *lot_box(5, A_FT)),
                     page_c=printed(printed(scan(1), *lot_box(1, C_FT)), *lot_box(5, C_FT), solid=True))
    ra, rc = data.sheet_rasters(SERIES, "sheetA", src), data.sheet_rasters(SERIES, "sheetC", src)
    for i, (win, lose) in ((1, ((ra, "sheetA"), (rc, "sheetC"))), (5, ((rc, "sheetC"), (ra, "sheetA")))):
        on_w, on_l = lot_px_on(src, win[1], i), lot_px_on(src, lose[1], i)
        assert on_w.sum() > 300 and on_l.sum() > 300, i
        assert np.array_equal(win[0].lot == i, on_w) and (win[0].block[on_w] == 1).all(), i
        assert not (lose[0].lot == i).any() and (lose[0].block[on_l] == data.UNKNOWN).all(), i
    t = data.lot_table(SERIES, src)
    assert (t.loc[1, "centre_sheet"], t.loc[5, "centre_sheet"]) == ("sheetA", "sheetC")


def test_single_candidate_unchanged(tmp_path):
    # a block inside one sheet's neatline only stays that sheet's, blank there or printed: lot 0 (on A only) blank on
    # A, lot 2 (on C only) blank on C, lots 3 (A) and 4 (C) printed
    src = two_sheets(tmp_path, page_a=blank(scan(0), *lot_box(0, A_FT)), page_c=blank(scan(1), *lot_box(2, C_FT)))
    ra, rc = data.sheet_rasters(SERIES, "sheetA", src), data.sheet_rasters(SERIES, "sheetC", src)
    for r, sheet, i in ((ra, "sheetA", 0), (rc, "sheetC", 2), (ra, "sheetA", 3), (rc, "sheetC", 4)):
        on = lot_px_on(src, sheet, i)
        other = lot_px_on(src, "sheetC" if sheet == "sheetA" else "sheetA", i)
        assert on.sum() > 100 and not other.any(), i
        assert np.array_equal(r.lot == i, on) and (r.block[on] == 1).all(), i
    t = data.lot_table(SERIES, src)
    assert list(t.loc[[0, 2, 3, 4], "centre_sheet"]) == ["sheetA", "sheetC", "sheetA", "sheetC"]


def test_every_px_owned_once(tmp_path):
    # blocks given against the draw order (lot 1's, blank on C) and with it (lot 5's, solid on C), streets by the draw
    # order: a place inside both neatlines is labelled on exactly one sheet, and a place inside one neatline on that
    # one.  Each sheet's px is carried to the other page through the placements; left out are px at a neatline's edge
    # and those whose nearest px on the other page reads another block (the two px are < 1 px apart)
    src = two_sheets(tmp_path, page_c=printed(blank(scan(1), *lot_box(1, C_FT)), *lot_box(5, C_FT), solid=True))
    r = {"sheetA": data.sheet_rasters(SERIES, "sheetA", src), "sheetC": data.sheet_rasters(SERIES, "sheetC", src)}
    tr = {"sheetA": placement(PAGE, A_FT), "sheetC": placement(PAGE, C_FT)}
    _, blocks = data.solver_grid(src)
    bid = ndimage.label(blocks)[0]
    h, w = PAGE
    inner = {k: cv2.erode(v.crop.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) for k, v in r.items()}
    checked = {}
    for me, other in (("sheetA", "sheetC"), ("sheetC", "sheetA")):
        ys, xs = np.nonzero(r[me].crop)
        world = tr[me].to_world(np.c_[xs, ys].astype(float))
        p = np.round(tr[other].to_pixel(world)).astype(int)
        on = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        held, deep = np.zeros(len(xs), bool), np.zeros(len(xs), bool)
        held[on] = r[other].crop[p[on, 1], p[on, 0]]
        deep[on] = inner[other][p[on, 1], p[on, 0]] & inner[me][ys[on], xs[on]]
        mine = r[me].block[ys, xs] != data.UNKNOWN
        assert mine[~held].all(), me                                 # inside this neatline only: this sheet's
        back = tr[other].to_world(p.astype(float))
        same = (hatching.grid_values(world[:, 0], world[:, 1], bid, fill=0) ==
                hatching.grid_values(back[:, 0], back[:, 1], bid, fill=0))
        both = deep & same
        theirs = r[other].block[p[both, 1], p[both, 0]] != data.UNKNOWN
        assert (mine[both] ^ theirs).all(), me                       # inside both: exactly one sheet's
        checked[me] = (int(both.sum()), int((mine[both] & (r[me].lot[ys[both], xs[both]] >= 0)).sum()))
    assert all(n > 2000 and lots > 300 for n, lots in checked.values()), checked
    assert not (r["sheetC"].lot == 1).any() and (r["sheetA"].lot == 1).any()
    assert not (r["sheetA"].lot == 5).any() and (r["sheetC"].lot == 5).any()


def test_lot_table_answers_and_reads(tmp_path):
    write_today(tmp_path)
    src = two_sheets(tmp_path,
                     reads_a=[("0001001", "1R", 0.8, "parcel"), ("0002001", "2R", 0.6, "parcel"),
                              ("0004001", "", 0.5, "parcel"), ("0003001", "HI", 0.9, "off_sheet"),
                              ("0006001", "HI", 0.7, "parcel"), ("0004001Z", "1R", 0.6, "parcel"),
                              ("0009001", "C", 0.9, "parcel"), ("0009002", "C", 0.9, "parcel"),
                              ("0009003", "C", 0.9, "parcel")],
                     reads_c=[("0002001", "C", 0.7, "parcel"), ("0003001", "LI", 0.9, "block"),
                              ("0005001", "U", 0.5, "parcel"), ("0003001B", "2R", 0.95, "parcel"),
                              ("0003001C", "1R", 0.5, "parcel")],
                     page_c=printed(printed(scan(1), *lot_box(1, C_FT), solid=True), *lot_box(5, C_FT), solid=True))
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "1921.json").write_text(json.dumps({"cases": [
        {"id": "c0001", "stratum": "random", "code": "1R", "blklot": "0001001", "block": "0001"},
        {"id": "c0002", "stratum": "disagree", "code": "2R", "blklot": "0002001", "block": "0002"},
        {"id": "c0003", "stratum": "random", "code": "2R", "blklot": "0004001", "block": "0004"}]}))
    (cases / "1921-likely.json").write_text(json.dumps({"cases": [
        {"id": "c0001", "stratum": "all", "code": "LI", "blklot": "0003001", "blklots": ["0003001"], "block": "0003"},
        {"id": "c0002", "stratum": "all", "code": "1R", "blklot": "0009009", "blklots": ["0009009", "0005001"],
         "block": "0005"}]}))
    answers = {"answers-1921": [("c0001", "ok", "1R"), ("c0002", "fix", "LI"), ("c0003", "canttell", "")],
               "answers-1921-likely": [("c0001", "unzoned", ""), ("c0002", "fix", "HI")]}
    for page, rows in answers.items():
        d = tmp_path / "answers" / page
        d.mkdir(parents=True)
        for case, ans, code in rows:
            (d / f"{case}.json").write_text(json.dumps({"case": case, "answer": ans, "code": code}))
    t = data.lot_table(SERIES, src)
    assert list(t["lot"]) == [0, 1, 2, 3, 4, 5]
    assert list(t["blklot"]) == [b for *_, b, _ in LOTS]
    assert list(t["quarter"]) == ["nw", "ne", "sw", "se", "ne", "se"]
    # one read per lot (R3'): a lot read on one sheet keeps it whichever sheet that is (lot 5: A read it, C owns its
    # centre); read on two, the centre's sheet picks (lot 1: C, which prints it solid, owns its centre).  Empty codes
    # and the reader's off-sheet marks are no reads.  R4: a read of today's parcel the solver's table lacks lands on the
    # solver lot under the parcel's representative point (lots 2 and 3); several on one lot from one sheet keep the
    # most confident (lot 2: 0.95 over its own 0.9 and 0.5); one on no solver lot, not in today's parcels or with no
    # geometry stays unattached
    assert list(t["read_code"]) == ["1R", "C", "2R", "1R", "U", "HI"]
    assert np.allclose(t["read_conf"].to_numpy(), [0.8, 0.7, 0.95, 0.6, 0.5, 0.7])
    assert list(t["read_sheet"]) == ["sheetA", "sheetC", "sheetC", "sheetA", "sheetC", "sheetA"]
    assert list(t["read_blklot"]) == ["0001001", "0002001", "0003001B", "0004001Z", "0005001", "0006001"]
    assert list(t["read_recovered"]) == [False, False, True, True, False, False]
    assert list(t["n_reads"]) == [1, 2, 1, 1, 1, 1]
    r = t.attrs["reads"]
    assert (r["direct"], r["recovered"]) == (6, 3)
    assert r["unattached"] == {"no_geometry": 1, "no_solver_lot": 1, "not_in_parcels": 1}
    assert (r["lots_several_reads_one_sheet"], r["lots_read_on_several_sheets"]) == (1, 1)
    # the owner's answers: ok -> the code shown, fix -> the fix, unzoned -> 2R, canttell dropped; a case whose own
    # blklot is not a solver lot lands on the lot of its footprint
    assert list(t["answer"]) == ["1R", "LI", "2R", "", "HI", ""]
    assert list(t["answer_page"]) == ["1921", "1921", "1921-likely", "", "1921-likely", ""]
    assert list(t["stratum"]) == ["random", "disagree", "likely", "", "likely", ""]


def test_no_rule_masks(tmp_path):
    # the ink is the scan's luminance, every px: hatch, solid ink, outlines, knockouts and text all kept
    src = two_sheets(tmp_path)
    page = scan(0)
    r = data.sheet_rasters(SERIES, "sheetA", src)
    assert r.ink.dtype == np.float32
    expect = page.astype(np.float32) / np.float32(255)
    assert np.array_equal(r.ink[r.crop], expect[r.crop])
    assert np.array_equal(r.ink, expect)


def test_full_tone_luminance(tmp_path):
    # a colour scan (1948) goes in as its luminance (Rec. 601), 0 = ink, not thresholded
    rgb = np.array([[[255, 255, 255], [0, 0, 0], [200, 100, 50], [12, 200, 240]]], np.uint8)
    f = tmp_path / "page.png"
    cv2.imwrite(str(f), rgb[..., ::-1])
    lum = data.load_luminance(f)
    r, g, b = (rgb[..., k].astype(np.float32) for k in range(3))
    expect = (np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * b) / np.float32(255)
    assert lum.dtype == np.float32 and lum.shape == (1, 4)
    assert np.allclose(lum, expect, atol=1e-6) and lum[0, 0] == 1.0 and lum[0, 1] == 0.0


def gap_blocks(width: int, *spans) -> np.ndarray:
    """A grid (20 rows) whose blocks are the column spans given (ends exclusive), every row: a 1-D map."""
    block = np.zeros((20, width), bool)
    for c0, c1 in spans:
        block[:, c0:c1] = True
    return block


def wline(*rc) -> shapely.LineString:
    """A centre line (EPSG:2227 ft) through the centres of grid px (row, col)."""
    return shapely.LineString([(grid.X0 + 4 * (c + 0.5), grid.Y1 - 4 * (r + 0.5)) for r, c in rc])


def centres_of(shape, *lines) -> "data.Centres":
    return data.centre_segments(np.array(lines, dtype=object), shape)


def test_corridor_between_two_blocks():
    # blocks at columns 0-9 and 20-29, a 10 px street between them, its centre line at column 14 (5 px from the west
    # block, 6 from the east): every px between the blocks is street, weighted by the learned S(d) = P(half-width > d)
    # and cut at each side's block; the half-widths learned are 5 and 6 (20 px each), so S is 1 below 5 px and 1/2 at
    # 5 px (column 19, beside the east block).  The blocks are not street; the px past the east block are ignored
    block = gap_blocks(40, (0, 10), (20, 30))
    lab = data.street_labels(block, centres_of(block.shape, wline((0, 14), (19, 14))))
    assert lab.street.dtype == np.uint8 and lab.weight.dtype == np.float16 and lab.centre.dtype == np.uint8
    assert (lab.street[:, 10:20] == 1).all()
    assert (lab.weight[:, 10:19] == 1).all() and (lab.weight[:, 19] == 0.5).all()
    assert (lab.street[block] == 0).all() and (lab.weight[block] == 1).all()
    assert (lab.street[:, 30:] == data.UNKNOWN).all() and (lab.weight[:, 30:] == 0).all()
    assert lab.centre[:, 14].all() and lab.centre.sum() == 20
    assert set(np.unique(lab.street)) <= {0, 1, data.UNKNOWN}


def shore_grid():
    """Rows 0-9 and 20-39 are blocks.  A street between them (centre row 14: 5 px to the block above, 6 below), and a
    shoreline street (centre row 42: 3 px below the block) with open water below it to the grid's edge."""
    block = np.zeros((60, 80), bool)
    block[0:10] = True
    block[20:40] = True
    return block, centres_of(block.shape, wline((14, 0), (14, 79)), wline((42, 0), (42, 79)))


def test_shoreline_open_side_is_soft():
    # every side is weighted by the learned survival S(d) = P(half-width > d): the half-widths learned are the
    # two-sided street's 5 and 6 (80 px each), so S is 1 below 5 px, 1/2 at 5 px and 0 from 6 px.  The shoreline's
    # block side is cut at its block (rows 40-41, S = 1 there); its open side follows S out to where it reaches 0, and
    # the water beyond is ignored.  An open side labels what a bounded side would
    block, centres = shore_grid()
    lab = data.street_labels(block, centres)
    assert (lab.street[40:42] == 1).all() and (lab.weight[40:42] == 1).all()
    assert (lab.street[10:20] == 1).all()
    assert (lab.weight[10:19] == 1).all() and (lab.weight[19] == 0.5).all()        # row 19: 5 px above the block
    for row in range(43, 60):
        d = row - 42
        assert np.allclose(lab.weight[row].astype(float), data.survival(lab.learned.pooled, np.full(80, d))), row
    assert (lab.street[43:48] == 1).all() and (lab.weight[43:47] == 1).all() and (lab.weight[47] == 0.5).all()
    assert (lab.street[48:] == data.UNKNOWN).all() and (lab.weight[48:] == 0).all()


def test_bridge_gets_only_soft_weights():
    # a centre line with no block on either side: only the learned survival weights, out to the largest learned
    # half-width, then ignored; nothing is not-street
    learned = data.street_labels(*shore_grid()).learned
    block = np.zeros((40, 40), bool)
    lab = data.street_labels(block, centres_of(block.shape, wline((0, 20), (39, 20))), learned=learned)
    d = np.abs(np.arange(40) - 20)
    for col in range(40):
        assert np.allclose(lab.weight[:, col].astype(float), data.survival(learned.pooled, np.full(40, d[col]))), col
    assert (lab.street[:, d <= 5] == 1).all() and (lab.street[:, d > 5] == data.UNKNOWN).all()
    assert not (lab.street == 0).any() and (lab.weight[:, d == 5] == 0.5).all()


def test_far_shore_parcel_does_not_make_full_corridor():
    # a centre line, open water, and a parcel 400 ft (100 px) away: the side is bounded, but its weights are still the
    # learned S(d), not 1; at 200 ft (50 px) about S(50 px), and they stop at the parcel
    learned = data.HalfWidths(pooled=np.arange(1.0, 121.0), own={}, report={})      # half-widths 1-120 px
    block = gap_blocks(130, (110, 130))
    lab = data.street_labels(block, centres_of(block.shape, wline((0, 10), (19, 10))), learned=learned)
    w = lab.weight.astype(float)
    assert np.allclose(w[:, 60], data.survival(learned.pooled, 50.0), atol=1e-3)
    assert np.allclose(data.survival(learned.pooled, 50.0), 70 / 120) and (w[:, 60] < 1).all()
    assert np.allclose(w[:, 109], data.survival(learned.pooled, 99.0), atol=1e-3)
    assert (lab.street[:, 11:110] == 1).all() and (lab.street[:, 110:] == 0).all()


def test_half_widths_learned_from_two_sided_px():
    # the distribution comes only from px bounded on both sides: not the shoreline's one side (3), and not a side whose
    # ray runs into a cross street's area before it meets a block (here up a cross street that ends 12 px away at a
    # block band), which measures the cross street's length, not this street's width
    block, centres = shore_grid()
    lab = data.street_labels(block, centres)
    assert np.array_equal(np.sort(lab.learned.pooled), np.r_[np.full(80, 5.0), np.full(80, 6.0)])
    block = np.zeros((30, 40), bool)
    block[0:3] = True                                         # a block band across the top
    for r0, r1 in ((3, 10), (20, 30)):
        block[r0:r1, 0:10] = True
        block[r0:r1, 20:40] = True
    lab = data.street_labels(block, centres_of(block.shape, wline((14, 0), (14, 39)), wline((3, 14), (29, 14))))
    assert lab.learned.pooled.size and set(np.unique(lab.learned.pooled)) <= {5.0, 6.0}
    i = int(np.flatnonzero((lab.sides.rows == 14) & (lab.sides.cols == 12))[0])
    assert np.isinf(lab.sides.width[i]).any()                 # its up side runs along the cross street


def test_no_centre_line_is_ignored():
    # a street between two blocks with no centre line today (columns 30-39) is ignored, not labelled not-street; the
    # one with a centre line (columns 10-19) is street.  With no centre line at all, every px off the blocks is ignored
    block = gap_blocks(60, (0, 10), (20, 30), (40, 50))
    lab = data.street_labels(block, centres_of(block.shape, wline((0, 14), (19, 14))))
    assert (lab.street[:, 10:20] == 1).all()
    assert (lab.street[:, 30:40] == data.UNKNOWN).all() and (lab.weight[:, 30:40] == 0).all()
    assert (lab.street[block] == 0).all()
    lab = data.street_labels(block, centres_of(block.shape))
    assert (lab.street[block] == 0).all() and (lab.street[~block] == data.UNKNOWN).all() and not lab.centre.any()
    assert (lab.weight[~block] == 0).all()


def test_centre_on_a_block_is_not_street():
    # a centre line crossing today's block (a park road, a street built over): the block stays not-street and the line
    # is no centre there, so it labels nothing off the block
    block = gap_blocks(40, (0, 20))
    lab = data.street_labels(block, centres_of(block.shape, wline((0, 10), (19, 10))))
    assert (lab.street[block] == 0).all() and (lab.street[~block] == data.UNKNOWN).all()
    assert not lab.centre.any()


def test_kind_pooled_unless_its_widths_differ():
    # each street type gets its own learned half-widths only when they differ from all types' by more than the DKW 95%
    # band for its number of lines; a type seen on one line is pooled
    block, _ = shore_grid()
    lines = np.array([wline((14, 0), (14, 79)), wline((42, 0), (42, 79))], dtype=object)
    centres = data.centre_segments(lines, block.shape, kinds=np.array(["AVE", "ST"]))
    lab = data.street_labels(block, centres)
    rep = lab.learned.report
    assert rep["AVE"]["own"] is False and rep["AVE"]["lines"] == 1 and rep["AVE"]["px"] == 80
    assert rep["ST"]["own"] is False and rep["ST"]["px"] == 0 and not lab.learned.own


def test_centre_lines_rasterised_on_the_grid():
    # today's centre lines (EPSG:2227 ft) on the solver's 4 ft grid, segment by segment, 8-connected, one px wide,
    # wherever they run, each px knowing its segment's direction; a line off the grid is left out
    lines = [shapely.LineString([(grid.X0 + 2, grid.Y1 - 42), (grid.X0 + 158, grid.Y1 - 42)]),       # row 10
             shapely.LineString([(grid.X0 + 30, grid.Y1 - 2), (grid.X0 + 70, grid.Y1 - 158)]),        # diagonal
             shapely.LineString([(grid.X0 - 500, grid.Y1 + 500), (grid.X0 - 100, grid.Y1 + 100)])]    # off the grid
    centres = data.centre_segments(np.array(lines, dtype=object), (40, 40))
    c = centres.seg >= 0
    assert c[10].all()
    assert np.allclose(centres.direction[centres.seg[10, 0]], [1.0, 0.0])
    assert np.allclose(centres.direction[centres.seg[39, 17]], np.array([40.0, 156.0]) / np.hypot(40, 156))
    n, _ = cv2.connectedComponents(c.astype(np.uint8), connectivity=8)
    assert n == 2                                                  # one piece (and the background)
    assert c[39, 17]                                               # the diagonal reaches the bottom row at x 70 ft


def test_street_type_from_name(tmp_path):
    # a line with no st_type takes its name's last word when that is a type other lines have (OCTAVIA ST -> ST); the
    # rest (freeways, THE EMBARCADERO) stay untyped
    f = tmp_path / "streets.parquet"
    line = shapely.to_wkb(shapely.LineString([(grid.X0, grid.Y1), (grid.X0 + 40, grid.Y1)]))
    pd.DataFrame({"st_type": ["ST", None, None, None], "geometry": [line] * 4,
                  "streetname": ["A ST", "OCTAVIA ST", "HWY 101 SOUTHBOUND", "THE EMBARCADERO"]}).to_parquet(f)
    lines = data.centre_lines(dataclasses.replace(sources(tmp_path), streets=f))
    assert list(lines.kinds) == ["ST", "ST", data.KIND_NONE, data.KIND_NONE]


def test_street_and_centre_carried_like_block(tmp_path):
    # the labels go to each page through the same placement and ownership as `block`: not-street exactly where block is
    # 1, ignored wherever block is unknown; the centre lines are drawn on the page through the placement, connected,
    # on street px only, and only on the sheet that owns them
    write_streets(tmp_path)
    src = two_sheets(tmp_path)
    for name, x_ft in (("sheetA", 20.37), ("sheetC", 80.37)):
        r = data.sheet_rasters(SERIES, name, src)
        assert r.street.dtype == np.uint8 and r.centre.dtype == np.uint8 and r.street_w.dtype == np.float16
        assert np.array_equal(r.street == 0, r.block == 1), name
        assert (r.street[r.block == data.UNKNOWN] == data.UNKNOWN).all(), name
        assert (r.street == 1).any() and (r.street[r.block == 0] == data.UNKNOWN).any(), name
        # the weight: 1 on not-street, 0 where ignored, in (0, 1] on street, soft somewhere off the lots
        w = r.street_w.astype(float)
        assert (w[r.street == 0] == 1).all() and (w[r.street == data.UNKNOWN] == 0).all(), name
        assert ((w[r.street == 1] > 0) & (w[r.street == 1] <= 1)).all() and (w[r.street == 1] < 1).any(), name
        assert r.centre.any() and (r.street[r.centre == 1] == 1).all(), name
        assert (r.centre[r.block == data.UNKNOWN] == 0).all(), name
        # the drawn centre px lie on the lines (STREETS), within a page px; the north-south line is A's alone
        ys, xs = np.nonzero(r.centre)
        world = placement(PAGE, x_ft).to_world(np.c_[xs, ys].astype(float))
        on_ew = np.abs(world[:, 1] - (grid.Y1 - 162.0)) <= FT_PER_PX
        on_ns = np.abs(world[:, 0] - (grid.X0 + 78.0)) <= FT_PER_PX
        assert (on_ew | on_ns).all() and on_ew.any(), name
        assert on_ns.any() == (name == "sheetA"), name
        assert cv2.connectedComponents(r.centre, connectivity=8)[0] == 2, name
    # with no centre lines every px off today's blocks is ignored
    bare = data.sheet_rasters(SERIES, "sheetA", dataclasses.replace(src, streets=None))
    assert not (bare.street == 1).any() and not bare.centre.any()
    assert np.array_equal(bare.street == 0, bare.block == 1)
    assert np.array_equal(bare.street_w == 1, bare.block == 1)


def test_run_writes_same_bytes(tmp_path):
    write_streets(tmp_path)
    src = two_sheets(tmp_path, reads_a=[("0001001", "1R", 0.8, "parcel")])
    (tmp_path / "legends.json").write_text(json.dumps({SERIES: {"sheet": "sheetA", "swatches": {
        "1R": [10, 20, 30, 40]}}}))

    def digest(folder):
        return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}

    one = data.run(SERIES, tmp_path / "one", workers=1, src=src)
    two = data.run(SERIES, tmp_path / "two", workers=1, src=src)
    assert digest(tmp_path / "one") == digest(tmp_path / "two")
    assert one["sheets_used"] == ["sheetA", "sheetC"]
    r = data.read_rasters(tmp_path / "one" / "sheetA.npz")
    direct = data.sheet_rasters(SERIES, "sheetA", src)
    for k in ("block", "street", "street_w", "centre"):
        assert np.array_equal(getattr(r, k), getattr(direct, k)), k
    assert (r.street == 1).any() and r.centre.any()
    lab = one["street_labels"]
    assert lab["lines"] == len(STREETS) and lab["centre_grid_px"] > 0 and "(none)" in lab["half_widths"]
    assert data.swatches(SERIES, src) == [("sheetA", "1R", (10, 20, 30, 40))]


def test_a_duplicate_scan_is_not_a_sheet(tmp_path, monkeypatch):
    # the owner's list names sheetA a duplicate of sheetC: only sheetC is read, and sheetA is reported as not used
    from sfzmaps import config
    src = two_sheets(tmp_path)
    assert data.sheet_names(src) == ["sheetA", "sheetC"]
    inputs = tmp_path / "inputs"
    (inputs / SERIES).mkdir(parents=True)
    (inputs / SERIES / "duplicate_scans.csv").write_text('sheet,kept,reason\nsheetA,sheetC,"coarser"\n')
    monkeypatch.setattr(config, "INPUTS_DIR", inputs)
    assert hatching.duplicate_scans(SERIES) == {"sheetA": "sheetC"}
    assert data.sheet_names(src) == ["sheetC"]
    assert data.not_used(src) == {"sheetA": ["a duplicate scan"]}
